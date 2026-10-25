"""AutoML científico H7 Rossmann com FLAML Tune e teste externo intocado.

Uso na raiz do repositório:
  python dados/scripts/executar_automl_rossmann_h7_flaml.py --modo verificar
  python -u dados/scripts/executar_automl_rossmann_h7_flaml.py --modo executar --tentativas 6 --tempo-algoritmo 900 --threads 2

Os artefatos são gravados em pasta própria; não sobrescreve experimento preliminar.
"""
import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'dados' / 'processados'
TRAIN = DATA / 'treinamento'
OUTPUT = DATA / 'experimentos' / 'cientifico_h7_flaml'
CORTE = pd.Timestamp('2015-01-23')
NUMERICAS = ['promocao', 'promocao_continua', 'mes_renovacao_promocao',
             'feriado', 'feriado_escolar', 'distancia_concorrente', 'dia_semana',
             'dia_mes', 'mes', 'semana_ano', 'lag_7', 'lag_14', 'lag_21',
             'lag_28', 'media_movel_7', 'media_movel_28']
CATEGORICAS = ['tipo_loja', 'sortimento']
META = ['loja_id', 'data', 'data_origem']


def carregar():
    df = pd.read_csv(DATA / 'atributos_h7.csv', parse_dates=['data', 'data_origem'], low_memory=False)
    if df.duplicated(['loja_id', 'data']).any():
        raise ValueError('Pares loja-data repetidos.')
    if not (df['data'] - df['data_origem']).eq(pd.Timedelta(days=7)).all():
        raise ValueError('Horizonte diferente de H7.')
    treino = df.loc[df['data'] <= CORTE].copy().reset_index(drop=True)
    teste = df.loc[df['data_origem'] >= CORTE].copy().reset_index(drop=True)
    if treino.empty or teste.empty or treino['data'].max() >= teste['data'].min():
        raise ValueError('Corte temporal externo inválido.')
    metadados = pd.read_csv(TRAIN / 'metadata_treino.csv', parse_dates=['data', 'data_origem'])
    if len(metadados) != len(treino) or not metadados[META].equals(treino[META]):
        raise ValueError('Metadados/índices de validação não correspondem ao dataset.')
    folds = []
    with np.load(TRAIN / 'indices_validacao_temporal.npz') as indices:
        for i in range(1, 5):
            ti = indices[f'treino_{i}'].astype(np.int64)
            vi = indices[f'validacao_{i}'].astype(np.int64)
            if (not len(ti) or not len(vi) or ti.min() < 0 or vi.min() < 0
                    or ti.max() >= len(treino) or vi.max() >= len(treino)
                    or np.intersect1d(ti, vi).size):
                raise ValueError(f'Índices inválidos na dobra {i}.')
            if treino.iloc[ti]['data'].max() > treino.iloc[vi]['data_origem'].min():
                raise ValueError(f'Vazamento temporal na dobra {i}.')
            folds.append((ti, vi))
    amostra = pd.read_csv(DATA / 'amostra_arima.csv')['loja_id'].astype(int).tolist()
    if len(amostra) != 63 or len(set(amostra)) != 63 or not set(amostra).issubset(set(teste.loja_id)):
        raise ValueError('A amostra fixa de 63 lojas é inválida.')
    return treino, teste, folds, amostra


def preparar(a, b):
    """Imputação e categorias aprendidas somente do treino da dobra."""
    a = a[NUMERICAS + CATEGORICAS].copy()
    b = b[NUMERICAS + CATEGORICAS].copy()
    mediana = pd.to_numeric(a['distancia_concorrente'], errors='coerce').median()
    if pd.isna(mediana):
        raise ValueError('Distância inteiramente ausente no treino.')
    for x in (a, b):
        for c in NUMERICAS:
            x[c] = pd.to_numeric(x[c], errors='coerce').astype('float32')
        x['distancia_concorrente'] = x['distancia_concorrente'].fillna(mediana)
        for c in CATEGORICAS:
            x[c] = x[c].fillna('desconhecido').astype(str)
    for c in CATEGORICAS:
        categorias = sorted(a[c].unique())
        a[c] = pd.Categorical(a[c], categories=categorias)
        b[c] = pd.Categorical(b[c], categories=categorias)
    x_a = pd.get_dummies(a, columns=CATEGORICAS, dtype='int8')
    x_b = pd.get_dummies(b, columns=CATEGORICAS, dtype='int8').reindex(columns=x_a.columns, fill_value=0)
    return x_a, x_b


def modelo(nome, cfg, threads):
    cfg = {k: int(v) if k in ('n_estimators', 'max_depth', 'num_leaves') else float(v)
           for k, v in cfg.items()}
    if nome == 'lightgbm':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=42, n_jobs=threads, verbosity=-1, **cfg)
    if nome == 'xgboost':
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=42, n_jobs=threads, tree_method='hist',
                            objective='reg:squarederror', **cfg)
    raise ValueError(nome)


def metricas(real, previsto):
    y = np.asarray(real, dtype=float)
    p = np.asarray(previsto, dtype=float)
    validos = y > 0
    return {'quantidade': int(len(y)),
            'rmse': float(np.sqrt(mean_squared_error(y, p))),
            'mae': float(mean_absolute_error(y, p)),
            'r2': float(r2_score(y, p)),
            'rmspe': float(np.sqrt(np.mean(((y[validos] - p[validos]) / y[validos]) ** 2))) if validos.any() else None}


def avaliar(nome, cfg, treino, dobras, threads):
    resultados = []
    soma_quadrados = 0.0
    soma_absolutos = 0.0
    total = 0
    for numero, (ti, vi) in enumerate(dobras, 1):
        a = treino.iloc[ti]
        b = treino.iloc[vi]
        x_a, x_b = preparar(a, b)
        m = modelo(nome, cfg, threads)
        m.fit(x_a, a['vendas'].to_numpy())
        pred = m.predict(x_b)
        real = b['vendas'].to_numpy(dtype=float)
        diff = real - np.asarray(pred, dtype=float)
        soma_quadrados += float(np.dot(diff, diff))
        soma_absolutos += float(np.abs(diff).sum())
        total += len(real)
        resultados.append({'dobra': numero, **metricas(real, pred)})
        del a, b, x_a, x_b, m, pred, real, diff
        gc.collect()
    return {'rmse': float(np.sqrt(soma_quadrados / total)),
            'mae': float(soma_absolutos / total), 'quantidade': total,
            'dobras': resultados}


def espaco(nome, tune):
    esp = {'n_estimators': tune.randint(80, 301),
           'learning_rate': tune.loguniform(0.015, 0.15),
           'max_depth': tune.randint(3, 10),
           'subsample': tune.uniform(0.75, 1.0),
           'colsample_bytree': tune.uniform(0.75, 1.0)}
    if nome == 'lightgbm':
        esp['num_leaves'] = tune.randint(15, 64)
    return esp


def executar():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modo', choices=['verificar', 'executar'], default='verificar')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--tentativas', type=int, default=6,
                        help='Número máximo de configurações por algoritmo, incluindo inicial.')
    parser.add_argument('--tempo-algoritmo', type=int, default=900,
                        help='Tempo alvo máximo para iniciar novas configurações por algoritmo, em segundos.')
    args = parser.parse_args()
    if args.threads < 1 or args.tentativas < 1 or args.tempo_algoritmo < 1:
        parser.error('Threads, tentativas e tempo-algoritmo devem ser positivos.')
    treino, teste, dobras, amostra = carregar()
    cobertura = {'treino_linhas': len(treino), 'teste_linhas': len(teste),
                 'lojas_treino': int(treino.loja_id.nunique()),
                 'lojas_teste': int(teste.loja_id.nunique()),
                 'lojas_amostra': len(amostra),
                 'linhas_amostra_teste': int(teste.loja_id.isin(amostra).sum()),
                 'dobras': [{'treino': len(ti), 'validacao': len(vi)} for ti, vi in dobras]}
    print(json.dumps(cobertura, indent=2, ensure_ascii=False), flush=True)
    if args.modo == 'verificar':
        print('Verificação concluída, nenhum modelo foi treinado.', flush=True)
        return
    from flaml import tune
    resultados = []
    inicio_execucao = time.monotonic()
    for nome in ('lightgbm', 'xgboost'):
        inicial = {'n_estimators': 120, 'learning_rate': .05, 'max_depth': 6,
                   'subsample': .9, 'colsample_bytree': .9}
        if nome == 'lightgbm':
            inicial['num_leaves'] = 31
        tentativas = []
        def objetivo(cfg):
            numero = len(tentativas) + 1
            parametros = dict(cfg)
            print(f'{nome}: tentativa {numero}/{args.tentativas}: {parametros}', flush=True)
            inicio = time.monotonic()
            resultado = avaliar(nome, parametros, treino, dobras, args.threads)
            registro = {'numero': numero, 'configuracao': parametros,
                        **resultado, 'tempo_segundos': round(time.monotonic() - inicio, 2)}
            tentativas.append(registro)
            print(f'  RMSE agregado={resultado["rmse"]:.3f}', flush=True)
            return {'rmse': resultado['rmse'], 'mae': resultado['mae']}
        analise = tune.run(objetivo, config=espaco(nome, tune),
                           points_to_evaluate=[inicial], metric='rmse', mode='min',
                           num_samples=args.tentativas,
                           time_budget_s=args.tempo_algoritmo,
                           use_ray=False, verbose=0)
        if not tentativas:
            raise RuntimeError(f'Nenhum candidato avaliado para {nome}.')
        vencedor_alg = min(tentativas, key=lambda t: t['rmse'])
        resultados.append({'algoritmo': nome, 'melhor': vencedor_alg,
                           'tentativas': tentativas,
                           'tentativas_executadas': len(tentativas),
                           'configuracao_flaml': dict(analise.best_config) if analise.best_config else None})
    selecionado = min(resultados, key=lambda r: r['melhor']['rmse'])
    nome = selecionado['algoritmo']
    config = selecionado['melhor']['configuracao']
    print(f'Vencedor: {nome}; RMSE de validação={selecionado["melhor"]["rmse"]:.3f}', flush=True)
    x_train, x_test = preparar(treino, teste)
    m = modelo(nome, config, args.threads)
    m.fit(x_train, treino['vendas'].to_numpy())
    pred = m.predict(x_test)
    df = teste[META + ['vendas']].copy()
    df['horizonte_dias'] = 7
    df['vendas_previstas'] = pred
    df = df[['loja_id', 'data_origem', 'data', 'horizonte_dias', 'vendas', 'vendas_previstas']]
    df_63 = df.loc[df.loja_id.isin(amostra)].copy()
    if len(df_63) != cobertura['linhas_amostra_teste']:
        raise RuntimeError('Recorte das 63 lojas perdeu observações.')
    relatorio = {'tipo': 'EXPERIMENTO_CIENTIFICO_H7_FLAML',
                 'observacao': 'Comparação dos dois candidatos por validação temporal; verificar admissibilidade temporal das covariáveis futuras.',
                 'corte': str(CORTE.date()), 'horizonte': 7,
                 'otimizacao': {'biblioteca': 'flaml.tune', 'max_tentativas_por_algoritmo': args.tentativas,
                               'tempo_alvo_segundos_por_algoritmo': args.tempo_algoritmo,
                               'threads': args.threads, 'semente_modelos': 42},
                 'cobertura': cobertura, 'resultados_validacao': resultados,
                 'vencedor': {'algoritmo': nome, 'configuracao': config,
                              'rmse_validacao': selecionado['melhor']['rmse']},
                 'teste_todas_lojas': metricas(df.vendas, df.vendas_previstas),
                 'teste_63_lojas': metricas(df_63.vendas, df_63.vendas_previstas),
                 'tempo_total_segundos': round(time.monotonic() - inicio_execucao, 2),
                 'gerado_em': datetime.now(timezone.utc).isoformat()}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({'modelo': m, 'algoritmo': nome, 'configuracao': config,
                 'colunas_features': x_train.columns.tolist(),
                 'origem': 'experimento_cientifico_h7_flaml'},
                OUTPUT / 'modelo_avaliacao_externa_h7.joblib')
    # Os arquivos científicos são separados dos resultados preliminares.
    df.to_csv(OUTPUT / 'previsoes_teste_h7_todas_lojas.csv', index=False, date_format='%Y-%m-%d')
    df_63.to_csv(OUTPUT / 'previsoes_teste_h7_63_lojas.csv', index=False, date_format='%Y-%m-%d')
    (OUTPUT / 'relatorio_h7.json').write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Artefatos:', OUTPUT, flush=True)
    print('RMSE externo nas 63 lojas:', relatorio['teste_63_lojas']['rmse'], flush=True)


if __name__ == '__main__':
    executar()
