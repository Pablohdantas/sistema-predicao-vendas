"""Experimento científico H7 Rossmann, isolado da API de produção.

Executar na raiz do repositório:
 python dados/scripts/executar_automl_rossmann_h7.py --modo verificar
 python dados/scripts/executar_automl_rossmann_h7.py --modo executar

IMPORTANTE: NÃO executa o mesmo otimizador FLAML do pipeline operacional.
Compara configurações pré-fixadas por validação temporal; eventual diferença
metodológica deve ser explicitada no TG, ou este módulo deve ser integrado à
rotina FLAML existente antes de ser chamado de resultado definitivo.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'dados' / 'processados'
TRAIN = DATA / 'treinamento'
OUTPUT = DATA / 'experimentos' / 'cientifico_h7'
CORTE = pd.Timestamp('2015-01-23')
FEATURES_NUM = ['promocao', 'promocao_continua', 'mes_renovacao_promocao',
 'feriado', 'feriado_escolar', 'distancia_concorrente', 'dia_semana',
 'dia_mes', 'mes', 'semana_ano', 'lag_7', 'lag_14', 'lag_21',
 'lag_28', 'media_movel_7', 'media_movel_28']
CATEGORICAS = ['tipo_loja', 'sortimento']
COLUNAS = ['loja_id', 'data', 'data_origem']


def dados_fonte():
    df = pd.read_csv(DATA / 'atributos_h7.csv', parse_dates=['data', 'data_origem'], low_memory=False)
    if df.duplicated(['loja_id', 'data']).any():
        raise ValueError('Duplicidade de loja e data no dataset H7')
    if not (df['data'] - df['data_origem']).eq(pd.Timedelta(days=7)).all():
        raise ValueError('Há horizonte diferente de sete dias')
    treino = df[df.data <= CORTE].copy().reset_index(drop=True)
    teste = df[df.data_origem >= CORTE].copy().reset_index(drop=True)
    if treino.empty or teste.empty:
        raise ValueError('Treino ou teste vazio')
    assert treino.data.max() < teste.data.min()
    return treino, teste


def configurar_features(train, validation):
    """Imputação/encoding ajustados APENAS nas linhas do treino de cada dobra."""
    a, b = train[FEATURES_NUM + CATEGORICAS].copy(), validation[FEATURES_NUM + CATEGORICAS].copy()
    mediana = pd.to_numeric(a['distancia_concorrente'], errors='coerce').median()
    if pd.isna(mediana):
        raise ValueError('Distância da concorrência inteiramente ausente no treinamento')
    for f in (a, b):
        for col in FEATURES_NUM:
            f[col] = pd.to_numeric(f[col], errors='coerce').astype('float32')
        f['distancia_concorrente'] = f['distancia_concorrente'].fillna(mediana)
        for cat in CATEGORICAS:
            f[cat] = f[cat].fillna('desconhecido').astype(str)
    # Categorias definidas estritamente pelos dados históricos da dobra.
    for cat in CATEGORICAS:
        cats = sorted(a[cat].unique())
        a[cat] = pd.Categorical(a[cat], categories=cats)
        b[cat] = pd.Categorical(b[cat], categories=cats)
    a = pd.get_dummies(a, columns=CATEGORICAS, dtype='int8')
    b = pd.get_dummies(b, columns=CATEGORICAS, dtype='int8').reindex(columns=a.columns, fill_value=0)
    return a, b


def folds_from_saved_metadata(treino):
    metadata = pd.read_csv(TRAIN / 'metadata_treino.csv', parse_dates=['data', 'data_origem'])
    if len(metadata) != len(treino) or not metadata[COLUNAS].equals(treino[COLUNAS]):
        raise ValueError('metadata_treino.csv não corresponde à ordem atual de atributos_h7.csv. Não utilizar índices antigos.')
    saved = np.load(TRAIN / 'indices_validacao_temporal.npz')
    folds=[]
    for n in range(1, 5):
        ti = saved[f'treino_{n}']
        vi = saved[f'validacao_{n}']
        if len(ti)==0 or len(vi)==0 or (np.intersect1d(ti, vi).size):
            raise ValueError(f'Dobra {n} inválida')
        if treino.iloc[ti].data.max() > treino.iloc[vi].data_origem.min():
            raise ValueError(f'Dobra {n}: vazamento temporal')
        folds.append((ti,vi))
    saved.close()
    return folds


def metricas(y, p):
    y,p=np.asarray(y,dtype=float), np.asarray(p,dtype=float)
    resultado={'quantidade':len(y), 'rmse':float(np.sqrt(mean_squared_error(y,p))),
               'mae':float(mean_absolute_error(y,p)), 'r2':float(r2_score(y,p))}
    validos=y>0
    resultado['rmspe']=float(np.sqrt(np.mean(((y[validos]-p[validos])/y[validos])**2))) if validos.any() else None
    return resultado


def construir_modelo(nome, params, threads):
    if nome=='lightgbm':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=42, n_jobs=threads, verbosity=-1, **params)
    from xgboost import XGBRegressor
    return XGBRegressor(random_state=42, n_jobs=threads, tree_method='hist', objective='reg:squarederror', **params)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--modo', choices=['verificar','executar'], default='verificar')
    parser.add_argument('--threads', type=int, default=2)
    args=parser.parse_args()
    treino,teste=dados_fonte()
    folds=folds_from_saved_metadata(treino)
    lojas_amostra=pd.read_csv(DATA / 'amostra_arima.csv').loja_id.astype(int).tolist()
    if len(lojas_amostra)!=63 or not set(lojas_amostra).issubset(set(teste.loja_id)):
        raise ValueError('Amostra científica não contém 63 lojas válidas no teste')
    cobertura={'treino_linhas':len(treino),'teste_linhas':len(teste),'lojas_treino':int(treino.loja_id.nunique()),
      'lojas_teste':int(teste.loja_id.nunique()),'lojas_amostra':len(lojas_amostra),
      'linhas_amostra_teste':int(teste.loja_id.isin(lojas_amostra).sum()),
      'dobras':[{'treino':len(t),'validacao':len(v)} for t,v in folds]}
    print(json.dumps(cobertura, indent=2, ensure_ascii=False),flush=True)
    if args.modo=='verificar':
        print('Verificação concluída, nenhum modelo foi treinado.')
        return
    # Configurações pré-definidas: desempenho NÃO equivale ao FLAML operacional.
    candidatos={'lightgbm':[
       {'n_estimators':120,'learning_rate':0.05,'max_depth':6,'num_leaves':31},
       {'n_estimators':180,'learning_rate':0.04,'max_depth':8,'num_leaves':31}],
      'xgboost':[
       {'n_estimators':120,'learning_rate':0.05,'max_depth':6,'subsample':0.9,'colsample_bytree':0.9},
       {'n_estimators':180,'learning_rate':0.04,'max_depth':5,'subsample':0.9,'colsample_bytree':0.9}]}
    resultados=[]
    for nome, configs in candidatos.items():
        for numero, cfg in enumerate(configs, 1):
            metricas_folds=[]
            erros=[]
            print(f'Validando {nome} configuração {numero}/{len(configs)}',flush=True)
            for n,(ti,vi) in enumerate(folds, 1):
                x_train,x_val=configurar_features(treino.iloc[ti],treino.iloc[vi])
                model=construir_modelo(nome,cfg,args.threads)
                model.fit(x_train,treino.iloc[ti].vendas.to_numpy())
                pred=model.predict(x_val)
                m=metricas(treino.iloc[vi].vendas.to_numpy(),pred)
                metricas_folds.append({'dobra':n, **m})
                erros.append((treino.iloc[vi].vendas.to_numpy()-pred)**2)
                print(f'  Dobra {n}: RMSE={m["rmse"]:.3f}',flush=True)
                del model,x_train,x_val
            rmse_geral=float(np.sqrt(np.mean(np.concatenate(erros))))
            resultados.append({'algoritmo':nome,'configuracao':cfg,'rmse_validacao':rmse_geral,'dobras':metricas_folds})
    vencedor=min(resultados,key=lambda r:r['rmse_validacao'])
    print('Vencedor:', vencedor['algoritmo'], 'RMSE validação',vencedor['rmse_validacao'], flush=True)
    x_train,x_test=configurar_features(treino,teste)
    model=construir_modelo(vencedor['algoritmo'],vencedor['configuracao'],args.threads)
    model.fit(x_train,treino.vendas.to_numpy())
    pred=model.predict(x_test)
    df=teste[COLUNAS+['vendas']].copy()
    df['horizonte_dias']=7
    df['vendas_previstas']=pred
    df=df[['loja_id','data_origem','data','horizonte_dias','vendas','vendas_previstas']]
    df_amostra=df[df.loja_id.isin(lojas_amostra)].copy()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT/'previsoes_teste_h7_todas_lojas.csv',index=False,date_format='%Y-%m-%d')
    df_amostra.to_csv(OUTPUT/'previsoes_teste_h7_63_lojas.csv',index=False,date_format='%Y-%m-%d')
    resumo={'tipo':'EXPERIMENTO_PRELIMINAR_H7_NAO_FLAML','data_corte':str(CORTE.date()),
      'aviso':'Competição de parâmetros fixos não equivale à otimização FLAML prevista no pipeline operacional.',
      'cobertura':cobertura,'candidatos':resultados,'vencedor':vencedor,
      'teste_todas_lojas':metricas(df.vendas,df.vendas_previstas),
      'teste_63_lojas':metricas(df_amostra.vendas,df_amostra.vendas_previstas),
      'gerado_em':datetime.now(timezone.utc).isoformat()}
    (OUTPUT/'relatorio_h7.json').write_text(json.dumps(resumo,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Artefatos:', OUTPUT,flush=True)
    print('RMSE teste 63 lojas:',resumo['teste_63_lojas']['rmse'],flush=True)

if __name__=='__main__':
    main()
