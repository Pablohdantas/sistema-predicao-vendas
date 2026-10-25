"""H2 Rossmann H7: estabilidade SHAP e concordancia com permutacao.
Executar na raiz do repositorio: python -u dados/scripts/avaliar_shap_global_h7.py
Nao treina ou altera o modelo AutoML.
"""
from pathlib import Path
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[2]
DADOS = ROOT / 'dados'
BASE = DADOS / 'processados' / 'experimentos' / 'cientifico_h7_flaml'
TREINO = DADOS / 'processados' / 'treinamento'
SAIDA = DADOS / 'processados' / 'experimentos' / 'shap_global_h7'
CHAVES = ['loja_id', 'data_origem', 'data']
SEMENTES = list(range(42, 52))
POR_LOJA_SHAP = 16
POR_LOJA_PERMUTACAO = 32


def amostra_estratificada(df, por_loja, semente):
    # Escolhe o mesmo numero de observacoes por loja, sem reposicao.
    grupos = [g.sample(n=min(por_loja, len(g)), random_state=semente)
              for _, g in df.groupby('loja_id', sort=True)]
    return pd.concat(grupos).sort_values('indice').reset_index(drop=True)


def shap_absoluto_medio(modelo, dados, nomes):
    matriz = xgb.DMatrix(dados[nomes], feature_names=nomes)
    contribuicoes = modelo.get_booster().predict(matriz, pred_contribs=True)
    if contribuicoes.shape != (len(dados), len(nomes) + 1):
        raise ValueError('Dimensao inesperada de SHAP + valor base.')
    previsoes = np.asarray(modelo.predict(dados[nomes]), dtype=float)
    if not np.allclose(contribuicoes.sum(axis=1), previsoes, atol=0.15, rtol=1e-5):
        raise ValueError('Contribuicoes SHAP nao reproduzem previsoes do modelo.')
    return np.mean(np.abs(contribuicoes[:, :-1]), axis=0)


def main():
    artefato = joblib.load(BASE / 'modelo_avaliacao_externa_h7.joblib')
    if not isinstance(artefato, dict) or 'modelo' not in artefato or 'colunas_features' not in artefato:
        raise ValueError('Estrutura inesperada do joblib.')
    modelo = artefato['modelo']
    nomes = list(artefato['colunas_features'])
    X = pd.read_csv(TREINO / 'X_teste.csv')
    meta = pd.read_csv(TREINO / 'metadata_teste.csv', parse_dates=['data_origem', 'data'])
    p = pd.read_csv(BASE / 'previsoes_teste_h7_63_lojas.csv', parse_dates=['data_origem', 'data'])
    if len(X) != len(meta) or len(X) != 169494 or len(p) != 9682:
        raise ValueError('Quantidade de linhas inesperada nos dados cientificos.')
    if len(nomes) != 23 or set(nomes) != set(X.columns) or len(set(nomes)) != len(nomes):
        raise ValueError('Atributos do joblib nao correspondem a X_teste.')
    if not {'vendas', 'vendas_previstas', 'horizonte_dias', *CHAVES}.issubset(p.columns):
        raise ValueError('Colunas incompletas nas previsoes FLAML.')
    if p.loja_id.nunique() != 63 or p.duplicated(CHAVES).any():
        raise ValueError('Amostra de 63 lojas invalida ou duplicada.')
    if not (p.horizonte_dias == 7).all() or not (p.data - p.data_origem == pd.Timedelta(days=7)).all():
        raise ValueError('Horizonte de previsao diferente de H7.')
    if meta.duplicated(CHAVES).any():
        raise ValueError('Metadados do conjunto externo duplicados.')
    meta = meta.reset_index(names='indice')
    dados = p.merge(meta[CHAVES + ['indice']], on=CHAVES, how='left', validate='one_to_one')
    if dados.indice.isna().any() or dados.indice.nunique() != 9682:
        raise ValueError('As 9682 previsoes nao foram alinhadas integralmente ao X_teste.')
    dados['indice'] = dados.indice.astype(int)
    dados = dados.sort_values('indice').reset_index(drop=True)
    X = X.loc[:, nomes]
    # O XGBoost foi treinado com NaN nativos; preserve-os no SHAP e na permutacao.
    # Infinitos nao foram encontrados no diagnostico e continuam proibidos.
    matriz_externa = X.iloc[dados.indice].to_numpy(dtype=float)
    if np.isinf(matriz_externa).any():
        raise ValueError('Atributos infinitos na amostra externa.')
    ausentes = int(np.isnan(matriz_externa).any(axis=1).sum())
    print(f'VALORES AUSENTES PRESERVADOS: {ausentes} linhas (NaN); nenhuma linha removida.', flush=True)
    verificacao = modelo.predict(X.iloc[dados.indice])
    maior_diferenca = float(np.max(np.abs(verificacao - dados.vendas_previstas.to_numpy())))
    if maior_diferenca > 0.15:
        raise ValueError(f'Modelo joblib nao reproduz previsoes FLAML: diferenca maxima {maior_diferenca:.6f}')
    print(f'ALINHAMENTO APROVADO: 63 lojas, {len(dados)} pares; diferenca maxima {maior_diferenca:.6f}', flush=True)

    rankings = []
    importancias = []
    for k, semente in enumerate(SEMENTES, 1):
        selecionada = amostra_estratificada(dados, POR_LOJA_SHAP, semente)
        Xi = X.iloc[selecionada.indice].reset_index(drop=True)
        valores = shap_absoluto_medio(modelo, Xi, nomes)
        importancias.append(valores)
        rankings.append(pd.Series(valores, index=nomes).rank(ascending=False, method='average'))
        print(f'SHAP subamostra {k}/10: {len(Xi)} observacoes', flush=True)
    correlacoes = []
    for i in range(10):
        for j in range(i + 1, 10):
            rho = float(spearmanr(rankings[i], rankings[j]).statistic)
            if not np.isfinite(rho):
                raise ValueError('Correlacao SHAP indefinida.')
            correlacoes.append({'amostra_a': i + 1, 'amostra_b': j + 1, 'spearman': rho})
    media_estabilidade = float(np.mean([c['spearman'] for c in correlacoes]))

    selecionada = amostra_estratificada(dados, POR_LOJA_PERMUTACAO, 2026)
    Xp = X.iloc[selecionada.indice].reset_index(drop=True)
    yp = selecionada.vendas.to_numpy(dtype=float)
    shap_perm = shap_absoluto_medio(modelo, Xp, nomes)
    perm = permutation_importance(modelo, Xp, yp, scoring='neg_mean_squared_error',
                                  n_repeats=3, random_state=42, n_jobs=1)
    rho_concordancia = float(spearmanr(shap_perm, perm.importances_mean).statistic)
    if not np.isfinite(rho_concordancia):
        raise ValueError('Correlacao SHAP versus permutacao indefinida.')
    tabela = pd.DataFrame({
        'atributo': nomes,
        'shap_abs_medio_10_subamostras': np.mean(importancias, axis=0),
        'shap_abs_medio_amostra_permutacao': shap_perm,
        'importancia_permutacao_delta_mse': perm.importances_mean,
        'desvio_permutacao': perm.importances_std,
    }).sort_values('shap_abs_medio_10_subamostras', ascending=False)
    SAIDA.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(SAIDA / 'importancias_globais_h7.csv', index=False)
    pd.DataFrame(correlacoes).to_csv(SAIDA / 'correlacoes_subamostras_shap.csv', index=False)
    resumo = {
        'tipo': 'avaliacao_h2_shap_global_h7',
        'pares_cientificos_alinhados': 9682,
        'lojas': 63,
        'atributos': len(nomes),
        'linhas_com_nan_preservadas': ausentes,
        'tratamento_ausentes': 'NaN nativo do XGBoost, sem imputacao ou exclusao',
        'maxima_divergencia_previsoes_joblib_flaml': maior_diferenca,
        'subamostras': 10,
        'observacoes_por_subamostra': int(sum(min(POR_LOJA_SHAP, n) for n in dados.groupby('loja_id').size())),
        'correlacoes_pares': len(correlacoes),
        'spearman_medio_estabilidade_shap': media_estabilidade,
        'limiar_estabilidade': 0.8,
        'estabilidade_atende': media_estabilidade >= 0.8,
        'observacoes_permutacao': len(Xp),
        'repeticoes_permutacao': 3,
        'spearman_shap_permutacao': rho_concordancia,
        'limiar_concordancia': 0.7,
        'concordancia_atende': rho_concordancia >= 0.7,
        'h2_atende_criterios': media_estabilidade >= 0.8 and rho_concordancia >= 0.7,
        'nota': 'SHAP TreeSHAP de XGBoost; 10 amostras estratificadas por loja e sobreposicao possivel. Permutacao marginal por atributo, que pode ser afetada por colinearidade. Analise externa sem retreinar.',
        'gerado_em_utc': datetime.now(timezone.utc).isoformat(),
    }
    (SAIDA / 'resumo_h2_shap_global_h7.json').write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(resumo, indent=2, ensure_ascii=False), flush=True)
    print('Arquivos gerados em:', SAIDA, flush=True)


if __name__ == '__main__':
    main()