"""Ensaio controlado AutoML x ARIMA H7; NÃO é o experimento final do TG.

Uso: python dados/scripts/comparar_arima_h7.py --execucao ID --max-origens 5

Seleciona as primeiras N origens cronológicas POR LOJA. Ordens ARIMA são
selecionadas pelo AIC com auto_arima apenas no histórico anterior ao teste.
O estado é atualizado diariamente com observações até a origem. Datas
fechadas (Open=0) recebem zero; lacunas genuínas são imputadas usando apenas
valores observados até aquela data, pela mediana histórica do dia da semana.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def metricas(y: np.ndarray, p: np.ndarray) -> dict:
    erro = y - p
    denom = float(np.sum((y - y.mean()) ** 2))
    pos = y > 0
    return {
        'quantidade': int(len(y)),
        'rmse': float(np.sqrt(np.mean(erro**2))),
        'mae': float(np.mean(np.abs(erro))),
        'r2': float(1 - np.sum(erro**2) / denom) if denom > 0 else None,
        'rmspe': float(np.sqrt(np.mean((erro[pos] / y[pos]) ** 2))) if pos.any() else None,
    }


def executar(execucao: str, max_origens: int) -> None:
    if max_origens < 1 or max_origens > 30:
        raise ValueError('Para este ensaio, --max-origens deve estar entre 1 e 30.')
    try:
        from pmdarima import auto_arima
    except ImportError as exc:
        raise RuntimeError('Instale pmdarima no ambiente virtual: pip install pmdarima') from exc

    previsoes_path = ROOT / 'backend' / 'execucoes' / execucao / 'multihorizonte' / 'previsoes_teste.csv'
    p = pd.read_csv(previsoes_path, parse_dates=['data_origem', 'data'])
    requeridas = {'loja_id', 'data_origem', 'data', 'horizonte_dias', 'vendas', 'vendas_previstas'}
    if not requeridas.issubset(p.columns):
        raise ValueError(f'Colunas ausentes: {sorted(requeridas - set(p.columns))}')
    p = p.loc[p.horizonte_dias == 7].copy()
    if p.empty or p.duplicated(['loja_id', 'data_origem', 'data']).any():
        raise ValueError('Previsões H7 vazias ou duplicadas.')
    if not (p['data'] - p['data_origem'] == pd.Timedelta(days=7)).all():
        raise ValueError('Horizontes H7 inconsistentes.')
    if not np.isfinite(p[['vendas', 'vendas_previstas']].to_numpy(dtype=float)).all():
        raise ValueError('Valores não finitos em vendas ou previsões.')
    p = (p.sort_values(['loja_id', 'data_origem'])
           .groupby('loja_id', group_keys=False).head(max_origens).copy())

    raw = pd.read_csv(ROOT / 'dados' / 'brutos' / 'train.csv',
                      usecols=['Store', 'Date', 'Sales', 'Open'], parse_dates=['Date'])
    raw = raw.loc[raw.Store.isin(p.loja_id.unique())].copy()
    if raw.duplicated(['Store', 'Date']).any():
        raise ValueError('Dados brutos possuem pares Store/Date duplicados.')
    raw.loc[raw.Open == 0, 'Sales'] = 0.0

    saidas, ajustes = [], []
    for loja, g in p.groupby('loja_id', sort=True):
        g = g.sort_values('data_origem')
        origem_inicial = g.data_origem.iloc[0]
        loja_raw = raw.loc[raw.Store == loja].set_index('Date').sort_index()
        calendario = pd.date_range(loja_raw.index.min(), g.data_origem.max(), freq='D')
        serie = loja_raw['Sales'].reindex(calendario).astype(float)
        # A imputação nunca usa observações posteriores à data preenchida.
        valores = []
        historico_dow = {dia: [] for dia in range(7)}
        historico_geral = []
        faltas = 0
        for data, observado in serie.items():
            if pd.isna(observado):
                faltas += 1
                candidatos = historico_dow[data.dayofweek]
                imputado = float(np.median(candidatos)) if candidatos else (
                    float(np.median(historico_geral)) if historico_geral else 0.0
                )
                valores.append(imputado)
                # Imputações não passam a ser consideradas observações reais.
            else:
                observado = float(observado)
                valores.append(observado)
                historico_dow[data.dayofweek].append(observado)
                historico_geral.append(observado)
        regular = pd.Series(valores, index=calendario, dtype=float)
        treino = regular.loc[:origem_inicial].to_numpy()
        if len(treino) < 180:
            raise ValueError(f'Loja {loja}: histórico inicial insuficiente ({len(treino)} dias).')
        modelo = auto_arima(
            treino, seasonal=False, start_p=0, start_q=0,
            max_p=2, max_q=2, max_d=1, max_order=4,
            stepwise=True, information_criterion='aic',
            suppress_warnings=True, error_action='ignore', trace=False,
        )
        ajuste = {'loja_id': int(loja), 'ordem': list(modelo.order),
                  'dias_treino_inicial': len(treino), 'lacunas_imputadas_ate_ultima_origem': faltas}
        ajustes.append(ajuste)
        ultima_origem = origem_inicial
        for linha in g.itertuples(index=False):
            origem = linha.data_origem
            if origem > ultima_origem:
                adicionais = regular.loc[ultima_origem + pd.Timedelta(days=1):origem].to_numpy()
                if len(adicionais):
                    modelo.update(adicionais, maxiter=0)
                ultima_origem = origem
            # Forecast de sete passos, realizado estritamente na data de origem.
            previsto = float(max(0.0, modelo.predict(n_periods=7)[-1]))
            saidas.append({'loja_id': int(loja), 'data_origem': origem,
                           'data': linha.data, 'vendas': float(linha.vendas),
                           'previsao_automl': float(linha.vendas_previstas),
                           'previsao_arima': previsto})
        print(f'Loja {loja}: ARIMA{modelo.order}, {len(g)} origens processadas.', flush=True)

    comparacao = pd.DataFrame(saidas).sort_values(['data_origem', 'loja_id'])
    y = comparacao.vendas.to_numpy(dtype=float)
    resumo = {
        'tipo': 'ENSAIO_PILOTO_comparacao_pareada_h7_automl_arima',
        'execucao': execucao, 'max_origens_por_loja': max_origens,
        'lojas': int(comparacao.loja_id.nunique()), 'pares': len(comparacao),
        'automl': metricas(y, comparacao.previsao_automl.to_numpy(dtype=float)),
        'arima': metricas(y, comparacao.previsao_arima.to_numpy(dtype=float)),
        'ajustes': ajustes,
        'limitacoes': [
            'Este é um teste do procedimento, não uma conclusão do TG.',
            'Ordens selecionadas apenas no histórico inicial; atualizações subsequentes de estado.',
            'Imputação de lacunas apenas com histórico passado; fechamentos observados = zero.',
            'A seleção de até N origens por loja não constitui a avaliação científica completa.',
        ],
    }
    destino = ROOT / 'dados' / 'processados' / 'experimentos' / 'ensaio' / execucao
    destino.mkdir(parents=True, exist_ok=True)
    comparacao.to_csv(destino / 'comparacao_arima_h7_piloto.csv', index=False, date_format='%Y-%m-%d')
    (destino / 'resumo_arima_h7_piloto.json').write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execucao', required=True)
    parser.add_argument('--max-origens', type=int, default=5)
    args = parser.parse_args()
    executar(args.execucao, args.max_origens)
