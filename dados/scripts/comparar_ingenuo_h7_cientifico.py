"""Compara o AutoML H7 ao ingênuo sazonal no recorte científico Rossmann.

Executar na raiz do repositório:
    python dados/scripts/comparar_ingenuo_h7_cientifico.py

O baseline para a data-alvo t é Sales da mesma loja em t-7, isto é,
na data_origem. Dia explicitamente fechado (Open=0) corresponde a zero;
dia sem registro bruto NÃO corresponde automaticamente a zero.
Não modifica nenhum artefato AutoML; grava apenas em baselines_h7/.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'dados'
DEFAULT_PREDS = DATA / 'processados/experimentos/cientifico_h7_flaml/previsoes_teste_h7_63_lojas.csv'
DEFAULT_RAW = DATA / 'brutos/train.csv'
DEFAULT_SAMPLE = DATA / 'processados/amostra_arima.csv'
DEFAULT_OUTPUT = DATA / 'processados/experimentos/baselines_h7'
KEY = ['loja_id', 'data_origem', 'data']


def metricas(y: np.ndarray, yhat: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    if len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(yhat).all():
        raise ValueError('Sem pares finitos para calcular métricas.')
    err = yhat - y
    ss_tot = float(np.square(y - y.mean()).sum())
    return {
        'quantidade': len(y),
        'rmse': float(np.sqrt(np.mean(err * err))),
        'mae': float(np.mean(np.abs(err))),
        'r2': float(1.0 - np.square(err).sum() / ss_tot) if ss_tot > 0 else None,
        'rmspe': float(np.sqrt(np.mean(np.square(err[y > 0] / y[y > 0])))) if np.any(y > 0) else None,
    }


def executar(previsoes: Path, bruto: Path, amostra: Path, saida: Path, permitir_incompleto: bool) -> dict:
    p = pd.read_csv(previsoes, parse_dates=['data_origem', 'data'])
    required = set(KEY + ['horizonte_dias', 'vendas', 'vendas_previstas'])
    if not required.issubset(p.columns):
        raise ValueError(f'Colunas ausentes nas previsões: {sorted(required - set(p.columns))}')
    if p.empty or p[KEY].isna().any().any() or p.duplicated(KEY).any():
        raise ValueError('Previsões vazias, com chaves nulas ou duplicadas.')
    if not p.horizonte_dias.eq(7).all() or not (p.data - p.data_origem).eq(pd.Timedelta(days=7)).all():
        raise ValueError('Previsões incompatíveis com o horizonte H7.')
    ids = pd.read_csv(amostra, usecols=['loja_id']).loja_id.astype(int)
    if len(ids) != 63 or ids.duplicated().any() or set(p.loja_id.unique()) != set(ids):
        raise ValueError('Previsões não correspondem exatamente às 63 lojas pré-selecionadas.')
    if not np.isfinite(p[['vendas', 'vendas_previstas']].to_numpy(dtype=float)).all():
        raise ValueError('Vendas reais ou previstas não finitas.')

    r = pd.read_csv(bruto, usecols=['Store', 'Date', 'Sales', 'Open'], parse_dates=['Date'])
    r = r.rename(columns={'Store': 'loja_id', 'Date': 'data_origem', 'Sales': 'vendas_origem', 'Open': 'aberta_origem'})
    r = r[r.loja_id.isin(ids)].copy()
    if r.duplicated(['loja_id', 'data_origem']).any():
        raise ValueError('Duplicidade loja-data no histórico bruto.')
    # Falta de informação não pode ser confundida com fechamento conhecido.
    r['previsao_ingenua'] = np.where(r.aberta_origem.eq(0), 0.0, r.vendas_origem)
    pares = p.merge(r[['loja_id', 'data_origem', 'previsao_ingenua', 'aberta_origem']],
                    on=['loja_id', 'data_origem'], how='left', validate='many_to_one', indicator=True)
    ausencias = int((pares['_merge'] != 'both').sum())
    nao_finitos = int((~np.isfinite(pd.to_numeric(pares.previsao_ingenua, errors='coerce'))).sum())
    validos = pares[(pares['_merge'] == 'both') & np.isfinite(pd.to_numeric(pares.previsao_ingenua, errors='coerce'))].copy()
    if (ausencias or nao_finitos) and not permitir_incompleto:
        raise ValueError(f'{ausencias} origens sem linha bruta e {nao_finitos} baselines ausentes/não finitos. '
                         'Corrija/investigue antes de comparar; para diagnóstico use --permitir-incompleto.')
    if len(validos) == 0:
        raise ValueError('Nenhum par válido.')
    validos = validos.drop(columns=['_merge']).sort_values(KEY).reset_index(drop=True)
    validos['erro_automl'] = validos['vendas_previstas'] - validos['vendas']
    validos['erro_ingenuo'] = validos['previsao_ingenua'] - validos['vendas']
    validos['diferencial_perda_quadratica'] = validos.erro_automl**2 - validos.erro_ingenuo**2
    resumo = {
        'tipo': 'avaliacao_cientifica_pareada_h7_automl_vs_ingenuo_sazonal',
        'previsoes_fonte': str(previsoes.relative_to(ROOT)) if previsoes.is_relative_to(ROOT) else str(previsoes),
        'historico_fonte': str(bruto.relative_to(ROOT)) if bruto.is_relative_to(ROOT) else str(bruto),
        'lojas_amostra': len(ids), 'lojas_com_pares_validos': int(validos.loja_id.nunique()),
        'pares_h7_origem': len(p), 'pares_validos': len(validos),
        'origens_sem_linha_bruta': ausencias, 'previsoes_ingenuas_nao_finitas': nao_finitos,
        'origens_distintas': int(validos.data_origem.nunique()),
        'automl': metricas(validos.vendas.to_numpy(), validos.vendas_previstas.to_numpy()),
        'ingenuo_sazonal': metricas(validos.vendas.to_numpy(), validos.previsao_ingenua.to_numpy()),
        'regra': 'Vendas na data_origem (= data prevista - 7 dias); fechamento explícito é zero; lacuna sem informação não é zero.',
        'alerta': 'Não há inferência estatística neste relatório; o teste Diebold-Mariano será realizado depois.',
        'gerado_em_utc': datetime.now(timezone.utc).isoformat(),
    }
    saida.mkdir(parents=True, exist_ok=True)
    validos.to_csv(saida / 'comparacao_ingenuo_h7_63_lojas.csv', index=False, date_format='%Y-%m-%d')
    (saida / 'relatorio_ingenuo_h7_63_lojas.json').write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding='utf-8')
    return resumo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previsoes', type=Path, default=DEFAULT_PREDS)
    parser.add_argument('--bruto', type=Path, default=DEFAULT_RAW)
    parser.add_argument('--amostra', type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument('--saida', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--permitir-incompleto', action='store_true', help='Somente diagnóstico: exclui pares sem baseline.')
    args = parser.parse_args()
    print(json.dumps(executar(args.previsoes, args.bruto, args.amostra, args.saida, args.permitir_incompleto),
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
