"""Baseline ARIMA H7 pareado ao AutoML FLAML (Rossmann, 63 lojas).

Executar da raiz do repositorio:
    python dados/scripts/comparar_arima_h7_cientifico.py --modo verificar
    python -u dados/scripts/comparar_arima_h7_cientifico.py --modo executar

Preserva uma previsao por (loja_id, data_origem, data) e atualiza o estado
com vendas conhecidas ate cada origem. Nao reestima ordens no teste externo.
Checkpoint atomico por loja permite retomar apos interrupcoes. Nao modifica
arquivos do AutoML, datasets nem o pipeline web.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DADOS = ROOT / 'dados'
FONTE = DADOS / 'processados' / 'experimentos' / 'cientifico_h7_flaml' / 'previsoes_teste_h7_63_lojas.csv'
AMOSTRA = DADOS / 'processados' / 'amostra_arima.csv'
RAW = DADOS / 'brutos' / 'train.csv'
SAIDA = DADOS / 'processados' / 'experimentos' / 'baselines_h7' / 'arima'
CORTE = pd.Timestamp('2015-01-23')
CAMPOS = ['loja_id', 'data_origem', 'data', 'vendas', 'previsao_automl', 'previsao_arima']


def _gravar_csv_atomico(df: pd.DataFrame, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(destino.suffix + '.tmp')
    df.to_csv(temporario, index=False, date_format='%Y-%m-%d')
    os.replace(temporario, destino)


def _gravar_json_atomico(obj: dict, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(destino.suffix + '.tmp')
    temporario.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporario, destino)


def metricas(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    erro = y - p
    positivos = y > 0
    denominador = np.sum((y - np.mean(y)) ** 2)
    return {
        'quantidade': int(len(y)),
        'rmse': float(np.sqrt(np.mean(erro ** 2))),
        'mae': float(np.mean(np.abs(erro))),
        'r2': float(1 - np.sum(erro ** 2) / denominador) if denominador > 0 else None,
        'rmspe': float(np.sqrt(np.mean((erro[positivos] / y[positivos]) ** 2))) if positivos.any() else None,
    }


def carregar_previsoes() -> pd.DataFrame:
    p = pd.read_csv(FONTE, parse_dates=['data_origem', 'data'])
    esperadas = {'loja_id', 'data_origem', 'data', 'horizonte_dias', 'vendas', 'vendas_previstas'}
    if not esperadas.issubset(p.columns):
        raise ValueError(f'Colunas ausentes na fonte: {sorted(esperadas - set(p.columns))}')
    amostra = pd.read_csv(AMOSTRA)
    ids = amostra.loja_id.astype(int).tolist()
    if len(ids) != 63 or len(set(ids)) != 63:
        raise ValueError('A amostra cientifica deve conter exatamente 63 lojas unicas.')
    p = p.sort_values(['loja_id', 'data_origem', 'data']).reset_index(drop=True)
    if len(p) != 9682 or p.loja_id.nunique() != 63 or set(p.loja_id) != set(ids):
        raise ValueError('Fonte FLAML nao corresponde aos 9682 pares das 63 lojas verificados.')
    if not (p.horizonte_dias == 7).all() or not (p.data - p.data_origem == pd.Timedelta(days=7)).all():
        raise ValueError('Horizonte H7 incoerente.')
    if p.duplicated(['loja_id', 'data_origem', 'data']).any():
        raise ValueError('Pares duplicados.')
    if (p.data_origem.min() < CORTE) or not np.isfinite(p[['vendas', 'vendas_previstas']].to_numpy(dtype=float)).all():
        raise ValueError('Dados do teste externo invalidos.')
    return p


def carregar_historico(ids: list[int]) -> pd.DataFrame:
    r = pd.read_csv(RAW, usecols=['Store', 'Date', 'Sales', 'Open'], parse_dates=['Date'])
    r = r.loc[r.Store.isin(ids)].copy()
    if r.duplicated(['Store', 'Date']).any():
        raise ValueError('Registros duplicados na Rossmann bruta.')
    r.loc[r.Open == 0, 'Sales'] = 0.0
    return r


def regularizar(loja_raw: pd.DataFrame, ultima_origem: pd.Timestamp) -> tuple[pd.Series, int]:
    calendario = pd.date_range(loja_raw.index.min(), ultima_origem, freq='D')
    dados = loja_raw['Sales'].reindex(calendario).astype(float)
    historico_dow = {i: [] for i in range(7)}
    historico_total: list[float] = []
    valores: list[float] = []
    faltas = 0
    for dia, valor in dados.items():
        if pd.isna(valor):
            faltas += 1
            anteriores = historico_dow[dia.dayofweek]
            substituto = float(np.median(anteriores)) if anteriores else (
                float(np.median(historico_total)) if historico_total else 0.0
            )
            valores.append(substituto)
        else:
            valor = float(valor)
            valores.append(valor)
            historico_dow[dia.dayofweek].append(valor)
            historico_total.append(valor)
    return pd.Series(valores, index=calendario), faltas


def verificar_checkpoint(loja: int, esperado: pd.DataFrame, arquivo: Path) -> pd.DataFrame:
    c = pd.read_csv(arquivo, parse_dates=['data_origem', 'data'])
    if list(c.columns) != CAMPOS:
        raise ValueError(f'Loja {loja}: cabecalho do checkpoint inesperado.')
    if len(c) != len(esperado) or c.duplicated(['loja_id', 'data_origem', 'data']).any():
        raise ValueError(f'Loja {loja}: checkpoint incompleto ou duplicado.')
    c = c.sort_values(['loja_id', 'data_origem']).reset_index(drop=True)
    original = esperado.sort_values(['loja_id', 'data_origem']).reset_index(drop=True)
    for col in ('loja_id', 'data_origem', 'data'):
        if not c[col].equals(original[col]):
            raise ValueError(f'Loja {loja}: checkpoint diverge nas chaves ({col}).')
    if not np.allclose(c.vendas, original.vendas, rtol=0, atol=1e-8) or not np.allclose(
        c.previsao_automl, original.vendas_previstas, rtol=0, atol=1e-5
    ) or not np.isfinite(c.previsao_arima).all():
        raise ValueError(f'Loja {loja}: checkpoint diverge nos valores.')
    return c


def executar(retomar: bool) -> None:
    from pmdarima import auto_arima

    p = carregar_previsoes()
    raw = carregar_historico(p.loja_id.unique().tolist())
    pastas = SAIDA / 'lojas'
    pastas.mkdir(parents=True, exist_ok=True)
    saidas = []
    ajustes = []
    for indice, (loja, grupo) in enumerate(p.groupby('loja_id', sort=True), 1):
        loja = int(loja)
        esperado = grupo.sort_values('data_origem').reset_index(drop=True)
        checkpoint = pastas / f'loja_{loja}.csv'
        detalhes = pastas / f'loja_{loja}.json'
        if retomar and checkpoint.exists() and detalhes.exists():
            c = verificar_checkpoint(loja, esperado, checkpoint)
            saidas.append(c)
            ajustes.append(json.loads(detalhes.read_text(encoding='utf-8')))
            print(f'[{indice}/63] Loja {loja}: recuperados {len(c)} pares.', flush=True)
            continue
        loja_raw = raw.loc[raw.Store == loja].set_index('Date').sort_index()
        if loja_raw.empty:
            raise ValueError(f'Loja {loja}: sem dados brutos.')
        regular, lacunas = regularizar(loja_raw, esperado.data_origem.max())
        primeira = esperado.data_origem.min()
        # Mesmo limite temporal do experimento: parametros ajustados somente ate o corte.
        inicial = regular.loc[:CORTE].to_numpy(dtype=float)
        if len(inicial) < 180:
            raise ValueError(f'Loja {loja}: menos de 180 dias iniciais.')
        modelo = auto_arima(
            inicial, seasonal=False, start_p=0, start_q=0,
            max_p=2, max_q=2, max_d=1, max_order=4,
            stepwise=True, information_criterion='aic',
            suppress_warnings=True, error_action='ignore', trace=False,
        )
        ultima = CORTE
        registros = []
        for linha in esperado.itertuples(index=False):
            origem = linha.data_origem
            if origem > ultima:
                novos = regular.loc[ultima + pd.Timedelta(days=1):origem].to_numpy(dtype=float)
                if len(novos):
                    modelo.update(novos, maxiter=0)
                ultima = origem
            previsto = max(0.0, float(modelo.predict(n_periods=7)[-1]))
            registros.append({
                'loja_id': loja, 'data_origem': origem, 'data': linha.data,
                'vendas': float(linha.vendas), 'previsao_automl': float(linha.vendas_previstas),
                'previsao_arima': previsto,
            })
        c = pd.DataFrame(registros, columns=CAMPOS)
        detalhes_loja = {
            'loja_id': loja, 'ordem': list(modelo.order),
            'dias_treino_inicial': int(len(inicial)),
            'origens_avaliadas': int(len(c)),
            'lacunas_imputadas_ate_ultima_origem': int(lacunas),
            'metodo_atualizacao': 'pmdarima.update(maxiter=0), sem reestimar coeficientes',
        }
        _gravar_csv_atomico(c, checkpoint)
        _gravar_json_atomico(detalhes_loja, detalhes)
        saidas.append(c)
        ajustes.append(detalhes_loja)
        print(f'[{indice}/63] Loja {loja}: ARIMA{modelo.order}, {len(c)} pares.', flush=True)

    resultado = pd.concat(saidas, ignore_index=True).sort_values(
        ['data_origem', 'loja_id', 'data']
    ).reset_index(drop=True)
    if len(resultado) != len(p) or resultado.loja_id.nunique() != 63:
        raise ValueError('A comparacao nao cobriu os 9682 pares.')
    chaves = ['loja_id', 'data_origem', 'data']
    unido = resultado.merge(p[chaves + ['vendas', 'vendas_previstas']], on=chaves, how='outer',
                           validate='one_to_one', indicator=True)
    if not (unido._merge == 'both').all() or not np.allclose(unido.vendas_x, unido.vendas_y, atol=1e-8) or not np.allclose(
        unido.previsao_automl, unido.vendas_previstas, atol=1e-5
    ):
        raise ValueError('A comparacao completa diverge das previsoes FLAML.')
    y = resultado.vendas.to_numpy(dtype=float)
    resumo = {
        'tipo': 'comparacao_pareada_cientifica_h7_automl_vs_arima',
        'fonte_automl': str(FONTE.relative_to(ROOT)),
        'fonte_historico': str(RAW.relative_to(ROOT)),
        'lojas': 63, 'pares': int(len(resultado)),
        'origens_distintas': int(resultado.data_origem.nunique()),
        'automl': metricas(y, resultado.previsao_automl.to_numpy(dtype=float)),
        'arima': metricas(y, resultado.previsao_arima.to_numpy(dtype=float)),
        'ajustes': ajustes,
        'decisoes': [
            'Ordens ARIMA selecionadas por AIC usando apenas o historico ate 2015-01-23.',
            'Previsao H7 e atualizacao de estados apenas ate cada origem de previsao.',
            'Fechamentos explicitos recebem venda zero; lacunas desconhecidas imputadas com historico passado.',
            'Previsoes avaliadas apenas em datas de teste validas do AutoML.',
            'Nao se aplica o teste Diebold-Mariano neste arquivo.',
        ],
        'gerado_em_utc': datetime.now(timezone.utc).isoformat(),
    }
    _gravar_csv_atomico(resultado, SAIDA / 'comparacao_arima_h7_63_lojas.csv')
    _gravar_json_atomico(resumo, SAIDA / 'resumo_arima_h7_63_lojas.json')
    print(json.dumps({k: v for k, v in resumo.items() if k != 'ajustes'}, indent=2, ensure_ascii=False))
    print('Artefatos:', SAIDA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modo', choices=['verificar', 'executar'], default='verificar')
    parser.add_argument('--sem-retomar', action='store_true', help='Refazer todas as lojas, ignorando checkpoints.')
    args = parser.parse_args()
    p = carregar_previsoes()
    resumo = {'status': 'fonte_verificada', 'linhas': len(p), 'lojas': int(p.loja_id.nunique()),
              'origens_distintas': int(p.data_origem.nunique()),
              'primeira_origem': str(p.data_origem.min().date()),
              'ultima_origem': str(p.data_origem.max().date())}
    print(json.dumps(resumo, ensure_ascii=False, indent=2), flush=True)
    if args.modo == 'executar':
        executar(retomar=not args.sem_retomar)


if __name__ == '__main__':
    main()
