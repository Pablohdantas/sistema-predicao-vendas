"""Teste Diebold-Mariano unilateral H7: AutoML versus ARIMA, 63 lojas.

Uso:
 python dados/scripts/teste_diebold_mariano_h7.py
 python dados/scripts/teste_diebold_mariano_h7.py --csv caminho/arquivo.csv

Não modifica os arquivos de entrada. Utiliza os pares produzidos pelo
comparador ARIMA e a média diária dos diferenciais de perda quadrática.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / 'dados' / 'processados' / 'experimentos' / 'baselines_h7' / 'arima'
DESTINO = DIR / 'diebold_mariano_h7.json'


def escolher_coluna(cols, candidatos, descricao):
    achados = [x for x in candidatos if x in cols]
    if len(achados) != 1:
        raise ValueError(f'Coluna de {descricao} não identificada univocamente. '
                         f'Candidatos encontrados: {achados}; colunas disponíveis: {sorted(cols)}')
    return achados[0]


def carregar(caminho):
    df = pd.read_csv(caminho, low_memory=False)
    cols = set(df.columns)
    for c in ['loja_id', 'data_origem', 'data']:
        if c not in cols:
            raise ValueError(f'Coluna obrigatória ausente: {c}. Colunas disponíveis: {sorted(cols)}')
    real = escolher_coluna(cols, ['vendas', 'vendas_reais', 'real'], 'vendas reais')
    automl = escolher_coluna(cols, ['vendas_previstas', 'previsao_automl', 'vendas_previstas_automl', 'previsto_automl'], 'AutoML')
    arima = escolher_coluna(cols, ['vendas_previstas_arima', 'previsao_arima', 'arima_previsto', 'previsao_estatistica', 'arima'], 'ARIMA')
    df['data_origem'] = pd.to_datetime(df['data_origem'], errors='raise')
    df['data'] = pd.to_datetime(df['data'], errors='raise')
    if len(df) != 9682 or df['loja_id'].nunique() != 63:
        raise ValueError('A comparação científica requer exatamente 9.682 pares e 63 lojas. '
                         f'Recebidos {len(df)} pares e {df.loja_id.nunique()} lojas.')
    if df.duplicated(['loja_id', 'data_origem', 'data']).any():
        raise ValueError('Pares duplicados encontrados.')
    if not ((df['data'] - df['data_origem']) == pd.Timedelta(days=7)).all():
        raise ValueError('Há previsões que não correspondem ao horizonte H7.')
    for col in [real, automl, arima]:
        df[col] = pd.to_numeric(df[col], errors='raise')
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f'Valores ausentes ou não finitos em {col}.')
    return df, real, automl, arima


def calcular(df, real, automl, arima):
    # D_t positivo significa que o AutoML teve menor erro quadrático.
    diferencial = (df[real] - df[arima]) ** 2 - (df[real] - df[automl]) ** 2
    serie = pd.DataFrame({'origem': df['data_origem'], 'diferencial': diferencial})\
        .groupby('origem', sort=True)['diferencial'].mean().sort_index()
    datas = pd.date_range(serie.index.min(), serie.index.max(), freq='D')
    if len(serie) != len(datas) or not serie.index.equals(datas):
        raise ValueError('As origens não formam uma sequência diária sem lacunas; '
                         'a correção HAC exige verificação metodológica antes de prosseguir.')
    d = serie.to_numpy(dtype=float)
    n = len(d)
    if n < 15:
        raise ValueError('Quantidade insuficiente de origens para o teste.')
    mu = d.mean()
    central = d - mu
    # Variância de longo prazo Bartlett/Newey-West; lag H-1 = 6.
    lag_max = 6
    cov0 = float(np.dot(central, central) / n)
    lrv = cov0
    autocovs = []
    for lag in range(1, lag_max + 1):
        gamma = float(np.dot(central[lag:], central[:-lag]) / n)
        peso = 1 - lag / (lag_max + 1)
        lrv += 2 * peso * gamma
        autocovs.append({'lag': lag, 'autocovariancia': gamma, 'peso_bartlett': peso})
    if not np.isfinite(lrv) or lrv <= 0:
        raise ValueError(f'Variância de longo prazo não positiva: {lrv}.')
    z = float(mu / np.sqrt(lrv / n))
    p = float(norm.sf(z))
    return {
        'tipo': 'diebold_mariano_unilateral_h7',
        'diferencial': 'erro_quadratico_arima_menos_erro_quadratico_automl',
        'hipotese_alternativa': 'AutoML tem menor perda quadratica media que ARIMA',
        'horizonte': 7, 'pares': int(len(df)), 'lojas': int(df['loja_id'].nunique()),
        'origens_distintas': n, 'primeira_origem': str(serie.index.min().date()),
        'ultima_origem': str(serie.index.max().date()),
        'agregacao': 'media dos diferenciais entre lojas por origem diaria',
        'variancia': 'Newey-West/Bartlett, autocovariancias ate lag 6, denominador n',
        'lag_maximo': lag_max, 'media_diferencial': float(mu),
        'variancia_longo_prazo': float(lrv), 'estatistica_normal_z': z,
        'p_valor_unilateral': p, 'alfa': 0.05, 'rejeita_hipotese_nula': bool(p < 0.05),
        'autocovariancias': autocovs,
        'nota': 'A inferencia depende da representatividade temporal e do protocolo de avaliacao. '
                'O teste nao prova causalidade nem garante desempenho futuro.',
        'gerado_em_utc': datetime.now(timezone.utc).isoformat(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=Path)
    args = ap.parse_args()
    if args.csv:
        csv = args.csv
    else:
        possibilidades = sorted(DIR.glob('*.csv'))
        if len(possibilidades) != 1:
            raise SystemExit('Informe --csv com o arquivo de pares do ARIMA. '
                             f'Encontrados: {[p.name for p in possibilidades]}')
        csv = possibilidades[0]
    if not csv.is_file():
        raise SystemExit(f'Arquivo inexistente: {csv}')
    df, real, automl, arima = carregar(csv)
    resultado = calcular(df, real, automl, arima)
    resultado['arquivo_fonte'] = str(csv)
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    DESTINO.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    print(f'\nRelatório salvo em {DESTINO}')


if __name__ == '__main__':
    main()
