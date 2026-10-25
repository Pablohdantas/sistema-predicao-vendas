"""Diagnostico somente-leitura antes do experimento cientifico Rossmann H7.

Uso: python dados/scripts/diagnosticar_experimento_rossmann.py \
  --execucao 17b9466beac84f039d97e8942462eb0d

Nao treina modelos, nao altera arquivos e nao carrega os 844 mil registros
em memoria de uma vez. Imprime JSON para registrar no TG.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def projeto() -> Path:
    return Path(__file__).resolve().parents[2]


def diagnosticar(raiz: Path, execucao: str | None) -> dict:
    amostra = raiz / 'dados/processados/amostra_arima.csv'
    template = raiz / 'dados/processados/template_rossmann.csv'
    if not amostra.is_file() or not template.is_file():
        raise FileNotFoundError('Nao encontrei amostra_arima.csv e/ou template_rossmann.csv')

    lojas_amostra = pd.read_csv(amostra, usecols=['loja_id'])['loja_id'].astype(int)
    if lojas_amostra.duplicated().any():
        raise ValueError('amostra_arima.csv contem lojas duplicadas')
    ids = set(lojas_amostra)
    quantidades = Counter()
    minimo = None
    maximo = None
    total = 0
    for bloco in pd.read_csv(template, usecols=['loja_id', 'data', 'vendas'], chunksize=75000):
        bloco['loja_id'] = pd.to_numeric(bloco['loja_id'], errors='raise').astype(int)
        data = pd.to_datetime(bloco['data'], errors='raise')
        if bloco['vendas'].isna().any():
            raise ValueError('Encontradas vendas ausentes no template')
        total += len(bloco)
        minimo = data.min() if minimo is None else min(minimo, data.min())
        maximo = data.max() if maximo is None else max(maximo, data.max())
        quantidades.update(bloco['loja_id'].tolist())
    presentes = set(quantidades)
    linhas_amostra = sum(quantidades[x] for x in ids)
    resposta = {
        'status': 'diagnostico_sem_treinamento',
        'template': {'linhas': total, 'lojas': len(presentes), 'inicio': str(minimo.date()), 'fim': str(maximo.date())},
        'amostra_arima': {'lojas': len(ids), 'lojas_ausentes_template': sorted(ids - presentes), 'linhas_template_da_amostra': linhas_amostra},
        'limite_produto': {'linhas': 5000, 'base_completa_ultrapassa': total > 5000, 'amostra_63_ultrapassa': linhas_amostra > 5000},
        'previsoes_externas': None,
        'alertas': [],
    }
    if execucao:
        arquivo = raiz / 'backend/execucoes' / execucao / 'multihorizonte/previsoes_teste.csv'
        if not arquivo.is_file():
            raise FileNotFoundError(f'Execucao nao encontrada: {arquivo}')
        df = pd.read_csv(arquivo, usecols=['loja_id', 'data_origem', 'data', 'horizonte_dias', 'vendas', 'vendas_previstas'])
        h7 = df.loc[df['horizonte_dias'] == 7].copy()
        h7['loja_id'] = h7['loja_id'].astype(int)
        h7['data_origem'] = pd.to_datetime(h7['data_origem'], errors='raise')
        h7['data'] = pd.to_datetime(h7['data'], errors='raise')
        incoerentes = int(((h7['data'] - h7['data_origem']).dt.days != 7).sum())
        duplicados = int(h7.duplicated(['loja_id', 'data_origem', 'data']).sum())
        lojas_h7 = set(h7['loja_id'])
        resposta['previsoes_externas'] = {
            'execucao': execucao,
            'linhas_todos_horizontes': len(df),
            'pares_h7': len(h7),
            'lojas_h7': len(lojas_h7),
            'lojas_h7_que_pertencem_amostra_63': len(lojas_h7 & ids),
            'pares_h7_da_amostra_63': int(h7['loja_id'].isin(ids).sum()),
            'lojas_da_amostra_sem_previsao_h7': sorted(ids - lojas_h7),
            'pares_h7_com_datas_incoerentes': incoerentes,
            'pares_h7_duplicados': duplicados,
        }
        if incoerentes or duplicados:
            resposta['alertas'].append('Ha inconsistencias nos pares H7: revisar antes de comparar.')
        if not ids.issubset(lojas_h7):
            resposta['alertas'].append('Esta execucao NAO cobre a amostra cientifica de 63 lojas.')
    if total > 5000:
        resposta['alertas'].append('Pipeline web atual limitado a 5000 linhas: nao aumentar sem revisar memoria e processamento.')
    resposta['alertas'].append('Limite de 7 horizontes cria varios exemplos por data; totais de linhas brutas nao equivalem aos exemplos gerados.')
    return resposta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execucao', default=None, help='ID de uma execucao do AutoML para verificar cobertura H7')
    args = parser.parse_args()
    print(json.dumps(diagnosticar(projeto(), args.execucao), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
