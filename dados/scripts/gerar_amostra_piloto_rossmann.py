"""Amostra fixa Rossmann para ensaio técnico do pipeline web (não evidência científica final).

Uso, na raiz: python dados/scripts/gerar_amostra_piloto_rossmann.py
Não sobrescreve template_rossmann.csv.
"""
import json
from pathlib import Path
import pandas as pd

DADOS = Path(__file__).resolve().parents[1]
ORIGEM = DADOS / 'processados/template_rossmann.csv'
DESTINO = DADOS / 'processados/amostra_piloto_rossmann.csv'
MANIFESTO = DADOS / 'processados/amostra_piloto_rossmann.json'
LOJAS = (1, 2, 3, 4)  # IDs fixados antes de inspecionar desempenho dos modelos


def main():
    dados = pd.read_csv(ORIGEM, dtype={'loja_id': 'string', 'data': 'string'})
    if not {'loja_id', 'data', 'vendas'}.issubset(dados.columns):
        raise ValueError('Template sem colunas obrigatórias.')
    pd.to_datetime(dados['data'], format='%Y-%m-%d', errors='raise')
    ids = pd.to_numeric(dados['loja_id'], errors='raise')
    amostra = dados.loc[ids.isin(LOJAS)].copy()
    if set(pd.to_numeric(amostra['loja_id']).astype(int)) != set(LOJAS):
        raise ValueError('Uma ou mais lojas predefinidas não foram encontradas.')
    if amostra.duplicated(['loja_id', 'data']).any():
        raise ValueError('Datas duplicadas para a mesma loja.')
    if len(amostra) > 5000:
        raise ValueError(f'Amostra com {len(amostra)} registros excede o limite atual de 5.000.')
    amostra = amostra.sort_values(['loja_id', 'data']).reset_index(drop=True)
    grupos = amostra.groupby('loja_id')['data'].agg(['count', 'min', 'max'])
    periodos = {}
    for loja, row in grupos.iterrows():
        abrangencia = (pd.Timestamp(row['max']) - pd.Timestamp(row['min'])).days + 1
        if abrangencia < 180:
            raise ValueError(f'Loja {loja}: abrangência inferior a 180 dias.')
        periodos[str(loja)] = {'registros': int(row['count']), 'inicio': row['min'], 'fim': row['max']}
    manifesto = {
        'finalidade': 'ensaio_tecnico_nao_experimento_cientifico_final',
        'criterio': 'lojas 1, 2, 3, 4 fixadas previamente',
        'origem': 'dados/processados/template_rossmann.csv',
        'lojas': list(LOJAS),
        'linhas': int(len(amostra)),
        'periodos_por_loja': periodos,
    }
    amostra.to_csv(DESTINO, index=False)
    MANIFESTO.write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifesto, ensure_ascii=False, indent=2))
    print(f'Arquivo de amostra: {DESTINO}')


if __name__ == '__main__':
    main()
