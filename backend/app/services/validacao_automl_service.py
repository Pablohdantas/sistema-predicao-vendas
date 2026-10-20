
"""
Prepara a validação temporal interna de cada execução AutoML.

Reutiliza as regras existentes em
dados/scripts/validacao_temporal.py.

Não executa treinamento nem utiliza o teste externo.
"""

import json
from pathlib import Path

import pandas as pd

from dados.scripts.validacao_temporal import criar_divisoes

from backend.app.services.divisao_temporal_service import (
    dividir_dataset_temporalmente
)


def preparar_validacao_automl(
    pasta_execucao: Path
) -> dict:

    pasta_execucao = Path(pasta_execucao)

    caminho_atributos = (
        pasta_execucao / "atributos_h7.csv"
    )

    if not caminho_atributos.is_file():
        raise FileNotFoundError(
            f"Atributos não encontrados: {caminho_atributos}"
        )

    df = pd.read_csv(
        caminho_atributos,
        parse_dates=["data", "data_origem"]
    )

    treino, teste, relatorio_divisao = (
        dividir_dataset_temporalmente(df)
    )

    # Índices das dobras referem-se exatamente
    # à ordem do DataFrame de treinamento.
    treino = treino.reset_index(drop=True)

    divisoes, relatorio_dobras = criar_divisoes(
        treino[["loja_id", "data", "data_origem"]]
    )

    relatorio = {
        "execucao_id": pasta_execucao.name,
        "horizonte_dias": 7,
        "data_corte": relatorio_divisao["data_corte"],
        "linhas_treino_externo": int(len(treino)),
        "linhas_teste_externo": int(len(teste)),
        "quantidade_dobras": len(divisoes),
        "dobras": relatorio_dobras
    }

    caminho_relatorio = (
        pasta_execucao / "validacao_automl.json"
    )

    caminho_relatorio.write_text(
        json.dumps(
            relatorio,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    return relatorio
