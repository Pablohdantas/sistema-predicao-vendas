
"""
Avaliação do modelo vencedor no teste temporal externo.

Usa o modelo já treinado e salvo pela execução.
Não reotimiza, não ajusta preprocessamento e não
utiliza as vendas reais do teste como atributos.

Salva previsões e métricas para o dashboard.
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.app.services.divisao_temporal_service import (
    dividir_dataset_temporalmente
)
from backend.app.services.preprocessamento_service import (
    transformar_atributos
)


def avaliar_modelo_externo(pasta_execucao: Path) -> dict:
    pasta = Path(pasta_execucao)

    caminho_atributos = pasta / "atributos_h7.csv"
    caminho_modelo = pasta / "modelo_vencedor.joblib"

    if not caminho_atributos.is_file():
        raise FileNotFoundError(
            "Atributos da execução não encontrados."
        )

    if not caminho_modelo.is_file():
        raise FileNotFoundError(
            "Modelo vencedor não encontrado. "
            "Execute o treinamento definitivo primeiro."
        )

    artefato = joblib.load(caminho_modelo)

    if not isinstance(artefato, dict):
        raise ValueError("Artefato de modelo inválido.")

    for chave in (
        "algoritmo",
        "modelo",
        "preprocessamento",
        "colunas_modelo"
    ):
        if chave not in artefato:
            raise ValueError(
                f"Artefato sem o campo: {chave}"
            )

    dados = pd.read_csv(
        caminho_atributos,
        dtype={"loja_id": "string"},
        parse_dates=["data", "data_origem"]
    )

    _, teste, divisao = dividir_dataset_temporalmente(
        dados
    )

    colunas_esperadas = list(
        artefato["colunas_modelo"]
    )

    x_teste = transformar_atributos(
        teste,
        artefato["preprocessamento"]
    )

    if list(x_teste.columns) != colunas_esperadas:
        raise ValueError(
            "As colunas do teste diferem das colunas "
            "utilizadas pelo modelo vencedor."
        )

    y_real = pd.to_numeric(
        teste["vendas"],
        errors="raise"
    ).to_numpy(dtype=float)

    y_previsto = np.asarray(
        artefato["modelo"].predict(x_teste),
        dtype=float
    ).reshape(-1)

    if len(y_real) != len(y_previsto):
        raise ValueError(
            "Quantidade de previsões inconsistente."
        )

    if not (
        np.isfinite(y_real).all()
        and np.isfinite(y_previsto).all()
    ):
        raise ValueError(
            "A avaliação contém valores não finitos."
        )

    erros = y_real - y_previsto

    rmse = float(
        np.sqrt(np.mean(np.square(erros)))
    )
    mae = float(
        np.mean(np.abs(erros))
    )

    previsoes = teste[
        ["loja_id", "data_origem", "data", "vendas"]
    ].copy()

    previsoes = previsoes.rename(
        columns={
            "data": "data_prevista",
            "vendas": "vendas_reais"
        }
    )

    previsoes["vendas_previstas"] = y_previsto
    previsoes["erro"] = erros
    previsoes["erro_absoluto"] = np.abs(erros)

    previsoes = previsoes.sort_values(
        ["data_prevista", "loja_id"]
    ).reset_index(drop=True)

    metricas = {
        "execucao_id": pasta.name,
        "algoritmo": artefato["algoritmo"],
        "tipo_avaliacao": "teste_temporal_externo",
        "horizonte_dias": 7,
        "rmse": rmse,
        "mae": mae,
        "quantidade_previsoes": int(
            len(previsoes)
        ),
        "quantidade_lojas": int(
            previsoes["loja_id"].nunique()
        ),
        "periodo_teste": {
            "inicio": str(
                previsoes["data_prevista"].min().date()
            ),
            "fim": str(
                previsoes["data_prevista"].max().date()
            )
        },
        "divisao_temporal": divisao,
        "arquivo_previsoes": "previsoes_teste_externo.csv"
    }

    caminho_previsoes = (
        pasta / "previsoes_teste_externo.csv"
    )
    caminho_metricas = (
        pasta / "avaliacao_externa.json"
    )

    # Escreve primeiro em arquivos temporários.
    temporario_csv = pasta / ".previsoes_teste_externo.tmp"
    temporario_json = pasta / ".avaliacao_externa.tmp"

    try:
        previsoes.to_csv(
            temporario_csv,
            index=False,
            date_format="%Y-%m-%d"
        )

        temporario_json.write_text(
            json.dumps(
                metricas,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        os.replace(
            temporario_csv,
            caminho_previsoes
        )

        os.replace(
            temporario_json,
            caminho_metricas
        )

    finally:
        temporario_csv.unlink(missing_ok=True)
        temporario_json.unlink(missing_ok=True)

    return metricas
