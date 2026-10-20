
"""
Serviço de previsão de vendas com XGBoost.

Carrega o modelo global treinado e aplica
o mesmo pré-processamento utilizado no treinamento.

Horizonte fixo: 7 dias.
"""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[3]

CAMINHO_MODELO = (
    RAIZ / "backend" / "modelos" / "modelo_global.joblib"
)


@lru_cache(maxsize=1)
def carregar_modelo():
    """
    Carrega o modelo uma única vez por processo.
    """

    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {CAMINHO_MODELO}"
        )

    artefato = joblib.load(CAMINHO_MODELO)

    if artefato["algoritmo"] != "xgboost":
        raise ValueError(
            "O artefato não contém um modelo XGBoost."
        )

    if artefato["horizonte_dias"] != 7:
        raise ValueError(
            "Horizonte do modelo diferente de 7 dias."
        )

    return artefato


def preparar_atributos(dados, preprocessamento):
    """
    Reproduz o pré-processamento do treinamento.
    """

    if isinstance(dados, dict):
        dados = pd.DataFrame([dados])

    df = dados.copy()

    colunas_numericas = preprocessamento[
        "colunas_numericas"
    ]

    colunas_categoricas = preprocessamento[
        "colunas_categoricas"
    ]

    categorias_treino = preprocessamento[
        "categorias_treino"
    ]

    colunas_modelo = preprocessamento[
        "colunas_modelo"
    ]

    colunas_necessarias = (
        colunas_numericas + colunas_categoricas
    )

    faltando = [
        coluna
        for coluna in colunas_necessarias
        if coluna not in df.columns
    ]

    if faltando:
        raise ValueError(
            f"Atributos ausentes: {faltando}"
        )

    # Imputação utilizada no treinamento.
    df["distancia_concorrente"] = (
        df["distancia_concorrente"].fillna(
            preprocessamento["mediana_distancia"]
        )
    )

    # Mesmas categorias utilizadas no treinamento.
    for coluna in colunas_categoricas:
        categorias = categorias_treino[coluna]

        valores_invalidos = (
            df[coluna].notna()
            & ~df[coluna].isin(categorias)
        )

        if valores_invalidos.any():
            raise ValueError(
                f"Categoria inválida em {coluna}."
            )

        df[coluna] = pd.Categorical(
            df[coluna],
            categories=categorias
        )

    # One-hot encoding.
    x = pd.get_dummies(
        df[colunas_necessarias],
        columns=colunas_categoricas,
        dummy_na=False,
        dtype="int8"
    )

    # Mantém exatamente as mesmas 23 colunas
    # e a mesma ordem do treinamento.
    x = x.reindex(
        columns=colunas_modelo,
        fill_value=0
    )

    if x.shape[1] != 23:
        raise ValueError(
            "Quantidade incorreta de atributos."
        )

    return x


def prever_vendas(dados):
    """
    Recebe os atributos de uma observação
    e retorna a previsão do XGBoost.

    Atenção:
    Os atributos históricos precisam ter sido
    calculados utilizando somente informações
    disponíveis até a data de origem da previsão.
    """

    artefato = carregar_modelo()

    x = preparar_atributos(
        dados,
        artefato["preprocessamento"]
    )

    modelo = artefato["modelo"]

    previsoes = np.asarray(
        modelo.predict(x),
        dtype=float
    )

    if not np.isfinite(previsoes).all():
        raise ValueError(
            "O modelo retornou previsões inválidas."
        )

    return previsoes.tolist()
