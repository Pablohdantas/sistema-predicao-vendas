
"""
Pré-processamento dinâmico para LightGBM e XGBoost.

O esquema é definido a partir do conjunto de treinamento.
O conjunto de teste utiliza exatamente esse mesmo esquema.
"""

import numpy as np
import pandas as pd


NUMERICAS_POSSIVEIS = [
    "promocao",
    "promocao_continua",
    "mes_renovacao_promocao",
    "feriado",
    "feriado_escolar",
    "distancia_concorrente",
    "dia_semana",
    "dia_mes",
    "mes",
    "semana_ano",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "media_movel_7",
    "media_movel_28",
    "observacoes_media_7",
    "observacoes_media_28",
]

CATEGORICAS_POSSIVEIS = [
    "tipo_loja",
    "sortimento",
]


def ajustar_preprocessamento(
    treino: pd.DataFrame
) -> dict:
    """
    Aprende o esquema de atributos somente no treino.
    """

    numericas = [
        coluna
        for coluna in NUMERICAS_POSSIVEIS
        if coluna in treino.columns
    ]

    categoricas = [
        coluna
        for coluna in CATEGORICAS_POSSIVEIS
        if coluna in treino.columns
    ]

    if not numericas:
        raise ValueError(
            "Nenhum atributo numérico disponível."
        )

    medianas = {}

    for coluna in numericas:
        valores = pd.to_numeric(
            treino[coluna],
            errors="raise"
        )

        mediana = valores.median()

        # Se a coluna inteira estiver ausente,
        # o modelo utilizará NaN.
        medianas[coluna] = (
            float(mediana)
            if pd.notna(mediana)
            else None
        )

    categorias_treino = {}

    for coluna in categoricas:
        categorias_treino[coluna] = sorted(
            treino[coluna]
            .fillna("desconhecido")
            .astype(str)
            .unique()
            .tolist()
        )

    preprocessamento = {
        "colunas_numericas": numericas,
        "colunas_categoricas": categoricas,
        "medianas": medianas,
        "categorias_treino": categorias_treino,
    }

    x_treino = transformar_atributos(
        treino,
        preprocessamento,
        definir_colunas=True
    )

    preprocessamento["colunas_modelo"] = (
        x_treino.columns.tolist()
    )

    return preprocessamento


def transformar_atributos(
    dados: pd.DataFrame,
    preprocessamento: dict,
    definir_colunas: bool = False
) -> pd.DataFrame:
    """
    Transforma os dados usando o esquema do treinamento.
    """

    numericas = preprocessamento[
        "colunas_numericas"
    ]

    categoricas = preprocessamento[
        "colunas_categoricas"
    ]

    colunas = numericas + categoricas

    faltantes = [
        coluna
        for coluna in colunas
        if coluna not in dados.columns
    ]

    if faltantes:
        raise ValueError(
            f"Atributos ausentes: {faltantes}"
        )

    df = dados[colunas].copy()

    for coluna in numericas:
        df[coluna] = pd.to_numeric(
            df[coluna],
            errors="raise"
        )

        # Apenas distância é preenchida pela mediana.
        # Lags e médias móveis mantêm NaN.
        if coluna == "distancia_concorrente":
            mediana = preprocessamento[
                "medianas"
            ][coluna]

            if mediana is not None:
                df[coluna] = df[coluna].fillna(
                    mediana
                )

    for coluna in categoricas:
        valores = (
            df[coluna]
            .fillna("desconhecido")
            .astype(str)
        )

        df[coluna] = pd.Categorical(
            valores,
            categories=preprocessamento[
                "categorias_treino"
            ][coluna]
        )

    x = pd.get_dummies(
        df,
        columns=categoricas,
        dtype="int8"
    )

    if not definir_colunas:
        x = x.reindex(
            columns=preprocessamento[
                "colunas_modelo"
            ],
            fill_value=0
        )

    if x.isna().all(axis=1).any():
        raise ValueError(
            "Foram encontradas observações "
            "sem nenhum atributo disponível."
        )

    return x
