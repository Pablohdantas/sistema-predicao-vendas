
"""
Engenharia de atributos para datasets enviados à API.

Produz previsões com horizonte de sete dias.

Dias ausentes no CSV são tratados como desconhecidos,
e não como dias com vendas iguais a zero.
"""

from pathlib import Path

import pandas as pd


HORIZONTE = 7
DEFASAGENS = [7, 14, 21, 28]
JANELAS = [7, 28]

COLUNAS_HISTORICAS = [
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "media_movel_7",
    "media_movel_28",
    "observacoes_media_7",
    "observacoes_media_28",
]


def gerar_atributos_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera atributos temporais para um CSV validado.

    O DataFrame original não é modificado.
    """

    resultado = df.copy()

    resultado["data"] = pd.to_datetime(
        resultado["data"],
        format="%Y-%m-%d",
        errors="raise"
    )

    resultado["vendas"] = pd.to_numeric(
        resultado["vendas"],
        errors="raise"
    )

    resultado = resultado.sort_values(
        ["loja_id", "data"]
    ).reset_index(drop=True)

    # Atributos conhecidos a partir do calendário.
    resultado["dia_semana"] = (
        resultado["data"].dt.dayofweek
    )
    resultado["dia_mes"] = resultado["data"].dt.day
    resultado["mes"] = resultado["data"].dt.month
    resultado["semana_ano"] = (
        resultado["data"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    # Construir calendário diário por loja.
    # Datas ausentes permanecem com vendas NaN.
    historico = resultado[
        ["loja_id", "data", "vendas"]
    ].copy()

    historico = historico.rename(
        columns={"vendas": "vendas_historicas"}
    )

    historico = historico.set_index(
        ["loja_id", "data"]
    ).sort_index()

    lojas = historico.index.get_level_values(
        "loja_id"
    ).unique()

    datas = pd.date_range(
        historico.index.get_level_values("data").min(),
        historico.index.get_level_values("data").max(),
        freq="D"
    )

    indice = pd.MultiIndex.from_product(
        [lojas, datas],
        names=["loja_id", "data"]
    )

    historico = (
        historico
        .reindex(indice)
        .reset_index()
    )

    grupos = historico.groupby(
        "loja_id"
    )["vendas_historicas"]

    # Defasagens em dias de calendário.
    # Para prever t a partir de t-7,
    # nenhum lag pode ser menor que 7.
    for lag in DEFASAGENS:
        historico[f"lag_{lag}"] = (
            grupos.shift(lag)
        )


    # Médias históricas calculadas com vendas conhecidas.
    # A janela termina na data de origem da previsão.
    # Dias sem registro permanecem desconhecidos (NaN).

    minimos = {
        7: 4,
        28: 14
    }

    for janela in JANELAS:
        coluna_media = f"media_movel_{janela}"
        coluna_contagem = f"observacoes_media_{janela}"

        historico[coluna_media] = grupos.transform(
            lambda serie: (
                serie.shift(HORIZONTE)
                .rolling(
                    window=janela,
                    min_periods=minimos[janela]
                )
                .mean()
            )
        )

        historico[coluna_contagem] = grupos.transform(
            lambda serie: (
                serie.shift(HORIZONTE)
                .rolling(
                    window=janela,
                    min_periods=1
                )
                .count()
            )
        )


    resultado = resultado.merge(
        historico[
            ["loja_id", "data"] + COLUNAS_HISTORICAS
        ],
        on=["loja_id", "data"],
        how="left",
        validate="one_to_one"
    )

    resultado["data_origem"] = (
        resultado["data"]
        - pd.Timedelta(days=HORIZONTE)
    )

    if len(resultado) != len(df):
        raise ValueError(
            "A engenharia de atributos alterou "
            "a quantidade de observações."
        )

    if not (
        resultado["data_origem"]
        == resultado["data"] - pd.Timedelta(days=7)
    ).all():
        raise ValueError(
            "Foram identificadas datas de origem incorretas."
        )

    return resultado


def preparar_atributos_execucao(
    pasta_execucao: Path
) -> dict:
    """
    Lê o dataset.csv de uma execução e salva
    um arquivo com os atributos gerados.

    Não modifica o dataset original.
    """

    pasta_execucao = Path(pasta_execucao)

    caminho_entrada = pasta_execucao / "dataset.csv"
    caminho_saida = pasta_execucao / "atributos_h7.csv"

    if not caminho_entrada.is_file():
        raise FileNotFoundError(
            f"Dataset não encontrado: {caminho_entrada}"
        )

    df = pd.read_csv(
        caminho_entrada,
        dtype={"loja_id": "string"},
        parse_dates=["data"]
    )

    resultado = gerar_atributos_dataset(df)

    resultado.to_csv(
        caminho_saida,
        index=False,
        date_format="%Y-%m-%d"
    )

    return {
        "arquivo": str(caminho_saida),
        "quantidade_linhas": int(len(resultado)),
        "quantidade_lojas": int(
            resultado["loja_id"].nunique()
        ),
        "horizonte_dias": HORIZONTE,
        "atributos_historicos": COLUNAS_HISTORICAS,
        "nulos_atributos_historicos": {
            coluna: int(resultado[coluna].isna().sum())
            for coluna in COLUNAS_HISTORICAS
        }
    }
