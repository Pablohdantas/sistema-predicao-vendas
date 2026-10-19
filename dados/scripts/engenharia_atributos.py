
from pathlib import Path

import pandas as pd

DADOS = Path(__file__).resolve().parents[1]
BRUTOS = DADOS / "brutos"
PROCESSADOS = DADOS / "processados"

HORIZONTE = 7
DEFASAGENS = [7, 14, 21, 28]
JANELAS = [7, 28]


def carregar_dados():
    df = pd.read_csv(
        PROCESSADOS / "template_rossmann.csv",
        parse_dates=["data"]
    )

    raw = pd.read_csv(
        BRUTOS / "train.csv",
        usecols=["Store", "Date", "Sales", "Open"],
        parse_dates=["Date"]
    )

    return df, raw


def construir_historico(raw):
    """Preserva dias fechados como zero e lacunas como NaN."""

    historico = raw.rename(
        columns={
            "Store": "loja_id",
            "Date": "data",
            "Sales": "vendas_historicas",
            "Open": "aberta"
        }
    ).copy()

    historico.loc[
        historico["aberta"] == 0,
        "vendas_historicas"
    ] = 0

    historico = historico.set_index(
        ["loja_id", "data"]
    ).sort_index()

    lojas = historico.index.get_level_values(0).unique()

    datas = pd.date_range(
        historico.index.get_level_values(1).min(),
        historico.index.get_level_values(1).max(),
        freq="D"
    )

    indice = pd.MultiIndex.from_product(
        [lojas, datas],
        names=["loja_id", "data"]
    )

    return historico.reindex(indice).reset_index()


def gerar_atributos(df, historico):
    df = df.copy()

    # Variáveis de calendário
    df["dia_semana"] = df["data"].dt.dayofweek
    df["dia_mes"] = df["data"].dt.day
    df["mes"] = df["data"].dt.month
    df["semana_ano"] = df["data"].dt.isocalendar().week.astype(int)

    # Os atributos são construídos usando a série diária
    # regularizada, e não posições entre dias abertos.
    historico = historico.sort_values(
        ["loja_id", "data"]
    ).copy()

    grupos = historico.groupby("loja_id")["vendas_historicas"]

    for lag in DEFASAGENS:
        historico[f"lag_{lag}"] = grupos.shift(lag)

    # Para uma previsão do dia t feita no dia t-7,
    # a janela mais recente termina em t-7.
    for janela in JANELAS:
        historico[f"media_movel_{janela}"] = (
            grupos.transform(
                lambda serie: (
                    serie.shift(HORIZONTE)
                    .rolling(
                        window=janela,
                        min_periods=janela
                    )
                    .mean()
                )
            )
        )

    colunas = [
        "loja_id",
        "data",
        *[f"lag_{lag}" for lag in DEFASAGENS],
        *[f"media_movel_{janela}" for janela in JANELAS]
    ]

    df = df.merge(
        historico[colunas],
        on=["loja_id", "data"],
        how="left",
        validate="one_to_one"
    )

    df["data_origem"] = (
        df["data"] - pd.Timedelta(days=HORIZONTE)
    )

    return df


def main():
    df, raw = carregar_dados()

    historico = construir_historico(raw)
    resultado = gerar_atributos(df, historico)

    assert len(resultado) == len(df), (
        "A engenharia de atributos alterou a quantidade de linhas."
    )

    assert (
        resultado["data_origem"]
        == resultado["data"] - pd.Timedelta(days=7)
    ).all()

    caminho = PROCESSADOS / "atributos_h7.csv"

    resultado.to_csv(
        caminho,
        index=False,
        date_format="%Y-%m-%d"
    )

    print("\nEngenharia de atributos concluída!")
    print(f"Linhas: {len(resultado)}")
    print(f"Lojas: {resultado['loja_id'].nunique()}")
    print(f"Arquivo: {caminho}")

    atributos = [
        *[f"lag_{lag}" for lag in DEFASAGENS],
        *[f"media_movel_{janela}" for janela in JANELAS]
    ]

    print("\nValores ausentes nos atributos:")
    print(resultado[atributos].isna().sum())

    print("\nExemplo:")
    print(
        resultado[
            ["loja_id", "data", "data_origem", "vendas", *atributos]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
