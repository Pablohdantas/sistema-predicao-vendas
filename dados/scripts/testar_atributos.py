
from pathlib import Path
import pandas as pd

DADOS = Path(__file__).resolve().parents[1]
PROCESSADOS = DADOS / "processados"
BRUTOS = DADOS / "brutos"

HORIZONTE = 7
DEFASAGENS = [7, 14, 21, 28]


def main():
    df = pd.read_csv(
        PROCESSADOS / "atributos_h7.csv",
        parse_dates=["data", "data_origem"]
    )

    raw = pd.read_csv(
        BRUTOS / "train.csv",
        usecols=["Store", "Date", "Sales", "Open"],
        parse_dates=["Date"]
    )

    # 1. Conferir a origem de cada previsão.
    assert (
        df["data_origem"]
        == df["data"] - pd.Timedelta(days=HORIZONTE)
    ).all(), "Existem datas de origem incorretas."

    # 2. Conferir a unicidade das observações.
    assert not df.duplicated(
        ["loja_id", "data"]
    ).any(), "Foram encontradas observações duplicadas."

    # 3. Recuperar vendas históricas por loja e data.
    raw["vendas_esperadas"] = raw["Sales"].astype(float)
    raw.loc[
        raw["Open"] == 0,
        "vendas_esperadas"
    ] = 0.0

    historico = raw.rename(
        columns={
            "Store": "loja_id",
            "Date": "data"
        }
    )[["loja_id", "data", "vendas_esperadas"]]

    # 4. Verificar cada lag contra a data correta.
    for lag in DEFASAGENS:
        verificacao = df[
            ["loja_id", "data", f"lag_{lag}"]
        ].copy()

        verificacao["data_historica"] = (
            verificacao["data"]
            - pd.Timedelta(days=lag)
        )

        referencia = historico.rename(
            columns={"data": "data_historica"}
        )

        verificacao = verificacao.merge(
            referencia,
            on=["loja_id", "data_historica"],
            how="left",
            validate="many_to_one"
        )

        diferentes = ~(
            verificacao[f"lag_{lag}"].eq(
                verificacao["vendas_esperadas"]
            )
            |
            (
                verificacao[f"lag_{lag}"].isna()
                & verificacao["vendas_esperadas"].isna()
            )
        )

        assert not diferentes.any(), (
            f"lag_{lag}: {int(diferentes.sum())} valores incorretos."
        )

        print(f"lag_{lag}: OK")

    # 5. Nenhuma defasagem pode consultar uma data futura.
    for lag in DEFASAGENS:
        assert lag >= HORIZONTE, (
            f"lag_{lag} utiliza informação posterior à origem."
        )

    print("\nTodos os testes de defasagem passaram!")


if __name__ == "__main__":
    main()
