
from pathlib import Path

import numpy as np
import pandas as pd


DADOS = Path(__file__).resolve().parents[1]
PROCESSADOS = DADOS / "processados"
BRUTOS = DADOS / "brutos"

JANELAS = [7, 28]
HORIZONTE = 7


def carregar_dados():
    atributos = pd.read_csv(
        PROCESSADOS / "atributos_h7.csv",
        parse_dates=["data", "data_origem"]
    )

    raw = pd.read_csv(
        BRUTOS / "train.csv",
        usecols=["Store", "Date", "Sales", "Open"],
        parse_dates=["Date"]
    )

    raw["vendas_historicas"] = raw["Sales"].astype(float)
    raw.loc[raw["Open"] == 0, "vendas_historicas"] = 0.0

    historico = raw.set_index(
        ["Store", "Date"]
    )["vendas_historicas"]

    return atributos, historico


def calcular_media_esperada(linha, historico, janela):
    loja = linha["loja_id"]
    origem = linha["data_origem"]

    # A janela termina na origem da previsão.
    datas = pd.date_range(
        end=origem,
        periods=janela,
        freq="D"
    )

    indice = pd.MultiIndex.from_product(
        [[loja], datas],
        names=["Store", "Date"]
    )

    valores = historico.reindex(indice)

    # A implementação usa min_periods=janela:
    # uma data histórica ausente invalida a média.
    if valores.isna().any():
        return np.nan

    return float(valores.mean())


def main():
    atributos, historico = carregar_dados()

    for janela in JANELAS:
        coluna = f"media_movel_{janela}"

        # Inclui exemplos com valores e valores ausentes.
        validas = atributos[atributos[coluna].notna()]
        ausentes = atributos[atributos[coluna].isna()]

        amostra = pd.concat([
            validas.sample(
                n=min(200, len(validas)),
                random_state=42
            ),
            ausentes.sample(
                n=min(100, len(ausentes)),
                random_state=42
            )
        ])

        erros = []

        for _, linha in amostra.iterrows():
            assert (
                linha["data_origem"]
                == linha["data"] - pd.Timedelta(days=HORIZONTE)
            )

            esperado = calcular_media_esperada(
                linha, historico, janela
            )
            obtido = linha[coluna]

            if pd.isna(esperado) and pd.isna(obtido):
                continue

            if not np.isclose(
                esperado, obtido,
                rtol=1e-9, atol=1e-6,
                equal_nan=True
            ):
                erros.append({
                    "loja": linha["loja_id"],
                    "data": str(linha["data"].date()),
                    "esperado": esperado,
                    "obtido": obtido
                })

        assert not erros, (
            f"{coluna}: {len(erros)} divergências. "
            f"Exemplo: {erros[:3]}"
        )

        print(
            f"{coluna}: OK "
            f"({len(amostra)} observações verificadas)"
        )

    print("\nTodos os testes de médias móveis passaram!")


if __name__ == "__main__":
    main()
