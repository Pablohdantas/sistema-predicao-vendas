
from pathlib import Path
import json

import pandas as pd


DADOS = Path(__file__).resolve().parents[1]
PROCESSADOS = DADOS / "processados"

CAMINHO_TEMPLATE = PROCESSADOS / "template_rossmann.csv"


def dividir_temporalmente(
    df: pd.DataFrame,
    proporcao_treino: float = 0.8
):
    """Divide os dados por datas, sem embaralhamento."""

    datas = sorted(df["data"].dropna().unique())

    if len(datas) < 2:
        raise ValueError("São necessárias pelo menos duas datas.")

    indice_corte = int(len(datas) * proporcao_treino)

    if not 0 < indice_corte < len(datas):
        raise ValueError("Proporção de treino inválida.")

    data_corte = datas[indice_corte - 1]

    treino = df[df["data"] <= data_corte].copy()
    teste = df[df["data"] > data_corte].copy()

    if treino.empty or teste.empty:
        raise ValueError("Treino ou teste ficou vazio.")

    return treino, teste, data_corte


def main():
    df = pd.read_csv(
        CAMINHO_TEMPLATE,
        parse_dates=["data"]
    )

    treino, teste, data_corte = dividir_temporalmente(df)

    relatorio = {
        "data_corte": str(data_corte.date()),
        "periodo_treino": [
            str(treino["data"].min().date()),
            str(treino["data"].max().date())
        ],
        "periodo_teste": [
            str(teste["data"].min().date()),
            str(teste["data"].max().date())
        ],
        "linhas_treino": len(treino),
        "linhas_teste": len(teste),
        "datas_treino": treino["data"].nunique(),
        "datas_teste": teste["data"].nunique(),
        "lojas_treino": treino["loja_id"].nunique(),
        "lojas_teste": teste["loja_id"].nunique()
    }

    PROCESSADOS.mkdir(parents=True, exist_ok=True)

    (PROCESSADOS / "divisao_temporal.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(json.dumps(relatorio, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
