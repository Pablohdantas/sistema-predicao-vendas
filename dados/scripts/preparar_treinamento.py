
import json
from pathlib import Path

import pandas as pd


DADOS = Path(__file__).resolve().parents[1]
PROCESSADOS = DADOS / "processados"

DATA_CORTE = pd.Timestamp("2015-01-23")
HORIZONTE = 7

NUMERICAS = [
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
]

CATEGORICAS = ["tipo_loja", "sortimento"]


def carregar_dados():
    return pd.read_csv(
        PROCESSADOS / "atributos_h7.csv",
        parse_dates=["data", "data_origem"]
    )


def dividir_dados(df):
    # O modelo só aprende com vendas conhecidas
    # até a data final do treinamento.
    treino = df.loc[
        df["data"] <= DATA_CORTE
    ].copy()

    # Cada previsão do teste deve ter sido produzida
    # após o encerramento do treinamento.
    teste = df.loc[
        df["data_origem"] >= DATA_CORTE
    ].copy()

    assert treino["data"].max() <= DATA_CORTE
    assert teste["data_origem"].min() >= DATA_CORTE

    assert (
        treino["data"].max() < teste["data"].min()
    )

    return treino, teste


def preparar_atributos(treino, teste):
    treino = treino.copy()
    teste = teste.copy()

    # A mediana é calculada somente no treino.
    mediana_distancia = treino[
        "distancia_concorrente"
    ].median()

    for conjunto in [treino, teste]:
        conjunto["distancia_concorrente"] = (
            conjunto["distancia_concorrente"]
            .fillna(mediana_distancia)
        )

    # As categorias são conhecidas somente pelo treino.
    # Valores desconhecidos no teste ficam como
    # categoria não reconhecida.
    for coluna in CATEGORICAS:
        treino[coluna] = (
            treino[coluna].fillna("desconhecido")
        )
        teste[coluna] = (
            teste[coluna].fillna("desconhecido")
        )

        categorias = sorted(
            treino[coluna].astype(str).unique()
        )

        treino[coluna] = pd.Categorical(
            treino[coluna].astype(str),
            categories=categorias
        )

        teste[coluna] = pd.Categorical(
            teste[coluna].astype(str),
            categories=categorias
        )

    # One-hot encoding com esquema fixado pelo treino.
    x_treino = pd.get_dummies(
        treino[NUMERICAS + CATEGORICAS],
        columns=CATEGORICAS,
        dtype="int8"
    )

    x_teste = pd.get_dummies(
        teste[NUMERICAS + CATEGORICAS],
        columns=CATEGORICAS,
        dtype="int8"
    )

    x_teste = x_teste.reindex(
        columns=x_treino.columns,
        fill_value=0
    )

    # Missing values dos lags e médias são mantidos.
    # LightGBM e XGBoost aceitam NaN.
    y_treino = treino["vendas"].copy()
    y_teste = teste["vendas"].copy()

    assert x_treino.columns.equals(x_teste.columns)
    assert not x_treino.isna().all(axis=1).any()
    assert not x_teste.isna().all(axis=1).any()

    return (
        x_treino, x_teste,
        y_treino, y_teste,
        mediana_distancia
    )


def main():
    df = carregar_dados()
    treino, teste = dividir_dados(df)

    (
        x_treino, x_teste,
        y_treino, y_teste,
        mediana_distancia
    ) = preparar_atributos(treino, teste)

    saida = PROCESSADOS / "treinamento"
    saida.mkdir(parents=True, exist_ok=True)

    x_treino.to_csv(saida / "X_treino.csv", index=False)
    x_teste.to_csv(saida / "X_teste.csv", index=False)

    y_treino.to_frame("vendas").to_csv(
        saida / "y_treino.csv", index=False
    )

    y_teste.to_frame("vendas").to_csv(
        saida / "y_teste.csv", index=False
    )

    # Metadados preservam a identificação de cada
    # previsão para avaliações futuras por loja/data.
    colunas_meta = [
        "loja_id", "data", "data_origem"
    ]

    treino[colunas_meta].to_csv(
        saida / "metadata_treino.csv", index=False
    )

    teste[colunas_meta].to_csv(
        saida / "metadata_teste.csv", index=False
    )

    relatorio = {
        "data_corte": str(DATA_CORTE.date()),
        "horizonte": HORIZONTE,
        "linhas_treino": len(x_treino),
        "linhas_teste": len(x_teste),
        "primeira_data_teste": str(
            teste["data"].min().date()
        ),
        "primeira_origem_teste": str(
            teste["data_origem"].min().date()
        ),
        "quantidade_atributos": len(x_treino.columns),
        "atributos": x_treino.columns.tolist(),
        "mediana_distancia_treino": float(
            mediana_distancia
        ),
        "nulos_treino": {
            k: int(v)
            for k, v in x_treino.isna().sum().items()
            if v > 0
        },
        "nulos_teste": {
            k: int(v)
            for k, v in x_teste.isna().sum().items()
            if v > 0
        }
    }

    (saida / "relatorio_preparacao.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(json.dumps(
        relatorio, indent=2, ensure_ascii=False
    ))

    print("\nPreparação concluída!")
    print(f"Arquivos salvos em: {saida}")


if __name__ == "__main__":
    main()
