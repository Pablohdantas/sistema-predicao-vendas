
"""
Avaliação externa do modelo global Rossmann.

- Carrega o modelo previamente treinado;
- Seleciona apenas previsões com origem a partir
  do corte temporal de 2015-01-23;
- Reproduz o pré-processamento salvo no modelo;
- Calcula RMSE e MAE;
- Salva as previsões e o relatório.

Não treina nem modifica o modelo.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


RAIZ = Path(__file__).resolve().parents[3]

PROCESSADOS = RAIZ / "dados" / "processados"
MODELOS = RAIZ / "backend" / "modelos"
RESULTADOS = RAIZ / "backend" / "resultados"

ARQUIVO_MODELO = MODELOS / "modelo_global.joblib"

DATA_CORTE = pd.Timestamp("2015-01-23")
HORIZONTE = 7


def carregar_modelo():
    artefato = joblib.load(ARQUIVO_MODELO)

    if artefato["horizonte_dias"] != HORIZONTE:
        raise ValueError("Horizonte incompatível.")

    if (
        artefato["data_corte_treinamento"]
        != str(DATA_CORTE.date())
    ):
        raise ValueError("Corte temporal incompatível.")

    return artefato


def carregar_teste():
    print("Carregando conjunto externo de teste...")

    df = pd.read_csv(
        PROCESSADOS / "atributos_h7.csv",
        parse_dates=["data", "data_origem"],
    )

    teste = df.loc[
        (df["data"] > DATA_CORTE)
        & (df["data_origem"] >= DATA_CORTE)
    ].copy()

    if teste.empty:
        raise ValueError("Conjunto de teste vazio.")

    diferenca = (
        teste["data"] - teste["data_origem"]
    ).dt.days

    if not (diferenca == HORIZONTE).all():
        raise ValueError("Horizonte de 7 dias inconsistente.")

    print(f"Registros: {len(teste)}")
    print(f"Lojas: {teste['loja_id'].nunique()}")
    print(
        "Período dos alvos: "
        f"{teste['data'].min().date()} até "
        f"{teste['data'].max().date()}"
    )

    return teste


def preparar_teste(teste, preprocessamento):
    dados = teste.copy()

    numericas = preprocessamento["colunas_numericas"]
    categoricas = preprocessamento["colunas_categoricas"]
    categorias_treino = preprocessamento["categorias_treino"]
    colunas_modelo = preprocessamento["colunas_modelo"]

    dados["distancia_concorrente"] = (
        dados["distancia_concorrente"].fillna(
            preprocessamento["mediana_distancia"]
        )
    )

    for coluna in categoricas:
        dados[coluna] = pd.Categorical(
            dados[coluna],
            categories=categorias_treino[coluna],
        )

    x_teste = pd.get_dummies(
        dados[numericas + categoricas],
        columns=categoricas,
        dummy_na=False,
        dtype="int8",
    )

    # Mesmas colunas e ordem utilizadas no ajuste.
    x_teste = x_teste.reindex(
        columns=colunas_modelo,
        fill_value=0,
    )

    if x_teste.columns.tolist() != colunas_modelo:
        raise ValueError("Esquema de atributos inconsistente.")

    if x_teste.shape[1] != 23:
        raise ValueError(
            f"Esperados 23 atributos, encontrados "
            f"{x_teste.shape[1]}."
        )

    y_teste = teste["vendas"].to_numpy(dtype=float)

    if not np.isfinite(y_teste).all():
        raise ValueError("Valores reais inválidos.")

    print(f"Atributos: {x_teste.shape[1]}")

    return x_teste, y_teste


def main():
    RESULTADOS.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("AVALIAÇÃO EXTERNA DO MODELO GLOBAL")
    print("========================================")

    artefato = carregar_modelo()
    modelo = artefato["modelo"]

    print(f"Algoritmo: {artefato['algoritmo']}")
    print("Modelo carregado sem novo treinamento.")

    teste = carregar_teste()

    x_teste, y_teste = preparar_teste(
        teste,
        artefato["preprocessamento"],
    )

    print("\nGerando previsões...")

    previsoes = modelo.predict(x_teste)

    if not np.isfinite(previsoes).all():
        raise ValueError("Foram geradas previsões inválidas.")

    rmse = float(np.sqrt(
        mean_squared_error(y_teste, previsoes)
    ))

    mae = float(
        mean_absolute_error(y_teste, previsoes)
    )

    resultado = pd.DataFrame({
        "loja_id": teste["loja_id"].to_numpy(),
        "data": teste["data"].to_numpy(),
        "data_origem": teste["data_origem"].to_numpy(),
        "vendas_reais": y_teste,
        "vendas_previstas": previsoes,
    })

    resultado["erro"] = (
        resultado["vendas_reais"]
        - resultado["vendas_previstas"]
    )

    resultado["erro_absoluto"] = (
        resultado["erro"].abs()
    )

    caminho_previsoes = (
        RESULTADOS / "previsoes_teste_externo.csv"
    )

    resultado.to_csv(
        caminho_previsoes,
        index=False,
        date_format="%Y-%m-%d",
    )

    relatorio = {
        "tipo": "avaliacao_externa",
        "algoritmo": artefato["algoritmo"],
        "horizonte_dias": HORIZONTE,
        "data_corte": str(DATA_CORTE.date()),
        "linhas_teste": len(teste),
        "quantidade_lojas": int(
            teste["loja_id"].nunique()
        ),
        "quantidade_atributos": x_teste.shape[1],
        "periodo_teste": [
            str(teste["data"].min().date()),
            str(teste["data"].max().date()),
        ],
        "rmse": rmse,
        "mae": mae,
        "modelo_retreinado": False,
        "observacao": (
            "Previsões com horizonte de sete dias "
            "e origem a partir do corte temporal. "
            "Os atributos históricos utilizam vendas "
            "observadas disponíveis até cada origem."
        ),
    }

    caminho_relatorio = (
        RESULTADOS / "relatorio_teste_externo.json"
    )

    caminho_relatorio.write_text(
        json.dumps(
            relatorio,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n========================================")
    print("AVALIAÇÃO EXTERNA CONCLUÍDA")
    print("========================================")

    print(f"Observações avaliadas: {len(teste)}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    print("\nArquivos gerados:")
    print(caminho_previsoes)
    print(caminho_relatorio)


if __name__ == "__main__":
    main()
