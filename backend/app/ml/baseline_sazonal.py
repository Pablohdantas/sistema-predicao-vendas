
"""
Baseline ingênuo sazonal para previsão de vendas.

Prevê as vendas de uma loja utilizando as vendas
registradas exatamente 7 dias antes.

Compara as previsões com as mesmas observações
utilizadas no teste experimental do LightGBM.

Não realiza treinamento de modelo.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


RAIZ = Path(__file__).resolve().parents[3]

BRUTOS = RAIZ / "dados" / "brutos"
RESULTADOS = RAIZ / "backend" / "resultados"

CAMINHO_AUTOML = (
    RESULTADOS / "previsoes_automl_inicial.csv"
)

CAMINHO_RAW = BRUTOS / "train.csv"


def carregar_dados():
    print("Carregando previsões do AutoML...")

    automl = pd.read_csv(
        CAMINHO_AUTOML,
        parse_dates=["data", "data_origem"]
    )

    raw = pd.read_csv(
        CAMINHO_RAW,
        usecols=["Store", "Date", "Sales", "Open"],
        parse_dates=["Date"]
    )

    return automl, raw


def gerar_baseline(automl, raw):
    historico = raw.rename(
        columns={
            "Store": "loja_id",
            "Date": "data_historica",
            "Sales": "vendas_historicas"
        }
    ).copy()

    # Fechamento conhecido equivale a venda observada zero.
    historico.loc[
        historico["Open"] == 0,
        "vendas_historicas"
    ] = 0.0

    # A previsão para o dia t utiliza as vendas do dia t-7.
    resultado = automl.copy()

    resultado["data_historica"] = (
        resultado["data"] - pd.Timedelta(days=7)
    )

    # Para h=7, a data histórica precisa
    # coincidir com a origem da previsão.
    assert (
        resultado["data_historica"]
        == resultado["data_origem"]
    ).all(), "Horizonte inconsistente."

    resultado = resultado.merge(
        historico[
            ["loja_id", "data_historica", "vendas_historicas"]
        ],
        on=["loja_id", "data_historica"],
        how="left",
        validate="many_to_one"
    )

    resultado = resultado.rename(
        columns={
            "vendas_historicas": "previsao_sazonal"
        }
    )

    return resultado


def avaliar(resultado):
    # Não inventar vendas quando não existe histórico.
    comparavel = resultado.dropna(
        subset=["previsao_sazonal"]
    ).copy()

    if comparavel.empty:
        raise ValueError(
            "Nenhuma observação possui histórico sazonal."
        )

    if len(comparavel) != len(resultado):
        print(
            "Atenção: algumas observações não possuem "
            "histórico sazonal disponível."
        )

    y_real = comparavel["vendas_reais"].to_numpy()
    y_sazonal = comparavel["previsao_sazonal"].to_numpy()
    y_automl = comparavel["vendas_previstas"].to_numpy()

    rmse_sazonal = float(
        np.sqrt(mean_squared_error(y_real, y_sazonal))
    )

    mae_sazonal = float(
        mean_absolute_error(y_real, y_sazonal)
    )

    # Recalcular AutoML sobre exatamente as mesmas linhas.
    rmse_automl = float(
        np.sqrt(mean_squared_error(y_real, y_automl))
    )

    mae_automl = float(
        mean_absolute_error(y_real, y_automl)
    )

    reducao_rmse = (
        (rmse_sazonal - rmse_automl)
        / rmse_sazonal * 100
        if rmse_sazonal > 0
        else None
    )

    comparavel["erro_sazonal"] = (
        comparavel["vendas_reais"]
        - comparavel["previsao_sazonal"]
    )

    comparavel["erro_quadratico_sazonal"] = (
        comparavel["erro_sazonal"] ** 2
    )

    relatorio = {
        "tipo": "comparacao_experimental",
        "horizonte": 7,
        "modelos": ["LightGBM", "Ingenuo sazonal"],
        "observacoes_originais": len(resultado),
        "observacoes_comparaveis": len(comparavel),
        "rmse_automl": rmse_automl,
        "rmse_sazonal": rmse_sazonal,
        "mae_automl": mae_automl,
        "mae_sazonal": mae_sazonal,
        "reducao_percentual_rmse": reducao_rmse
    }

    return comparavel, relatorio


def main():
    automl, raw = carregar_dados()

    resultado = gerar_baseline(automl, raw)

    comparavel, relatorio = avaliar(resultado)

    RESULTADOS.mkdir(parents=True, exist_ok=True)

    caminho_csv = (
        RESULTADOS / "comparacao_baseline_sazonal.csv"
    )

    caminho_json = (
        RESULTADOS / "relatorio_baseline_sazonal.json"
    )

    comparavel.to_csv(
        caminho_csv,
        index=False,
        date_format="%Y-%m-%d"
    )

    caminho_json.write_text(
        json.dumps(
            relatorio,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nComparação concluída!")
    print(f"Observações: {len(comparavel)}")
    print(f"RMSE LightGBM: {relatorio['rmse_automl']:.2f}")
    print(f"RMSE Sazonal: {relatorio['rmse_sazonal']:.2f}")
    print(f"MAE LightGBM: {relatorio['mae_automl']:.2f}")
    print(f"MAE Sazonal: {relatorio['mae_sazonal']:.2f}")

    reducao = relatorio["reducao_percentual_rmse"]

    if reducao is not None:
        print(f"Redução percentual do RMSE: {reducao:.2f}%")

    print("\nArquivos gerados:")
    print(caminho_csv)
    print(caminho_json)


if __name__ == "__main__":
    main()
