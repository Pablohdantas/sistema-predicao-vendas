
"""
Comparação externa entre XGBoost e baseline ingênuo sazonal.

O baseline prevê as vendas da loja na data t
usando as vendas observadas na data t-7.

Utiliza exatamente as previsões do teste externo
já geradas pelo modelo XGBoost.

Não modifica nem treina modelos.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[3]
RESULTADOS = RAIZ / "backend" / "resultados"
BRUTOS = RAIZ / "dados" / "brutos"

HORIZONTE = 7


def calcular_metricas(reais, previstas):
    erros = reais - previstas

    return {
        "rmse": float(np.sqrt(np.mean(erros ** 2))),
        "mae": float(np.mean(np.abs(erros))),
    }


def main():
    print("Carregando previsões externas...")

    previsoes = pd.read_csv(
        RESULTADOS / "previsoes_teste_externo.csv",
        parse_dates=["data", "data_origem"],
    )

    print("Carregando histórico original...")

    historico = pd.read_csv(
        BRUTOS / "train.csv",
        usecols=["Store", "Date", "Sales", "Open"],
        parse_dates=["Date"],
    )

    historico = historico.rename(
        columns={
            "Store": "loja_id",
            "Date": "data_origem",
            "Sales": "vendas_semana_anterior",
        }
    )

    # Uma loja fechada com Open=0 apresenta
    # vendas observadas iguais a zero.
    historico.loc[
        historico["Open"] == 0,
        "vendas_semana_anterior"
    ] = 0.0

    # Confirma que cada origem está sete dias
    # antes da respectiva data prevista.
    diferenca = (
        previsoes["data"] - previsoes["data_origem"]
    ).dt.days

    if not (diferenca == HORIZONTE).all():
        raise ValueError(
            "Foram encontradas previsões fora "
            "do horizonte de sete dias."
        )

    historico = historico[
        ["loja_id", "data_origem", "vendas_semana_anterior"]
    ]

    comparacao = previsoes.merge(
        historico,
        on=["loja_id", "data_origem"],
        how="left",
        validate="many_to_one",
    )

    total_original = len(comparacao)

    faltantes = int(
        comparacao["vendas_semana_anterior"]
        .isna()
        .sum()
    )

    # A comparação deve conter as mesmas
    # observações para os dois métodos.
    comparacao = comparacao.dropna(
        subset=["vendas_semana_anterior"]
    ).copy()

    if comparacao.empty:
        raise ValueError(
            "Nenhuma observação comparável encontrada."
        )

    reais = comparacao["vendas_reais"].to_numpy(
        dtype=float
    )

    previstas_xgboost = (
        comparacao["vendas_previstas"]
        .to_numpy(dtype=float)
    )

    previstas_sazonal = (
        comparacao["vendas_semana_anterior"]
        .to_numpy(dtype=float)
    )

    metricas_xgboost = calcular_metricas(
        reais, previstas_xgboost
    )

    metricas_sazonal = calcular_metricas(
        reais, previstas_sazonal
    )

    rmse_sazonal = metricas_sazonal["rmse"]
    rmse_xgboost = metricas_xgboost["rmse"]

    reducao_rmse = (
        100 * (rmse_sazonal - rmse_xgboost)
        / rmse_sazonal
        if rmse_sazonal > 0
        else None
    )

    comparacao["erro_xgboost"] = (
        comparacao["vendas_reais"]
        - comparacao["vendas_previstas"]
    )

    comparacao["erro_sazonal"] = (
        comparacao["vendas_reais"]
        - comparacao["vendas_semana_anterior"]
    )

    caminho_csv = (
        RESULTADOS / "comparacao_baseline_externo.csv"
    )

    caminho_json = (
        RESULTADOS / "relatorio_baseline_externo.json"
    )

    comparacao.to_csv(
        caminho_csv,
        index=False,
        date_format="%Y-%m-%d",
    )

    relatorio = {
        "tipo": "comparacao_externa_pareada",
        "horizonte_dias": HORIZONTE,
        "observacoes_originais": total_original,
        "observacoes_sem_historico": faltantes,
        "observacoes_comparaveis": len(comparacao),
        "xgboost": metricas_xgboost,
        "baseline_sazonal": metricas_sazonal,
        "reducao_percentual_rmse": reducao_rmse,
        "observacao": (
            "Ambos os métodos são avaliados "
            "nas mesmas observações com histórico "
            "disponível na origem da previsão."
        ),
    }

    caminho_json.write_text(
        json.dumps(
            relatorio,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n========================================")
    print("COMPARAÇÃO EXTERNA CONCLUÍDA")
    print("========================================")

    print(f"Observações originais: {total_original}")
    print(f"Sem histórico sazonal: {faltantes}")
    print(
        f"Observações comparáveis: {len(comparacao)}"
    )

    print(
        f"\nRMSE XGBoost: "
        f"{metricas_xgboost['rmse']:.2f}"
    )

    print(
        f"RMSE sazonal: "
        f"{metricas_sazonal['rmse']:.2f}"
    )

    print(
        f"MAE XGBoost: "
        f"{metricas_xgboost['mae']:.2f}"
    )

    print(
        f"MAE sazonal: "
        f"{metricas_sazonal['mae']:.2f}"
    )

    if reducao_rmse is not None:
        print(
            f"Redução relativa do RMSE: "
            f"{reducao_rmse:.2f}%"
        )

    print("\nArquivos gerados:")
    print(caminho_csv)
    print(caminho_json)


if __name__ == "__main__":
    main()
