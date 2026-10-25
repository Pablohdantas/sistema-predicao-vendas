"""Comparação pareada H7 entre AutoML e baseline ingênuo sazonal.

Baseline: previsão da loja para data-alvo t = vendas observadas em t-7
(que coincide com a data de origem para H7).

Uso (na raiz do repositório):
  python dados/scripts/comparar_baseline_sazonal.py --execucao ID --modo ensaio
  python dados/scripts/comparar_baseline_sazonal.py --execucao ID --modo cientifico

Modo científico usa APENAS lojas em amostra_arima.csv e exige cobertura das 63
lojas nos dados previstos; não reamostra nem retreina o AutoML.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]


def metricas(df: pd.DataFrame, coluna: str) -> dict:
    y = df["vendas"].to_numpy(dtype=float)
    p = df[coluna].to_numpy(dtype=float)
    erro = y - p
    rmse = float(np.sqrt(np.mean(erro**2)))
    mae = float(np.mean(np.abs(erro)))
    denominador_r2 = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1 - np.sum(erro**2) / denominador_r2) if denominador_r2 > 0 else None
    positivos = y > 0
    rmspe = float(np.sqrt(np.mean((erro[positivos] / y[positivos]) ** 2))) if positivos.any() else None
    return {"quantidade": len(df), "rmse": rmse, "mae": mae, "r2": r2, "rmspe": rmspe}


def executar(execucao: str, modo: str) -> None:
    pasta = RAIZ / "backend" / "execucoes" / execucao / "multihorizonte"
    caminho = pasta / "previsoes_teste.csv"
    if not caminho.is_file():
        raise FileNotFoundError(f"Previsões externas não encontradas: {caminho}")

    p = pd.read_csv(caminho, parse_dates=["data_origem", "data"])
    obrigatorias = {"loja_id", "data_origem", "data", "horizonte_dias", "vendas", "vendas_previstas"}
    if not obrigatorias.issubset(p.columns):
        raise ValueError(f"Colunas ausentes: {sorted(obrigatorias - set(p.columns))}")
    p = p.loc[p["horizonte_dias"] == 7].copy()
    if p.empty:
        raise ValueError("Execução não possui previsões externas H7.")
    if p.duplicated(["loja_id", "data_origem", "data"]).any():
        raise ValueError("Previsões H7 duplicadas por loja/origem/data.")
    if not (p["data"] - p["data_origem"] == pd.Timedelta(days=7)).all():
        raise ValueError("Há exemplos cujo horizonte não corresponde a sete dias de calendário.")
    for coluna in ("vendas", "vendas_previstas"):
        p[coluna] = pd.to_numeric(p[coluna], errors="raise")
        if not np.isfinite(p[coluna]).all():
            raise ValueError(f"Valores inválidos em {coluna}.")

    amostra_path = RAIZ / "dados" / "processados" / "amostra_arima.csv"
    amostra = pd.read_csv(amostra_path, usecols=["loja_id"])
    lojas_amostra = set(amostra["loja_id"].astype(int))
    if len(lojas_amostra) != len(amostra):
        raise ValueError("IDs duplicados na amostra estatística.")
    lojas_execucao = set(p["loja_id"].astype(int))
    if modo == "cientifico":
        ausentes = sorted(lojas_amostra - lojas_execucao)
        if ausentes:
            raise ValueError(
                f"Execução ainda não contém previsões H7 para {len(ausentes)} das "
                f"{len(lojas_amostra)} lojas da amostra; não é possível chamar de comparação científica."
            )
        p = p.loc[p["loja_id"].isin(lojas_amostra)].copy()

    # Somente o dado bruto na data de origem; nunca lê vendas futuras para prever.
    raw = pd.read_csv(
        RAIZ / "dados" / "brutos" / "train.csv",
        usecols=["Store", "Date", "Sales", "Open"],
        parse_dates=["Date"],
    ).rename(columns={"Store": "loja_id", "Date": "data_origem", "Sales": "vendas_origem"})
    if raw.duplicated(["loja_id", "data_origem"]).any():
        raise ValueError("Datas duplicadas no histórico bruto por loja.")
    # Loja fechada conhecida em t-7 corresponde a 0, não a ausência de informação.
    raw.loc[raw["Open"] == 0, "vendas_origem"] = 0
    raw = raw[["loja_id", "data_origem", "vendas_origem"]]
    comparacao = p.merge(raw, on=["loja_id", "data_origem"], how="left", validate="many_to_one", indicator=True)
    faltantes = int((comparacao["_merge"] != "both").sum())
    comparacao = comparacao.drop(columns="_merge")
    comparacao["vendas_origem"] = pd.to_numeric(comparacao["vendas_origem"], errors="coerce")
    validos = np.isfinite(comparacao["vendas_origem"])
    comparacao = comparacao.loc[validos].copy()
    if comparacao.empty:
        raise ValueError("Nenhuma previsão H7 possui venda conhecida na data de origem.")
    comparacao = comparacao.rename(columns={"vendas_origem": "previsao_ingenua_sazonal"})
    comparacao = comparacao.sort_values(["data_origem", "loja_id", "data"])

    resumo = {
        "tipo": "comparacao_pareada_h7_automl_vs_ingenuo_sazonal",
        "modo": modo,
        "execucao": execucao,
        "amostra_estatistica_lojas": len(lojas_amostra),
        "lojas_comparadas": int(comparacao["loja_id"].nunique()),
        "previsoes_h7_selecionadas": len(p),
        "pares_validos": len(comparacao),
        "origens_sem_observacao": faltantes,
        "pares_descartados_total": len(p) - len(comparacao),
        "automl": metricas(comparacao, "vendas_previstas"),
        "ingenuo_sazonal": metricas(comparacao, "previsao_ingenua_sazonal"),
        "observacao": "As duas métricas foram calculadas exclusivamente nos mesmos pares loja/origem/alvo.",
    }
    destino = RAIZ / "dados" / "processados" / "experimentos" / ("ensaio" if modo == "ensaio" else "cientifico") / execucao
    destino.mkdir(parents=True, exist_ok=True)
    comparacao.to_csv(destino / "comparacao_ingenuo_h7.csv", index=False, date_format="%Y-%m-%d")
    (destino / "resumo_ingenuo_h7.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    print(f"\nArtefatos salvos em: {destino}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execucao", required=True, help="ID de execução do pipeline multihorizonte")
    parser.add_argument("--modo", required=True, choices=["ensaio", "cientifico"])
    args = parser.parse_args()
    executar(args.execucao, args.modo)
