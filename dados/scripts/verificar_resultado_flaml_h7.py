"""Valida artefatos FLAML/H7, a amostra estratificada e o pareamento.

Uso: python dados/scripts/verificar_resultado_flaml_h7.py
Não modifica arquivos nem reexecuta modelos.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DADOS = ROOT / "dados" / "processados"
SAIDA = DADOS / "experimentos" / "cientifico_h7_flaml"
COLUNAS = ["loja_id", "data_origem", "data", "horizonte_dias", "vendas", "vendas_previstas"]


def exigir(condicao, mensagem):
    if not condicao:
        raise ValueError(mensagem)


def ler_csv(caminho):
    df = pd.read_csv(caminho, parse_dates=["data_origem", "data"])
    exigir(all(c in df for c in COLUNAS), f"Colunas faltando: {caminho}")
    exigir(not df.empty, f"Arquivo vazio: {caminho}")
    exigir(not df[COLUNAS].isna().any().any(), f"Nulos em campos obrigatórios: {caminho}")
    exigir(df["horizonte_dias"].eq(7).all(), f"Horizonte diferente de H7: {caminho}")
    exigir(((df["data"] - df["data_origem"]).dt.days == 7).all(), f"Datas incoerentes: {caminho}")
    exigir(not df.duplicated(["loja_id", "data_origem", "data", "horizonte_dias"]).any(),
           f"Pares repetidos: {caminho}")
    for c in ("vendas", "vendas_previstas"):
        exigir(np.isfinite(pd.to_numeric(df[c], errors="coerce")).all(), f"Valores não finitos: {caminho}/{c}")
    exigir(df["data_origem"].min() >= pd.Timestamp("2015-01-23"), "Teste começou antes do corte")
    return df


def rmse(df):
    return float(np.sqrt(np.mean((df["vendas"].to_numpy(float) - df["vendas_previstas"].to_numpy(float)) ** 2)))


def main():
    rel = json.loads((SAIDA / "relatorio_h7.json").read_text(encoding="utf-8"))
    completo = ler_csv(SAIDA / "previsoes_teste_h7_todas_lojas.csv")
    amostra = ler_csv(SAIDA / "previsoes_teste_h7_63_lojas.csv")
    ids = pd.read_csv(DADOS / "amostra_arima.csv")["loja_id"].astype(int)
    exigir(len(ids) == 63 and ids.nunique() == 63, "Amostra estatística não tem 63 lojas únicas")
    exigir(set(amostra["loja_id"].unique()) == set(ids), "Amostra não cobre as 63 lojas")
    esperado = completo.loc[completo["loja_id"].isin(ids)].copy()
    chaves = ["loja_id", "data_origem", "data", "horizonte_dias"]
    cruzado = esperado.merge(amostra, on=chaves, how="outer", validate="one_to_one", indicator=True,
                             suffixes=("_completo", "_amostra"))
    exigir(cruzado["_merge"].eq("both").all(), "Recorte não corresponde aos pares da base completa")
    for campo in ("vendas", "vendas_previstas"):
        exigir(np.allclose(cruzado[f"{campo}_completo"], cruzado[f"{campo}_amostra"], rtol=1e-9, atol=1e-6),
               f"Campo divergente: {campo}")
    medido = rmse(amostra)
    relatado = float(rel["teste_63_lojas"]["rmse"])
    exigir(abs(medido - relatado) < 1e-5, "RMSE do CSV difere do relatório")
    exigir(len(amostra) == int(rel["teste_63_lojas"]["quantidade"]), "Quantidade do relatório difere do CSV")
    resumo = {
        "status": "APROVADO",
        "algoritmo_vencedor": rel["vencedor"]["algoritmo"],
        "rmse_validacao": rel["vencedor"]["rmse_validacao"],
        "linhas_teste_todas_lojas": len(completo),
        "lojas_teste_todas": int(completo.loja_id.nunique()),
        "pares_teste_63_lojas": len(amostra),
        "lojas_amostra": int(amostra.loja_id.nunique()),
        "origens_distintas_63_lojas": int(amostra.data_origem.nunique()),
        "primeira_origem_63_lojas": str(amostra.data_origem.min().date()),
        "ultima_origem_63_lojas": str(amostra.data_origem.max().date()),
        "rmse_externo_63_confirmado": medido,
        "pares_duplicados": 0,
        "divergencias_recorte": 0,
    }
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
