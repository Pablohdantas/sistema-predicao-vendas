
"""
Explicação individual de uma previsão já calculada com SHAP.

Carrega os resultados da explicabilidade global e
seleciona uma das 500 observações analisadas.

Não treina modelos nem recalcula valores SHAP.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[4]
PASTA_SHAP = RAIZ / "backend" / "resultados" / "shap"

INDICE_OBSERVACAO = 0
QUANTIDADE_ATRIBUTOS = 10


def main():
    print("====================================")
    print("EXPLICAÇÃO INDIVIDUAL COM SHAP")
    print("====================================")

    observacoes = pd.read_csv(
        PASTA_SHAP / "observacoes_explicadas.csv"
    )

    with np.load(
        PASTA_SHAP / "valores_shap_amostra.npz"
    ) as arquivo:
        valores = arquivo["valores_shap"]
        bases = arquivo["valores_base"]
        atributos = arquivo["atributos"]

    if len(observacoes) != len(valores):
        raise ValueError(
            "Quantidade de observações inconsistente."
        )

    if valores.shape[1] != len(atributos):
        raise ValueError(
            "Quantidade de atributos inconsistente."
        )

    if not 0 <= INDICE_OBSERVACAO < len(observacoes):
        raise ValueError("Índice da observação inválido.")

    observacao = observacoes.iloc[INDICE_OBSERVACAO]

    contribuicoes = valores[INDICE_OBSERVACAO]

    valor_base = float(
        np.asarray(bases[INDICE_OBSERVACAO]).item()
    )

    previsao_reconstruida = (
        valor_base + float(contribuicoes.sum())
    )

    previsao_modelo = float(
        observacao["vendas_previstas"]
    )

    diferenca = abs(
        previsao_modelo - previsao_reconstruida
    )

    if diferenca > 0.1:
        raise ValueError(
            "A reconstrução da previsão é inconsistente."
        )

    importancia = pd.DataFrame({
        "atributo": atributos,
        "contribuicao_shap": contribuicoes,
    })

    importancia["impacto_absoluto"] = (
        importancia["contribuicao_shap"].abs()
    )

    importancia = importancia.sort_values(
        "impacto_absoluto",
        ascending=False
    )

    principais = importancia.head(
        QUANTIDADE_ATRIBUTOS
    ).copy()

    resultado = {
        "indice_amostra": INDICE_OBSERVACAO,
        "loja_id": int(observacao["loja_id"]),
        "data": str(observacao["data"]),
        "data_origem": str(observacao["data_origem"]),
        "vendas_reais": float(observacao["vendas"]),
        "vendas_previstas": previsao_modelo,
        "valor_base_shap": valor_base,
        "previsao_reconstruida": previsao_reconstruida,
        "diferenca_reconstrucao": diferenca,
        "principais_contribuicoes": principais[
            ["atributo", "contribuicao_shap"]
        ].to_dict(orient="records"),
    }

    destino = PASTA_SHAP / "explicacao_individual.json"

    destino.write_text(
        json.dumps(
            resultado,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(f"\nLoja: {resultado['loja_id']}")
    print(f"Data: {resultado['data']}")
    print(f"Vendas reais: {resultado['vendas_reais']:.2f}")
    print(f"Previsão: {previsao_modelo:.2f}")
    print(f"Valor-base SHAP: {valor_base:.2f}")

    print("\nPrincipais contribuições:")
    for _, linha in principais.iterrows():
        print(
            f"{linha['atributo']}: "
            f"{linha['contribuicao_shap']:+.2f}"
        )

    print(
        f"\nDiferença de reconstrução: {diferenca:.6f}"
    )

    print(f"Relatório salvo em: {destino}")


if __name__ == "__main__":
    main()
