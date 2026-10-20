
"""
Explicabilidade individual com SHAP.

Suporta o modelo experimental e os modelos vencedores
AutoML, tanto LightGBM quanto XGBoost.

As contribuições explicam a previsão do modelo,
não representam relações causais.
"""

from functools import lru_cache

import numpy as np
import shap

from backend.app.services.previsao_service import (
    carregar_modelo,
    preparar_entrada_modelo
)


@lru_cache(maxsize=4)
def carregar_explicador(
    execucao_id: str | None = None
):
    """
    Cria um explicador para o modelo correspondente.

    Cada execução possui seu próprio modelo.
    """

    artefato = carregar_modelo(execucao_id)

    return shap.TreeExplainer(
        artefato["modelo"],
        model_output="raw"
    )


def explicar_previsao(
    dados: dict,
    quantidade_atributos: int = 10,
    execucao_id: str | None = None
) -> dict:
    """
    Retorna previsão, valor-base e contribuições SHAP
    para exatamente uma observação.
    """

    artefato, x = preparar_entrada_modelo(
        dados,
        execucao_id
    )

    if len(x) != 1:
        raise ValueError(
            "A explicação individual exige "
            "exatamente uma observação."
        )

    if not 1 <= quantidade_atributos <= x.shape[1]:
        raise ValueError(
            "A quantidade de atributos deve estar "
            f"entre 1 e {x.shape[1]}."
        )

    modelo = artefato["modelo"]

    previsao = float(
        modelo.predict(x)[0]
    )

    explicador = carregar_explicador(execucao_id)

    explicacao = explicador(x)

    valores = np.asarray(
        explicacao.values,
        dtype=float
    )

    if valores.shape != (1, x.shape[1]):
        raise ValueError(
            "Formato inesperado das contribuições SHAP."
        )

    valores_shap = valores[0]

    bases = np.asarray(
        explicacao.base_values,
        dtype=float
    ).reshape(-1)

    if len(bases) != 1:
        raise ValueError(
            "Formato inesperado do valor-base SHAP."
        )

    valor_base = float(bases[0])

    previsao_reconstruida = (
        valor_base + float(valores_shap.sum())
    )

    diferenca = abs(
        previsao - previsao_reconstruida
    )

    if not np.isfinite(
        [previsao, valor_base, diferenca]
    ).all():
        raise ValueError(
            "A explicação contém valores inválidos."
        )

    if not np.isfinite(valores_shap).all():
        raise ValueError(
            "Contribuições SHAP inválidas."
        )

    if not np.isclose(
        previsao,
        previsao_reconstruida,
        atol=0.1,
        rtol=1e-4
    ):
        raise ValueError(
            "As contribuições SHAP não "
            "reconstruíram a previsão."
        )

    contribuicoes = [
        {
            "atributo": str(coluna),
            "contribuicao": float(valor)
        }
        for coluna, valor in zip(
            x.columns,
            valores_shap
        )
    ]

    contribuicoes.sort(
        key=lambda item: abs(item["contribuicao"]),
        reverse=True
    )

    return {
        "algoritmo": artefato["algoritmo"],
        "horizonte_dias": int(
            artefato["horizonte_dias"]
        ),
        "previsao": round(previsao, 2),
        "valor_base_shap": round(valor_base, 2),
        "previsao_reconstruida": round(
            previsao_reconstruida,
            2
        ),
        "diferenca_reconstrucao": float(diferenca),
        "principais_contribuicoes": (
            contribuicoes[:quantidade_atributos]
        ),
        "observacao": (
            "As contribuições SHAP explicam a "
            "previsão do modelo, não relações causais."
        )
    }
