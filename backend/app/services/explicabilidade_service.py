
"""
Serviço de explicabilidade individual com SHAP.

Calcula uma explicação para os atributos recebidos,
usando o mesmo XGBoost e o mesmo pré-processamento
do serviço de previsão.

Não treina nem modifica o modelo.
"""

from functools import lru_cache

import numpy as np
import shap

from backend.app.services.previsao_service import (
    carregar_modelo,
    preparar_atributos,
)


@lru_cache(maxsize=1)
def carregar_explicador():
    """
    Cria o explicador SHAP uma única vez
    por processo.
    """
    artefato = carregar_modelo()

    return shap.TreeExplainer(
        artefato["modelo"]
    )


def explicar_previsao(dados, quantidade_atributos=10):
    """
    Retorna a previsão, o valor-base SHAP e
    as contribuições de seus atributos.

    Recebe os mesmos 18 atributos originais
    utilizados pelo serviço de previsão.
    """

    if not 1 <= quantidade_atributos <= 23:
        raise ValueError(
            "A quantidade de atributos deve "
            "estar entre 1 e 23."
        )

    artefato = carregar_modelo()

    x = preparar_atributos(
        dados,
        artefato["preprocessamento"]
    )

    if len(x) != 1:
        raise ValueError(
            "A explicação individual exige "
            "exatamente uma observação."
        )

    modelo = artefato["modelo"]

    previsao = float(
        modelo.predict(x)[0]
    )

    explicador = carregar_explicador()

    explicacao = explicador(x)

    valores_shap = np.asarray(
        explicacao.values[0],
        dtype=float
    )

    valor_base = float(
        np.asarray(
            explicacao.base_values[0]
        ).item()
    )

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

    if not np.isfinite(
        valores_shap
    ).all():
        raise ValueError(
            "Foram encontrados valores SHAP inválidos."
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

    contribuicoes = []

    for atributo, valor in zip(
        x.columns,
        valores_shap
    ):
        contribuicoes.append({
            "atributo": str(atributo),
            "contribuicao": float(valor)
        })

    contribuicoes.sort(
        key=lambda item: abs(
            item["contribuicao"]
        ),
        reverse=True
    )

    return {
        "algoritmo": "XGBoost",
        "horizonte_dias": 7,
        "previsao": round(previsao, 2),
        "valor_base_shap": round(valor_base, 2),
        "previsao_reconstruida": round(
            previsao_reconstruida, 2
        ),
        "diferenca_reconstrucao": diferenca,
        "principais_contribuicoes": (
            contribuicoes[:quantidade_atributos]
        ),
        "observacao": (
            "Contribuições SHAP explicam a "
            "previsão do modelo, não relações causais."
        )
    }
