
"""
Rotas de previsão semanal e explicabilidade SHAP.

POST /{execucao_id}/lojas/{loja_id}/semanal
    Retorna as sete previsões diárias.

POST /{execucao_id}/lojas/{loja_id}/semanal/explicacao/{dia}
    Explica uma previsão escolhida, com SHAP.

Preserva os modelos e os resultados anteriores.
"""

from functools import lru_cache

import joblib
import numpy as np
import shap

from fastapi import APIRouter, HTTPException, Path

from backend.app.services.treinamento_service import (
    pasta_da_execucao,
)
from backend.app.services.atualizacao_historico_service import (
    ler_historico,
)
from backend.app.services.atributos_multihorizonte_service import (
    gerar_entradas_futuras_loja,
)
from backend.app.services.preprocessamento_service import (
    transformar_atributos,
)


router = APIRouter(
    prefix="/previsoes/execucoes",
    tags=["Previsões semanais"],
)


@lru_cache(maxsize=3)
def carregar_modelo_semanal(execucao_id: str):
    """
    Carrega um modelo semanal produzido pelo próprio
    sistema. Cada execução tem artefato imutável.

    Nunca carregue arquivos joblib de fontes externas
    não confiáveis.
    """

    pasta = pasta_da_execucao(execucao_id)

    caminho = (
        pasta
        / "multihorizonte"
        / "modelo_multihorizonte.joblib"
    )

    if not caminho.is_file():
        raise FileNotFoundError(
            "Esta execução não possui modelo "
            "semanal treinado."
        )

    artefato = joblib.load(caminho)

    if not isinstance(artefato, dict):
        raise ValueError(
            "Artefato semanal inválido."
        )

    obrigatorios = {
        "tipo",
        "horizontes",
        "algoritmo",
        "modelo",
        "preprocessamento",
    }

    if not obrigatorios.issubset(artefato):
        raise ValueError(
            "O modelo semanal está incompleto."
        )

    if (
        artefato["tipo"] != "multihorizonte"
        or artefato["horizontes"] != list(range(1, 8))
        or artefato["algoritmo"] not in ("lgbm", "xgboost")
    ):
        raise ValueError(
            "Artefato semanal incompatível."
        )

    return artefato


def preparar_previsoes(execucao_id: str, loja_id: int):
    """
    Reconstrói atributos com o histórico atualizado.

    Nenhuma venda futura é fornecida ao modelo.
    """

    pasta = pasta_da_execucao(execucao_id)

    artefato = carregar_modelo_semanal(pasta.name)

    historico = ler_historico(pasta)

    entradas = gerar_entradas_futuras_loja(
        historico,
        loja_id,
    )

    x = transformar_atributos(
        entradas,
        artefato["preprocessamento"],
    )

    if len(x) != 7:
        raise ValueError(
            "Foram geradas menos de sete entradas."
        )

    valores_brutos = np.asarray(
        artefato["modelo"].predict(x),
        dtype=float,
    ).reshape(-1)

    if (
        len(valores_brutos) != 7
        or not np.isfinite(valores_brutos).all()
    ):
        raise ValueError(
            "O modelo retornou previsões inválidas."
        )

    valores = np.maximum(valores_brutos, 0)

    return artefato, entradas, x, valores_brutos, valores


def traduzir_erro(erro: Exception):
    if isinstance(erro, (FileNotFoundError, LookupError)):
        raise HTTPException(
            status_code=404,
            detail=str(erro),
        ) from erro

    if isinstance(erro, (ValueError, KeyError, TypeError)):
        raise HTTPException(
            status_code=422,
            detail=str(erro),
        ) from erro

    raise erro


@router.post("/{execucao_id}/lojas/{loja_id}/semanal")
def criar_previsao_semanal(
    execucao_id: str,
    loja_id: int = Path(gt=0),
):
    try:
        artefato, entradas, _, _, valores = (
            preparar_previsoes(execucao_id, loja_id)
        )

        previsoes = []

        for indice, (_, linha) in enumerate(
            entradas.iterrows()
        ):
            previsoes.append({
                "horizonte_dias": int(
                    linha["horizonte_dias"]
                ),
                "data_prevista": (
                    linha["data"].strftime("%Y-%m-%d")
                ),
                "vendas_previstas": round(
                    float(valores[indice]),
                    2,
                ),
            })

        if [
            item["horizonte_dias"]
            for item in previsoes
        ] != list(range(1, 8)):
            raise ValueError(
                "Horizontes inconsistentes."
            )

        return {
            "execucao_id": execucao_id,
            "loja_id": loja_id,
            "data_origem": (
                entradas["data_origem"]
                .iloc[0]
                .strftime("%Y-%m-%d")
            ),
            "algoritmo": artefato["algoritmo"],
            "tipo_modelo": "multihorizonte",
            "horizonte_dias": 7,
            "previsoes": previsoes,
            "total_semanal_previsto": round(
                sum(
                    item["vendas_previstas"]
                    for item in previsoes
                ),
                2,
            ),
            "observacao": (
                "As sete previsões usam informações "
                "disponíveis até a data de origem. "
                "Variáveis comerciais futuras "
                "desconhecidas não são utilizadas "
                "nesta versão."
            ),
        }

    except Exception as erro:
        traduzir_erro(erro)


@lru_cache(maxsize=3)
def carregar_explicador_semanal(execucao_id: str):
    """
    Cria o explicador SHAP do modelo semanal.
    """

    artefato = carregar_modelo_semanal(execucao_id)

    return shap.TreeExplainer(
        artefato["modelo"],
        model_output="raw",
    )


@router.post(
    "/{execucao_id}/lojas/{loja_id}"
    "/semanal/explicacao/{dia}"
)
def explicar_previsao_semanal(
    execucao_id: str,
    loja_id: int = Path(gt=0),
    dia: int = Path(ge=1, le=7),
):
    """
    Calcula SHAP para um horizonte específico.

    As contribuições representam o funcionamento
    do modelo, e não relações de causa e efeito.
    """

    try:
        artefato, entradas, x, valores_brutos, valores = (
            preparar_previsoes(execucao_id, loja_id)
        )

        indice = dia - 1

        x_dia = x.iloc[[indice]]

        explicador = carregar_explicador_semanal(
            execucao_id
        )

        explicacao = explicador(x_dia)

        contribuicoes = np.asarray(
            explicacao.values,
            dtype=float,
        )

        if contribuicoes.shape != (1, x_dia.shape[1]):
            raise ValueError(
                "Formato SHAP incompatível."
            )

        contribuicoes = contribuicoes[0]

        bases = np.asarray(
            explicacao.base_values,
            dtype=float,
        ).reshape(-1)

        if len(bases) != 1:
            raise ValueError(
                "Valor-base SHAP inválido."
            )

        valor_base = float(bases[0])

        bruto = float(valores_brutos[indice])

        reconstruida = (
            valor_base + float(contribuicoes.sum())
        )

        diferenca = abs(bruto - reconstruida)

        if not np.isfinite(
            [valor_base, bruto, reconstruida, diferenca]
        ).all() or not np.isfinite(contribuicoes).all():
            raise ValueError(
                "Valores SHAP inválidos."
            )

        if not np.isclose(
            bruto,
            reconstruida,
            atol=0.1,
            rtol=1e-4,
        ):
            raise ValueError(
                "SHAP não reconstruiu a saída do modelo."
            )

        itens = [
            {
                "atributo": str(coluna),
                "contribuicao": round(float(valor), 4),
            }
            for coluna, valor in zip(
                x_dia.columns,
                contribuicoes,
            )
        ]

        itens.sort(
            key=lambda item: abs(item["contribuicao"]),
            reverse=True,
        )

        linha = entradas.iloc[indice]

        return {
            "execucao_id": execucao_id,
            "loja_id": loja_id,
            "data_origem": (
                linha["data_origem"].strftime("%Y-%m-%d")
            ),
            "data_prevista": (
                linha["data"].strftime("%Y-%m-%d")
            ),
            "horizonte_dias": dia,
            "algoritmo": artefato["algoritmo"],
            "vendas_previstas": round(
                float(valores[indice]),
                2,
            ),
            "explicabilidade": {
                "metodo": "SHAP",
                "valor_base": round(valor_base, 2),
                "previsao_bruta": round(bruto, 2),
                "diferenca_reconstrucao": diferenca,
                "principais_contribuicoes": itens[:10],
                "observacao": (
                    "SHAP mostra quais atributos aumentam "
                    "ou reduzem a saída do modelo em relação "
                    "ao valor-base. Não demonstra causalidade. "
                    "Caso a saída do modelo seja negativa, "
                    "a previsão exibida é limitada a zero."
                ),
            },
        }

    except Exception as erro:
        traduzir_erro(erro)
