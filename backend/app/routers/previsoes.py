
"""
Rotas de previsão de vendas.

Rotas antigas:
- POST /previsoes/
- POST /previsoes/explicacao

Rotas AutoML:
- POST /previsoes/execucoes/{execucao_id}
- POST /previsoes/execucoes/{execucao_id}/explicacao
"""

from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.services.previsao_service import (
    carregar_modelo,
    prever_vendas,
    validar_execucao_id
)

from backend.app.services.explicabilidade_service import (
    explicar_previsao
)


router = APIRouter(
    prefix="/previsoes",
    tags=["Previsões"]
)


class AtributosPrevisao(BaseModel):
    """
    Formato preservado da API experimental.
    """

    promocao: float
    promocao_continua: float
    mes_renovacao_promocao: float | None = None
    feriado: float
    feriado_escolar: float
    distancia_concorrente: float | None = None

    dia_semana: int = Field(ge=1, le=7)
    dia_mes: int = Field(ge=1, le=31)
    mes: int = Field(ge=1, le=12)
    semana_ano: int = Field(ge=1, le=53)

    lag_7: float | None = None
    lag_14: float | None = None
    lag_21: float | None = None
    lag_28: float | None = None
    media_movel_7: float | None = None
    media_movel_28: float | None = None

    tipo_loja: str
    sortimento: str


class SolicitacaoPrevisao(BaseModel):
    """Requisição do modelo experimental antigo."""

    loja_id: int = Field(gt=0)
    data_origem: date
    data: date
    atributos: AtributosPrevisao


class SolicitacaoPrevisaoExecucao(BaseModel):
    """
    Requisição dinâmica.

    Aceita as variáveis disponíveis na execução,
    incluindo observacoes_media_7 e
    observacoes_media_28 quando utilizadas.

    dia_semana segue a engenharia dinâmica:
    segunda=0, domingo=6.
    """

    loja_id: int = Field(gt=0)
    data_origem: date
    data: date
    atributos: dict[str, Any] = Field(min_length=1)


def validar_horizonte(solicitacao):
    """Exige uma previsão exatamente sete dias à frente."""

    if solicitacao.data != (
        solicitacao.data_origem + timedelta(days=7)
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "A data prevista deve estar exatamente "
                "7 dias após a data de origem."
            )
        )


def responder_previsao(
    solicitacao,
    previsao: float,
    algoritmo: str,
    execucao_id: str | None = None
) -> dict:
    """Constrói a resposta padronizada."""

    resposta = {
        "loja_id": solicitacao.loja_id,
        "data_origem": solicitacao.data_origem,
        "data_prevista": solicitacao.data,
        "horizonte_dias": 7,
        "vendas_previstas": round(float(previsao), 2),
        "algoritmo": algoritmo
    }

    if execucao_id is not None:
        resposta["execucao_id"] = execucao_id

    return resposta


def responder_explicacao(
    solicitacao,
    resultado: dict,
    execucao_id: str | None = None
) -> dict:
    """Constrói a resposta de previsão com SHAP."""

    resposta = {
        "loja_id": solicitacao.loja_id,
        "data_origem": solicitacao.data_origem,
        "data_prevista": solicitacao.data,
        "horizonte_dias": 7,
        "algoritmo": resultado["algoritmo"],
        "vendas_previstas": resultado["previsao"],
        "explicabilidade": {
            "metodo": "SHAP",
            "valor_base": resultado["valor_base_shap"],
            "principais_contribuicoes": (
                resultado["principais_contribuicoes"]
            ),
            "diferenca_reconstrucao": (
                resultado["diferenca_reconstrucao"]
            ),
            "observacao": resultado["observacao"]
        }
    }

    if execucao_id is not None:
        resposta["execucao_id"] = execucao_id

    return resposta


# ============================================================
# ROTAS EXPERIMENTAIS EXISTENTES
# ============================================================

@router.post("/")
def criar_previsao(solicitacao: SolicitacaoPrevisao):
    """Usa o modelo experimental da Rossmann."""

    validar_horizonte(solicitacao)

    try:
        dados = solicitacao.atributos.model_dump()

        previsao = prever_vendas(dados)[0]
        algoritmo = carregar_modelo()["algoritmo"]

        return responder_previsao(
            solicitacao,
            previsao,
            algoritmo
        )

    except FileNotFoundError as erro:
        raise HTTPException(
            status_code=404,
            detail=str(erro)
        ) from erro

    except ValueError as erro:
        raise HTTPException(
            status_code=422,
            detail=str(erro)
        ) from erro


@router.post("/explicacao")
def criar_previsao_com_explicacao(
    solicitacao: SolicitacaoPrevisao
):
    """Usa o SHAP do modelo experimental."""

    validar_horizonte(solicitacao)

    try:
        resultado = explicar_previsao(
            solicitacao.atributos.model_dump(),
            quantidade_atributos=10
        )

        return responder_explicacao(
            solicitacao,
            resultado
        )

    except FileNotFoundError as erro:
        raise HTTPException(
            status_code=404,
            detail=str(erro)
        ) from erro

    except ValueError as erro:
        raise HTTPException(
            status_code=422,
            detail=str(erro)
        ) from erro


# ============================================================
# ROTAS DINÂMICAS AUTOML
# ============================================================

@router.post("/execucoes/{execucao_id}")
def criar_previsao_execucao(
    execucao_id: str,
    solicitacao: SolicitacaoPrevisaoExecucao
):
    """Prevê com o algoritmo vencedor da execução."""

    validar_horizonte(solicitacao)

    try:
        execucao_id = validar_execucao_id(execucao_id)

        artefato = carregar_modelo(execucao_id)

        previsao = prever_vendas(
            solicitacao.atributos,
            execucao_id=execucao_id
        )[0]

        return responder_previsao(
            solicitacao,
            previsao,
            artefato["algoritmo"],
            execucao_id
        )

    except FileNotFoundError as erro:
        raise HTTPException(
            status_code=404,
            detail=str(erro)
        ) from erro

    except ValueError as erro:
        raise HTTPException(
            status_code=422,
            detail=str(erro)
        ) from erro


@router.post("/execucoes/{execucao_id}/explicacao")
def criar_explicacao_execucao(
    execucao_id: str,
    solicitacao: SolicitacaoPrevisaoExecucao
):
    """Gera previsão e SHAP do vencedor da execução."""

    validar_horizonte(solicitacao)

    try:
        execucao_id = validar_execucao_id(execucao_id)

        resultado = explicar_previsao(
            solicitacao.atributos,
            quantidade_atributos=10,
            execucao_id=execucao_id
        )

        return responder_explicacao(
            solicitacao,
            resultado,
            execucao_id
        )

    except FileNotFoundError as erro:
        raise HTTPException(
            status_code=404,
            detail=str(erro)
        ) from erro

    except ValueError as erro:
        raise HTTPException(
            status_code=422,
            detail=str(erro)
        ) from erro
