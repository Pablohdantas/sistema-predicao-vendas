
"""
Endpoints para iniciar e acompanhar treinamentos AutoML.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.services.treinamento_service import (
    iniciar_treinamento,
    consultar_treinamento
)


router = APIRouter(
    prefix="/treinamentos",
    tags=["Treinamentos AutoML"]
)


class ConfiguracaoTreinamento(BaseModel):
    """
    Configuração do treinamento.

    O modo de teste não produz modelo definitivo.
    """

    modo_teste: bool = True

    tentativas: int = Field(
        default=2,
        ge=1,
        le=30
    )

    tempo_por_algoritmo: int = Field(
        default=60,
        ge=1,
        le=1800
    )


@router.post(
    "/{execucao_id}",
    status_code=status.HTTP_202_ACCEPTED
)
def iniciar(
    execucao_id: str,
    configuracao: ConfiguracaoTreinamento
):
    """Agenda o treinamento da execução."""

    try:
        return iniciar_treinamento(
            execucao_id=execucao_id,
            modo_teste=configuracao.modo_teste,
            tentativas=configuracao.tentativas,
            tempo_por_algoritmo=(
                configuracao.tempo_por_algoritmo
            )
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

    except RuntimeError as erro:
        raise HTTPException(
            status_code=409,
            detail=str(erro)
        ) from erro


@router.get("/{execucao_id}")
def consultar(execucao_id: str):
    """Consulta status e métricas da execução."""

    try:
        return consultar_treinamento(execucao_id)

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
