
"""
Rotas para previsões automáticas por loja.

Não exige atributos técnicos fornecidos pelo usuário.
"""

from fastapi import APIRouter, HTTPException, Path

from backend.app.services.previsao_automatica_service import (
    prever_loja_automaticamente
)


router = APIRouter(
    prefix="/previsoes/execucoes",
    tags=["Previsões automáticas"]
)


@router.post("/{execucao_id}/lojas/{loja_id}/automatica")
def criar_previsao_automatica(
    execucao_id: str,
    loja_id: int = Path(gt=0)
):
    """
    Prevê as vendas da loja para sete dias após
    sua última data registrada no dataset.
    """

    try:
        return prever_loja_automaticamente(
            execucao_id=execucao_id,
            loja_id=loja_id
        )

    except FileNotFoundError as erro:
        raise HTTPException(
            status_code=404,
            detail=str(erro)
        ) from erro

    except LookupError as erro:
        raise HTTPException(
            status_code=404,
            detail=str(erro)
        ) from erro

    except ValueError as erro:
        raise HTTPException(
            status_code=422,
            detail=str(erro)
        ) from erro
