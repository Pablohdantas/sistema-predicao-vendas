
"""
Rotas de previsão automática e explicabilidade SHAP
por loja, sem exigir atributos técnicos do usuário.
"""

from fastapi import APIRouter, HTTPException, Path

from backend.app.services.previsao_automatica_service import (
    gerar_entrada_automatica,
    prever_loja_automaticamente,
)
from backend.app.services.explicabilidade_service import (
    explicar_previsao,
)


router = APIRouter(
    prefix="/previsoes/execucoes",
    tags=["Previsões automáticas"],
)


@router.post("/{execucao_id}/lojas/{loja_id}/automatica")
def criar_previsao_automatica(
    execucao_id: str,
    loja_id: int = Path(gt=0),
):
    """Prevê as vendas a partir do histórico da loja."""
    try:
        return prever_loja_automaticamente(
            execucao_id=execucao_id,
            loja_id=loja_id,
        )
    except (FileNotFoundError, LookupError) as erro:
        raise HTTPException(
            status_code=404,
            detail=str(erro),
        ) from erro
    except ValueError as erro:
        raise HTTPException(
            status_code=422,
            detail=str(erro),
        ) from erro


@router.post(
    "/{execucao_id}/lojas/{loja_id}/automatica/explicacao"
)
def explicar_previsao_automatica(
    execucao_id: str,
    loja_id: int = Path(gt=0),
):
    """
    Reconstrói os atributos da previsão automática
    e calcula SHAP para o modelo vencedor.
    """
    try:
        entrada = gerar_entrada_automatica(
            execucao_id=execucao_id,
            loja_id=loja_id,
        )

        resultado = explicar_previsao(
            entrada["atributos"],
            quantidade_atributos=10,
            execucao_id=entrada["execucao_id"],
        )

        return {
            "execucao_id": entrada["execucao_id"],
            "loja_id": entrada["loja_id"],
            "data_origem": entrada["data_origem"],
            "data_prevista": entrada["data"],
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
                "observacao": resultado["observacao"],
            },
        }
    except (FileNotFoundError, LookupError) as erro:
        raise HTTPException(
            status_code=404,
            detail=str(erro),
        ) from erro
    except ValueError as erro:
        raise HTTPException(
            status_code=422,
            detail=str(erro),
        ) from erro
