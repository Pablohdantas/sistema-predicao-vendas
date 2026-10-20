
"""
Endpoints para atualização do histórico de vendas.
"""

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile
)

from backend.app.services.atualizacao_historico_service import (
    TAMANHO_MAXIMO_ATUALIZACAO,
    atualizar_historico
)


router = APIRouter(
    prefix="/historico",
    tags=["Histórico de vendas"]
)


@router.post("/execucoes/{execucao_id}/atualizar")
async def adicionar_vendas(
    execucao_id: str,
    arquivo: UploadFile = File(...)
):
    """
    Recebe um CSV com novas vendas.

    Não altera o modelo nem retreina o AutoML.
    """

    try:
        conteudo = await arquivo.read(
            TAMANHO_MAXIMO_ATUALIZACAO + 1
        )

        return atualizar_historico(
            execucao_id,
            conteudo,
            arquivo.filename or ""
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

    finally:
        await arquivo.close()
