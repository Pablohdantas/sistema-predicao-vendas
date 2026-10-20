
"""
Rotas para recebimento e validação de datasets.
"""

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile
)

from backend.app.services.validacao_csv_service import (
    validar_csv,
    TAMANHO_MAXIMO_BYTES
)

from backend.app.services.execucoes_service import (
    salvar_execucao
)


router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"]
)


@router.post("/upload")
async def upload_dataset(
    arquivo: UploadFile = File(...)
):
    """
    Valida um CSV e, caso seja aprovado,
    cria uma execução independente.

    Ainda não executa o AutoML.
    """

    try:
        conteudo = await arquivo.read(
            TAMANHO_MAXIMO_BYTES + 1
        )

        relatorio = validar_csv(
            conteudo,
            arquivo.filename or ""
        )

        if not relatorio["valido"]:
            return {
                **relatorio,
                "execucao_id": None,
                "status": "rejeitado"
            }

        resultado = salvar_execucao(
            conteudo,
            relatorio
        )

        return resultado

    except ValueError as erro:
        raise HTTPException(
            status_code=422,
            detail=str(erro)
        ) from erro

    finally:
        await arquivo.close()
