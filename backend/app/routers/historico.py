"""Rotas para inserir vendas reais e consultar avaliações semanais."""
from pathlib import Path
import json

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.services.atualizacao_historico_service import (
    TAMANHO_MAXIMO_ATUALIZACAO,
    atualizar_historico,
)
from backend.app.services.treinamento_service import pasta_da_execucao

router = APIRouter(prefix="/historico", tags=["Histórico de vendas"])


@router.post("/execucoes/{execucao_id}/atualizar")
async def adicionar_vendas(execucao_id: str, arquivo: UploadFile = File(...)):
    try:
        conteudo = await arquivo.read(TAMANHO_MAXIMO_ATUALIZACAO + 1)
        return atualizar_historico(execucao_id, conteudo, arquivo.filename or "")
    except FileNotFoundError as erro:
        raise HTTPException(status_code=404, detail="Execução não encontrada.") from erro
    except ValueError as erro:
        raise HTTPException(status_code=422, detail=str(erro)) from erro
    finally:
        await arquivo.close()


@router.get("/execucoes/{execucao_id}/avaliacoes")
def listar_avaliacoes_semanais(execucao_id: str):
    try:
        pasta = pasta_da_execucao(execucao_id)
    except FileNotFoundError as erro:
        raise HTTPException(status_code=404, detail="Execução não encontrada.") from erro
    except ValueError as erro:
        raise HTTPException(status_code=422, detail=str(erro)) from erro
    avaliacoes = []
    for caminho in sorted((pasta / "avaliacoes_semanais").glob("loja_*.json")):
        try:
            avaliacoes.append(json.loads(caminho.read_text(encoding="utf-8")))
        except (ValueError, OSError) as erro:
            raise HTTPException(status_code=500, detail="Avaliação semanal armazenada inválida.") from erro
    return {"execucao_id": pasta.name, "avaliacoes": avaliacoes}
