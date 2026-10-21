
"""
Rotas de previsão automática, listagem de lojas
e explicabilidade SHAP por execução.
"""

import re
from pathlib import Path as CaminhoArquivo

import pandas as pd
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

PASTA_EXECUCOES = (
    CaminhoArquivo(__file__).resolve().parents[2] / "execucoes"
)


@router.get("/{execucao_id}/lojas")
def listar_lojas_execucao(execucao_id: str):
    """
    Lista todas as lojas do histórico disponível.

    Se houver atualização de vendas, consulta o
    histórico atualizado; caso contrário, usa
    o dataset original.
    """
    if not re.fullmatch(r"[0-9a-f]{32}", execucao_id):
        raise HTTPException(
            status_code=422,
            detail="Identificador de execução inválido.",
        )

    pasta = PASTA_EXECUCOES / execucao_id

    if not pasta.is_dir():
        raise HTTPException(
            status_code=404,
            detail="Execução não encontrada.",
        )

    historico_atualizado = pasta / "historico_atualizado.csv"
    dataset_original = pasta / "dataset.csv"

    arquivo = (
        historico_atualizado
        if historico_atualizado.is_file()
        else dataset_original
    )

    if not arquivo.is_file():
        raise HTTPException(
            status_code=404,
            detail="Histórico de vendas não encontrado.",
        )

    try:
        identificadores = set()

        for bloco in pd.read_csv(
            arquivo,
            usecols=["loja_id"],
            chunksize=100_000,
        ):
            identificadores.update(
                int(valor)
                for valor in bloco["loja_id"].dropna().unique()
            )

        lojas = sorted(identificadores)

    except (OSError, ValueError, KeyError) as erro:
        raise HTTPException(
            status_code=422,
            detail=f"Não foi possível ler as lojas: {erro}",
        ) from erro

    return {
        "execucao_id": execucao_id,
        "quantidade_lojas": len(lojas),
        "lojas": lojas,
    }


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
