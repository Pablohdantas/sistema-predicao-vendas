
"""
Consulta HTTP da avaliação externa e das previsões.

Os arquivos são produzidos pelo serviço de avaliação.
Estas rotas não treinam nem reavaliam os modelos.
"""

import json

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.app.services.treinamento_service import (
    pasta_da_execucao
)


router = APIRouter(
    prefix="/avaliacoes",
    tags=["Avaliação externa"]
)


def localizar_execucao(execucao_id: str):
    try:
        return pasta_da_execucao(execucao_id)
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


@router.get("/{execucao_id}")
def consultar_metricas(execucao_id: str):
    pasta = localizar_execucao(execucao_id)
    caminho = pasta / "avaliacao_externa.json"

    if not caminho.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                "Avaliação externa ainda não disponível "
                "para esta execução."
            )
        )

    try:
        with caminho.open(encoding="utf-8") as arquivo:
            return json.load(arquivo)
    except (OSError, json.JSONDecodeError) as erro:
        raise HTTPException(
            status_code=500,
            detail="Não foi possível ler a avaliação externa."
        ) from erro


@router.get("/{execucao_id}/previsoes")
def consultar_previsoes(
    execucao_id: str,
    loja_id: int | None = Query(default=None, gt=0),
    limite: int = Query(default=100, ge=1, le=1000)
):
    pasta = localizar_execucao(execucao_id)
    caminho = pasta / "previsoes_teste_externo.csv"

    if not caminho.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                "Previsões externas ainda não disponíveis "
                "para esta execução."
            )
        )

    registros = []

    try:
        # Leitura em blocos para não carregar todo
        # o CSV na memória em bases grandes.
        for bloco in pd.read_csv(
            caminho,
            dtype={"loja_id": "int64"},
            chunksize=10000
        ):
            if loja_id is not None:
                bloco = bloco.loc[
                    bloco["loja_id"] == loja_id
                ]

            if bloco.empty:
                continue

            restantes = limite - len(registros)

            registros.extend(
                bloco.head(restantes).to_dict(
                    orient="records"
                )
            )

            if len(registros) >= limite:
                break

    except (OSError, ValueError, pd.errors.ParserError) as erro:
        raise HTTPException(
            status_code=500,
            detail="Não foi possível ler as previsões externas."
        ) from erro

    return {
        "execucao_id": pasta.name,
        "loja_id": loja_id,
        "limite": limite,
        "quantidade_retornada": len(registros),
        "previsoes": registros
    }
