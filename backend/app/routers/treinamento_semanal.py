
"""
Treinamento AutoML multihorizonte.

Permite iniciar e acompanhar o treinamento de
modelos para previsão dos próximos sete dias.

Não exige treinamento convencional prévio.
Preserva os modelos e avaliações já existentes.

Execução local com um único processo Uvicorn.
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.services.treinamento_service import (
    pasta_da_execucao,
)
from backend.app.services.pipeline_multihorizonte_service import (
    executar_pipeline_multihorizonte,
)


router = APIRouter(
    prefix="/treinamentos-semanais",
    tags=["Treinamento AutoML semanal"],
)

ARQUIVO_STATUS = "status_multihorizonte.json"

CONTROLE = threading.Lock()
EXECUCAO_ATIVA: str | None = None


class ConfiguracaoTreinamentoSemanal(BaseModel):
    tentativas: int = Field(default=3, ge=1, le=30)
    tempo_por_algoritmo: int = Field(
        default=30,
        ge=1,
        le=1800,
    )


def agora() -> str:
    return datetime.now(timezone.utc).isoformat()


def caminho_modelo(pasta: Path) -> Path:
    return (
        pasta
        / "multihorizonte"
        / "modelo_multihorizonte.joblib"
    )


def salvar_status(pasta: Path, dados: dict) -> None:
    destino = pasta / ARQUIVO_STATUS

    temporario = pasta / (
        f".{ARQUIVO_STATUS}."
        f"{threading.get_ident()}.tmp"
    )

    try:
        temporario.write_text(
            json.dumps(
                dados,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        os.replace(temporario, destino)

    finally:
        temporario.unlink(missing_ok=True)


def consultar_status(execucao_id: str) -> dict:
    pasta = pasta_da_execucao(execucao_id)
    caminho = pasta / ARQUIVO_STATUS
    modelo = caminho_modelo(pasta)

    if not caminho.is_file():
        if modelo.is_file():
            return {
                "execucao_id": pasta.name,
                "status": "concluido",
                "tipo": "multihorizonte",
                "erro": None,
            }

        return {
            "execucao_id": pasta.name,
            "status": "nao_iniciado",
            "tipo": "multihorizonte",
            "erro": None,
        }

    dados = json.loads(
        caminho.read_text(encoding="utf-8")
    )

    if dados.get("status") == "em_andamento":
        with CONTROLE:
            ativa = EXECUCAO_ATIVA == pasta.name

        if not ativa:
            if modelo.is_file():
                # O artefato pode ter sido salvo antes
                # de uma interrupção do servidor.
                dados["status"] = "concluido"
                dados["erro"] = None
            else:
                dados["status"] = "interrompido"
                dados["erro"] = (
                    "Treinamento interrompido. "
                    "Confira os arquivos da execução."
                )

            dados["atualizado_em"] = agora()
            salvar_status(pasta, dados)

    return dados


def executar_tarefa(
    execucao_id: str,
    tentativas: int,
    tempo_por_algoritmo: int,
) -> None:
    global EXECUCAO_ATIVA

    pasta = pasta_da_execucao(execucao_id)

    try:
        relatorio = executar_pipeline_multihorizonte(
            execucao_id=execucao_id,
            tentativas=tentativas,
            tempo_por_algoritmo=tempo_por_algoritmo,
        )

        salvar_status(
            pasta,
            {
                "execucao_id": execucao_id,
                "status": "concluido",
                "tipo": "multihorizonte",
                "algoritmo_vencedor": relatorio[
                    "algoritmo_vencedor"
                ],
                "otimizacao_flaml": relatorio[
                    "otimizacao_flaml"
                ],
                "avaliacao_externa": relatorio[
                    "avaliacao_externa"
                ],
                "tentativas": tentativas,
                "tempo_por_algoritmo": tempo_por_algoritmo,
                "atualizado_em": agora(),
                "erro": None,
            },
        )

    except Exception as erro:
        print(
            f"Erro no AutoML semanal {execucao_id}: "
            f"{type(erro).__name__}: {erro}",
            flush=True,
        )

        salvar_status(
            pasta,
            {
                "execucao_id": execucao_id,
                "status": "falhou",
                "tipo": "multihorizonte",
                "atualizado_em": agora(),
                "erro": (
                    f"{type(erro).__name__}: {erro}"
                ),
            },
        )

    finally:
        with CONTROLE:
            if EXECUCAO_ATIVA == execucao_id:
                EXECUCAO_ATIVA = None


@router.get("/{execucao_id}")
def obter_status_semanal(execucao_id: str):
    try:
        return consultar_status(execucao_id)

    except FileNotFoundError as erro:
        raise HTTPException(
            status_code=404,
            detail=str(erro),
        ) from erro

    except ValueError as erro:
        raise HTTPException(
            status_code=422,
            detail=str(erro),
        ) from erro


@router.post("/{execucao_id}", status_code=202)
def iniciar_treinamento_semanal(
    execucao_id: str,
    configuracao: ConfiguracaoTreinamentoSemanal,
):
    global EXECUCAO_ATIVA

    try:
        pasta = pasta_da_execucao(execucao_id)

    except FileNotFoundError as erro:
        raise HTTPException(
            status_code=404,
            detail=str(erro),
        ) from erro

    except ValueError as erro:
        raise HTTPException(
            status_code=422,
            detail=str(erro),
        ) from erro

    # Não é necessário treinar o modelo convencional.

    with CONTROLE:
        if EXECUCAO_ATIVA is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Outro treinamento semanal "
                    "já está em andamento."
                ),
            )

        if (pasta / "multihorizonte").exists():
            raise HTTPException(
                status_code=409,
                detail=(
                    "Esta execução já possui "
                    "arquivos do modelo semanal."
                ),
            )

        caminho = pasta / ARQUIVO_STATUS

        if caminho.is_file():
            anterior = json.loads(
                caminho.read_text(encoding="utf-8")
            )

            if anterior.get("status") in (
                "em_andamento",
                "concluido",
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "O treinamento semanal já foi "
                        "iniciado ou concluído."
                    ),
                )

        dados = {
            "execucao_id": pasta.name,
            "status": "em_andamento",
            "tipo": "multihorizonte",
            "tentativas": configuracao.tentativas,
            "tempo_por_algoritmo": (
                configuracao.tempo_por_algoritmo
            ),
            "iniciado_em": agora(),
            "atualizado_em": agora(),
            "erro": None,
        }

        salvar_status(pasta, dados)
        EXECUCAO_ATIVA = pasta.name

        tarefa = threading.Thread(
            target=executar_tarefa,
            args=(
                pasta.name,
                configuracao.tentativas,
                configuracao.tempo_por_algoritmo,
            ),
            daemon=True,
            name=f"semanal-{pasta.name[:8]}",
        )

        try:
            tarefa.start()

        except Exception as erro:
            EXECUCAO_ATIVA = None

            dados["status"] = "falhou"
            dados["erro"] = "Falha ao iniciar treinamento."
            dados["atualizado_em"] = agora()

            salvar_status(pasta, dados)

            raise HTTPException(
                status_code=500,
                detail=str(erro),
            ) from erro

    return dados
