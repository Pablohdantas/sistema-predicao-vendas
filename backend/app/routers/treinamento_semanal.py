
"""
Rotas para treinamento experimental multihorizonte.

Permite:
- Iniciar o treinamento semanal em uma thread.
- Consultar o status do treinamento.
- Reconhecer modelos semanais já treinados.
- Impedir treinamentos simultâneos.
- Preservar os modelos convencionais existentes.

Esta implementação é destinada a uma instância local
do FastAPI, executada com um único processo.
"""

import json
import os
import threading

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.app.services.treinamento_service import (
    consultar_treinamento,
    pasta_da_execucao,
)

from backend.app.services.pipeline_multihorizonte_service import (
    executar_pipeline_multihorizonte,
)


router = APIRouter(
    prefix="/treinamentos-semanais",
    tags=["Treinamento semanal"],
)

ARQUIVO_STATUS = "status_multihorizonte.json"

CONTROLE = threading.Lock()

EXECUCAO_ATIVA: str | None = None


def agora() -> str:
    """Retorna a data e a hora atual em UTC."""
    return datetime.now(timezone.utc).isoformat()


def caminho_modelo(pasta: Path) -> Path:
    """Retorna o caminho do modelo semanal."""
    return (
        pasta
        / "multihorizonte"
        / "modelo_multihorizonte.joblib"
    )


def salvar_status(
    pasta: Path,
    dados: dict,
) -> None:
    """
    Salva o status de forma atômica.

    Evita que o frontend leia um JSON parcialmente
    escrito durante a atualização.
    """

    destino = pasta / ARQUIVO_STATUS

    temporario = pasta / (
        f".{ARQUIVO_STATUS}."
        f"{threading.get_ident()}.tmp"
    )

    try:
        temporario.write_text(
            json.dumps(
                dados,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        os.replace(temporario, destino)

    finally:
        if temporario.exists():
            temporario.unlink()


def consultar_status(
    execucao_id: str,
) -> dict:
    """
    Consulta o treinamento semanal.

    Reconhece também modelos gerados anteriormente
    pelo script de treinamento, mesmo quando ainda
    não existe um arquivo de status.
    """

    pasta = pasta_da_execucao(execucao_id)

    caminho = pasta / ARQUIVO_STATUS

    modelo = caminho_modelo(pasta)

    if not caminho.is_file():

        if modelo.is_file():
            return {
                "execucao_id": pasta.name,
                "status": "concluido",
                "tipo": "piloto_multihorizonte",
                "erro": None,
            }

        return {
            "execucao_id": pasta.name,
            "status": "nao_iniciado",
            "tipo": "piloto_multihorizonte",
            "erro": None,
        }

    dados = json.loads(
        caminho.read_text(encoding="utf-8")
    )

    if dados.get("status") == "em_andamento":

        with CONTROLE:
            ativa = EXECUCAO_ATIVA == pasta.name

        if not ativa:
            dados["status"] = "interrompido"
            dados["erro"] = (
                "A tarefa não está ativa neste servidor. "
                "O treinamento pode ter sido interrompido."
            )
            dados["atualizado_em"] = agora()

            salvar_status(pasta, dados)

    return dados


def executar_tarefa(
    execucao_id: str,
) -> None:
    """
    Executa o pipeline semanal em segundo plano.

    Não altera o treinamento convencional.
    """

    global EXECUCAO_ATIVA

    pasta = pasta_da_execucao(execucao_id)

    try:
        relatorio = executar_pipeline_multihorizonte(
            execucao_id
        )

        dados = {
            "execucao_id": execucao_id,
            "status": "concluido",
            "tipo": "piloto_multihorizonte",
            "algoritmo_vencedor": relatorio[
                "algoritmo_vencedor"
            ],
            "avaliacao_externa": relatorio[
                "avaliacao_externa"
            ],
            "atualizado_em": agora(),
            "erro": None,
        }

        salvar_status(pasta, dados)

    except Exception as erro:

        print(
            f"Erro no treinamento semanal "
            f"{execucao_id}: "
            f"{type(erro).__name__}: {erro}",
            flush=True,
        )

        dados = {
            "execucao_id": execucao_id,
            "status": "falhou",
            "tipo": "piloto_multihorizonte",
            "atualizado_em": agora(),
            "erro": (
                f"{type(erro).__name__}: {erro}"
            ),
        }

        salvar_status(pasta, dados)

    finally:
        with CONTROLE:
            if EXECUCAO_ATIVA == execucao_id:
                EXECUCAO_ATIVA = None


@router.get("/{execucao_id}")
def obter_status_semanal(
    execucao_id: str,
):
    """Consulta o status do treinamento semanal."""

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


@router.post(
    "/{execucao_id}",
    status_code=202,
)
def iniciar_treinamento_semanal(
    execucao_id: str,
):
    """
    Inicia o treinamento multihorizonte.

    Esta versão experimental só permite iniciar
    o treinamento semanal após a conclusão do
    treinamento convencional.
    """

    global EXECUCAO_ATIVA

    try:
        pasta = pasta_da_execucao(execucao_id)

        status_original = consultar_treinamento(
            execucao_id
        )

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

    if (
        status_original.get("status") != "concluido"
        or status_original.get("modo")
        != "treinamento_definitivo"
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "Conclua primeiro o treinamento "
                "definitivo convencional."
            ),
        )

    # Impede sobrescrever modelos já existentes.
    if (pasta / "multihorizonte").exists():
        raise HTTPException(
            status_code=409,
            detail=(
                "Esta execução já possui arquivos "
                "do modelo semanal."
            ),
        )

    # Esta consulta acontece FORA do bloqueio.
    # Assim, evitamos bloquear a própria thread
    # ao tentar consultar o status.
    status_anterior = consultar_status(
        execucao_id
    )

    if status_anterior["status"] == "concluido":
        raise HTTPException(
            status_code=409,
            detail=(
                "O treinamento semanal "
                "já foi concluído."
            ),
        )

    with CONTROLE:

        if EXECUCAO_ATIVA is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Já existe outro treinamento semanal "
                    "em andamento neste servidor."
                ),
            )

        if (pasta / "multihorizonte").exists():
            raise HTTPException(
                status_code=409,
                detail=(
                    "Esta execução já possui "
                    "um modelo semanal."
                ),
            )

        dados = {
            "execucao_id": execucao_id,
            "status": "em_andamento",
            "tipo": "piloto_multihorizonte",
            "iniciado_em": agora(),
            "atualizado_em": agora(),
            "erro": None,
        }

        salvar_status(pasta, dados)

        EXECUCAO_ATIVA = execucao_id

        tarefa = threading.Thread(
            target=executar_tarefa,
            args=(execucao_id,),
            daemon=True,
            name=f"semanal-{execucao_id[:8]}",
        )

        try:
            tarefa.start()

        except Exception as erro:

            EXECUCAO_ATIVA = None

            dados["status"] = "falhou"
            dados["erro"] = (
                "Não foi possível iniciar "
                "a thread de treinamento."
            )
            dados["atualizado_em"] = agora()

            salvar_status(pasta, dados)

            raise HTTPException(
                status_code=500,
                detail=str(erro),
            ) from erro

    return dados
