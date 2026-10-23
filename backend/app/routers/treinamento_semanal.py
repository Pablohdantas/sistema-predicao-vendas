"""Rotas do treinamento AutoML multihorizonte.

O treinamento convencional nao e necessario. Este executor e LOCAL,
para um unico processo Uvicorn; nao e um sistema distribuido de jobs.
"""

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.services.treinamento_service import pasta_da_execucao
from backend.app.services.pipeline_multihorizonte_service import (
    executar_pipeline_multihorizonte,
)

router = APIRouter(
    prefix="/treinamentos-semanais",
    tags=["Treinamento AutoML semanal"],
)

ARQUIVO_STATUS = "status_multihorizonte.json"
ARQUIVOS_OBRIGATORIOS = (
    "modelo_multihorizonte.joblib",
    "relatorio.json",
    "previsoes_teste.csv",
    "previsoes_futuras.json",
)
CONTROLE = threading.RLock()
EXECUCAO_ATIVA: str | None = None


class ConfiguracaoTreinamentoSemanal(BaseModel):
    tentativas: int = Field(default=3, ge=1, le=30)
    tempo_por_algoritmo: int = Field(default=30, ge=1, le=1800)


def agora() -> str:
    return datetime.now(timezone.utc).isoformat()


def _caminho_status(pasta: Path) -> Path:
    return pasta / ARQUIVO_STATUS


def salvar_status(pasta: Path, dados: dict) -> None:
    """Grava o JSON por substituicao atomica no mesmo filesystem."""
    destino = _caminho_status(pasta)
    temporario = pasta / f".{ARQUIVO_STATUS}.{uuid.uuid4().hex}.tmp"
    try:
        with temporario.open("w", encoding="utf-8") as arquivo:
            json.dump(dados, arquivo, ensure_ascii=False, indent=2)
            arquivo.flush()
            os.fsync(arquivo.fileno())
        os.replace(temporario, destino)
    finally:
        temporario.unlink(missing_ok=True)


def _ler_status(pasta: Path) -> dict | None:
    caminho = _caminho_status(pasta)
    if not caminho.is_file():
        return None
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, ValueError) as erro:
        raise ValueError("Status do treinamento esta corrompido.") from erro
    if not isinstance(dados, dict):
        raise ValueError("Formato do status de treinamento invalido.")
    return dados


def verificar_artefatos(pasta: Path) -> tuple[bool, str]:
    """Verifica completude basica sem desserializar arquivos joblib."""
    destino = pasta / "multihorizonte"
    faltantes = [
        nome for nome in ARQUIVOS_OBRIGATORIOS
        if not (destino / nome).is_file()
        or (destino / nome).stat().st_size == 0
    ]
    if faltantes:
        return False, "Artefatos ausentes ou vazios: " + ", ".join(faltantes)

    try:
        relatorio = json.loads((destino / "relatorio.json").read_text(encoding="utf-8"))
        futuras = json.loads((destino / "previsoes_futuras.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "Relatorio ou previsoes futuras com JSON invalido."

    if not isinstance(relatorio, dict) or not isinstance(futuras, list) or not futuras:
        return False, "Relatorio ou previsoes futuras incompletos."
    if not relatorio.get("algoritmo_vencedor") or not relatorio.get("avaliacao_externa"):
        return False, "Relatorio nao contem resultado da avaliacao externa."

    # Treinamentos atuais geram dois modelos; pilotos antigos geravam apenas um.
    if relatorio.get("modelo_operacional"):
        avaliacao = destino / "modelo_avaliacao_externa.joblib"
        if not avaliacao.is_file() or avaliacao.stat().st_size == 0:
            return False, "Modelo de avaliacao externa ausente ou vazio."
    return True, ""


def consultar_status(execucao_id: str) -> dict:
    pasta = pasta_da_execucao(execucao_id)
    with CONTROLE:
        dados = _ler_status(pasta)
        ativa = EXECUCAO_ATIVA == pasta.name
        completos, motivo = verificar_artefatos(pasta)

        if dados is None:
            if completos:
                # Compatibilidade apenas de leitura com pilotos antigos.
                return {
                    "execucao_id": pasta.name,
                    "status": "concluido",
                    "tipo": "multihorizonte",
                    "legado_sem_status": True,
                    "erro": None,
                }
            return {
                "execucao_id": pasta.name,
                "status": "interrompido" if (pasta / "multihorizonte").exists() else "nao_iniciado",
                "tipo": "multihorizonte",
                "erro": motivo if (pasta / "multihorizonte").exists() else None,
            }

        if dados.get("status") == "em_andamento" and not ativa:
            dados["status"] = "interrompido"
            dados["erro"] = (
                "O processo responsavel pelo treinamento nao esta ativo. "
                "Os artefatos existentes precisam ser verificados antes de qualquer nova tentativa."
            )
            dados["atualizado_em"] = agora()
            salvar_status(pasta, dados)
        elif dados.get("status") == "concluido" and not completos:
            dados["status"] = "falhou"
            dados["erro"] = "Treinamento marcado como concluido, mas incompleto: " + motivo
            dados["atualizado_em"] = agora()
            salvar_status(pasta, dados)
        return dados


def executar_tarefa(execucao_id: str, tentativas: int, tempo_por_algoritmo: int) -> None:
    global EXECUCAO_ATIVA
    pasta = pasta_da_execucao(execucao_id)
    try:
        relatorio = executar_pipeline_multihorizonte(
            execucao_id=execucao_id,
            tentativas=tentativas,
            tempo_por_algoritmo=tempo_por_algoritmo,
        )
        completos, motivo = verificar_artefatos(pasta)
        if not completos:
            raise RuntimeError("Treinamento nao produziu todos os artefatos: " + motivo)
        salvar_status(pasta, {
            "execucao_id": pasta.name,
            "status": "concluido",
            "tipo": "multihorizonte",
            "algoritmo_vencedor": relatorio["algoritmo_vencedor"],
            "otimizacao_flaml": relatorio["otimizacao_flaml"],
            "avaliacao_externa": relatorio["avaliacao_externa"],
            "tentativas": tentativas,
            "tempo_por_algoritmo": tempo_por_algoritmo,
            "atualizado_em": agora(),
            "erro": None,
        })
    except Exception as erro:
        # Detalhes tecnicos ficam no servidor; mensagens da API devem ser controladas.
        print(
            f"Erro no AutoML semanal {execucao_id}: {type(erro).__name__}: {erro}",
            flush=True,
        )
        try:
            salvar_status(pasta, {
                "execucao_id": pasta.name,
                "status": "falhou",
                "tipo": "multihorizonte",
                "atualizado_em": agora(),
                "erro": f"Falha no treinamento: {type(erro).__name__}. Consulte os logs da API.",
            })
        except OSError as falha_status:
            print(f"Falha ao registrar status de erro: {falha_status}", flush=True)
    finally:
        with CONTROLE:
            if EXECUCAO_ATIVA == execucao_id:
                EXECUCAO_ATIVA = None


@router.get("/{execucao_id}")
def obter_status_semanal(execucao_id: str):
    try:
        return consultar_status(execucao_id)
    except FileNotFoundError as erro:
        raise HTTPException(status_code=404, detail=str(erro)) from erro
    except ValueError as erro:
        raise HTTPException(status_code=422, detail=str(erro)) from erro


@router.post("/{execucao_id}", status_code=202)
def iniciar_treinamento_semanal(
    execucao_id: str,
    configuracao: ConfiguracaoTreinamentoSemanal,
):
    global EXECUCAO_ATIVA
    try:
        pasta = pasta_da_execucao(execucao_id)
    except FileNotFoundError as erro:
        raise HTTPException(status_code=404, detail=str(erro)) from erro
    except ValueError as erro:
        raise HTTPException(status_code=422, detail=str(erro)) from erro

    with CONTROLE:
        if EXECUCAO_ATIVA is not None:
            raise HTTPException(status_code=409, detail="Outro treinamento ja esta em andamento.")
        if (pasta / "multihorizonte").exists():
            raise HTTPException(
                status_code=409,
                detail="Ja existem artefatos nesta execucao. Nao serao sobrescritos.",
            )
        try:
            anterior = _ler_status(pasta)
        except ValueError as erro:
            raise HTTPException(status_code=409, detail=str(erro)) from erro
        if anterior and anterior.get("status") in ("em_andamento", "concluido"):
            raise HTTPException(
                status_code=409,
                detail="O treinamento desta execucao ja foi iniciado ou concluido.",
            )

        dados = {
            "execucao_id": pasta.name,
            "status": "em_andamento",
            "tipo": "multihorizonte",
            "tentativas": configuracao.tentativas,
            "tempo_por_algoritmo": configuracao.tempo_por_algoritmo,
            "iniciado_em": agora(),
            "atualizado_em": agora(),
            "erro": None,
        }
        salvar_status(pasta, dados)
        EXECUCAO_ATIVA = pasta.name
        tarefa = threading.Thread(
            target=executar_tarefa,
            args=(pasta.name, configuracao.tentativas, configuracao.tempo_por_algoritmo),
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
            raise HTTPException(status_code=500, detail="Falha ao iniciar treinamento.") from erro
    return dados
