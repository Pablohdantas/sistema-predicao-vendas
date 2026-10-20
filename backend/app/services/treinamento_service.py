
"""
Gerenciamento local dos treinamentos AutoML.

- Uma execução de treinamento por vez.
- Processamento em thread separada.
- Status persistido em JSON.
- Preparação automática dos atributos.
- Compatível com os modos teste e definitivo.

Projetado inicialmente para uma instância da API,
com um único processo de servidor.
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from backend.app.services.atributos_service import (
    preparar_atributos_execucao
)
from backend.app.services.automl_service import (
    executar_pipeline_automl
)
from backend.app.services.execucoes_service import (
    PASTA_EXECUCOES
)
from backend.app.services.previsao_service import (
    validar_execucao_id
)


NOME_STATUS = "status_treinamento.json"

CONTROLE = threading.Lock()
EXECUCAO_ATIVA = None
THREAD_ATIVA = None


def agora() -> str:
    """Retorna o instante atual em formato ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def pasta_da_execucao(execucao_id: str) -> Path:
    """Localiza somente execuções com identificador válido."""

    identificador = validar_execucao_id(execucao_id)
    pasta = PASTA_EXECUCOES / identificador

    if not pasta.is_dir():
        raise FileNotFoundError(
            f"Execução não encontrada: {identificador}"
        )

    if not (pasta / "dataset.csv").is_file():
        raise FileNotFoundError(
            "O dataset desta execução não foi encontrado."
        )

    return pasta


def gravar_status(pasta: Path, dados: dict) -> None:
    """
    Grava o status de forma atômica para evitar
    leituras de arquivos JSON incompletos.
    """

    destino = pasta / NOME_STATUS
    temporario = pasta / (
        f".{NOME_STATUS}.{threading.get_ident()}.tmp"
    )

    try:
        temporario.write_text(
            json.dumps(
                dados,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )
        os.replace(temporario, destino)
    finally:
        if temporario.exists():
            temporario.unlink()


def consultar_treinamento(execucao_id: str) -> dict:
    """
    Consulta o estado de uma execução.

    Um status 'em_andamento' deixado por uma
    interrupção do servidor será identificado
    quando não existir tarefa ativa nesta instância.
    """

    global EXECUCAO_ATIVA

    pasta = pasta_da_execucao(execucao_id)
    caminho = pasta / NOME_STATUS

    if not caminho.is_file():
        return {
            "execucao_id": pasta.name,
            "status": "aguardando_treinamento",
            "etapa": None,
            "modo": None,
            "resultado": None
        }

    dados = json.loads(
        caminho.read_text(encoding="utf-8")
    )

    if dados.get("status") == "em_andamento":
        with CONTROLE:
            ativa = EXECUCAO_ATIVA == pasta.name

        if not ativa:
            dados["status"] = "interrompido"
            dados["etapa"] = "servidor_interrompido"
            dados["atualizado_em"] = agora()
            dados["erro"] = (
                "A tarefa não está ativa neste processo. "
                "O treinamento pode ter sido interrompido."
            )
            gravar_status(pasta, dados)

    return dados


def executar_em_segundo_plano(
    execucao_id: str,
    modo_teste: bool,
    tentativas: int,
    tempo_por_algoritmo: int
) -> None:
    """Executa o pipeline e atualiza seu status."""

    global EXECUCAO_ATIVA, THREAD_ATIVA

    pasta = PASTA_EXECUCOES / execucao_id
    caminho_status = pasta / NOME_STATUS

    try:
        dados_status = json.loads(
            caminho_status.read_text(encoding="utf-8")
        )

        dados_status["etapa"] = "preparando_atributos"
        dados_status["atualizado_em"] = agora()
        gravar_status(pasta, dados_status)

        if not (pasta / "atributos_h7.csv").is_file():
            preparar_atributos_execucao(pasta)

        dados_status["etapa"] = "otimizando_modelos"
        dados_status["atualizado_em"] = agora()
        gravar_status(pasta, dados_status)

        resultado = executar_pipeline_automl(
            pasta_execucao=pasta,
            modo_teste=modo_teste,
            tentativas=tentativas,
            tempo_por_algoritmo=tempo_por_algoritmo
        )

        dados_status.update({
            "status": "concluido",
            "etapa": "finalizado",
            "atualizado_em": agora(),
            "finalizado_em": agora(),
            "resultado": {
                "melhor_algoritmo_na_validacao": (
                    resultado["melhor_algoritmo_na_validacao"]
                ),
                "algoritmo_vencedor": (
                    resultado["algoritmo_vencedor"]
                ),
                "rmse_selecao": resultado["rmse_selecao"],
                "mae_selecao": resultado["mae_selecao"],
                "modelo_salvo": resultado["modelo_salvo"],
                "quantidade_dobras": (
                    resultado["quantidade_dobras"]
                ),
                "tempo_total_segundos": (
                    resultado["tempo_total_segundos"]
                )
            },
            "erro": None
        })

        gravar_status(pasta, dados_status)

    except Exception as erro:
        # Registra a falha e libera a fila.
        print(
            f"Erro no treinamento {execucao_id}: "
            f"{type(erro).__name__}: {erro}",
            flush=True
        )

        dados_status = {
            "execucao_id": execucao_id,
            "status": "falhou",
            "etapa": "erro",
            "modo": (
                "teste_integracao"
                if modo_teste
                else "treinamento_definitivo"
            ),
            "atualizado_em": agora(),
            "erro": (
                f"{type(erro).__name__}: {erro}"
            ),
            "resultado": None
        }

        gravar_status(pasta, dados_status)

    finally:
        with CONTROLE:
            if EXECUCAO_ATIVA == execucao_id:
                EXECUCAO_ATIVA = None
                THREAD_ATIVA = None


def iniciar_treinamento(
    execucao_id: str,
    modo_teste: bool = True,
    tentativas: int = 2,
    tempo_por_algoritmo: int = 60
) -> dict:
    """
    Reserva a execução e inicia o trabalho em thread.

    Retorna imediatamente, sem esperar os modelos.
    """

    global EXECUCAO_ATIVA, THREAD_ATIVA

    pasta = pasta_da_execucao(execucao_id)
    identificador = pasta.name

    if tentativas < 1 or tempo_por_algoritmo < 1:
        raise ValueError(
            "Tentativas e tempo devem ser positivos."
        )

    with CONTROLE:
        if EXECUCAO_ATIVA is not None:
            raise RuntimeError(
                "Já existe um treinamento em andamento: "
                f"{EXECUCAO_ATIVA}"
            )

        caminho_status = pasta / NOME_STATUS

        if caminho_status.is_file():
            anterior = json.loads(
                caminho_status.read_text(encoding="utf-8")
            )

            if anterior.get("status") == "em_andamento":
                raise RuntimeError(
                    "Esta execução já possui um treinamento "
                    "em andamento. Consulte seu status."
                )

            if (
                anterior.get("status") == "concluido"
                and anterior.get("modo") == "treinamento_definitivo"
            ):
                raise RuntimeError(
                    "Esta execução já possui um treinamento "
                    "definitivo concluído."
                )

        if (
            not modo_teste
            and (pasta / "modelo_vencedor.joblib").exists()
        ):
            raise RuntimeError(
                "Esta execução já possui um modelo definitivo."
            )

        status = {
            "execucao_id": identificador,
            "status": "em_andamento",
            "etapa": "inicializando",
            "modo": (
                "teste_integracao"
                if modo_teste
                else "treinamento_definitivo"
            ),
            "tentativas_por_algoritmo": tentativas,
            "tempo_por_algoritmo": tempo_por_algoritmo,
            "iniciado_em": agora(),
            "atualizado_em": agora(),
            "finalizado_em": None,
            "resultado": None,
            "erro": None
        }

        gravar_status(pasta, status)

        EXECUCAO_ATIVA = identificador

        thread = threading.Thread(
            target=executar_em_segundo_plano,
            args=(
                identificador,
                modo_teste,
                tentativas,
                tempo_por_algoritmo
            ),
            daemon=True,
            name=f"automl-{identificador[:8]}"
        )

        THREAD_ATIVA = thread

        try:
            thread.start()
        except Exception:
            EXECUCAO_ATIVA = None
            THREAD_ATIVA = None

            status["status"] = "falhou"
            status["etapa"] = "erro_inicializacao"
            status["erro"] = (
                "Não foi possível iniciar a tarefa."
            )
            gravar_status(pasta, status)
            raise

    return status
