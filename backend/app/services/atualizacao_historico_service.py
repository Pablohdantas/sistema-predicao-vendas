"""Atualização semanal com comparação de previsões previamente registradas.

Contrato: um CSV por atualização, contendo sete dias consecutivos para
cada loja informada. As previsões da semana devem existir antes do upload.
O CSV original, os modelos e a avaliação externa não são modificados.
Execução local, um processo de API. Consulte as limitações transacionais.
"""

from datetime import timedelta
from io import BytesIO
from pathlib import Path
import json
import os
import threading
import uuid

import joblib
import numpy as np
import pandas as pd

from dados.scripts.template import NOMES, validar_template
from backend.app.services.treinamento_service import pasta_da_execucao
from backend.app.services.registro_previsoes_service import (
    consultar_previsao_registrada,
    registrar_previsao,
)

TAMANHO_MAXIMO_ATUALIZACAO = 10 * 1024 * 1024
CONTROLE_ATUALIZACAO = threading.Lock()
OBRIGATORIAS = {"loja_id", "data", "vendas"}


def ler_historico(pasta: Path) -> pd.DataFrame:
    """Lê o último histórico confirmado ou o CSV de origem."""
    atualizado = pasta / "historico_atualizado.csv"
    caminho = atualizado if atualizado.is_file() else pasta / "dataset.csv"
    return pd.read_csv(caminho, dtype={"loja_id": "string", "data": "string"})


def possui_modelo_concluido(pasta: Path) -> bool:
    """Verifica o modelo operacional e o estado oficial do AutoML."""
    modelo = pasta / "multihorizonte"
    if not all((modelo / nome).is_file() for nome in (
        "modelo_multihorizonte.joblib", "relatorio.json", "previsoes_teste.csv",
        "previsoes_futuras.json",
    )):
        return False
    status = pasta / "status_multihorizonte.json"
    if not status.is_file():
        return False
    try:
        dados = json.loads(status.read_text(encoding="utf-8"))
    except (OSError, ValueError) as erro:
        raise ValueError("Status semanal ilegível.") from erro
    return isinstance(dados, dict) and dados.get("status") == "concluido"


def _validar_semanas(execucao_id: str, historico: pd.DataFrame, novos: pd.DataFrame) -> list[dict]:
    """Exige sete datas e compara apenas com previsões anteriores ao upload."""
    ultimas_datas = pd.to_datetime(historico["data"], format="%Y-%m-%d", errors="raise")
    ultimas = historico.assign(_data=ultimas_datas).groupby("loja_id")["_data"].max()
    avaliacoes = []

    for loja, grupo in novos.groupby("loja_id", sort=True):
        loja = str(loja)
        if loja not in ultimas.index:
            raise ValueError(f"Loja desconhecida: {loja}.")
        if len(grupo) != 7:
            raise ValueError(f"A loja {loja} precisa fornecer exatamente sete dias.")
        if not loja.isdecimal() or int(loja) < 1:
            raise ValueError(f"Identificador de loja não suportado: {loja}.")

        origem = ultimas.loc[loja]
        datas_esperadas = pd.date_range(origem + timedelta(days=1), periods=7, freq="D")
        datas_reais = pd.DatetimeIndex(pd.to_datetime(grupo["data"], format="%Y-%m-%d", errors="raise"))
        if not datas_reais.is_unique or not datas_reais.sort_values().equals(datas_esperadas):
            raise ValueError(
                f"A loja {loja} deve informar os sete dias consecutivos "
                f"de {datas_esperadas[0].date()} a {datas_esperadas[-1].date()}."
            )

        registro = consultar_previsao_registrada(execucao_id, int(loja), origem.strftime("%Y-%m-%d"))
        if registro is None:
            raise ValueError(
                f"A loja {loja} não possui previsão registrada em {origem.date()}. "
                "Consulte as previsões antes de atualizar as vendas."
            )
        previstos = registro.get("previsoes")
        if not isinstance(previstos, list) or len(previstos) != 7:
            raise ValueError(f"Registro de previsão incompleto da loja {loja}.")
        # Registros operacionais usam data_prevista; aceita registros
        # antigos com data, sem alterar previsões já persistidas.
        previstos_por_data = {}
        for item in previstos:
            if not isinstance(item, dict):
                raise ValueError(f"Previsão registrada inválida da loja {loja}.")
            data = item.get("data_prevista", item.get("data"))
            try:
                valor = float(item["vendas_previstas"])
            except (KeyError, TypeError, ValueError) as erro:
                raise ValueError(f"Valor previsto inválido da loja {loja}.") from erro
            if not isinstance(data, str) or not np.isfinite(valor) or valor < 0:
                raise ValueError(f"Data ou valor previsto inválido da loja {loja}.")
            if data in previstos_por_data:
                raise ValueError(f"Datas de previsão repetidas da loja {loja}.")
            previstos_por_data[data] = valor
        if len(previstos_por_data) != 7 or set(previstos_por_data) != set(datas_esperadas.strftime("%Y-%m-%d")):
            raise ValueError(f"Datas previstas incompatíveis para loja {loja}.")

        ordenado = grupo.sort_values("data")
        real = pd.to_numeric(ordenado["vendas"], errors="raise").to_numpy(dtype=float)
        previsto = np.array([previstos_por_data[d] for d in ordenado["data"]], dtype=float)
        if not np.isfinite(real).all() or not np.isfinite(previsto).all() or (real < 0).any():
            raise ValueError(f"Vendas ou previsões inválidas para loja {loja}.")
        diferencas = real - previsto
        avaliacoes.append({
            "loja_id": int(loja),
            "data_origem": origem.strftime("%Y-%m-%d"),
            "inicio": datas_esperadas[0].strftime("%Y-%m-%d"),
            "fim": datas_esperadas[-1].strftime("%Y-%m-%d"),
            "mae": round(float(np.mean(np.abs(diferencas))), 4),
            "rmse": round(float(np.sqrt(np.mean(diferencas ** 2))), 4),
            "total_previsto": round(float(previsto.sum()), 2),
            "total_real": round(float(real.sum()), 2),
            "diferenca_total": round(float(diferencas.sum()), 2),
            "dias": [
                {"data": data, "previsto": round(float(p), 2), "real": round(float(r), 2),
                 "erro": round(float(r - p), 2)}
                for data, r, p in zip(ordenado["data"], real, previsto)
            ],
        })
    return avaliacoes


def _escrever_atomico(destino: Path, conteudo: str) -> None:
    """Substitui arquivo completo; evita leitura de JSON/CSV parcialmente escrito."""
    temporario = destino.parent / f".{destino.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporario.open("w", encoding="utf-8") as arquivo:
            arquivo.write(conteudo)
            arquivo.flush()
            os.fsync(arquivo.fileno())
        os.replace(temporario, destino)
    finally:
        temporario.unlink(missing_ok=True)


def _proximas_previsoes(pasta: Path, combinado: pd.DataFrame, lojas: set[int]) -> list[dict]:
    """Usa o modelo operacional existente, sem qualquer chamada de fit()."""
    from backend.app.services.pipeline_multihorizonte_service import gerar_previsoes_semanais

    artefato = joblib.load(pasta / "multihorizonte" / "modelo_multihorizonte.joblib")
    if not isinstance(artefato, dict) or artefato.get("tipo") != "multihorizonte":
        raise ValueError("Modelo operacional inválido.")
    resultados = gerar_previsoes_semanais(combinado, artefato["modelo"], artefato["preprocessamento"])
    return [resultado for resultado in resultados if int(resultado["loja_id"]) in lojas]


def atualizar_historico(execucao_id: str, conteudo: bytes, nome_arquivo: str) -> dict:
    """Valida, avalia e atualiza sete dias completos de cada loja enviada."""
    pasta = pasta_da_execucao(execucao_id)
    if not possui_modelo_concluido(pasta):
        raise ValueError("Conclua o AutoML semanal antes da atualização.")
    if not nome_arquivo.lower().endswith(".csv"):
        raise ValueError("A atualização deve ser um CSV.")
    if not conteudo or len(conteudo) > TAMANHO_MAXIMO_ATUALIZACAO:
        raise ValueError("O CSV deve conter dados e ter até 10 MB.")

    try:
        novos = pd.read_csv(BytesIO(conteudo), dtype={"loja_id": "string", "data": "string"}, encoding="utf-8-sig")
    except (UnicodeError, ValueError, pd.errors.ParserError) as erro:
        raise ValueError("CSV de atualização inválido.") from erro
    if novos.empty or novos.columns.duplicated().any():
        raise ValueError("CSV vazio ou com colunas duplicadas.")
    faltantes = OBRIGATORIAS - set(novos.columns)
    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(faltantes)}")
    desconhecidas = set(novos.columns) - set(NOMES)
    if desconhecidas:
        raise ValueError(f"Colunas não reconhecidas: {sorted(desconhecidas)}")
    if novos[list(OBRIGATORIAS)].isna().any().any():
        raise ValueError("Os campos obrigatórios não podem estar vazios.")

    with CONTROLE_ATUALIZACAO:
        if not possui_modelo_concluido(pasta):
            raise ValueError("O treinamento semanal não está concluído.")
        historico = ler_historico(pasta)
        colunas_historico = historico.columns.tolist()
        extras = set(novos.columns) - set(colunas_historico)
        if extras:
            raise ValueError(f"Colunas inexistentes no histórico original: {sorted(extras)}")
        for coluna in colunas_historico:
            if coluna not in novos.columns:
                novos[coluna] = pd.NA
        novos = novos[colunas_historico].copy()
        verificacao = validar_template(novos, min_linhas=1, min_dias=1)
        if not verificacao.valido:
            raise ValueError("Atualização inválida: " + "; ".join(verificacao.erros))

        avaliacoes = _validar_semanas(execucao_id, historico, novos)
        combinado = pd.concat([historico, novos], ignore_index=True)
        combinado = combinado.sort_values(["loja_id", "data"]).reset_index(drop=True)
        validacao = validar_template(combinado, min_linhas=90, min_dias=90)
        if not validacao.valido:
            raise ValueError("Histórico resultante inválido: " + "; ".join(validacao.erros))

        # Antecipadamente testa a previsão seguinte; evita atualizar a base se o modelo falhar.
        lojas = {item["loja_id"] for item in avaliacoes}
        proximas = _proximas_previsoes(pasta, combinado, lojas)
        if len(proximas) != len(lojas):
            raise RuntimeError("Não foi possível gerar a próxima previsão de todas as lojas.")

        pasta_avaliacoes = pasta / "avaliacoes_semanais"
        pasta_avaliacoes.mkdir(exist_ok=True)
        destinos = [pasta_avaliacoes / f"loja_{a['loja_id']}_{a['data_origem']}.json" for a in avaliacoes]
        if any(d.exists() for d in destinos):
            raise ValueError("Esta semana já foi avaliada; não é permitido duplicar a atualização.")

        # Resultados escritos antes do CSV. Se a atualização do CSV falhar,
        # apaga-se o que foi criado, preservando o estado anterior.
        criados = []
        try:
            for destino, avaliacao in zip(destinos, avaliacoes):
                _escrever_atomico(destino, json.dumps(avaliacao, ensure_ascii=False, indent=2, allow_nan=False))
                criados.append(destino)
            destino_historico = pasta / "historico_atualizado.csv"
            _escrever_atomico(destino_historico, combinado.to_csv(index=False))
        except Exception:
            for destino in criados:
                destino.unlink(missing_ok=True)
            raise

        # O CSV está confirmado. Caso esta etapa falhe, uma nova consulta de
        # previsão pode recuperar a geração sem retreinamento.
        resultados_seguintes = [registrar_previsao(execucao_id, item) for item in proximas]
        return {
            "execucao_id": pasta.name,
            "status": "historico_atualizado",
            "registros_adicionados": int(len(novos)),
            "registros_totais": int(len(combinado)),
            "quantidade_lojas": int(combinado["loja_id"].nunique()),
            "ultimas_datas_por_loja": combinado.groupby("loja_id")["data"].max().to_dict(),
            "modelo_retreinado": False,
            "avaliacao_externa_preservada": True,
            "avaliacoes_semanais": avaliacoes,
            "proximas_previsoes": resultados_seguintes,
        }
