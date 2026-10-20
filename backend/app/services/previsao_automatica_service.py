
"""
Previsão automática para uma loja de uma execução AutoML.

Utiliza o histórico salvo da loja e prevê vendas
para sete dias após a última observação registrada.

Não exige que o usuário informe lags, médias móveis
ou atributos técnicos.

Informações comerciais futuras desconhecidas
permanecem ausentes (NaN).
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from backend.app.services.atributos_service import (
    gerar_atributos_dataset
)
from backend.app.services.previsao_service import (
    carregar_modelo,
    prever_vendas
)
from backend.app.services.treinamento_service import (
    consultar_treinamento,
    pasta_da_execucao
)
from backend.app.services.atualizacao_historico_service import (
    ler_historico
)

HORIZONTE = 7

ATRIBUTOS_ESTATICOS = (
    "tipo_loja",
    "sortimento",
    "distancia_concorrente"
)

VARIAVEIS_FUTURAS_DESCONHECIDAS = (
    "promocao",
    "promocao_continua",
    "mes_renovacao_promocao",
    "feriado",
    "feriado_escolar"
)


def gerar_entrada_automatica(
    execucao_id: str,
    loja_id: int
) -> dict:
    """
    Monta uma observação futura usando exclusivamente
    o histórico disponível até a data de origem.

    O valor de vendas da data futura nunca é fornecido.
    """

    if loja_id <= 0:
        raise ValueError(
            "O identificador da loja deve ser positivo."
        )

    pasta = pasta_da_execucao(execucao_id)

    status = consultar_treinamento(pasta.name)

    if (
        status.get("status") != "concluido"
        or status.get("modo") != "treinamento_definitivo"
    ):
        raise ValueError(
            "Esta execução ainda não possui um "
            "treinamento definitivo concluído."
        )

    dados = ler_historico(pasta)

    dados["data"] = pd.to_datetime(
        dados["data"],
        format="%Y-%m-%d",
        errors="raise"
    )

    ids = pd.to_numeric(
        dados["loja_id"],
        errors="coerce"
    )

    historico = dados.loc[
        ids == loja_id
    ].copy()

    if historico.empty:
        raise LookupError(
            f"Loja {loja_id} não encontrada nesta execução."
        )

    historico = historico.sort_values(
        "data"
    ).reset_index(drop=True)

    ultima_observacao = historico.iloc[-1]

    data_origem = pd.Timestamp(
        ultima_observacao["data"]
    ).normalize()

    data_prevista = (
        data_origem + timedelta(days=HORIZONTE)
    )

    # Prepara um registro do dia futuro, mas sem
    # inventar vendas ou eventos comerciais futuros.
    futuro = {
        coluna: np.nan
        for coluna in historico.columns
    }

    futuro["loja_id"] = str(loja_id)
    futuro["data"] = data_prevista
    futuro["vendas"] = np.nan

    # Características que podem ser consideradas
    # estáticas na versão atual da ferramenta.
    for coluna in ATRIBUTOS_ESTATICOS:
        if coluna in historico.columns:
            valores = historico[coluna].dropna()

            if not valores.empty:
                futuro[coluna] = valores.iloc[-1]

    # Eventos e decisões comerciais do dia previsto
    # não podem ser inferidos do último dia observado.
    for coluna in VARIAVEIS_FUTURAS_DESCONHECIDAS:
        if coluna in futuro:
            futuro[coluna] = np.nan

    futuro_df = pd.DataFrame([futuro])

    historico_com_futuro = pd.concat(
        [historico, futuro_df],
        ignore_index=True
    )

    # Reutiliza EXATAMENTE a mesma engenharia
    # de atributos empregada no treinamento.
    atributos = gerar_atributos_dataset(
        historico_com_futuro
    )

    linha_futura = atributos.loc[
        atributos["data"] == data_prevista
    ]

    if len(linha_futura) != 1:
        raise ValueError(
            "Não foi possível gerar uma única "
            "observação para a data futura."
        )

    linha = linha_futura.iloc[0]

    if pd.notna(linha["vendas"]):
        raise ValueError(
            "A previsão futura contém uma venda real. "
            "Operação interrompida para evitar vazamento."
        )

    if pd.Timestamp(linha["data_origem"]) != data_origem:
        raise ValueError(
            "A data de origem calculada está incorreta."
        )

    atributos_modelo = linha.drop(
        labels=[
            "loja_id",
            "data",
            "data_origem",
            "vendas"
        ],
        errors="ignore"
    ).to_dict()

    return {
        "execucao_id": pasta.name,
        "loja_id": loja_id,
        "data_origem": data_origem.date(),
        "data": data_prevista.date(),
        "atributos": atributos_modelo,
        "variaveis_futuras_desconhecidas": [
            coluna
            for coluna in VARIAVEIS_FUTURAS_DESCONHECIDAS
            if coluna in historico.columns
        ]
    }


def prever_loja_automaticamente(
    execucao_id: str,
    loja_id: int
) -> dict:
    """
    Calcula os atributos e realiza a previsão
    com o modelo vencedor da execução.
    """

    entrada = gerar_entrada_automatica(
        execucao_id,
        loja_id
    )

    artefato = carregar_modelo(
        entrada["execucao_id"]
    )

    previsao = prever_vendas(
        entrada["atributos"],
        execucao_id=entrada["execucao_id"]
    )[0]

    if not np.isfinite(previsao):
        raise ValueError(
            "O modelo retornou uma previsão inválida."
        )

    return {
        "execucao_id": entrada["execucao_id"],
        "loja_id": entrada["loja_id"],
        "data_origem": entrada["data_origem"],
        "data_prevista": entrada["data"],
        "horizonte_dias": HORIZONTE,
        "vendas_previstas": round(float(previsao), 2),
        "algoritmo": artefato["algoritmo"],
        "modo": "previsao_automatica",
        "variaveis_futuras_desconhecidas": (
            entrada["variaveis_futuras_desconhecidas"]
        ),
        "observacao": (
            "Previsão para sete dias após a última venda "
            "registrada. Variáveis comerciais futuras "
            "desconhecidas foram mantidas ausentes. "
            "A previsão não representa necessariamente "
            "as vendas de todos os próximos sete dias."
        )
    }
