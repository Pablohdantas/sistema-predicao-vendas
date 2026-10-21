
"""
Validação temporal expansiva para modelos multihorizonte.

Cada dobra respeita duas regras:

1. Os alvos do treinamento estão no máximo
   na data de corte.
2. As origens das previsões de validação estão
   na data de corte ou depois dela.

O teste externo deve ser separado ANTES
de chamar este serviço.
"""

import numpy as np
import pandas as pd


HORIZONTES = set(range(1, 8))


def criar_dobras_multihorizonte(
    treino_externo: pd.DataFrame,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """
    Retorna pares de índices posicionais compatíveis
    com DataFrame.iloc para treino e validação.

    Usa janelas expansivas:
      - 90 a 179 dias: 2 dobras
      - 180 a 299 dias: 3 dobras
      - 300 dias ou mais: 4 dobras

    O número de dias é estimado pelo período
    original disponível na base de treinamento.
    """

    obrigatorias = {
        "data",
        "data_origem",
        "horizonte_dias",
    }

    faltantes = obrigatorias - set(
        treino_externo.columns
    )

    if faltantes:
        raise ValueError(
            f"Colunas ausentes: {sorted(faltantes)}"
        )

    if treino_externo.empty:
        raise ValueError(
            "Treinamento externo vazio."
        )

    dados = treino_externo.copy()

    dados["data"] = pd.to_datetime(
        dados["data"],
        errors="raise",
    )

    dados["data_origem"] = pd.to_datetime(
        dados["data_origem"],
        errors="raise",
    )

    if dados[
        ["data", "data_origem", "horizonte_dias"]
    ].isna().any().any():
        raise ValueError(
            "Datas ou horizontes ausentes."
        )

    diferencas = (
        dados["data"] - dados["data_origem"]
    ).dt.days

    if not (
        diferencas == dados["horizonte_dias"]
    ).all():
        raise ValueError(
            "Horizontes temporais inconsistentes."
        )

    dias_periodo = (
        dados["data"].max()
        - dados["data_origem"].min()
    ).days + 1

    if dias_periodo < 180:
        quantidade_dobras = 2
    elif dias_periodo < 300:
        quantidade_dobras = 3
    else:
        quantidade_dobras = 4

    datas = pd.DatetimeIndex(
        sorted(dados["data"].unique())
    )

    # Reserva a primeira metade das datas-alvo
    # disponíveis para o treinamento inicial.
    cortes_indices = np.linspace(
        int(len(datas) * 0.5),
        len(datas) - 1,
        quantidade_dobras + 1,
        dtype=int,
    )

    divisoes = []
    relatorios = []

    for numero in range(quantidade_dobras):
        indice_corte = int(
            cortes_indices[numero]
        )

        indice_fim = int(
            cortes_indices[numero + 1]
        )

        data_corte = datas[indice_corte]
        data_fim = datas[indice_fim]

        # Treino: somente vendas-alvo conhecidas
        # até a data de corte.
        mascara_treino = (
            dados["data"] <= data_corte
        )

        # Validação: todas as previsões precisam
        # partir da data de corte ou de data posterior.
        mascara_validacao = (
            (dados["data_origem"] >= data_corte)
            & (dados["data"] <= data_fim)
        )

        indices_treino = np.flatnonzero(
            mascara_treino.to_numpy()
        )

        indices_validacao = np.flatnonzero(
            mascara_validacao.to_numpy()
        )

        if (
            len(indices_treino) == 0
            or len(indices_validacao) == 0
        ):
            raise ValueError(
                f"Dobra {numero + 1} vazia."
            )

        treino = dados.iloc[indices_treino]
        validacao = dados.iloc[indices_validacao]

        if (
            treino["data"].max()
            > validacao["data_origem"].min()
        ):
            raise ValueError(
                "Vazamento temporal na dobra "
                f"{numero + 1}."
            )

        horizontes_treino = set(
            treino["horizonte_dias"].astype(int)
        )

        horizontes_validacao = set(
            validacao["horizonte_dias"].astype(int)
        )

        if (
            horizontes_treino != HORIZONTES
            or horizontes_validacao != HORIZONTES
        ):
            raise ValueError(
                "Nem todos os horizontes estão "
                f"presentes na dobra {numero + 1}."
            )

        divisoes.append(
            (indices_treino, indices_validacao)
        )

        relatorios.append({
            "dobra": numero + 1,
            "data_corte": str(
                data_corte.date()
            ),
            "data_fim_validacao": str(
                data_fim.date()
            ),
            "linhas_treino": len(indices_treino),
            "linhas_validacao": len(
                indices_validacao
            ),
            "horizontes": sorted(
                horizontes_validacao
            ),
        })

    return divisoes, {
        "tipo": "temporal_expansiva_multihorizonte",
        "quantidade_dobras": len(divisoes),
        "dias_periodo_treino": dias_periodo,
        "dobras": relatorios,
    }
