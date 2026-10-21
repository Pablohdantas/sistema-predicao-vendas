
"""
Divisão temporal para previsão de 1 a 7 dias.

Separa treino e teste por datas, garantindo que:

- Nenhum alvo de treinamento ultrapasse o corte.
- Nenhuma previsão de teste utilize origem anterior ao corte.
- O teste externo não participe da seleção do modelo.
- Todos os horizontes estejam representados.
"""

import pandas as pd


PROPORCAO_TREINO = 0.80
MIN_DIAS_DISTINTOS = 90
HORIZONTES = set(range(1, 8))


def dividir_dataset_multihorizonte(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Divide exemplos supervisionados por data-alvo,
    respeitando a disponibilidade das informações
    na origem de cada previsão.
    """

    if df.empty:
        raise ValueError("Dataset vazio.")

    obrigatorias = {
        "loja_id",
        "data",
        "data_origem",
        "horizonte_dias",
        "vendas",
    }

    faltantes = obrigatorias - set(df.columns)

    if faltantes:
        raise ValueError(
            f"Colunas obrigatórias ausentes: "
            f"{sorted(faltantes)}"
        )

    dados = df.copy()

    dados["data"] = pd.to_datetime(
        dados["data"],
        errors="raise",
    )

    dados["data_origem"] = pd.to_datetime(
        dados["data_origem"],
        errors="raise",
    )

    dados["horizonte_dias"] = pd.to_numeric(
        dados["horizonte_dias"],
        errors="raise",
    )

    if dados[
        ["data", "data_origem", "horizonte_dias", "vendas"]
    ].isna().any().any():
        raise ValueError(
            "Há datas, horizontes ou vendas ausentes."
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

    horizontes = set(
        dados["horizonte_dias"].astype(int).unique()
    )

    if horizontes != HORIZONTES:
        raise ValueError(
            "O dataset precisa conter todos os "
            "horizontes de 1 a 7 dias."
        )

    datas = pd.DatetimeIndex(
        sorted(dados["data"].unique())
    )

    if len(datas) < MIN_DIAS_DISTINTOS - 1:
        raise ValueError(
            "Histórico insuficiente para "
            "avaliação multihorizonte."
        )

    indice_corte = (
        int(len(datas) * PROPORCAO_TREINO) - 1
    )

    data_corte = datas[indice_corte]

    # O modelo só aprende vendas observadas
    # até a data de corte.
    treino = dados.loc[
        dados["data"] <= data_corte
    ].copy()

    # Todas as entradas de teste precisam ter
    # origem na data de corte ou depois dela.
    teste = dados.loc[
        dados["data_origem"] >= data_corte
    ].copy()

    if treino.empty or teste.empty:
        raise ValueError(
            "Treinamento ou teste externo vazio."
        )

    if treino["data"].max() > data_corte:
        raise ValueError(
            "Treino ultrapassou o corte temporal."
        )

    if teste["data_origem"].min() < data_corte:
        raise ValueError(
            "Teste utiliza informação posterior "
            "à origem da previsão."
        )

    if treino["data"].max() >= teste["data"].min():
        raise ValueError(
            "Há sobreposição dos alvos de "
            "treinamento e teste."
        )

    horizontes_treino = set(
        treino["horizonte_dias"].astype(int).unique()
    )

    horizontes_teste = set(
        teste["horizonte_dias"].astype(int).unique()
    )

    if (
        horizontes_treino != HORIZONTES
        or horizontes_teste != HORIZONTES
    ):
        raise ValueError(
            "Treino ou teste não possui "
            "todos os sete horizontes."
        )

    relatorio = {
        "tipo": "multihorizonte",
        "horizontes": sorted(HORIZONTES),
        "data_corte": str(data_corte.date()),
        "proporcao_treino": PROPORCAO_TREINO,
        "linhas_treino": int(len(treino)),
        "linhas_teste": int(len(teste)),
        "dias_distintos": int(len(datas)),
        "primeira_origem_teste": str(
            teste["data_origem"].min().date()
        ),
        "primeira_data_teste": str(
            teste["data"].min().date()
        ),
        "observacoes_teste_por_horizonte": {
            str(horizonte): int(
                (teste["horizonte_dias"] == horizonte).sum()
            )
            for horizonte in range(1, 8)
        },
    }

    return (
        treino.sort_values(
            ["data", "loja_id", "horizonte_dias"]
        ).reset_index(drop=True),
        teste.sort_values(
            ["data", "loja_id", "horizonte_dias"]
        ).reset_index(drop=True),
        relatorio,
    )
