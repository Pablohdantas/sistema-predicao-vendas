
"""
Divisão temporal para datasets enviados à ferramenta AutoML.

Utiliza 80% do intervalo cronológico para treinamento
e reserva a parte final para avaliação externa.

Respeita o horizonte de previsão de sete dias.
"""

import pandas as pd


HORIZONTE = 7
PROPORCAO_TREINO = 0.80


def dividir_dataset_temporalmente(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:

    if df.empty:
        raise ValueError("Dataset vazio.")

    dados = df.copy()

    dados["data"] = pd.to_datetime(dados["data"])
    dados["data_origem"] = pd.to_datetime(
        dados["data_origem"]
    )

    datas = pd.DatetimeIndex(
        sorted(dados["data"].unique())
    )

    if len(datas) < 180:
        raise ValueError(
            "São necessários pelo menos 180 dias distintos."
        )

    indice_corte = int(
        len(datas) * PROPORCAO_TREINO
    ) - 1

    data_corte = datas[indice_corte]

    # Treinamento: somente vendas observadas
    # até a data de corte.
    treino = dados.loc[
        dados["data"] <= data_corte
    ].copy()

    # Teste: previsões cuja origem ocorre
    # na data de corte ou depois dela.
    teste = dados.loc[
        dados["data_origem"] >= data_corte
    ].copy()

    if treino.empty or teste.empty:
        raise ValueError(
            "A divisão produziu treino ou teste vazio."
        )

    if not (
        treino["data"].max() < teste["data"].min()
    ):
        raise ValueError(
            "Foi detectada sobreposição temporal."
        )

    if not (
        teste["data_origem"].min() >= data_corte
    ):
        raise ValueError(
            "O teste utiliza origem anterior ao corte."
        )

    relatorio = {
        "data_corte": str(data_corte.date()),
        "horizonte_dias": HORIZONTE,
        "proporcao_treino": PROPORCAO_TREINO,
        "linhas_treino": int(len(treino)),
        "linhas_teste": int(len(teste)),
        "lojas_treino": int(treino["loja_id"].nunique()),
        "lojas_teste": int(teste["loja_id"].nunique()),
        "primeira_data_teste": str(
            teste["data"].min().date()
        ),
        "primeira_origem_teste": str(
            teste["data_origem"].min().date()
        )
    }

    return treino, teste, relatorio
