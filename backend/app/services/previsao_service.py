
"""
Serviço de previsão de vendas.

Compatibilidade:
- Sem execucao_id: utiliza o modelo experimental antigo.
- Com execucao_id: utiliza o vencedor AutoML da execução.

Os atributos históricos devem conter somente informações
disponíveis até a data de origem da previsão.
"""

import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.app.services.preprocessamento_service import (
    transformar_atributos
)


RAIZ = Path(__file__).resolve().parents[3]

CAMINHO_MODELO_LEGADO = (
    RAIZ / "backend" / "modelos" / "modelo_global.joblib"
)

PASTA_EXECUCOES = RAIZ / "backend" / "execucoes"

PADRAO_EXECUCAO = re.compile(r"^[a-fA-F0-9]{32}$")


def validar_execucao_id(execucao_id: str) -> str:
    """Impede IDs inválidos ou caminhos arbitrários."""

    if not PADRAO_EXECUCAO.fullmatch(execucao_id):
        raise ValueError(
            "ID de execução inválido. "
            "Informe o identificador de 32 caracteres."
        )

    return execucao_id.lower()


@lru_cache(maxsize=4)
def carregar_modelo(execucao_id: str | None = None) -> dict:
    """
    Carrega o artefato experimental ou o de uma execução.

    Mantém poucos modelos em cache para limitar o uso
    de memória. Reinicie a API se substituir um modelo
    já carregado.
    """

    if execucao_id is None:
        caminho = CAMINHO_MODELO_LEGADO
    else:
        execucao_id = validar_execucao_id(execucao_id)

        caminho = (
            PASTA_EXECUCOES
            / execucao_id
            / "modelo_vencedor.joblib"
        )

    if not caminho.is_file():
        raise FileNotFoundError(
            f"Modelo não encontrado: {caminho}"
        )

    artefato = joblib.load(caminho)

    if not isinstance(artefato, dict):
        raise ValueError("Artefato de modelo inválido.")

    obrigatorias = {
        "algoritmo",
        "modelo",
        "preprocessamento",
        "horizonte_dias"
    }

    faltando = obrigatorias - set(artefato)

    if faltando:
        raise ValueError(
            f"Artefato incompleto: {sorted(faltando)}"
        )

    if artefato["algoritmo"] not in ("xgboost", "lgbm"):
        raise ValueError(
            "Algoritmo do artefato não suportado."
        )

    if int(artefato["horizonte_dias"]) != 7:
        raise ValueError(
            "Horizonte do modelo diferente de 7 dias."
        )

    return artefato


def preparar_atributos(
    dados: dict | pd.DataFrame,
    preprocessamento: dict,
    *,
    dinamico: bool = False
) -> pd.DataFrame:
    """
    Reproduz o preprocessamento correspondente ao modelo.

    O modelo dinâmico usa o serviço criado para o AutoML.
    O experimental preserva sua transformação original.
    """

    if isinstance(dados, dict):
        df = pd.DataFrame([dados])
    elif isinstance(dados, pd.DataFrame):
        df = dados.copy()
    else:
        raise ValueError(
            "Os atributos devem ser um objeto ou DataFrame."
        )

    if df.empty:
        raise ValueError("Nenhuma observação foi fornecida.")

    if dinamico:
        # Preserva exatamente a codificação utilizada
        # durante o treinamento da execução.
        return transformar_atributos(
            df,
            preprocessamento
        )

    # Compatibilidade com o modelo experimental antigo.
    colunas_numericas = preprocessamento[
        "colunas_numericas"
    ]
    colunas_categoricas = preprocessamento[
        "colunas_categoricas"
    ]
    categorias_treino = preprocessamento[
        "categorias_treino"
    ]
    colunas_modelo = preprocessamento[
        "colunas_modelo"
    ]

    necessarias = (
        colunas_numericas + colunas_categoricas
    )

    faltando = [
        coluna for coluna in necessarias
        if coluna not in df.columns
    ]

    if faltando:
        raise ValueError(
            f"Atributos ausentes: {faltando}"
        )

    if "distancia_concorrente" in colunas_numericas:
        df["distancia_concorrente"] = (
            pd.to_numeric(
                df["distancia_concorrente"],
                errors="coerce"
            ).fillna(
                preprocessamento["mediana_distancia"]
            )
        )

    for coluna in colunas_categoricas:
        categorias = categorias_treino[coluna]

        invalidas = (
            df[coluna].notna()
            & ~df[coluna].isin(categorias)
        )

        if invalidas.any():
            raise ValueError(
                f"Categoria inválida em {coluna}."
            )

        df[coluna] = pd.Categorical(
            df[coluna],
            categories=categorias
        )

    x = pd.get_dummies(
        df[necessarias],
        columns=colunas_categoricas,
        dummy_na=False,
        dtype="int8"
    )

    x = x.reindex(
        columns=colunas_modelo,
        fill_value=0
    )

    if x.shape[1] != len(colunas_modelo):
        raise ValueError(
            "Quantidade incorreta de atributos."
        )

    return x


def preparar_entrada_modelo(
    dados: dict | pd.DataFrame,
    execucao_id: str | None = None
) -> tuple[dict, pd.DataFrame]:
    """
    Carrega o modelo solicitado e prepara seus atributos.
    Usado tanto na previsão quanto na explicabilidade.
    """

    artefato = carregar_modelo(execucao_id)

    x = preparar_atributos(
        dados,
        artefato["preprocessamento"],
        dinamico=execucao_id is not None
    )

    if (
        execucao_id is not None
        and "colunas_modelo" in artefato
    ):
        esperadas = artefato["colunas_modelo"]

        if x.columns.tolist() != list(esperadas):
            raise ValueError(
                "As colunas não correspondem "
                "ao treinamento desta execução."
            )

    return artefato, x


def prever_vendas(
    dados: dict | pd.DataFrame,
    execucao_id: str | None = None
) -> list[float]:
    """Prevê vendas usando o modelo solicitado."""

    artefato, x = preparar_entrada_modelo(
        dados,
        execucao_id
    )

    previsoes = np.asarray(
        artefato["modelo"].predict(x),
        dtype=float
    )

    if not np.isfinite(previsoes).all():
        raise ValueError(
            "O modelo retornou previsões inválidas."
        )

    return previsoes.tolist()
