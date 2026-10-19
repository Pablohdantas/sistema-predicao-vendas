
"""
Ajuste do modelo global de previsão de vendas Rossmann.

Objetivos:
- Carregar o algoritmo e os hiperparâmetros selecionados;
- Utilizar somente dados até 2015-01-23;
- Reproduzir os 23 atributos da seleção temporal;
- Treinar o modelo global;
- Salvar o modelo e o pré-processamento;
- Gerar relatório técnico para o TCC.

O teste externo não é utilizado nesta etapa.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
import lightgbm

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RAIZ = Path(__file__).resolve().parents[3]

PROCESSADOS = RAIZ / "dados" / "processados"
RESULTADOS = RAIZ / "backend" / "resultados"
MODELOS = RAIZ / "backend" / "modelos"

MODELOS.mkdir(parents=True, exist_ok=True)
RESULTADOS.mkdir(parents=True, exist_ok=True)

DATA_CORTE = pd.Timestamp("2015-01-23")
SEMENTE = 42

ARQUIVO_SELECAO = (
    RESULTADOS
    / "selecao_modelo_1115_lojas_15_tentativas.json"
)

ARQUIVO_MODELO = MODELOS / "modelo_global.joblib"

ARQUIVO_RELATORIO = (
    RESULTADOS / "relatorio_ajuste_global.json"
)

ARQUIVO_REFERENCIA = (
    PROCESSADOS / "treinamento" / "X_treino.csv"
)

NUMERICAS = [
    "promocao",
    "promocao_continua",
    "mes_renovacao_promocao",
    "feriado",
    "feriado_escolar",
    "distancia_concorrente",
    "dia_semana",
    "dia_mes",
    "mes",
    "semana_ano",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "media_movel_7",
    "media_movel_28",
]

CATEGORICAS = [
    "tipo_loja",
    "sortimento",
]


# ============================================================
# 1. CARREGAR A CONFIGURAÇÃO SELECIONADA
# ============================================================

def carregar_configuracao():
    if not ARQUIVO_SELECAO.exists():
        raise FileNotFoundError(
            f"Relatório de seleção não encontrado: "
            f"{ARQUIVO_SELECAO}"
        )

    with ARQUIVO_SELECAO.open(
        encoding="utf-8"
    ) as arquivo:
        selecao = json.load(arquivo)

    algoritmo = selecao["algoritmo_selecionado"]
    parametros = selecao["configuracao_selecionada"]

    if algoritmo not in ("xgboost", "lgbm"):
        raise ValueError(
            f"Algoritmo não suportado: {algoritmo}"
        )

    if selecao.get("quantidade_lojas") != 1115:
        raise ValueError(
            "A seleção não corresponde às 1.115 lojas."
        )

    if selecao.get("divisoes") != 4:
        raise ValueError(
            "A seleção não corresponde às quatro dobras."
        )

    return algoritmo, parametros, selecao


# ============================================================
# 2. CARREGAR O CONJUNTO DE TREINAMENTO
# ============================================================

def carregar_treinamento():
    print("\nCarregando atributos...")

    caminho = PROCESSADOS / "atributos_h7.csv"

    df = pd.read_csv(
        caminho,
        parse_dates=["data", "data_origem"]
    )

    treino = df.loc[
        df["data"] <= DATA_CORTE
    ].copy()

    if treino.empty:
        raise ValueError(
            "Nenhum registro de treinamento encontrado."
        )

    if treino["data"].max() > DATA_CORTE:
        raise ValueError(
            "Existem registros posteriores ao corte."
        )

    if treino["vendas"].isna().any():
        raise ValueError(
            "Existem vendas ausentes no treinamento."
        )

    if (treino["vendas"] < 0).any():
        raise ValueError(
            "Existem vendas negativas no treinamento."
        )

    print(f"Registros: {len(treino)}")
    print(f"Lojas: {treino['loja_id'].nunique()}")

    print(
        "Período: "
        f"{treino['data'].min().date()} até "
        f"{treino['data'].max().date()}"
    )

    return treino


# ============================================================
# 3. PREPARAR E VALIDAR OS 23 ATRIBUTOS
# ============================================================

def preparar_atributos(treino):
    print("\nPreparando atributos...")

    treino = treino.copy()

    # A mediana é calculada exclusivamente com
    # os registros do treinamento externo.
    mediana = treino[
        "distancia_concorrente"
    ].median()

    if pd.isna(mediana):
        raise ValueError(
            "Não foi possível calcular a mediana "
            "da distância ao concorrente."
        )

    treino["distancia_concorrente"] = (
        treino["distancia_concorrente"]
        .fillna(mediana)
    )

    # Categorias aprendidas exclusivamente
    # com os dados de treinamento.
    categorias_treino = {}

    for coluna in CATEGORICAS:
        categorias = sorted(
            treino[coluna]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        categorias_treino[coluna] = categorias

        treino[coluna] = pd.Categorical(
            treino[coluna],
            categories=categorias
        )

    # CORREÇÃO:
    # Não criar colunas artificiais de categoria ausente.
    # A seleção temporal utilizou 23 atributos.
    x_treino = pd.get_dummies(
        treino[NUMERICAS + CATEGORICAS],
        columns=CATEGORICAS,
        dummy_na=False,
        dtype="int8"
    )

    y_treino = treino["vendas"].astype(float)

    # Comparação com o esquema de atributos
    # previamente gerado no projeto.
    if not ARQUIVO_REFERENCIA.exists():
        raise FileNotFoundError(
            f"Arquivo de referência não encontrado: "
            f"{ARQUIVO_REFERENCIA}"
        )

    colunas_referencia = pd.read_csv(
        ARQUIVO_REFERENCIA,
        nrows=0
    ).columns.tolist()

    colunas_modelo = x_treino.columns.tolist()

    if colunas_modelo != colunas_referencia:
        somente_modelo = sorted(
            set(colunas_modelo)
            - set(colunas_referencia)
        )

        somente_referencia = sorted(
            set(colunas_referencia)
            - set(colunas_modelo)
        )

        raise ValueError(
            "Os atributos do modelo global não "
            "correspondem ao esquema de referência.\n"
            f"Quantidade atual: {len(colunas_modelo)}\n"
            f"Quantidade referência: "
            f"{len(colunas_referencia)}\n"
            f"Apenas no modelo: {somente_modelo}\n"
            f"Apenas na referência: {somente_referencia}\n"
            "Verifique também a ordem das colunas."
        )

    if len(colunas_modelo) != 23:
        raise ValueError(
            "Esperados 23 atributos, encontrados "
            f"{len(colunas_modelo)}."
        )

    if not np.isfinite(y_treino.to_numpy()).all():
        raise ValueError(
            "Existem valores inválidos na variável-alvo."
        )

    if x_treino.columns.duplicated().any():
        raise ValueError(
            "Existem nomes de atributos duplicados."
        )

    preprocessamento = {
        "colunas_numericas": NUMERICAS.copy(),
        "colunas_categoricas": CATEGORICAS.copy(),
        "categorias_treino": categorias_treino,
        "colunas_modelo": colunas_modelo,
        "mediana_distancia": float(mediana),
    }

    print(f"Atributos: {len(colunas_modelo)}")
    print(f"Mediana da distância: {mediana:.2f}")

    print(
        "Valores ausentes nos atributos: "
        f"{int(x_treino.isna().sum().sum())}"
    )

    print("Esquema de 23 atributos validado.")

    return x_treino, y_treino, preprocessamento


# ============================================================
# 4. CRIAR MODELO COM HIPERPARÂMETROS SELECIONADOS
# ============================================================

def criar_modelo(algoritmo, config):
    parametros = {
        "n_estimators": int(
            config["n_estimators"]
        ),
        "learning_rate": float(
            config["learning_rate"]
        ),
        "max_depth": int(
            config["max_depth"]
        ),
        "subsample": float(
            config["subsample"]
        ),
        "colsample_bytree": float(
            config["colsample_bytree"]
        ),
        "n_jobs": 2,
        "random_state": SEMENTE,
    }

    if algoritmo == "xgboost":
        return XGBRegressor(
            **parametros,
            objective="reg:squarederror",
            tree_method="hist",
            verbosity=0
        )

    if algoritmo == "lgbm":
        return LGBMRegressor(
            **parametros,
            num_leaves=int(
                config["num_leaves"]
            ),
            verbosity=-1
        )

    raise ValueError(
        f"Algoritmo não reconhecido: {algoritmo}"
    )


# ============================================================
# 5. AJUSTAR E SALVAR O MODELO GLOBAL
# ============================================================

def main():
    algoritmo, parametros, selecao = (
        carregar_configuracao()
    )

    print("========================================")
    print("AJUSTE DO MODELO GLOBAL")
    print("========================================")

    print(f"Algoritmo selecionado: {algoritmo}")

    print("\nHiperparâmetros:")
    print(
        json.dumps(
            parametros,
            indent=2,
            ensure_ascii=False
        )
    )

    treino = carregar_treinamento()

    (
        x_treino,
        y_treino,
        preprocessamento
    ) = preparar_atributos(treino)

    modelo = criar_modelo(
        algoritmo,
        parametros
    )

    print("\nIniciando ajuste do modelo global...")

    inicio = time.perf_counter()

    modelo.fit(
        x_treino,
        y_treino
    )

    duracao = (
        time.perf_counter() - inicio
    )

    print(
        f"Ajuste concluído em {duracao:.2f} segundos."
    )

    artefato = {
        "modelo": modelo,
        "algoritmo": algoritmo,
        "hiperparametros": parametros,
        "preprocessamento": preprocessamento,
        "horizonte_dias": 7,
        "data_corte_treinamento": str(
            DATA_CORTE.date()
        ),
        "semente": SEMENTE,
    }

    print("\nSalvando modelo...")

    joblib.dump(
        artefato,
        ARQUIVO_MODELO,
        compress=3
    )

    # Confirmar se o artefato foi salvo corretamente.
    recuperado = joblib.load(
        ARQUIVO_MODELO
    )

    if recuperado["algoritmo"] != algoritmo:
        raise RuntimeError(
            "O algoritmo recuperado não corresponde "
            "ao algoritmo treinado."
        )

    if (
        recuperado["preprocessamento"]["colunas_modelo"]
        != preprocessamento["colunas_modelo"]
    ):
        raise RuntimeError(
            "Os atributos recuperados estão inconsistentes."
        )

    if (
        recuperado["modelo"].n_features_in_
        != len(preprocessamento["colunas_modelo"])
    ):
        raise RuntimeError(
            "O modelo recuperado possui número "
            "incorreto de atributos."
        )

    relatorio = {
        "tipo": "ajuste_modelo_global",
        "algoritmo": algoritmo,
        "hiperparametros": parametros,
        "horizonte_dias": 7,
        "semente": SEMENTE,
        "data_corte_treinamento": str(
            DATA_CORTE.date()
        ),
        "periodo_treinamento": [
            str(treino["data"].min().date()),
            str(treino["data"].max().date())
        ],
        "linhas_treinamento": int(
            len(treino)
        ),
        "quantidade_lojas": int(
            treino["loja_id"].nunique()
        ),
        "quantidade_atributos": int(
            len(preprocessamento["colunas_modelo"])
        ),
        "atributos": (
            preprocessamento["colunas_modelo"]
        ),
        "mediana_distancia": (
            preprocessamento["mediana_distancia"]
        ),
        "tempo_ajuste_segundos": round(
            duracao, 3
        ),
        "modelo_salvo_em": str(
            ARQUIVO_MODELO
        ),
        "selecao_rmse": selecao["rmse_selecao"],
        "selecao_mae": selecao["mae_selecao"],
        "versoes_bibliotecas": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "observacao": (
            "Modelo ajustado com registros até "
            "2015-01-23 e com o mesmo esquema "
            "de 23 atributos da seleção temporal. "
            "O teste externo não foi utilizado."
        ),
    }

    ARQUIVO_RELATORIO.write_text(
        json.dumps(
            relatorio,
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )

    print("\n========================================")
    print("AJUSTE GLOBAL CONCLUÍDO")
    print("========================================")

    print(f"Algoritmo: {algoritmo}")
    print(f"Registros: {len(x_treino)}")
    print(f"Lojas: {treino['loja_id'].nunique()}")
    print(f"Atributos: {x_treino.shape[1]}")
    print(f"Tempo de ajuste: {duracao:.2f}s")

    print("\nArquivos gerados:")
    print(f"- {ARQUIVO_MODELO}")
    print(f"- {ARQUIVO_RELATORIO}")

    print(
        "\nO conjunto externo de teste não foi utilizado."
    )


if __name__ == "__main__":
    main()
