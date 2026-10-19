
"""
Treinamento experimental com FLAML e LightGBM.

Objetivos:
- Utilizar a quarta divisão temporal;
- Selecionar 20 lojas para o teste inicial;
- Treinar um modelo LightGBM com AutoML;
- Evitar vazamento temporal no pré-processamento;
- Calcular RMSE e MAE;
- Salvar métricas, configurações e previsões individuais.

IMPORTANTE:
Este é um teste de integração.
Os resultados ainda não representam a avaliação final do TCC.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RAIZ = Path(__file__).resolve().parents[3]

PROCESSADOS = RAIZ / "dados" / "processados"
TREINAMENTO = PROCESSADOS / "treinamento"
RESULTADOS = RAIZ / "backend" / "resultados"

RESULTADOS.mkdir(parents=True, exist_ok=True)

DATA_CORTE = pd.Timestamp("2015-01-23")

N_LOJAS = 20
DOBRA = 4
TEMPO_LIMITE = 60
MAX_ITERACOES = 15
SEMENTE = 42

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
# 1. CARREGAMENTO DOS DADOS
# ============================================================

def carregar_dados():
    print("Carregando dados...")

    df = pd.read_csv(
        PROCESSADOS / "atributos_h7.csv",
        parse_dates=["data", "data_origem"]
    )

    metadados = pd.read_csv(
        TREINAMENTO / "metadata_treino.csv",
        parse_dates=["data", "data_origem"]
    )

    base_treino = df.loc[
        df["data"] <= DATA_CORTE
    ].reset_index(drop=True)

    colunas = [
        "loja_id",
        "data",
        "data_origem",
    ]

    assert base_treino[colunas].equals(
        metadados[colunas]
    ), "A ordem dos metadados não corresponde aos atributos."

    with np.load(
        TREINAMENTO / "indices_validacao_temporal.npz"
    ) as indices:

        idx_treino = indices[f"treino_{DOBRA}"].copy()
        idx_validacao = indices[f"validacao_{DOBRA}"].copy()

    treino = base_treino.iloc[idx_treino].copy()
    validacao = base_treino.iloc[idx_validacao].copy()

    return treino, validacao


# ============================================================
# 2. SELEÇÃO DAS LOJAS
# ============================================================

def selecionar_lojas(treino, validacao):
    lojas_comuns = sorted(
        set(treino["loja_id"])
        & set(validacao["loja_id"])
    )

    if len(lojas_comuns) < N_LOJAS:
        raise ValueError(
            "Não há lojas suficientes para o teste."
        )

    lojas = lojas_comuns[:N_LOJAS]

    treino = treino.loc[
        treino["loja_id"].isin(lojas)
    ].copy()

    validacao = validacao.loc[
        validacao["loja_id"].isin(lojas)
    ].copy()

    # Remove origens anteriores ao primeiro registro
    # disponível no próprio conjunto de treinamento.
    inicio_por_loja = (
        treino.groupby("loja_id")["data"].min()
    )

    treino = treino.join(
        inicio_por_loja.rename("inicio_historico"),
        on="loja_id"
    )

    treino = treino.loc[
        treino["data_origem"]
        >= treino["inicio_historico"]
    ].drop(columns="inicio_historico")

    if treino.empty or validacao.empty:
        raise ValueError(
            "O treinamento ou a validação ficou vazio."
        )

    # Os alvos utilizados para ajustar o modelo devem
    # estar disponíveis na primeira origem da validação.
    if (
        treino["data"].max()
        > validacao["data_origem"].min()
    ):
        raise ValueError(
            "Possível vazamento temporal identificado."
        )

    return treino, validacao, lojas


# ============================================================
# 3. PREPARAÇÃO DAS VARIÁVEIS
# ============================================================

def preparar_variaveis(treino, validacao):
    treino = treino.copy()
    validacao = validacao.copy()

    # Calcula a mediana usando somente o treinamento.
    mediana_distancia = (
        treino["distancia_concorrente"].median()
    )

    if pd.isna(mediana_distancia):
        raise ValueError(
            "Não foi possível calcular a mediana da distância."
        )

    for conjunto in [treino, validacao]:
        conjunto["distancia_concorrente"] = (
            conjunto["distancia_concorrente"]
            .fillna(mediana_distancia)
        )

    # As categorias são definidas exclusivamente
    # com os valores presentes no treinamento.
    for coluna in CATEGORICAS:
        categorias = sorted(
            treino[coluna]
            .dropna()
            .astype(str)
            .unique()
        )

        treino[coluna] = pd.Categorical(
            treino[coluna],
            categories=categorias
        )

        validacao[coluna] = pd.Categorical(
            validacao[coluna],
            categories=categorias
        )

    x_treino = pd.get_dummies(
        treino[NUMERICAS + CATEGORICAS],
        columns=CATEGORICAS,
        dummy_na=True,
        dtype="int8"
    )

    x_validacao = pd.get_dummies(
        validacao[NUMERICAS + CATEGORICAS],
        columns=CATEGORICAS,
        dummy_na=True,
        dtype="int8"
    )

    x_validacao = x_validacao.reindex(
        columns=x_treino.columns,
        fill_value=0
    )

    y_treino = treino["vendas"].astype(float)
    y_validacao = validacao["vendas"].astype(float)

    if not x_treino.columns.equals(x_validacao.columns):
        raise ValueError(
            "As colunas de treino e validação são diferentes."
        )

    if y_treino.isna().any() or y_validacao.isna().any():
        raise ValueError(
            "Existem valores ausentes na variável-alvo."
        )

    return (
        x_treino,
        x_validacao,
        y_treino,
        y_validacao,
        float(mediana_distancia),
    )


# ============================================================
# 4. TREINAMENTO COM FLAML
# ============================================================

def treinar_modelo(
    x_treino,
    y_treino,
    x_validacao,
    y_validacao,
):
    print("\nIniciando treinamento FLAML...")

    automl = AutoML()

    inicio = time.perf_counter()

    automl.fit(
        X_train=x_treino,
        y_train=y_treino,
        X_val=x_validacao,
        y_val=y_validacao,
        task="regression",
        metric="rmse",
        estimator_list=["lgbm"],
        eval_method="holdout",
        time_budget=TEMPO_LIMITE,
        max_iter=MAX_ITERACOES,
        sample=False,
        n_jobs=2,
        mem_thres=2 * 1024**3,
        seed=SEMENTE,
        retrain_full=False,
        verbose=3,
    )

    tempo = time.perf_counter() - inicio

    previsoes = automl.predict(x_validacao)

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_validacao,
                previsoes
            )
        )
    )

    mae = float(
        mean_absolute_error(
            y_validacao,
            previsoes
        )
    )

    return automl, previsoes, rmse, mae, tempo


# ============================================================
# 5. SALVAMENTO DAS PREVISÕES INDIVIDUAIS
# ============================================================

def salvar_previsoes(
    validacao,
    y_validacao,
    previsoes,
):
    resultados = pd.DataFrame({
        "loja_id": validacao["loja_id"].to_numpy(),
        "data": validacao["data"].to_numpy(),
        "data_origem": validacao["data_origem"].to_numpy(),
        "vendas_reais": y_validacao.to_numpy(),
        "vendas_previstas": previsoes,
    })

    resultados["erro"] = (
        resultados["vendas_reais"]
        - resultados["vendas_previstas"]
    )

    resultados["erro_absoluto"] = (
        resultados["erro"].abs()
    )

    resultados["erro_quadratico"] = (
        resultados["erro"] ** 2
    )

    caminho = (
        RESULTADOS / "previsoes_automl_inicial.csv"
    )

    resultados.to_csv(
        caminho,
        index=False,
        date_format="%Y-%m-%d"
    )

    return caminho


# ============================================================
# 6. RELATÓRIO DO TREINAMENTO
# ============================================================

def salvar_relatorio(
    automl,
    treino,
    validacao,
    x_treino,
    x_validacao,
    lojas,
    mediana,
    rmse,
    mae,
    tempo,
):
    relatorio = {
        "tipo": "teste_tecnico",
        "algoritmo": automl.best_estimator,
        "dobra": DOBRA,
        "horizonte": 7,
        "semente": SEMENTE,
        "tempo_limite_segundos": TEMPO_LIMITE,
        "max_iteracoes": MAX_ITERACOES,
        "lojas": [int(loja) for loja in lojas],
        "linhas_treino": len(x_treino),
        "linhas_validacao": len(x_validacao),
        "quantidade_atributos": len(x_treino.columns),
        "atributos": x_treino.columns.tolist(),
        "periodo_treino": [
            str(treino["data"].min().date()),
            str(treino["data"].max().date()),
        ],
        "periodo_validacao": [
            str(validacao["data"].min().date()),
            str(validacao["data"].max().date()),
        ],
        "rmse": rmse,
        "mae": mae,
        "tempo_segundos": round(tempo, 4),
        "mediana_distancia": mediana,
        "melhores_parametros": automl.best_config,
        "arquivos_gerados": [
            "teste_automl_inicial.json",
            "previsoes_automl_inicial.csv",
        ],
    }

    caminho = RESULTADOS / "teste_automl_inicial.json"

    caminho.write_text(
        json.dumps(
            relatorio,
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )

    return caminho


# ============================================================
# 7. EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    treino, validacao = carregar_dados()

    treino, validacao, lojas = selecionar_lojas(
        treino,
        validacao
    )

    (
        x_treino,
        x_validacao,
        y_treino,
        y_validacao,
        mediana,
    ) = preparar_variaveis(treino, validacao)

    print(f"Lojas selecionadas: {lojas}")
    print(f"Registros de treino: {len(x_treino)}")
    print(f"Registros de validação: {len(x_validacao)}")
    print(f"Atributos: {len(x_treino.columns)}")
    print(f"Mediana utilizada: {mediana:.2f}")

    (
        automl,
        previsoes,
        rmse,
        mae,
        tempo,
    ) = treinar_modelo(
        x_treino,
        y_treino,
        x_validacao,
        y_validacao
    )

    caminho_previsoes = salvar_previsoes(
        validacao,
        y_validacao,
        previsoes
    )

    caminho_relatorio = salvar_relatorio(
        automl,
        treino,
        validacao,
        x_treino,
        x_validacao,
        lojas,
        mediana,
        rmse,
        mae,
        tempo
    )

    print("\nTreinamento concluído!")
    print(f"Modelo: {automl.best_estimator}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"Tempo: {tempo:.2f} segundos")

    print("\nArquivos gerados:")
    print(f"- {caminho_relatorio}")
    print(f"- {caminho_previsoes}")


if __name__ == "__main__":
    main()
