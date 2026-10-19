
"""
Validação experimental do AutoML com FLAML.

Executa LightGBM e XGBoost em quatro divisões temporais,
utilizando uma amostra de 20 lojas.

Cada divisão possui seu próprio:
- conjunto de ajuste;
- conjunto de validação;
- tratamento de valores ausentes;
- codificação de categorias;
- processo de seleção de hiperparâmetros.

Os resultados são preliminares e não utilizam o teste externo.
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

DATA_CORTE = pd.Timestamp("2015-01-23")

N_LOJAS = 20
N_DIVISOES = 4
TEMPO_LIMITE = 30
MAX_ITERACOES = 20
SEMENTE = 42

ALGORITMOS = ["lgbm", "xgboost"]

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

CATEGORICAS = ["tipo_loja", "sortimento"]


# ============================================================
# CARREGAMENTO
# ============================================================

def carregar_dados():
    df = pd.read_csv(
        PROCESSADOS / "atributos_h7.csv",
        parse_dates=["data", "data_origem"]
    )

    base = df.loc[
        df["data"] <= DATA_CORTE
    ].reset_index(drop=True)

    metadata = pd.read_csv(
        TREINAMENTO / "metadata_treino.csv",
        parse_dates=["data", "data_origem"]
    )

    colunas = ["loja_id", "data", "data_origem"]

    if not base[colunas].equals(metadata[colunas]):
        raise ValueError(
            "Os metadados não correspondem aos atributos."
        )

    with np.load(
        TREINAMENTO / "indices_validacao_temporal.npz"
    ) as arquivo:
        divisoes = [
            (
                arquivo[f"treino_{i}"].copy(),
                arquivo[f"validacao_{i}"].copy()
            )
            for i in range(1, N_DIVISOES + 1)
        ]

    for idx_treino, idx_val in divisoes:
        if (
            idx_treino.max() >= len(base)
            or idx_val.max() >= len(base)
        ):
            raise ValueError("Índices temporais inválidos.")

    return base, divisoes


# ============================================================
# SELEÇÃO REPRODUZÍVEL DE LOJAS
# ============================================================

def selecionar_lojas(base, divisoes):
    comuns = None

    for idx_treino, idx_val in divisoes:
        lojas_treino = set(
            base.iloc[idx_treino]["loja_id"]
        )
        lojas_val = set(
            base.iloc[idx_val]["loja_id"]
        )

        elegiveis = lojas_treino & lojas_val

        if comuns is None:
            comuns = elegiveis
        else:
            comuns &= elegiveis

    lojas = sorted(comuns)[:N_LOJAS]

    if len(lojas) < N_LOJAS:
        raise ValueError(
            "Não existem 20 lojas presentes em todas as dobras."
        )

    return lojas


# ============================================================
# PREPARAÇÃO POR DOBRA
# ============================================================

def preparar_dobra(base, indices, lojas):
    idx_treino, idx_val = indices

    treino = base.iloc[idx_treino].copy()
    validacao = base.iloc[idx_val].copy()

    treino = treino[
        treino["loja_id"].isin(lojas)
    ].copy()

    validacao = validacao[
        validacao["loja_id"].isin(lojas)
    ].copy()

    # Remover origens anteriores ao primeiro registro
    # disponível de cada loja.
    inicio = treino.groupby("loja_id")["data"].min()

    treino = treino.join(
        inicio.rename("inicio_historico"),
        on="loja_id"
    )

    treino = treino[
        treino["data_origem"]
        >= treino["inicio_historico"]
    ].drop(columns="inicio_historico")

    if treino.empty or validacao.empty:
        raise ValueError("Dobra sem observações suficientes.")

    # Nenhum alvo usado no ajuste deve ocorrer
    # depois da primeira origem de validação.
    if (
        treino["data"].max()
        > validacao["data_origem"].min()
    ):
        raise ValueError("Vazamento temporal detectado.")

    # A mediana é aprendida apenas nos dados de ajuste.
    mediana = treino["distancia_concorrente"].median()

    if pd.isna(mediana):
        raise ValueError("Mediana de distância indisponível.")

    for conjunto in (treino, validacao):
        conjunto["distancia_concorrente"] = (
            conjunto["distancia_concorrente"]
            .fillna(mediana)
        )

    # Definir categorias apenas a partir do ajuste.
    for coluna in CATEGORICAS:
        categorias = sorted(
            treino[coluna].dropna().astype(str).unique()
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

    x_val = pd.get_dummies(
        validacao[NUMERICAS + CATEGORICAS],
        columns=CATEGORICAS,
        dummy_na=True,
        dtype="int8"
    )

    x_val = x_val.reindex(
        columns=x_treino.columns,
        fill_value=0
    )

    y_treino = treino["vendas"].astype(float)
    y_val = validacao["vendas"].astype(float)

    if not x_treino.columns.equals(x_val.columns):
        raise ValueError("Atributos inconsistentes.")

    if y_treino.isna().any() or y_val.isna().any():
        raise ValueError("Valores-alvo ausentes.")

    return (
        treino, validacao,
        x_treino, x_val,
        y_treino, y_val,
        float(mediana)
    )


# ============================================================
# EXECUÇÃO DO AUTOML
# ============================================================

def executar_dobra(numero, base, indices, lojas):
    (
        treino, validacao,
        x_treino, x_val,
        y_treino, y_val,
        mediana
    ) = preparar_dobra(base, indices, lojas)

    print(f"\n{'=' * 55}")
    print(f"DOBRA {numero}/{N_DIVISOES}")
    print(f"{'=' * 55}")
    print(f"Treino: {len(treino)} registros")
    print(f"Validação: {len(validacao)} registros")
    print(f"Atributos: {len(x_treino.columns)}")
    print(f"Mediana da distância: {mediana:.2f}")
    print(
        f"Fim do ajuste: "
        f"{treino['data'].max().date()}"
    )
    print(
        f"Primeira origem: "
        f"{validacao['data_origem'].min().date()}"
    )

    automl = AutoML()

    inicio = time.perf_counter()

    automl.fit(
        X_train=x_treino,
        y_train=y_treino,
        X_val=x_val,
        y_val=y_val,
        task="regression",
        metric="rmse",
        estimator_list=ALGORITMOS,
        eval_method="holdout",
        time_budget=TEMPO_LIMITE,
        max_iter=MAX_ITERACOES,
        sample=False,
        n_jobs=2,
        mem_thres=2 * 1024**3,
        seed=SEMENTE,
        retrain_full=False,
        verbose=2
    )

    tempo = time.perf_counter() - inicio

    previsoes = automl.predict(x_val)

    rmse = float(np.sqrt(
        mean_squared_error(y_val, previsoes)
    ))

    mae = float(
        mean_absolute_error(y_val, previsoes)
    )

    previsoes_df = pd.DataFrame({
        "loja_id": validacao["loja_id"].to_numpy(),
        "data": validacao["data"].to_numpy(),
        "data_origem": validacao["data_origem"].to_numpy(),
        "vendas_reais": y_val.to_numpy(),
        "vendas_previstas": previsoes
    })

    previsoes_df["erro"] = (
        previsoes_df["vendas_reais"]
        - previsoes_df["vendas_previstas"]
    )

    previsoes_df.to_csv(
        RESULTADOS / f"validacao_dobra_{numero}.csv",
        index=False,
        date_format="%Y-%m-%d"
    )

    resultado = {
        "dobra": numero,
        "modelo_vencedor": automl.best_estimator,
        "rmse": rmse,
        "mae": mae,
        "tempo_segundos": round(tempo, 3),
        "linhas_treino": len(treino),
        "linhas_validacao": len(validacao),
        "periodo_validacao": [
            str(validacao["data"].min().date()),
            str(validacao["data"].max().date())
        ],
        "mediana_distancia": mediana,
        "melhores_parametros": automl.best_config
    }

    print(f"\nModelo selecionado: {automl.best_estimator}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"Tempo: {tempo:.2f} segundos")

    return resultado


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    RESULTADOS.mkdir(parents=True, exist_ok=True)

    print("Carregando dados e divisões temporais...")

    base, divisoes = carregar_dados()
    lojas = selecionar_lojas(base, divisoes)

    print(f"Lojas comuns selecionadas: {lojas}")

    resultados = []

    for numero, indices in enumerate(divisoes, start=1):
        resultado = executar_dobra(
            numero, base, indices, lojas
        )
        resultados.append(resultado)

    resumo = {
        "tipo": "validacao_interna_experimental",
        "algoritmos": ALGORITMOS,
        "horizonte": 7,
        "quantidade_lojas": len(lojas),
        "lojas": [int(loja) for loja in lojas],
        "quantidade_divisoes": N_DIVISOES,
        "tempo_limite_por_dobra": TEMPO_LIMITE,
        "resultados": resultados,
        "rmse_medio": float(np.mean([
            r["rmse"] for r in resultados
        ])),
        "mae_medio": float(np.mean([
            r["mae"] for r in resultados
        ]))
    }

    caminho = (
        RESULTADOS / "relatorio_validacao_automl.json"
    )

    caminho.write_text(
        json.dumps(
            resumo,
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )

    print(f"\n{'=' * 55}")
    print("RESUMO DAS QUATRO DIVISÕES")
    print(f"{'=' * 55}")

    for resultado in resultados:
        print(
            f"Dobra {resultado['dobra']}: "
            f"{resultado['modelo_vencedor']} | "
            f"RMSE={resultado['rmse']:.2f} | "
            f"MAE={resultado['mae']:.2f}"
        )

    print(f"\nRMSE médio: {resumo['rmse_medio']:.2f}")
    print(f"MAE médio: {resumo['mae_medio']:.2f}")
    print(f"\nRelatório salvo em: {caminho}")


if __name__ == "__main__":
    main()
