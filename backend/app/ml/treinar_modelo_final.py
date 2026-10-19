
"""
Seleção de hiperparâmetros usando validação temporal.

Objetivos:
- Comparar LightGBM e XGBoost;
- Avaliar cada configuração nas quatro dobras;
- Otimizar o RMSE agregado de validação;
- Aplicar pré-processamento independente por dobra;
- Salvar a melhor configuração encontrada.

ETAPA ATUAL: teste de seleção com 20 lojas.

Não utiliza o conjunto externo de teste.
Não gera o modelo global definitivo nesta execução.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from flaml import tune
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RAIZ = Path(__file__).resolve().parents[3]

PROCESSADOS = RAIZ / "dados" / "processados"
TREINAMENTO = PROCESSADOS / "treinamento"
RESULTADOS = RAIZ / "backend" / "resultados"

RESULTADOS.mkdir(parents=True, exist_ok=True)

DATA_CORTE = pd.Timestamp("2015-01-23")

N_LOJAS = None
N_DIVISOES = 4
N_TENTATIVAS = 15
TEMPO_POR_ALGORITMO = 360
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

CATEGORICAS = ["tipo_loja", "sortimento"]


# ============================================================
# CARREGAMENTO
# ============================================================

def carregar_dados():
    print("Carregando atributos...")

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
            "Metadados e atributos estão inconsistentes."
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

    return base, divisoes


# ============================================================
# SELEÇÃO DAS LOJAS
# ============================================================


def selecionar_lojas(base, divisoes):
    """
    Seleciona lojas disponíveis no treinamento geral.

    Não exige que cada loja esteja presente nas quatro
    validações, pois algumas possuem lacunas históricas.
    """

    lojas_disponiveis = sorted(
        base["loja_id"].dropna().unique()
    )

    if N_LOJAS is None:
        lojas = lojas_disponiveis
    else:
        lojas = lojas_disponiveis[:N_LOJAS]

    if not lojas:
        raise ValueError("Nenhuma loja disponível.")

    print(
        f"Total de lojas selecionadas: {len(lojas)}"
    )

    for numero, (idx_treino, idx_val) in enumerate(
        divisoes, start=1
    ):
        lojas_treino = set(
            base.iloc[idx_treino]["loja_id"]
        )

        lojas_validacao = set(
            base.iloc[idx_val]["loja_id"]
        )

        qtd_treino = len(
            lojas_treino.intersection(lojas)
        )

        qtd_validacao = len(
            lojas_validacao.intersection(lojas)
        )

        print(
            f"Dobra {numero}: "
            f"{qtd_treino} lojas no treinamento | "
            f"{qtd_validacao} na validação"
        )

    return lojas



# ============================================================
# PREPARAÇÃO INDEPENDENTE DE CADA DOBRA
# ============================================================

def preparar_dobra(base, indices, lojas):
    idx_treino, idx_validacao = indices

    treino = base.iloc[idx_treino].copy()
    validacao = base.iloc[idx_validacao].copy()

    treino = treino.loc[
        treino["loja_id"].isin(lojas)
    ].copy()

    validacao = validacao.loc[
        validacao["loja_id"].isin(lojas)
    ].copy()

    if treino.empty or validacao.empty:
        raise ValueError("Treino ou validação vazio.")

    # Segurança temporal: nenhum alvo do ajuste
    # pode ocorrer após a primeira origem de validação.
    if (
        treino["data"].max()
        > validacao["data_origem"].min()
    ):
        raise ValueError(
            "Vazamento temporal identificado."
        )

    # Mediana calculada somente nos dados de ajuste.
    mediana = treino["distancia_concorrente"].median()

    if pd.isna(mediana):
        raise ValueError(
            "Não foi possível calcular a mediana."
        )

    for conjunto in (treino, validacao):
        conjunto["distancia_concorrente"] = (
            conjunto["distancia_concorrente"]
            .fillna(mediana)
        )

    # Categorias aprendidas apenas com o ajuste.
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

    y_treino = treino["vendas"].to_numpy(dtype=float)
    y_validacao = validacao["vendas"].to_numpy(
        dtype=float
    )

    if not np.isfinite(y_treino).all():
        raise ValueError("Alvos de treino inválidos.")

    if not np.isfinite(y_validacao).all():
        raise ValueError("Alvos de validação inválidos.")

    return {
        "x_treino": x_treino,
        "x_validacao": x_validacao,
        "y_treino": y_treino,
        "y_validacao": y_validacao,
        "linhas_treino": len(treino),
        "linhas_validacao": len(validacao),
        "mediana_distancia": float(mediana),
    }


# ============================================================
# CONSTRUÇÃO DOS MODELOS
# ============================================================

def criar_modelo(algoritmo, config):
    parametros = {
        "n_estimators": int(config["n_estimators"]),
        "learning_rate": float(config["learning_rate"]),
        "max_depth": int(config["max_depth"]),
        "subsample": float(config["subsample"]),
        "colsample_bytree": float(
            config["colsample_bytree"]
        ),
        "n_jobs": 2,
        "random_state": SEMENTE,
    }

    if algoritmo == "lgbm":
        return LGBMRegressor(
            **parametros,
            num_leaves=int(config["num_leaves"]),
            verbosity=-1
        )

    if algoritmo == "xgboost":
        return XGBRegressor(
            **parametros,
            objective="reg:squarederror",
            tree_method="hist",
            verbosity=0
        )

    raise ValueError(f"Algoritmo desconhecido: {algoritmo}")


# ============================================================
# AVALIAÇÃO NAS QUATRO DOBRAS
# ============================================================

def avaliar_configuracao(algoritmo, config, dobras):
    erros_quadraticos = 0.0
    erros_absolutos = 0.0
    total_observacoes = 0

    resultados_dobras = []

    for numero, dados in enumerate(dobras, start=1):
        modelo = criar_modelo(algoritmo, config)

        modelo.fit(
            dados["x_treino"],
            dados["y_treino"]
        )

        previsoes = modelo.predict(
            dados["x_validacao"]
        )

        erros = dados["y_validacao"] - previsoes

        soma_quadrados = float(
            np.sum(np.square(erros))
        )

        soma_absolutos = float(
            np.sum(np.abs(erros))
        )

        quantidade = len(erros)

        rmse_dobra = float(
            np.sqrt(soma_quadrados / quantidade)
        )

        mae_dobra = float(
            mean_absolute_error(
                dados["y_validacao"],
                previsoes
            )
        )

        resultados_dobras.append({
            "dobra": numero,
            "rmse": rmse_dobra,
            "mae": mae_dobra,
            "observacoes": quantidade
        })

        erros_quadraticos += soma_quadrados
        erros_absolutos += soma_absolutos
        total_observacoes += quantidade

    # RMSE agregado sobre todas as previsões
    # das quatro dobras, não média simples de RMSEs.
    rmse_agregado = float(
        np.sqrt(
            erros_quadraticos / total_observacoes
        )
    )

    mae_agregado = float(
        erros_absolutos / total_observacoes
    )

    return {
        "rmse": rmse_agregado,
        "mae": mae_agregado,
        "resultados_dobras": resultados_dobras,
        "observacoes": total_observacoes
    }


# ============================================================
# OTIMIZAÇÃO COM FLAML TUNE
# ============================================================

def otimizar_algoritmo(algoritmo, dobras):
    print(f"\n{'=' * 55}")
    print(f"OTIMIZAÇÃO: {algoritmo.upper()}")
    print(f"{'=' * 55}")

    # Espaço de busca moderado para o teste inicial.
    espaco = {
        "n_estimators": tune.randint(50, 251),
        "learning_rate": tune.loguniform(0.01, 0.2),
        "max_depth": tune.randint(3, 9),
        "subsample": tune.uniform(0.7, 1.0),
        "colsample_bytree": tune.uniform(0.7, 1.0),
    }

    inicial = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }

    if algoritmo == "lgbm":
        espaco["num_leaves"] = tune.randint(8, 33)
        inicial["num_leaves"] = 15

    tentativas = []
    contador = 0

    def funcao_objetivo(config):
        nonlocal contador
        contador += 1

        inicio = time.perf_counter()

        resultado = avaliar_configuracao(
            algoritmo,
            config,
            dobras
        )

        duracao = time.perf_counter() - inicio

        registro = {
            "tentativa": contador,
            "algoritmo": algoritmo,
            "configuracao": dict(config),
            "rmse": resultado["rmse"],
            "mae": resultado["mae"],
            "tempo_segundos": round(duracao, 3),
            "dobras": resultado["resultados_dobras"]
        }

        tentativas.append(registro)

        print(
            f"Tentativa {contador}: "
            f"RMSE={resultado['rmse']:.2f} | "
            f"MAE={resultado['mae']:.2f} | "
            f"Tempo={duracao:.2f}s",
            flush=True
        )

        return {"rmse": resultado["rmse"]}

    analise = tune.run(
        funcao_objetivo,
        config=espaco,
        points_to_evaluate=[inicial],
        metric="rmse",
        mode="min",
        num_samples=N_TENTATIVAS,
        time_budget_s=TEMPO_POR_ALGORITMO,
        use_ray=False,
        verbose=0,
    )

    if analise.best_config is None:
        raise RuntimeError(
            f"Nenhuma configuração válida para {algoritmo}."
        )

    melhor_config = dict(analise.best_config)

    resultado_final = avaliar_configuracao(
        algoritmo,
        melhor_config,
        dobras
    )

    print(
        f"\nMelhor RMSE {algoritmo}: "
        f"{resultado_final['rmse']:.2f}"
    )

    return {
        "algoritmo": algoritmo,
        "melhor_configuracao": melhor_config,
        "rmse": resultado_final["rmse"],
        "mae": resultado_final["mae"],
        "dobras": resultado_final["resultados_dobras"],
        "tentativas": tentativas,
    }


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    inicio_total = time.perf_counter()

    base, divisoes = carregar_dados()

    lojas = selecionar_lojas(base, divisoes)

    print(f"Quantidade de lojas utilizadas: {len(lojas)}")
    print("Preparando as quatro dobras...")

    dobras = []

    for numero, indices in enumerate(divisoes, start=1):
        dados = preparar_dobra(base, indices, lojas)
        dobras.append(dados)

        print(
            f"Dobra {numero}: "
            f"{dados['linhas_treino']} treino | "
            f"{dados['linhas_validacao']} validação"
        )

    # Otimizar separadamente para garantir que
    # ambos os algoritmos sejam avaliados.
    resultados = [
        otimizar_algoritmo("lgbm", dobras),
        otimizar_algoritmo("xgboost", dobras),
    ]

    melhor = min(
        resultados,
        key=lambda r: r["rmse"]
    )

    tempo_total = time.perf_counter() - inicio_total

    relatorio = {
        "tipo": "selecao_preliminar_hiperparametros",
        "metodo": "flaml_tune_validacao_temporal",
        "horizonte": 7,
        "semente": SEMENTE,
        "quantidade_lojas": len(lojas),
        "lojas": [int(loja) for loja in lojas],
        "divisoes": N_DIVISOES,
        "tentativas_por_algoritmo": N_TENTATIVAS,
        "tempo_limite_por_algoritmo": TEMPO_POR_ALGORITMO,
        "criterio_selecao": "rmse_agregado",
        "resultados": resultados,
        "algoritmo_selecionado": melhor["algoritmo"],
        "configuracao_selecionada": (
            melhor["melhor_configuracao"]
        ),
        "rmse_selecao": melhor["rmse"],
        "mae_selecao": melhor["mae"],
        "tempo_total_segundos": round(tempo_total, 3),
        "observacao": (
            "Seleção preliminar com 20 lojas. "
            "As mesmas dobras são utilizadas para "
            "otimização; as métricas não representam "
            "avaliação externa independente."
        )
    }

    caminho = (
        RESULTADOS / "selecao_modelo_final_teste.json"
    )

    caminho.write_text(
        json.dumps(
            relatorio,
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )

    print(f"\n{'=' * 55}")
    print("RESULTADO DA SELEÇÃO")
    print(f"{'=' * 55}")

    for resultado in resultados:
        print(
            f"{resultado['algoritmo']}: "
            f"RMSE={resultado['rmse']:.2f} | "
            f"MAE={resultado['mae']:.2f}"
        )

    print(
        f"\nAlgoritmo selecionado: "
        f"{melhor['algoritmo']}"
    )

    print(
        f"RMSE de seleção: "
        f"{melhor['rmse']:.2f}"
    )

    print(
        f"Tempo total: "
        f"{tempo_total:.2f}s"
    )

    print(f"Relatório: {caminho}")

    print(
        "\nEtapa de seleção preliminar concluída. "
        "O modelo global ainda não foi treinado."
    )


if __name__ == "__main__":
    main()
