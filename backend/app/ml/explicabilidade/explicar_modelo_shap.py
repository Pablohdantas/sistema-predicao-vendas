
"""
Explicabilidade global do modelo XGBoost com SHAP.

Objetivos:
- Carregar o modelo global já treinado;
- Selecionar 500 observações do teste externo;
- Reproduzir o pré-processamento salvo;
- Calcular os valores SHAP;
- Identificar a importância global dos 23 atributos;
- Gerar gráficos e relatórios.

Não treina nem modifica o modelo.
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RAIZ = Path(__file__).resolve().parents[4]

PROCESSADOS = RAIZ / "dados" / "processados"
MODELOS = RAIZ / "backend" / "modelos"
RESULTADOS = RAIZ / "backend" / "resultados"
SAIDA = RESULTADOS / "shap"

SAIDA.mkdir(parents=True, exist_ok=True)

ARQUIVO_MODELO = MODELOS / "modelo_global.joblib"
ARQUIVO_DADOS = PROCESSADOS / "atributos_h7.csv"

DATA_CORTE = pd.Timestamp("2015-01-23")

AMOSTRA = 500
SEMENTE = 42
HORIZONTE = 7


# ============================================================
# 1. CARREGAR MODELO
# ============================================================

def carregar_modelo():
    print("Carregando modelo global...")

    artefato = joblib.load(ARQUIVO_MODELO)

    if artefato["algoritmo"] != "xgboost":
        raise ValueError(
            "Este experimento espera um modelo XGBoost."
        )

    if artefato["horizonte_dias"] != HORIZONTE:
        raise ValueError(
            "Horizonte do modelo incompatível."
        )

    return artefato


# ============================================================
# 2. SELECIONAR AMOSTRA DO TESTE EXTERNO
# ============================================================

def carregar_amostra():
    print("Carregando observações externas...")

    df = pd.read_csv(
        ARQUIVO_DADOS,
        parse_dates=["data", "data_origem"]
    )

    teste = df.loc[
        (df["data"] > DATA_CORTE)
        & (df["data_origem"] >= DATA_CORTE)
    ].copy()

    diferencas = (
        teste["data"] - teste["data_origem"]
    ).dt.days

    if not (diferencas == HORIZONTE).all():
        raise ValueError(
            "Existem observações com horizonte incorreto."
        )

    if teste.empty:
        raise ValueError(
            "Nenhuma observação externa disponível."
        )

    quantidade = min(AMOSTRA, len(teste))

    # Amostragem reproduzível sem reposição.
    rng = np.random.default_rng(SEMENTE)

    posicoes = rng.choice(
        len(teste),
        size=quantidade,
        replace=False
    )

    amostra = (
        teste.iloc[np.sort(posicoes)]
        .reset_index(drop=True)
    )

    print(f"Observações selecionadas: {len(amostra)}")
    print(f"Lojas representadas: {amostra['loja_id'].nunique()}")

    return amostra


# ============================================================
# 3. REPRODUZIR PRÉ-PROCESSAMENTO
# ============================================================

def preparar_atributos(amostra, preprocessamento):
    dados = amostra.copy()

    numericas = preprocessamento[
        "colunas_numericas"
    ]

    categoricas = preprocessamento[
        "colunas_categoricas"
    ]

    categorias_treino = preprocessamento[
        "categorias_treino"
    ]

    colunas_modelo = preprocessamento[
        "colunas_modelo"
    ]

    dados["distancia_concorrente"] = (
        dados["distancia_concorrente"].fillna(
            preprocessamento["mediana_distancia"]
        )
    )

    for coluna in categoricas:
        dados[coluna] = pd.Categorical(
            dados[coluna],
            categories=categorias_treino[coluna]
        )

    x = pd.get_dummies(
        dados[numericas + categoricas],
        columns=categoricas,
        dummy_na=False,
        dtype="int8"
    )

    x = x.reindex(
        columns=colunas_modelo,
        fill_value=0
    )

    if x.columns.tolist() != colunas_modelo:
        raise ValueError(
            "A ordem dos atributos não corresponde ao modelo."
        )

    if x.shape[1] != 23:
        raise ValueError(
            f"Esperados 23 atributos, encontrados {x.shape[1]}."
        )

    return x


# ============================================================
# 4. CALCULAR SHAP
# ============================================================

def calcular_shap(modelo, x):
    print("\nCalculando explicações SHAP...")

    inicio = time.perf_counter()

    explicador = shap.TreeExplainer(modelo)
    explicacoes = explicador(x)

    valores = np.asarray(explicacoes.values)

    if valores.shape != x.shape:
        raise ValueError(
            "Dimensões dos valores SHAP inconsistentes."
        )

    if not np.isfinite(valores).all():
        raise ValueError(
            "Foram encontrados valores SHAP inválidos."
        )

    tempo = time.perf_counter() - inicio

    print(f"Cálculo concluído em {tempo:.2f} segundos.")

    return explicacoes, valores, tempo


# ============================================================
# 5. IMPORTÂNCIA GLOBAL
# ============================================================

def salvar_importancia_global(x, valores):
    importancia = pd.DataFrame({
        "atributo": x.columns,
        "media_abs_shap": np.mean(
            np.abs(valores),
            axis=0
        )
    })

    importancia = importancia.sort_values(
        "media_abs_shap",
        ascending=False
    ).reset_index(drop=True)

    destino = SAIDA / "importancia_global.csv"

    importancia.to_csv(
        destino,
        index=False
    )

    return importancia, destino


# ============================================================
# 6. GRÁFICOS
# ============================================================

def gerar_graficos(x, valores):
    print("Gerando gráficos...")

    grafico_barras = (
        SAIDA / "importancia_global_barras.png"
    )

    grafico_resumo = (
        SAIDA / "resumo_shap.png"
    )

    # Importância global média absoluta.
    plt.figure(figsize=(11, 8))

    shap.summary_plot(
        valores,
        x,
        plot_type="bar",
        max_display=20,
        show=False
    )

    plt.tight_layout()
    plt.savefig(
        grafico_barras,
        dpi=160,
        bbox_inches="tight"
    )
    plt.close("all")

    # Distribuição de contribuições SHAP.
    plt.figure(figsize=(11, 8))

    shap.summary_plot(
        valores,
        x,
        max_display=20,
        show=False
    )

    plt.tight_layout()
    plt.savefig(
        grafico_resumo,
        dpi=160,
        bbox_inches="tight"
    )
    plt.close("all")

    return grafico_barras, grafico_resumo


# ============================================================
# 7. EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    print("========================================")
    print("EXPLICABILIDADE GLOBAL COM SHAP")
    print("========================================")

    artefato = carregar_modelo()
    amostra = carregar_amostra()

    x = preparar_atributos(
        amostra,
        artefato["preprocessamento"]
    )

    print(f"Atributos utilizados: {x.shape[1]}")

    modelo = artefato["modelo"]

    # Confirmar que o modelo gera previsões na amostra.
    previsoes = np.asarray(
        modelo.predict(x),
        dtype=float
    )

    if not np.isfinite(previsoes).all():
        raise ValueError("Previsões inválidas.")

    explicacoes, valores, tempo = calcular_shap(
        modelo,
        x
    )

    # Teste de consistência:
    # valor base + soma SHAP ~= previsão do modelo.
    valores_base = np.asarray(
        explicacoes.base_values
    )

    reconstruidas = (
        valores_base + valores.sum(axis=1)
    )

    diferenca_maxima = float(
        np.max(
            np.abs(previsoes - reconstruidas)
        )
    )

    print(
        "Diferença máxima na reconstrução SHAP: "
        f"{diferenca_maxima:.6f}"
    )

    if not np.allclose(
        previsoes,
        reconstruidas,
        rtol=1e-4,
        atol=0.1
    ):
        raise ValueError(
            "As contribuições SHAP não reconstruíram "
            "adequadamente as previsões."
        )

    importancia, arquivo_importancia = (
        salvar_importancia_global(x, valores)
    )

    arquivo_barras, arquivo_resumo = (
        gerar_graficos(x, valores)
    )

    # Preservar valores SHAP para futura análise.
    np.savez_compressed(
        SAIDA / "valores_shap_amostra.npz",
        valores_shap=valores,
        valores_base=valores_base,
        atributos=np.asarray(x.columns, dtype=str)
    )

    # Identificar as observações explicadas.
    identificacao = amostra[
        ["loja_id", "data", "data_origem", "vendas"]
    ].copy()

    identificacao["vendas_previstas"] = previsoes

    identificacao.to_csv(
        SAIDA / "observacoes_explicadas.csv",
        index=False,
        date_format="%Y-%m-%d"
    )

    relatorio = {
        "tipo": "explicabilidade_global_shap",
        "algoritmo": artefato["algoritmo"],
        "horizonte_dias": HORIZONTE,
        "semente": SEMENTE,
        "tamanho_amostra": len(amostra),
        "lojas_representadas": int(
            amostra["loja_id"].nunique()
        ),
        "quantidade_atributos": x.shape[1],
        "periodo_amostra": [
            str(amostra["data"].min().date()),
            str(amostra["data"].max().date())
        ],
        "tempo_calculo_segundos": round(
            tempo, 4
        ),
        "diferenca_maxima_reconstrucao": (
            diferenca_maxima
        ),
        "principais_atributos": (
            importancia.head(10).to_dict(
                orient="records"
            )
        ),
        "observacao": (
            "Importância global calculada pela média "
            "dos valores SHAP absolutos em uma amostra "
            "reproduzível do teste externo. "
            "As contribuições não estabelecem causalidade."
        )
    }

    arquivo_relatorio = (
        SAIDA / "relatorio_shap_global.json"
    )

    arquivo_relatorio.write_text(
        json.dumps(
            relatorio,
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )

    print("\n========================================")
    print("RESULTADOS SHAP")
    print("========================================")

    print("\n10 atributos mais importantes:")

    print(
        importancia.head(10).to_string(
            index=False
        )
    )

    print("\nArquivos gerados:")
    print(arquivo_importancia)
    print(arquivo_barras)
    print(arquivo_resumo)
    print(arquivo_relatorio)
    print(SAIDA / "valores_shap_amostra.npz")
    print(SAIDA / "observacoes_explicadas.csv")


if __name__ == "__main__":
    main()
