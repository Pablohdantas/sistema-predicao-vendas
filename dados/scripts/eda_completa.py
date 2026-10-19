"""
EDA completa sobre o TEMPLATE (a entrada real do sistema).

REGRA METODOLÓGICA: toda análise que usa `vendas` olha SOMENTE a janela de treino
(80% iniciais do calendário). O teste nunca é consultado, para não haver vazamento
na escolha de limites de outlier, na amostra do ARIMA ou nas conclusões da EDA.

Saídas:
  dados/figuras/*.png
  dados/processados/resumo_eda.json
  dados/processados/amostra_arima.csv

Uso (na raiz do projeto, com a .venv ativada):
    python dados/scripts/eda_completa.py
"""
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera arquivos sem precisar de interface gráfica
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller

warnings.filterwarnings("ignore", message="adfuller currently returns")

DADOS = Path(__file__).resolve().parents[1]
PROCESSADOS = DADOS / "processados"
FIGURAS = DADOS / "figuras"
FIGURAS.mkdir(exist_ok=True)

SEMENTE = 42
PROPORCAO_TREINO = 0.8
N_AMOSTRA_ARIMA = 60
MIN_DIAS_AMOSTRA = 400  # lojas elegíveis para o ARIMA: histórico mínimo no treino
DIAS = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]

resumo: dict = {}


def br(valor: float, casas: int = 0) -> str:
    """Formato brasileiro: 296.240 e 40,5."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def eixo_br(casas: int) -> FuncFormatter:
    return FuncFormatter(lambda v, _: br(v, casas))


def salvar(fig, nome: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURAS / nome, dpi=200)
    plt.close(fig)
    print(f"  figura salva: {nome}")


def data_de_corte(datas: pd.Series, proporcao: float) -> pd.Timestamp:
    """Corte pelo CALENDÁRIO (igual para todas as lojas), não pelas datas presentes."""
    inicio, fim = datas.min(), datas.max()
    total_dias = (fim - inicio).days + 1
    return inicio + pd.Timedelta(days=int(total_dias * proporcao))


# ---------------------------------------------------------------- carga e corte
df = pd.read_csv(PROCESSADOS / "template_rossmann.csv", parse_dates=["data"])
corte = data_de_corte(df["data"], PROPORCAO_TREINO)
treino = df[df["data"] < corte].copy()
print(f"Data de corte: {corte.date()} | treino: {len(treino):,} linhas, "
      f"{treino['loja_id'].nunique()} lojas ({treino['data'].min().date()} a {treino['data'].max().date()})")
resumo["data_de_corte_primeiro_dia_do_teste"] = str(corte.date())
resumo["treino"] = {"linhas": int(len(treino)), "lojas": int(treino["loja_id"].nunique()),
                    "periodo": [str(treino["data"].min().date()), str(treino["data"].max().date())]}

# média da loja no treino: serve para comparar efeitos sem confundir com o porte da loja
treino["media_loja"] = treino.groupby("loja_id")["vendas"].transform("mean")
treino["razao"] = treino["vendas"] / treino["media_loja"]
treino["dow"] = treino["data"].dt.dayofweek

# ---------------------------------------------------------------- 1. série agregada
print("\n1. Série agregada")
calendario = pd.date_range(treino["data"].min(), treino["data"].max())
diaria = treino.groupby("data")["vendas"].sum().reindex(calendario)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(diaria.index, diaria.values / 1e6, lw=0.6, alpha=0.5, label="Vendas diárias (todas as lojas)")
ax.plot(diaria.index, (diaria.fillna(0).rolling(7, min_periods=7).mean()) / 1e6, lw=1.8, label="Média móvel de 7 dias")
ax.set_ylabel("Vendas (milhões)")
ax.set_title("Vendas totais por dia (janela de treino)")
ax.legend()
salvar(fig, "01_serie_agregada.png")

# ---------------------------------------------------------------- 2. dia da semana
print("2. Dia da semana")
por_dow = treino.groupby("dow").agg(razao=("razao", "mean"), media=("vendas", "mean"),
                                    n=("vendas", "size")).reindex(range(7))
por_dow["n"] = por_dow["n"].fillna(0)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar([DIAS[i] for i in por_dow.index], por_dow["razao"].fillna(0))
ax.axhline(1, color="k", lw=0.8)
for i, (r, n) in enumerate(zip(por_dow["razao"], por_dow["n"])):
    ax.text(i, (0 if np.isnan(r) else r), f"n={br(n)}", ha="center", va="bottom", fontsize=7)
ax.yaxis.set_major_formatter(eixo_br(1))
ax.set_ylabel("Venda relativa à média da própria loja (1,0 = média)")
ax.set_title("Vendas por dia da semana (treino)")
salvar(fig, "02_dia_da_semana.png")
resumo["venda_relativa_por_dia_da_semana"] = {DIAS[i]: (None if np.isnan(v) else round(float(v), 3))
                                              for i, v in por_dow["razao"].items()}
resumo["venda_media_bruta_por_dia_da_semana"] = {DIAS[i]: (None if np.isnan(v) else round(float(v), 1))
                                                 for i, v in por_dow["media"].items()}
dom = treino[treino["dow"] == 6]
resumo["domingo"] = {"lojas_que_abriram": int(dom["loja_id"].nunique()), "linhas": int(len(dom)),
                     "obs": "poucas lojas abrem aos domingos; a média bruta de domingo reflete essas lojas, não o varejo em geral"}

# ---------------------------------------------------------------- 3. efeito das variáveis
print("3. Efeito das variáveis (descritivo, NÃO causal)")
efeitos = {}
for col in ["promocao", "promocao_continua", "mes_renovacao_promocao", "feriado_escolar", "feriado"]:
    com, sem = treino.loc[treino[col] == 1, "razao"], treino.loc[treino[col] == 0, "razao"]
    if len(com) > 0 and len(sem) > 0:
        efeitos[col] = {"variacao_percentual": round(float(com.mean() / sem.mean() - 1) * 100, 2),
                        "linhas_com_evento": int(len(com))}
resumo["efeito_descritivo_sobre_razao_a_media_da_loja"] = efeitos
fig, ax = plt.subplots(figsize=(8, 4))
nomes = list(efeitos)
valores = [efeitos[n]["variacao_percentual"] for n in nomes]
ax.barh(nomes, valores)
for i, n in enumerate(nomes):
    sinal = "+" if valores[i] > 0 else ""
    rotulo = f"{sinal}{br(valores[i], 1)}%  (n={br(efeitos[n]['linhas_com_evento'])})"
    ax.text(valores[i], i, f" {rotulo} " if valores[i] >= 0 else f" {rotulo} ",
            ha="left" if valores[i] >= 0 else "right", va="center", fontsize=8)
ax.axvline(0, color="k", lw=0.8)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # só valores inteiros nos ticks
ax.xaxis.set_major_formatter(eixo_br(0))
ax.margins(x=0.4)
ax.set_xlabel("Variação das vendas relativas à média da própria loja (%)")
ax.set_title("Efeito descritivo de cada variável (treino)")
salvar(fig, "03_efeito_variaveis.png")

# ---------------------------------------------------------------- 4. heterogeneidade
print("4. Heterogeneidade entre lojas")
lojas = treino.groupby("loja_id").agg(media=("vendas", "mean"), desvio=("vendas", "std"),
                                      dias=("vendas", "size"), tipo_loja=("tipo_loja", "first"))
lojas["cv"] = lojas["desvio"] / lojas["media"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(lojas["media"], bins=40)
axes[0].set_xlabel("Venda média diária da loja")
axes[0].set_ylabel("Nº de lojas")
axes[0].set_title("Distribuição do porte das lojas")
grupos = [(t, g["media"].values) for t, g in lojas.groupby(lojas["tipo_loja"].fillna("n/d"))]
axes[1].boxplot([g for _, g in grupos], tick_labels=[t for t, _ in grupos])
axes[1].set_xlabel("Tipo de loja")
axes[1].set_title("Venda média por tipo de loja")
salvar(fig, "04_heterogeneidade_lojas.png")
resumo["heterogeneidade"] = {
    "media_da_loja_min": round(float(lojas["media"].min()), 1),
    "media_da_loja_mediana": round(float(lojas["media"].median()), 1),
    "media_da_loja_max": round(float(lojas["media"].max()), 1),
    "coeficiente_de_variacao_mediano": round(float(lojas["cv"].median()), 3),
}

# ---------------------------------------------------------------- 5. ACF e ADF
print("5. Autocorrelação e estacionariedade")
agregada = diaria.fillna(0)
valores_acf = acf(agregada, nlags=35)
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(len(valores_acf)), valores_acf)
for lag in (7, 14, 21, 28, 35):
    ax.axvline(lag, color="r", lw=0.6, ls="--")
ax.set_xlabel("Defasagem (dias)")
ax.set_ylabel("Autocorrelação")
ax.yaxis.set_major_formatter(eixo_br(1))
ax.set_title("Autocorrelação das vendas totais diárias (treino); tracejado = múltiplos de 7 dias")
salvar(fig, "05_acf_agregada.png")
adf_agregada = adfuller(agregada, autolag="AIC")
resumo["acf_lag_7"] = round(float(valores_acf[7]), 3)
resumo["acf_por_lag"] = {str(l): round(float(valores_acf[l]), 3) for l in (7, 14, 21, 28, 35)}
resumo["adf_serie_agregada"] = {"estatistica": round(float(adf_agregada[0]), 3), "p_valor": round(float(adf_agregada[1]), 4)}

# ---------------------------------------------------------------- 6. lacunas de calendário
print("6. Lacunas de calendário por loja (excluindo domingos)")
cal = pd.date_range(treino["data"].min(), treino["data"].max())
nao_domingo = pd.Series((cal.dayofweek != 6).astype(int), index=cal).cumsum()
ini = treino.groupby("loja_id")["data"].min()
fim = treino.groupby("loja_id")["data"].max()
esperados = nao_domingo.loc[fim.values].values - nao_domingo.loc[ini.values].values + 1
presentes = treino[treino["data"].dt.dayofweek != 6].groupby("loja_id").size().reindex(ini.index).fillna(0).values
lacunas = pd.Series(esperados - presentes, index=ini.index)
resumo["lacunas_calendario"] = {
    "lojas_com_mais_de_30_dias_de_lacuna": int((lacunas > 30).sum()),
    "mediana_de_dias_de_lacuna": float(lacunas.median()),
    "maximo": int(lacunas.max()),
    "lojas_mais_afetadas": {int(k): int(v) for k, v in lacunas.sort_values(ascending=False).head(5).items()},
}

# ---------------------------------------------------------------- 7. outliers (IQR por loja)
print("7. Diagnóstico de outliers (IQR por loja, SOMENTE diagnóstico)")
q1 = treino.groupby("loja_id")["vendas"].transform(lambda s: s.quantile(0.25))
q3 = treino.groupby("loja_id")["vendas"].transform(lambda s: s.quantile(0.75))
iqr = q3 - q1
abaixo, acima = treino["vendas"] < q1 - 1.5 * iqr, treino["vendas"] > q3 + 1.5 * iqr
resumo["outliers_iqr_treino"] = {
    "share_abaixo": round(float(abaixo.mean()), 4),
    "share_acima": round(float(acima.mean()), 4),
    "share_promocao_nos_outliers_superiores": round(float(treino.loc[acima, "promocao"].mean()), 3) if acima.any() else None,
    "share_promocao_geral": round(float(treino["promocao"].mean()), 3),
    "lojas_com_mais_outliers": {int(k): round(float(v), 3) for k, v in
                                (abaixo | acima).groupby(treino["loja_id"]).mean().sort_values(ascending=False).head(5).items()},
}

# ---------------------------------------------------------------- 8. amostra para o ARIMA
print("8. Amostra estratificada de lojas para o ARIMA")
elegiveis = lojas[lojas["dias"] >= MIN_DIAS_AMOSTRA].copy()
elegiveis["tipo"] = elegiveis["tipo_loja"].fillna("n/d")
n_quantis = min(4, elegiveis["media"].nunique())
elegiveis["faixa_venda"] = pd.qcut(elegiveis["media"], n_quantis, labels=False, duplicates="drop")
n_alvo = min(N_AMOSTRA_ARIMA, len(elegiveis))
fracao = n_alvo / len(elegiveis)
partes = []
for _, g in elegiveis.groupby(["tipo", "faixa_venda"]):
    partes.append(g.sample(n=max(1, round(len(g) * fracao)), random_state=SEMENTE))
amostra = pd.concat(partes)[["tipo", "faixa_venda", "media", "dias"]].sort_index()
amostra.to_csv(PROCESSADOS / "amostra_arima.csv", index_label="loja_id")
resumo["amostra_arima"] = {"lojas_elegiveis": int(len(elegiveis)), "lojas_na_amostra": int(len(amostra)),
                           "semente": SEMENTE, "criterio": f"estratificada por tipo_loja x quartil de venda; >= {MIN_DIAS_AMOSTRA} dias no treino"}

# ADF por loja na amostra: testa o argumento do TG de que as séries não seriam estacionárias
p_valores = []
for loja in amostra.index:
    serie = treino.loc[treino["loja_id"] == loja].sort_values("data")["vendas"].values
    try:
        p_valores.append(adfuller(serie, autolag="AIC")[1])
    except Exception:
        pass
p_valores = np.array(p_valores)
resumo["adf_amostra_lojas"] = {"lojas_testadas": int(len(p_valores)),
                               "share_estacionarias_a_5pct": round(float((p_valores < 0.05).mean()), 3) if len(p_valores) else None}

# ---------------------------------------------------------------- salvar
(PROCESSADOS / "resumo_eda.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
print("\n" + "=" * 70 + "\nRESUMO\n" + "=" * 70)
print(json.dumps(resumo, indent=2, ensure_ascii=False))