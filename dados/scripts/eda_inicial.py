from pathlib import Path

import pandas as pd

DADOS = Path(__file__).resolve().parents[1]
RAW = DADOS / "brutos"


def titulo(texto: str) -> None:
    print(f"\n{'=' * 70}\n{texto}\n{'=' * 70}")


# StateHoliday vem com tipos misturados (0, '0', 'a', 'b', 'c'): ler como texto.
train = pd.read_csv(RAW / "train.csv", dtype={"StateHoliday": str}, parse_dates=["Date"])
store = pd.read_csv(RAW / "store.csv")

# ---------------------------------------------------------------- 1. visão geral
titulo("1. VISÃO GERAL")
print(f"train: {train.shape[0]:,} linhas x {train.shape[1]} colunas")
print(f"store: {store.shape[0]:,} linhas x {store.shape[1]} colunas")
print(f"Período: {train['Date'].min().date()} a {train['Date'].max().date()}")
print(f"Lojas em train: {train['Store'].nunique()} | lojas em store: {store['Store'].nunique()}")
sem_cadastro = set(train["Store"]) - set(store["Store"])
print(f"Lojas do train sem cadastro em store.csv: {len(sem_cadastro)}")
dups = train.duplicated(subset=["Store", "Date"]).sum()
print(f"Linhas duplicadas (Store, Date): {dups}")

# ---------------------------------------------------------------- 2. nulos
titulo("2. VALORES NULOS")
print("train:")
nulos_train = train.isna().sum()[lambda s: s > 0]
print(nulos_train.to_string() if len(nulos_train) else "  nenhum")
print("\nstore:")
nulos_store = store.isna().sum()[lambda s: s > 0]
print(nulos_store.to_string() if len(nulos_store) else "  nenhum")
# Promo2 = 0 deveria explicar os NaN de Promo2Since*/PromoInterval
sem_promo2 = store[store["Promo2"] == 0]
print(f"\nLojas sem Promo2: {len(sem_promo2)} | com NaN em Promo2SinceWeek entre elas: "
      f"{sem_promo2['Promo2SinceWeek'].isna().sum()}  (NaN estrutural, NÃO imputar com mediana)")

# ---------------------------------------------------------------- 3. categorias
titulo("3. VARIÁVEIS CATEGÓRICAS")
print("StateHoliday:\n", train["StateHoliday"].value_counts(dropna=False).to_string())
for col in ["StoreType", "Assortment", "PromoInterval"]:
    print(f"\n{col}:\n", store[col].value_counts(dropna=False).to_string())

# ---------------------------------------------------------------- 4. fechadas / zeros
titulo("4. LOJAS FECHADAS E VENDAS ZERO")
fechado = train["Open"] == 0
print(f"Open = 0: {fechado.sum():,} linhas ({fechado.mean():.1%})")
print(f"  dessas, com Sales > 0 (inconsistência): {(fechado & (train['Sales'] > 0)).sum()}")
aberto_zero = (train["Open"] == 1) & (train["Sales"] == 0)
print(f"Open = 1 e Sales = 0: {aberto_zero.sum()} linhas  -> decidir tratamento")

nao_domingo = train[fechado & (train["DayOfWeek"] != 7)]
por_loja = nao_domingo.groupby("Store").size().sort_values(ascending=False)
print(f"\nLojas com > 30 dias fechados fora de domingo (ex.: reformas): {(por_loja > 30).sum()}")
print(por_loja.head(10).to_string())

# ---------------------------------------------------------------- 5. distribuição
titulo("5. DISTRIBUIÇÃO DE VENDAS (apenas Open = 1)")
vendas = train.loc[train["Open"] == 1, "Sales"]
print(vendas.describe().round(1).to_string())
print(f"\nAssimetria: {vendas.skew():.2f} (alta => considerar RMSPE e olhar por loja)")

# ---------------------------------------------------------------- 6. corte 80/20
titulo("6. CORTE TEMPORAL 80/20 (por DATA, igual para todas as lojas)")
datas = sorted(train["Date"].unique())
corte = pd.Timestamp(datas[int(len(datas) * 0.8)])
print(f"Dias distintos: {len(datas)} | primeira data do TESTE: {corte.date()}")
tr, te = train[train["Date"] < corte], train[train["Date"] >= corte]
print(f"Treino: {len(tr):,} linhas ({tr['Date'].min().date()} a {tr['Date'].max().date()})")
print(f"Teste : {len(te):,} linhas ({te['Date'].min().date()} a {te['Date'].max().date()})")
print("\nStateHoliday no TREINO:\n", tr["StateHoliday"].value_counts().to_string())
print("\nStateHoliday no TESTE:\n", te["StateHoliday"].value_counts().to_string())
print(f"\nShare de dias com Promo -> treino: {tr['Promo'].mean():.1%} | teste: {te['Promo'].mean():.1%}")

# ---------------------------------------------------------------- 7. Customers
titulo("7. LEMBRETE")
print("Customers NÃO é conhecida no momento da previsão -> excluir do modelo (vazamento).")