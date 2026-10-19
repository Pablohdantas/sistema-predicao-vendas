"""Adaptador Rossmann -> template CSV.

Executa apenas transformações determinísticas. Imputação estatística,
engenharia de atributos e treinamento serão realizados depois da divisão
temporal, para prevenir vazamento de dados.
"""
from pathlib import Path

import pandas as pd

from template import NOMES

# A Rossmann utiliza 'Sept' em PromoInterval.
MESES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec",
}


def carregar_raw(pasta: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os arquivos originais, sem modificá-los."""
    train = pd.read_csv(
        pasta / "train.csv", dtype={"StateHoliday": str}, parse_dates=["Date"]
    )
    store = pd.read_csv(pasta / "store.csv")
    return train, store


def _promo2(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Deriva indicadores temporais de Promo2 e PromoInterval.

    Sem participação, os indicadores são zero. Se uma loja participa mas
    não informa o início, o resultado é mantido como ausente, evitando
    interpretar informação desconhecida como ausência de promoção.
    """
    inicio = pd.to_datetime([
        pd.Timestamp.fromisocalendar(int(ano), int(semana), 1)
        if pd.notna(ano) and pd.notna(semana) else pd.NaT
        for ano, semana in zip(df["Promo2SinceYear"], df["Promo2SinceWeek"])
    ])
    inicio = pd.Series(inicio, index=df.index)
    participa = (df["Promo2"] == 1) & (df["Date"] >= inicio)
    mes_abrev = df["Date"].dt.month.map(MESES)
    janela = pd.Series(False, index=df.index)
    for intervalo in df["PromoInterval"].dropna().unique():
        meses_do_grupo = [mes.strip() for mes in intervalo.split(",")]
        janela |= (df["PromoInterval"] == intervalo) & mes_abrev.isin(meses_do_grupo)

    promocao_continua = participa.astype("Int64")
    renovacao = (participa & janela).astype("Int64")
    inicio_desconhecido = (df["Promo2"] == 1) & inicio.isna()
    promocao_continua.loc[inicio_desconhecido] = pd.NA
    renovacao.loc[inicio_desconhecido] = pd.NA
    return promocao_continua, renovacao


def construir_template(train: pd.DataFrame, store: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Produz o template histórico de dias operados e relatório de qualidade."""
    rel: dict = {"linhas_brutas": int(len(train))}
    rel["periodo_bruto"] = [str(train["Date"].min().date()), str(train["Date"].max().date())]
    df = train.copy()

    # O calendário original permanece nos arquivos brutos. O template
    # histórico mantém somente observações consideradas válidas para análise.
    fechada = df["Open"] == 0
    rel["removidas_loja_fechada"] = int(fechada.sum())
    rel["dias_fechados_por_loja"] = {
        str(k): int(v)
        for k, v in df.loc[fechada].groupby("Store").size().items()
    }
    df = df.loc[~fechada].copy()

    aberta_sem_venda = df["Sales"] == 0
    rel["removidas_aberta_sem_venda"] = int(aberta_sem_venda.sum())
    df = df.loc[~aberta_sem_venda].copy()

    neg = df["Sales"] < 0
    rel["removidas_vendas_negativas"] = int(neg.sum())
    df = df.loc[~neg].copy()
    dup = df.duplicated(subset=["Store", "Date"], keep="first")
    rel["removidas_duplicadas"] = int(dup.sum())
    df = df.loc[~dup].copy()

    df = df.merge(store, on="Store", how="left", validate="many_to_one")
    rel["linhas_sem_cadastro_de_loja"] = int(df["StoreType"].isna().sum())

    validos = {"0", "a", "b", "c"}
    invalidos = set(df["StateHoliday"].dropna().unique()) - validos
    if invalidos:
        raise ValueError(f"StateHoliday com valores inesperados: {invalidos}")
    df["feriado"] = df["StateHoliday"].isin(["a", "b", "c"]).astype(int)

    promo_continua, mes_renovacao = _promo2(df)
    sem_inicio = (
        (store["Promo2"] == 1)
        & (store["Promo2SinceYear"].isna() | store["Promo2SinceWeek"].isna())
    )
    rel["lojas_promo2_sem_data_inicio"] = int(sem_inicio.sum())

    out = pd.DataFrame({
        "loja_id": df["Store"].astype(int),
        "data": df["Date"],
        "vendas": df["Sales"].astype(float),
        "promocao": df["Promo"].astype(int),
        "promocao_continua": promo_continua,
        "mes_renovacao_promocao": mes_renovacao,
        "feriado": df["feriado"],
        "feriado_escolar": df["SchoolHoliday"].astype(int),
        "distancia_concorrente": df["CompetitionDistance"],
        "tipo_loja": df["StoreType"],
        "sortimento": df["Assortment"],
    })[NOMES]  # Customers não é conhecido na data da previsão.

    out = out.sort_values(["loja_id", "data"]).reset_index(drop=True)
    rel["linhas_finais"] = int(len(out))
    rel["lojas_finais"] = int(out["loja_id"].nunique())
    rel["periodo_final"] = [str(out["data"].min().date()), str(out["data"].max().date())]
    rel["participacao_em_linhas"] = {
        coluna: round(float(out[coluna].mean()), 4)
        for coluna in ["promocao", "promocao_continua", "mes_renovacao_promocao", "feriado", "feriado_escolar"]
    }
    rel["nulos_por_coluna"] = {
        coluna: int(qtd) for coluna, qtd in out.isna().sum().items() if qtd > 0
    }
    rel["colunas_descartadas"] = [
        "Customers", "Open", "DayOfWeek (derivável da data)",
        "StateHoliday (convertida em feriado 0/1)",
        "Competition*Since*, Promo2Since*, PromoInterval (transformadas ou não utilizadas)",
    ]
    return out, rel
