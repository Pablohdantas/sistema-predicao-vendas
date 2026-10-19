"""Contrato do CSV histórico padronizado (uma linha por loja e dia operado).

A validação não modifica o DataFrame recebido. Datas devem seguir AAAA-MM-DD.
"""
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Coluna:
    nome: str
    tipo: str
    obrigatoria: bool
    descricao: str


COLUNAS = [
    Coluna("loja_id", "id", True, "Identificador da loja/unidade"),
    Coluna("data", "data", True, "Data no formato AAAA-MM-DD"),
    Coluna("vendas", "numero", True, "VARIÁVEL-ALVO: vendas do dia (>= 0)"),
    Coluna("promocao", "binario", False, "1 se havia promoção no dia"),
    Coluna("promocao_continua", "binario", False, "1 se a loja já havia aderido à Promo2 na data"),
    Coluna("mes_renovacao_promocao", "binario", False, "1 se a promoção contínua é renovada no mês"),
    Coluna("feriado", "binario", False, "1 se o dia era feriado oficial"),
    Coluna("feriado_escolar", "binario", False, "1 se o dia era feriado escolar"),
    Coluna("distancia_concorrente", "numero", False, "Distância ao concorrente mais próximo (m)"),
    Coluna("tipo_loja", "categoria", False, "Tipo/formato da loja"),
    Coluna("sortimento", "categoria", False, "Nível de sortimento"),
]

NOMES = [c.nome for c in COLUNAS]
OBRIGATORIAS = [c.nome for c in COLUNAS if c.obrigatoria]
BINARIAS = [c.nome for c in COLUNAS if c.tipo == "binario"]
MIN_LINHAS_TOTAL = 1000
MIN_DIAS_DISTINTOS = 180
MIN_DIAS_HISTORICO = 56
MAX_MEDIANA_INTERVALO_DIAS = 3


@dataclass
class ResultadoValidacao:
    erros: list = field(default_factory=list)
    avisos: list = field(default_factory=list)

    @property
    def valido(self) -> bool:
        return not self.erros


def validar_template(df: pd.DataFrame) -> ResultadoValidacao:
    """Valida tipos, chaves, domínio dos atributos e histórico disponível."""
    r = ResultadoValidacao()
    if df.empty:
        r.erros.append("O arquivo não contém nenhuma linha")
        return r

    faltando = [c for c in OBRIGATORIAS if c not in df.columns]
    if faltando:
        r.erros.append(f"Colunas obrigatórias ausentes: {faltando}")
        return r

    extras = [c for c in df.columns if c not in NOMES]
    if extras:
        r.avisos.append(f"Colunas fora do template serão ignoradas: {extras}")

    loja = pd.to_numeric(df["loja_id"], errors="coerce")
    if loja.isna().any() or (loja % 1 != 0).any() or (loja <= 0).any():
        r.erros.append("loja_id deve ser um identificador inteiro positivo (sem vazios)")

    data_texto = df["data"].astype("string")
    formato_ok = data_texto.str.fullmatch(r"\d{4}-\d{2}-\d{2}").fillna(False)
    datas = pd.to_datetime(data_texto, format="%Y-%m-%d", errors="coerce")
    datas_invalidas = (~formato_ok) | datas.isna()
    if datas_invalidas.any():
        r.erros.append(f"data inválida em {int(datas_invalidas.sum())} linhas (use AAAA-MM-DD)")

    vendas = pd.to_numeric(df["vendas"], errors="coerce")
    if vendas.isna().any():
        r.erros.append(f"vendas não numérica ou vazia em {int(vendas.isna().sum())} linhas")
    if (vendas < 0).any():
        r.erros.append(f"vendas negativa em {int((vendas < 0).sum())} linhas")

    for col in BINARIAS:
        if col in df.columns:
            valores = pd.to_numeric(df[col], errors="coerce")
            # Ausência de início de Promo2 pode ser desconhecida, não zero.
            permite_nulo = col in {"promocao_continua", "mes_renovacao_promocao"}
            invalidos = ~valores.isin([0, 1])
            if permite_nulo:
                invalidos &= df[col].notna()
            if invalidos.any():
                r.erros.append(f"{col} deve conter apenas 0 ou 1" + (" (vazio permitido)" if permite_nulo else " (sem vazios)"))

    if "distancia_concorrente" in df.columns:
        dist = pd.to_numeric(df["distancia_concorrente"], errors="coerce")
        if (dist < 0).any():
            r.erros.append("distancia_concorrente negativa")
        if ((dist.isna()) & df["distancia_concorrente"].notna()).any():
            r.erros.append("distancia_concorrente contém valores não numéricos")

    if len(df) < MIN_LINHAS_TOTAL:
        r.erros.append(f"Base muito pequena: {len(df)} linhas (mínimo {MIN_LINHAS_TOTAL})")

    if not datas_invalidas.any():
        duplicadas = df.assign(_d=datas).duplicated(subset=["loja_id", "_d"]).sum()
        if duplicadas:
            r.erros.append(f"{int(duplicadas)} linhas duplicadas para (loja_id, data)")
        n_dias = int(datas.nunique())
        if n_dias < MIN_DIAS_DISTINTOS:
            r.erros.append(f"Histórico curto: {n_dias} dias distintos (mínimo {MIN_DIAS_DISTINTOS})")
        ordenado = pd.DataFrame({"loja_id": df["loja_id"].values, "_d": datas.values}).sort_values(["loja_id", "_d"])
        intervalos = ordenado.groupby("loja_id")["_d"].diff().dt.days.dropna()
        if len(intervalos) and intervalos.median() > MAX_MEDIANA_INTERVALO_DIAS:
            r.erros.append(
                "Os dados devem ter frequência DIÁRIA (um registro por loja por dia de operação); "
                f"mediana do intervalo encontrada: {intervalos.median():.0f} dias"
            )
        dias = df.groupby("loja_id").size()
        curtas = int((dias < MIN_DIAS_HISTORICO).sum())
        if curtas:
            r.avisos.append(f"{curtas} lojas com menos de {MIN_DIAS_HISTORICO} dias de histórico")
    return r
