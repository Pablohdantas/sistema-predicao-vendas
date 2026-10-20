
"""
Contrato do CSV histórico padronizado.

Uma linha por loja e dia operado.
Datas no formato AAAA-MM-DD.

Os limites originais permanecem como padrão para
preservar a compatibilidade com os experimentos.

A API pode fornecer limites menores explicitamente.
"""

from dataclasses import dataclass, field

import numpy as np
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
    Coluna("vendas", "numero", True, "Vendas do dia (>= 0)"),
    Coluna("promocao", "binario", False, "Promoção no dia"),
    Coluna("promocao_continua", "binario", False, "Promo2 ativa"),
    Coluna(
        "mes_renovacao_promocao",
        "binario",
        False,
        "Renovação da promoção contínua no mês"
    ),
    Coluna("feriado", "binario", False, "Feriado oficial"),
    Coluna("feriado_escolar", "binario", False, "Feriado escolar"),
    Coluna(
        "distancia_concorrente",
        "numero",
        False,
        "Distância ao concorrente em metros"
    ),
    Coluna("tipo_loja", "categoria", False, "Tipo de loja"),
    Coluna("sortimento", "categoria", False, "Sortimento")
]

NOMES = [coluna.nome for coluna in COLUNAS]
OBRIGATORIAS = [
    coluna.nome for coluna in COLUNAS
    if coluna.obrigatoria
]
BINARIAS = [
    coluna.nome for coluna in COLUNAS
    if coluna.tipo == "binario"
]

# Padrões preservados dos experimentos originais.
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


def validar_template(
    df: pd.DataFrame,
    min_linhas: int = MIN_LINHAS_TOTAL,
    min_dias: int = MIN_DIAS_DISTINTOS
) -> ResultadoValidacao:
    """
    Valida o conteúdo sem modificar o DataFrame.

    Os limites podem ser personalizados pela API.
    """

    r = ResultadoValidacao()

    if df.empty:
        r.erros.append("O arquivo não contém nenhuma linha")
        return r

    faltando = [
        coluna for coluna in OBRIGATORIAS
        if coluna not in df.columns
    ]

    if faltando:
        r.erros.append(
            f"Colunas obrigatórias ausentes: {faltando}"
        )
        return r

    extras = [
        coluna for coluna in df.columns
        if coluna not in NOMES
    ]

    if extras:
        r.avisos.append(
            f"Colunas fora do template serão ignoradas: {extras}"
        )

    loja = pd.to_numeric(
        df["loja_id"],
        errors="coerce"
    )

    if (
        loja.isna().any()
        or (~np.isfinite(loja)).any()
        or (loja % 1 != 0).any()
        or (loja <= 0).any()
    ):
        r.erros.append(
            "loja_id deve ser inteiro positivo, sem vazios"
        )

    data_texto = df["data"].astype("string")

    formato_ok = data_texto.str.fullmatch(
        r"\d{4}-\d{2}-\d{2}"
    ).fillna(False)

    datas = pd.to_datetime(
        data_texto,
        format="%Y-%m-%d",
        errors="coerce"
    )

    datas_invalidas = ~formato_ok | datas.isna()

    if datas_invalidas.any():
        r.erros.append(
            f"data inválida em {int(datas_invalidas.sum())} "
            "linhas (use AAAA-MM-DD)"
        )

    vendas = pd.to_numeric(
        df["vendas"],
        errors="coerce"
    )

    vendas_invalidas = (
        vendas.isna()
        | ~np.isfinite(vendas)
    )

    if vendas_invalidas.any():
        r.erros.append(
            "vendas não numérica, vazia ou não finita "
            f"em {int(vendas_invalidas.sum())} linhas"
        )

    if (vendas < 0).any():
        r.erros.append(
            f"vendas negativa em {int((vendas < 0).sum())} linhas"
        )

    for coluna in BINARIAS:
        if coluna not in df.columns:
            continue

        valores = pd.to_numeric(
            df[coluna],
            errors="coerce"
        )

        permite_nulo = coluna in {
            "promocao_continua",
            "mes_renovacao_promocao"
        }

        invalidos = ~valores.isin([0, 1])

        if permite_nulo:
            invalidos &= df[coluna].notna()

        if invalidos.any():
            r.erros.append(
                f"{coluna} deve conter apenas 0 ou 1"
                + (
                    " (vazio permitido)"
                    if permite_nulo
                    else " (sem vazios)"
                )
            )

    if "distancia_concorrente" in df.columns:
        distancia = pd.to_numeric(
            df["distancia_concorrente"],
            errors="coerce"
        )

        if (distancia < 0).any():
            r.erros.append(
                "distancia_concorrente negativa"
            )

        invalidas = (
            (~np.isfinite(distancia))
            & df["distancia_concorrente"].notna()
        )

        if invalidas.any():
            r.erros.append(
                "distancia_concorrente contém "
                "valores não numéricos ou não finitos"
            )

    if len(df) < min_linhas:
        r.erros.append(
            f"Base muito pequena: {len(df)} linhas "
            f"(mínimo {min_linhas})"
        )

    if not datas_invalidas.any():
        duplicadas = (
            df.assign(_data_validada=datas)
            .duplicated(
                subset=["loja_id", "_data_validada"]
            )
            .sum()
        )

        if duplicadas:
            r.erros.append(
                f"{int(duplicadas)} linhas duplicadas "
                "para (loja_id, data)"
            )

        quantidade_dias = int(datas.nunique())

        if quantidade_dias < min_dias:
            r.erros.append(
                f"Histórico curto: {quantidade_dias} "
                f"dias distintos (mínimo {min_dias})"
            )

        ordenado = pd.DataFrame({
            "loja_id": df["loja_id"].values,
            "_data": datas.values
        }).sort_values(["loja_id", "_data"])

        intervalos = (
            ordenado.groupby("loja_id")["_data"]
            .diff()
            .dt.days
            .dropna()
        )

        if (
            len(intervalos)
            and intervalos.median()
            > MAX_MEDIANA_INTERVALO_DIAS
        ):
            r.erros.append(
                "Os dados devem ter frequência DIÁRIA. "
                "Mediana do intervalo encontrada: "
                f"{intervalos.median():.0f} dias"
            )

        dias_por_loja = df.groupby("loja_id").size()

        curtas = int(
            (dias_por_loja < MIN_DIAS_HISTORICO).sum()
        )

        if curtas:
            r.avisos.append(
                f"{curtas} lojas com menos de "
                f"{MIN_DIAS_HISTORICO} dias de histórico"
            )

        if quantidade_dias < 180:
            r.avisos.append(
                "Histórico inferior a 180 dias: "
                "a validação será reduzida e "
                "as métricas podem ser menos estáveis."
            )

    return r
