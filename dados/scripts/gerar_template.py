"""Gera e valida o template histórico Rossmann."""
import json
import sys
from pathlib import Path

from rossmann import carregar_raw, construir_template
from template import validar_template


def main() -> int:
    dados = Path(__file__).resolve().parents[1]
    raw = dados / "brutos"
    processed = dados / "processados"
    processed.mkdir(parents=True, exist_ok=True)

    train, store = carregar_raw(raw)
    template, relatorio = construir_template(train, store)
    resultado = validar_template(template)
    relatorio["validacao"] = {
        "valido": resultado.valido,
        "erros": resultado.erros,
        "avisos": resultado.avisos,
    }

    (processed / "relatorio_qualidade.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(relatorio, indent=2, ensure_ascii=False))

    if not resultado.valido:
        print("\nTemplate inválido. CSV não foi atualizado.")
        return 1

    caminho_csv = processed / "template_rossmann.csv"
    template.to_csv(caminho_csv, index=False, date_format="%Y-%m-%d")
    print(f"\nTemplate salvo em: {caminho_csv}")
    print(template.head(8).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
