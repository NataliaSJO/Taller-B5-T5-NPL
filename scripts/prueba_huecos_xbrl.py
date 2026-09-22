"""Huecos reales del corpus: preguntas cuya respuesta correcta es «no está».

El enunciado los señala expresamente —Amazon no reporta GrossProfit, Liabilities
ni ResearchAndDevelopmentExpense en us-gaap, y Meta y Alphabet tampoco
GrossProfit— y dice que lo relevante es que el agente lo diga en vez de
inventarse una cifra. El golden propio no cubre ese caso, así que se comprueba
aparte antes del holdout.

Uso:
    python scripts/prueba_huecos_xbrl.py
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from agente import interfaz as s  # noqa: E402

CASOS = [
    ("¿Cuál fue el beneficio bruto (gross profit) de Amazon en el ejercicio fiscal 2024?",
     "AMZN", 2024, "GrossProfit"),
    ("¿Cuánto gastó Amazon en investigación y desarrollo en el ejercicio fiscal 2025?",
     "AMZN", 2025, "ResearchAndDevelopmentExpense"),
    ("¿Cuál fue el beneficio bruto de Meta en el ejercicio fiscal 2025?",
     "META", 2025, "GrossProfit"),
]


def main():
    salida = s.nueva_salida("huecos_xbrl")
    salida.mkdir(parents=True, exist_ok=True)
    filas = []
    for pregunta, ticker, ejercicio, concepto in CASOS:
        assert s.hecho_xbrl(ticker, ejercicio, concepto) is None, "el hueco ya no lo es"
        print(f"[hueco] {ticker} FY{ejercicio} {concepto}", flush=True)
        fila = {"ticker": ticker, "ejercicio": ejercicio, "concepto": concepto,
                "pregunta": pregunta}
        try:
            r = s.ejecutar(pregunta)
            resp = s.respuesta_de(r)
            fila.update(
                fuente=resp.get("fuente"), cifra=resp.get("cifra"),
                respuesta=resp.get("respuesta"),
                # Lo correcto es abstenerse: sin cifra y declarando que no está.
                correcto=bool(resp.get("fuente") == "ninguna" and resp.get("cifra") is None),
                uso_list_available=any(t["name"] == "list_available" for t in s.llamadas_de(r)),
                llamadas=[t["name"] for t in s.llamadas_de(r)],
                coste_usd=r.get("coste_usd"), latencia_s=r.get("latencia_s"))
        except Exception as exc:
            fila["error"] = f"{type(exc).__name__}: {exc}"
            fila["correcto"] = False
        filas.append(fila)
        s.guardar_json(salida / "huecos.json", filas)
        print(f"   fuente={fila.get('fuente')} cifra={fila.get('cifra')} "
              f"correcto={fila.get('correcto')}", flush=True)
    aciertos = sum(bool(f.get("correcto")) for f in filas)
    print(json.dumps({"casos": len(filas), "se_abstiene_correctamente": aciertos,
                      "coste_usd": sum(f.get("coste_usd") or 0 for f in filas),
                      "salida": str(salida)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
