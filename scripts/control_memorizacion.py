"""Control de memorización: las mismas preguntas, sin herramientas ni corpus.

Seis tecnológicas famosas en un modelo grande: parte del acierto puede venir del
preentrenamiento y no del 10-K. El enunciado penaliza acertar por el camino
equivocado, así que conviene poner número a ese riesgo en lugar de suponerlo.

Mide dos cosas sobre las mismas preguntas del golden:

- cuántas cifras acierta el modelo *de memoria*, contra el parquet XBRL;
- cuántas citas se inventa, comprobando si la frase existe en el corpus.

Uso:
    python scripts/control_memorizacion.py [--golden ruta.jsonl] [--salida carpeta]
"""
import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from agente import interfaz as s  # noqa: E402

SYSTEM_SIN_HERRAMIENTAS = """Eres un analista de informes 10-K. NO tienes herramientas
ni acceso al corpus: responde sólo con lo que recuerdes. Si no lo recuerdas con
seguridad, usa fuente 'ninguna' y no inventes. Si recuerdas una cifra, ponla en
cifra con su unidad. Si recuerdas una frase del informe, ponla en cita."""


def ejecutar_sin_herramientas(pregunta):
    registro = s.RegistroLLM()
    comienzo = time.perf_counter()
    modelo = s.modelo().with_structured_output(s.RespuestaFinanciera, include_raw=True)
    salida = modelo.invoke([{"role": "system", "content": SYSTEM_SIN_HERRAMIENTAS},
                            {"role": "user", "content": pregunta}],
                           config={"callbacks": [registro]})
    return (s.como_dict(salida.get("parsed")) or {},
            {"latencia_s": time.perf_counter() - comienzo, **registro.resumen()})


def cita_existe(texto):
    """¿La frase citada está en el corpus, o se la ha inventado el modelo?"""
    if not texto:
        return None
    literal = s.miax_s2.normalizar(texto)
    _, por_id, _ = s.datos()
    return any(literal in s.miax_s2.normalizar(c["texto"]) for c in por_id.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--golden", type=Path, default=s.RUTA_GOLDEN)
    p.add_argument("--salida", type=Path, default=None)
    args = p.parse_args()

    preguntas = s.leer_jsonl(args.golden)
    salida = args.salida or s.nueva_salida("control_memorizacion")
    salida.mkdir(parents=True, exist_ok=True)
    filas = []
    for i, item in enumerate(preguntas, 1):
        print(f"[memoria] {i}/{len(preguntas)} {item['id']}", flush=True)
        fila = {"id": item["id"], "familia": item["familia"], "cifra_de_memoria": None,
                "cita_inventada": None, "se_abstiene": None, "coste_usd": None, "error": None}
        try:
            r, uso = ejecutar_sin_herramientas(item["pregunta"])
            fila["coste_usd"] = uso.get("coste_usd")
            fila["se_abstiene"] = r.get("fuente") == "ninguna"
            if item.get("cifra_esperada") is not None:
                fila["cifra_de_memoria"] = bool(
                    r.get("cifra") is not None
                    and s.miax_s2.cuadra(r["cifra"], item["cifra_esperada"], s.TOLERANCIA))
            existe = cita_existe(r.get("cita"))
            fila["cita_inventada"] = None if existe is None else not existe
            fila["respuesta"] = r
        except Exception as exc:  # se conserva el fallo, no se convierte en acierto
            fila["error"] = f"{type(exc).__name__}: {exc}"
        filas.append(fila)
        s.guardar_json(salida / "control.json", filas)

    numericas = [f for f in filas if f["cifra_de_memoria"] is not None]
    con_cita = [f for f in filas if f["cita_inventada"] is not None]
    resumen = {
        "preguntas": len(filas),
        "con_cifra_esperada": len(numericas),
        "aciertos_de_memoria": sum(f["cifra_de_memoria"] for f in numericas),
        "tasa_memoria": (sum(f["cifra_de_memoria"] for f in numericas) / len(numericas)
                         if numericas else None),
        "citas_emitidas": len(con_cita),
        "citas_inventadas": sum(f["cita_inventada"] for f in con_cita),
        "abstenciones": sum(bool(f["se_abstiene"]) for f in filas),
        "coste_usd": sum(f["coste_usd"] or 0 for f in filas),
        "modelo": s.MODELO,
    }
    s.guardar_json(salida / "resumen.json", resumen)
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    print("salida:", salida)


if __name__ == "__main__":
    main()
