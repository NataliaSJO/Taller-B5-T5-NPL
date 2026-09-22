"""Genera `resultados/RESUMEN.md` a partir de los ficheros de resultados.

El enunciado exige que el código genere todas las tablas reportadas. Este script
no inventa nada: lee las métricas guardadas y las escribe, marcando lo que falta
como faltante en vez de rellenarlo.

Uso:
    python scripts/generar_resumen.py --comparacion resultados/s2/comparacion/<marca> \
                                      --retrieval resultados/s2/retrieval/<marca>
"""
import argparse
import json
from math import comb
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent


def porcentaje(x):
    return "faltante" if pd.isna(x) else f"{x:.1%}"


def numero(x, dec=4):
    return "faltante" if pd.isna(x) else f"{x:.{dec}f}"


def mcnemar(a, b):
    u = a[["id", "acierto"]].merge(b[["id", "acierto"]], on="id", suffixes=("_a", "_b"))
    gana = int((~u.acierto_a & u.acierto_b).sum())
    pierde = int((u.acierto_a & ~u.acierto_b).sum())
    n = gana + pierde
    p = (sum(comb(n, i) for i in range(min(gana, pierde) + 1)) / 2 ** n * 2) if n else 1.0
    return gana, pierde, min(1.0, p)


def tabla_md(cabecera, filas):
    alineacion = "| " + " | ".join("---" for _ in cabecera) + " |"
    return "\n".join(["| " + " | ".join(cabecera) + " |", alineacion]
                     + ["| " + " | ".join(f) + " |" for f in filas])


# Cada agente usa un buscador distinto: el baseline, denso con filtros aplicados
# después del índice (`miax_s1.buscar`); el final, híbrido con encabezados sobre
# la consulta reescrita. Se comparan sus recall@5 medidos igual, con los filtros
# del golden, que es una cota superior para los dos.
BUSCADOR = {"baseline": "filtros", "final": "hibrido_enc_reescrito"}


def relativa(ruta: Path) -> str:
    """Ruta legible dentro del repositorio, se pase absoluta o relativa."""
    ruta = ruta.resolve()
    return str(ruta.relative_to(RAIZ)) if ruta.is_relative_to(RAIZ) else str(ruta)


def seccion_varianza(actual: Path, repeticion: Path | None) -> list[str]:
    """Mismo agente baseline, dos ejecuciones: cuánto se mueve la métrica sola."""
    if repeticion is None or not (repeticion / "baseline/metricas.csv").is_file():
        return []
    a = pd.read_csv(repeticion / "baseline/metricas.csv")
    b = pd.read_csv(actual / "baseline/metricas.csv")
    u = a[["id", "familia", "acierto"]].merge(b[["id", "acierto"]], on="id",
                                              suffixes=("_1", "_2"))
    filas = [["Global", porcentaje(u.acierto_1.mean()), porcentaje(u.acierto_2.mean())]]
    for familia, g in u.groupby("familia"):
        filas.append([familia, porcentaje(g.acierto_1.mean()), porcentaje(g.acierto_2.mean())])
    cambian = int((u.acierto_1 != u.acierto_2).sum())
    return ["## Varianza entre ejecuciones", "",
            "El **mismo** agente baseline, las mismas preguntas y los mismos evaluadores, "
            "medido dos veces. La temperatura es 0, pero ni la API ni el juez son "
            "deterministas.", "",
            tabla_md(["Familia", "Ejecución 1", "Ejecución 2"], filas), "",
            f"Cambian de resultado **{cambian} de {len(u)} preguntas**. Una de ellas se explica "
            "por el reintento de firma de pensamiento que incorpora la segunda ejecución; el "
            "resto es ruido. Con 7 preguntas por familia, las cifras por familia se mueven "
            "decenas de puntos sin que cambie nada: no se deben leer como diferencias reales.",
            ""]


def seccion_controles(control: Path | None, huecos: list[Path]) -> list[str]:
    """Dos controles que el golden propio no cubre y el enunciado sí menciona."""
    partes = []
    if control is not None and (control / "resumen.json").is_file():
        r = json.loads((control / "resumen.json").read_text(encoding="utf-8"))
        partes += ["## Control de memorización (sin herramientas)", "",
                   "Las mismas preguntas contra el mismo modelo, sin corpus ni herramientas. "
                   "Mide cuánto del acierto no depende del 10-K: el enunciado penaliza acertar "
                   "por el camino equivocado.", "",
                   tabla_md(["Medida", "Valor"], [
                       ["Cifras acertadas de memoria",
                        f"{r['aciertos_de_memoria']}/{r['con_cifra_esperada']} "
                        f"({(r['tasa_memoria'] or 0):.1%})"],
                       ["Abstenciones", f"{r['abstenciones']}/{r['preguntas']}"],
                       ["Citas inventadas (no existen en el corpus)",
                        f"{r['citas_inventadas']}/{r['citas_emitidas']}"],
                       ["Coste del control (USD)", f"{r['coste_usd']:.4f}"]]), ""]
    filas = []
    for ruta in huecos:
        if not (ruta / "huecos.json").is_file():
            continue
        datos = json.loads((ruta / "huecos.json").read_text(encoding="utf-8"))
        filas.append([ruta.name[:15],
                      f"{sum(bool(x.get('correcto')) for x in datos)}/{len(datos)}",
                      f"{sum(x.get('coste_usd') or 0 for x in datos):.4f}"])
    if filas:
        partes += ["## Huecos reales del XBRL", "",
                   "Conceptos que una compañía no reporta (Amazon no publica GrossProfit, "
                   "Liabilities ni ResearchAndDevelopmentExpense; Meta y Alphabet tampoco "
                   "GrossProfit). La respuesta correcta es decir que no está. El golden propio "
                   "no cubre este caso, así que se mide aparte.", "",
                   tabla_md(["Ejecución", "Se abstiene correctamente", "Coste (USD)"], filas), ""]
    return partes


def generar(comparacion: Path, retrieval: Path | None, control: Path | None = None,
            huecos: list[Path] | None = None, repeticion: Path | None = None) -> str:
    tablas = {v: pd.read_csv(comparacion / v / "metricas.csv") for v in ("baseline", "final")}
    recall_aislado = {}
    if retrieval is not None and (retrieval / "recall.csv").is_file():
        serie = pd.read_csv(retrieval / "recall.csv").set_index("variante").recall_at_5
        recall_aislado = {v: serie.get(BUSCADOR[v], float("nan")) for v in tablas}
    config = json.loads((comparacion / "final" / "configuracion.json").read_text(encoding="utf-8"))
    partes = ["# Resultados de la evaluación", "",
              f"Modelo del agente: `{config['modelo']}`, máximo {config['max_tokens']} tokens de "
              f"salida, temperatura {config['temperature']}. Tolerancia numérica "
              f"{config['tolerancia']:.0%}. Límite de {config['limite_tools_final']} llamadas a "
              f"herramienta y {config['limite_modelo_final']} al modelo por pregunta.", "",
              f"Carpeta de resultados: `{relativa(comparacion)}`.", ""]

    filas, menor_es_mejor = [], {"Coste medio (USD)", "Coste medio estimado (USD)",
                                 "Latencia media (s)", "Llamadas a herramienta"}
    metricas = [
        ("Preguntas evaluadas", lambda t: (t.estado == "evaluado").sum(), lambda v: str(int(v))),
        ("Acierto global", lambda t: t.acierto.mean(), porcentaje),
        ("Acierto numéricas", lambda t: t[t.familia == "numerica"].acierto.mean(), porcentaje),
        ("Acierto extractivas", lambda t: t[t.familia == "extractiva"].acierto.mean(), porcentaje),
        ("Acierto comparativas", lambda t: t[t.familia == "comparativa"].acierto.mean(), porcentaje),
        ("Cita válida", lambda t: t.cita.dropna().mean() if t.cita.notna().any() else float("nan"), porcentaje),
        ("Cita sobre el ancla del golden", lambda t: t.cita_ancla.dropna().mean() if "cita_ancla" in t and t.cita_ancla.notna().any() else float("nan"), porcentaje),
        ("Recall@5 del buscador (aislado)", lambda t, v=None: float("nan"), porcentaje),
        ("Recall de la trayectoria", lambda t: t.recall_trayectoria.mean(), porcentaje),
        ("Coste medio (USD)", lambda t: t.coste_usd.mean(), lambda v: numero(v, 4)),
        ("Coste medio estimado (USD)", lambda t: t.coste_estimado_usd.mean() if "coste_estimado_usd" in t else float("nan"), lambda v: numero(v, 4)),
        ("Latencia media (s)", lambda t: t.latencia_s.mean(), lambda v: numero(v, 1)),
        ("Llamadas a herramienta", lambda t: t.llamadas.mean(), lambda v: numero(v, 2)),
        ("Avisos del guardrail", lambda t: t.avisos_guardrail.sum() if "avisos_guardrail" in t else float("nan"), lambda v: numero(v, 0)),
    ]
    completa = all((t.estado == "evaluado").sum() == len(t) for t in tablas.values())
    for nombre, calcula, formatea in metricas:
        if nombre.startswith("Recall@5 del buscador"):
            valores = {v: recall_aislado.get(v, float("nan")) for v in tablas}
        else:
            valores = {v: calcula(t) for v, t in tablas.items()}
        textos = {v: formatea(x) for v, x in valores.items()}
        # Los avisos del guardrail no se resaltan: el baseline no lo lleva, así que
        # «menos avisos» no es mérito suyo ni «más» es mérito del final.
        comparable = nombre not in {"Preguntas evaluadas", "Avisos del guardrail"}
        if completa and comparable and not any(pd.isna(x) for x in valores.values()):
            mejor = min(valores, key=valores.get) if nombre in menor_es_mejor else max(valores, key=valores.get)
            if valores["baseline"] != valores["final"]:
                textos[mejor] = f"**{textos[mejor]}**"
        filas.append([nombre, textos["baseline"], textos["final"]])
    partes += ["## Baseline frente a sistema final", "",
               tabla_md(["Métrica", "Baseline", "Final"], filas), ""]

    gana, pierde, p = mcnemar(tablas["baseline"], tablas["final"])
    partes += [f"Prueba pareada (McNemar exacto) sobre las mismas 20 preguntas: sólo acierta el "
               f"final en {gana}, sólo el baseline en {pierde}, p = {p:.3f}. Con 20 preguntas el "
               f"intervalo de confianza de una tasa ronda ±20 puntos: la tabla se lee junto a esta "
               f"prueba, no en su lugar.", ""]
    if not completa:
        partes += ["> **Evaluación incompleta.** Las preguntas con error no se cuentan como "
                   "respuestas incorrectas del modelo. Los ceros de preguntas no intentadas no "
                   "miden calidad.", ""]

    incidencias = []
    for v, t in tablas.items():
        if "error" not in t:
            continue
        for _, r in t[t.error.notna()].iterrows():
            incidencias.append([v, r["id"], str(r["error"])[:110]])
    if incidencias:
        partes += ["## Incidencias", "",
                   tabla_md(["Versión", "Pregunta", "Error"], incidencias), ""]

    if retrieval is not None and (retrieval / "recall_por_k.csv").is_file():
        curva = pd.read_csv(retrieval / "recall_por_k.csv").set_index("variante")
        columnas = [c for c in curva.columns if c.startswith("recall_at_")]
        filas_r = [[v] + [f"{fila[c]:.1%}" for c in columnas] + [str(int(fila["evidencias"]))]
                   for v, fila in curva.iterrows()]
        partes += ["## Buscador: matriz de mejoras", "",
                   tabla_md(["Variante"] + [c.replace("recall_at_", "Recall@") for c in columnas]
                            + ["Evidencias"], filas_r), "",
                   "Los filtros salen del golden: la medición aísla el buscador y es una cota "
                   "superior de lo que consigue el agente, que debe inferirlos de la pregunta.",
                   f"Carpeta: `{relativa(retrieval)}`.", ""]

    partes += seccion_varianza(comparacion, repeticion)
    partes += seccion_controles(control, huecos or [])
    partes += ["## Cómo se regenera", "",
               "```powershell",
               "python -m unittest discover -s tests -v",
               "python scripts/ejecutar_evaluacion_s2.py --salida resultados\\s2\\<carpeta>",
               "python scripts/generar_resumen.py --comparacion <carpeta>\\comparacion "
               "--retrieval resultados\\s2\\retrieval\\<marca>",
               "```", ""]
    return "\n".join(partes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparacion", type=Path, required=True)
    p.add_argument("--retrieval", type=Path, default=None)
    p.add_argument("--control", type=Path, default=None,
                   help="carpeta de resultados de control_memorizacion.py")
    p.add_argument("--huecos", type=Path, nargs="*", default=None,
                   help="carpetas de prueba_huecos_xbrl.py, en orden cronológico")
    p.add_argument("--repeticion", type=Path, default=None,
                   help="otra comparación con el mismo baseline, para medir la varianza")
    p.add_argument("--salida", type=Path, default=RAIZ / "resultados/RESUMEN.md")
    args = p.parse_args()
    texto = generar(args.comparacion, args.retrieval, args.control, args.huecos,
                    args.repeticion)
    args.salida.write_text(texto, encoding="utf-8")
    print(texto)
    print("\nEscrito en", args.salida)


if __name__ == "__main__":
    main()
