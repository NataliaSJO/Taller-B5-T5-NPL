"""Genera el informe Markdown y PDF exclusivamente desde resultados guardados."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def generar(comparacion, retrieval, salida):
    salida.mkdir(parents=True, exist_ok=True)
    resumen = pd.read_csv(comparacion / "comparacion.csv").set_index("version")
    recall = pd.read_csv(retrieval / "recall.csv")
    completa = bool((resumen.evaluadas == resumen.preguntas).all())
    estilos = getSampleStyleSheet()
    contenido, markdown = [], []

    def texto(t, titulo=False):
        markdown.append(("## " if titulo else "") + t)
        contenido.extend([Paragraph(escape(t), estilos["Heading2" if titulo else "BodyText"]), Spacer(1, 0.22 * cm)])

    def tabla(filas):
        markdown.append("\n".join(["| " + " | ".join(map(str, filas[0])) + " |",
                                     "| " + " | ".join("---" for _ in filas[0]) + " |"] +
                                    ["| " + " | ".join(map(str, r)) + " |" for r in filas[1:]]))
        celdas = [[Paragraph(escape(str(v)), estilos["BodyText"]) for v in r] for r in filas]
        t = Table(celdas, colWidths=[16.5 * cm / len(filas[0])] * len(filas[0]), repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E4EDF5")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C4CED8")),
                               ("TOPPADDING", (0, 0), (-1, -1), 6),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        contenido.extend([t, Spacer(1, 0.35 * cm)])

    texto("Agente sobre informes 10-K: robustez y evaluación", True)
    texto("Estado: " + ("comparación completa." if completa else
          "evaluación incompleta. Informe provisional; no permite concluir qué agente es mejor."))
    config = json.loads((comparacion / "final/configuracion.json").read_text(encoding="utf-8"))
    texto(f"Modelo: {config['modelo']}. Temperatura: {config['temperature']}; "
          f"máximo de tokens de salida por llamada: {config['max_tokens']}; k: {config['k']}.")
    texto("Diseño y funcionamiento", True)
    texto("Se comparan el baseline propio y el agente final con las mismas 20 preguntas: "
          "7 numéricas, 7 extractivas y 6 comparativas. El corpus contiene 48 secciones de "
          "seis compañías en dos ejercicios; las comparativas aportan evidencia de ambos años.")
    texto("El agente consulta get_xbrl_fact para cifras, search_filings para texto, read_section "
          "para una sección completa y list_available para comprobar cobertura. El sistema final "
          "incorpora filtros, fusión BM25/densa por posiciones y reescritura al inglés. Conserva "
          "decimales XBRL, comprueba cifras y permite una corrección antes de abstenerse; limita "
          "a ocho llamadas a herramientas y diez al modelo, con reescrituras adicionales registradas.")
    texto("El acierto exige superar los criterios aplicables de cifra, cita y trayectoria. "
          "La cita se comprueba literalmente, con sus metadatos y procedencia, antes del juez "
          "semántico. Las preguntas numéricas sin anclas no necesitan juez. Los fallos técnicos "
          "se conservan y no se presentan como respuestas evaluadas.")
    texto("Comparación de los agentes", True)
    filas = [["Métrica", "Baseline", "Final"]]
    for columna, etiqueta in [("evaluadas", "Preguntas evaluadas / 20"), ("acierto", "Acierto global"),
            ("acierto_numerica", "Acierto numéricas"), ("acierto_extractiva", "Acierto extractivas"),
            ("acierto_comparativa", "Acierto comparativas"), ("recall_trayectoria", "Recall de la trayectoria"),
            ("coste_medio_usd", "Coste medio agente (USD)"), ("costes_conocidos", "Preguntas con coste conocido"),
            ("latencia_media_s", "Latencia media (s)"), ("llamadas_medias", "Llamadas a herramientas")]:
        valores = [resumen.loc[v, columna] for v in ["baseline", "final"]]
        textos = ["No disponible" if pd.isna(v) else f"{v:.4f}" for v in valores]
        comparable = completa and columna not in {"evaluadas", "costes_conocidos"}
        if columna == "coste_medio_usd":
            comparable = comparable and bool((resumen.costes_conocidos == resumen.preguntas).all())
        if comparable:
            mejor = min(valores) if columna in {"coste_medio_usd", "latencia_media_s", "llamadas_medias"} else max(valores)
            textos = [t + " *" if pd.notna(v) and v == mejor else t for t, v in zip(textos, valores)]
        filas.append([etiqueta, *textos])
    tabla(filas)
    texto("* Mejor valor observado (ambos si empatan), sólo con evaluación completa. Las tasas "
          "usan todas las preguntas, incluidos errores y pendientes. Las medias de coste y latencia "
          "excluyen valores desconocidos; con ejecución incompleta no describen una comparación válida. "
          "El coste del agente incluye reescritura; el juez se contabiliza por separado.")
    if completa:
        delta = 100 * (resumen.loc["final", "acierto"] - resumen.loc["baseline", "acierto"])
        texto(f"La variación del acierto global del sistema final es {delta:+.1f} puntos porcentuales. "
              "Las mejoras de búsqueda no garantizan el acierto final: también influyen el enrutamiento, "
              "la respuesta estructurada y la evidencia citada.")
    else:
        texto("No se interpreta la diferencia de acierto entre agentes porque faltan evaluaciones. "
              "Los ceros de preguntas no intentadas no demuestran mala calidad del modelo.")
    texto("Medición aislada de la búsqueda", True)
    config_retrieval = json.loads((retrieval / "configuracion.json").read_text(encoding="utf-8"))
    texto(f"Esta medición se conserva como experimento separado: modelo {config_retrieval['modelo']}, "
          f"máximo de salida {config_retrieval['max_tokens']} tokens. La comparación de agentes "
          f"utiliza {config['max_tokens']} tokens. No se han repetido las consultas de este "
          "experimento al cambiar el límite de salida de los agentes.")
    tabla([["Variante", "Recall@5", "Evidencias"]] +
          [[r.variante, f"{r.recall_at_5:.2%}", str(r.evidencias)] for r in recall.itertuples()])
    rec = recall.set_index("variante").recall_at_5
    texto(f"Los filtros cambian el recall en {(rec['filtros']-rec['denso'])*100:+.1f} puntos; "
          f"la fusión híbrida añade {(rec['hibrido']-rec['filtros'])*100:+.1f}; la reescritura añade "
          f"{(rec['hibrido_reescrito']-rec['hibrido'])*100:+.1f}. "
          "En esta medición, la fusión no mejoró el resultado del filtrado. Los filtros proceden "
          "del golden: esta prueba aísla el buscador y no mide si el agente sabe elegirlos. "
          "El recall de la trayectoria agrega todas las búsquedas del agente y no equivale al recall@5.")
    texto("Revisión de respuestas y costes", True)
    for version in ["baseline", "final"]:
        registros = [json.loads(l) for l in (comparacion / version / "respuestas.jsonl").read_text(encoding="utf-8").splitlines()]
        estados = pd.Series([r["metricas"]["estado"] for r in registros]).value_counts().to_dict()
        texto(f"{version}: " + "; ".join(f"{k}: {v}" for k, v in estados.items()) + ".")
        costes = [r["metricas"].get("coste_juez_usd") for r in registros]
        conocidos = [c for c in costes if c is not None]
        texto(f"Coste registrado del juez: {sum(conocidos):.6f} USD en {len(conocidos)} preguntas con coste de juez informado.")
        ejemplos = [r for r in registros if r["metricas"]["estado"] == "evaluado" and not r["metricas"]["acierto"]][:2]
        if not ejemplos:
            ejemplos = [r for r in registros if r["metricas"]["estado"] == "evaluado"][:1]
        for r in ejemplos:
            respuesta = r["resultado"].get("structured_response", {})
            texto(f"Ejemplo {r['id']}: {r['pregunta']}")
            prosa = respuesta.get("respuesta", "Sin respuesta estructurada")
            texto(("Respuesta (extracto): " + prosa[:700] + "…") if len(prosa) > 700 else "Respuesta: " + prosa)
            texto("Criterios: " + "; ".join(f"{k}={r['metricas'].get(k)}" for k in ["cifra", "cita", "trayectoria", "acierto"]))
            motivo = r["resultado"].get("juicio_cita", {}).get("motivo")
            if motivo:
                texto("Motivo del juez de citas: " + motivo)
            elif r["metricas"].get("cita") is False:
                texto("La cita no superó las comprobaciones previas de literalidad, metadatos "
                      "o evidencia recuperada; no se solicitó juicio semántico.")
        errores = list(dict.fromkeys(r["metricas"].get("error", "") for r in registros if r["metricas"].get("error")))
        for error in errores[:3]:
            texto("Incidencia registrada: " + error)
    texto("Reproducibilidad y límites", True)
    texto("Ejecutar scripts/ejecutar_evaluacion_s2.py con --salida y una carpeta nueva. La clave "
          "se introduce de forma privada o mediante OPENROUTER_API_KEY. El notebook permite también "
          "ejecutar los bloques en orden. scripts/generar_informe_s2.py regenera este informe a partir "
          "de los CSV y las trazas. Las carpetas incluyen configuración, preguntas y hashes.")
    texto("La validación del golden es local contra el corpus, no el validador externo del aula. "
          "Las 10 preguntas ciegas no están disponibles y su resultado y diferencia frente al golden "
          "quedan pendientes. Una muestra de 20 preguntas y un juez automático no garantizan "
          "generalización; se conservan respuestas y trazas para revisión humana.")
    texto("La latencia depende de la red, la carga del proveedor y la caché local. Las versiones "
          "se ejecutan secuencialmente, primero baseline y después final; no es un ensayo "
          "aleatorizado de rendimiento. Los extractos del informe no sustituyen las respuestas "
          "completas guardadas en respuestas.jsonl.")
    texto("Fuentes locales: " + str(comparacion) + " ; " + str(retrieval))
    (salida / "informe.md").write_text("\n\n".join(markdown) + "\n", encoding="utf-8")
    SimpleDocTemplate(str(salida / "informe.pdf"), title="Evaluación del agente 10-K",
                      rightMargin=2*cm, leftMargin=2*cm, topMargin=1.8*cm,
                      bottomMargin=1.8*cm).build(contenido)
    return salida / "informe.pdf"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparacion", type=Path, required=True)
    p.add_argument("--retrieval", type=Path, required=True)
    p.add_argument("--salida", type=Path, required=True)
    args = p.parse_args()
    print(generar(args.comparacion, args.retrieval, args.salida))
