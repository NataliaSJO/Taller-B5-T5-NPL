"""Recupera el notebook S2 desde el módulo conservado, guardando una copia previa.

Uso desde la raíz: .\.venv\Scripts\python scripts/recuperar_notebook_s2.py
Si el notebook ya está adaptado, no se sobrescribe salvo con --forzar.
"""
import argparse
import ast
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import nbformat as nbf

RAIZ = Path(__file__).resolve().parents[1]
RUTA = RAIZ / "src/S2_Robustez_y_Evaluacion_Alumno.ipynb"
MODULO = RAIZ / "agente/interfaz.py"


def recuperar(forzar=False):
    actual = nbf.read(RUTA, as_version=4)
    if any("s2-exportar" in c.metadata.get("tags", []) for c in actual.cells) and not forzar:
        raise SystemExit("El notebook ya está adaptado. No se sobrescribe. Usa --forzar sólo si quieres reconstruirlo.")
    fuente = MODULO.read_text(encoding="utf-8")
    ast.parse(fuente)
    partes = re.split(r"(?=^# %% )", fuente, flags=re.M)
    bloques = {p.splitlines()[0][5:]: p.rstrip() for p in partes[1:]}
    if len(bloques) != 9:
        raise ValueError("Se esperaban los nueve bloques conservados en agente/interfaz.py")

    marca = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    copia = RAIZ / "resultados/s2/recuperacion" / marca
    copia.mkdir(parents=True)
    (copia / "notebook_antes.ipynb").write_bytes(RUTA.read_bytes())
    (copia / "interfaz_conservada.py").write_bytes(MODULO.read_bytes())
    celdas = []

    def md(texto):
        celdas.append(nbf.v4.new_markdown_cell(texto))

    def codigo(texto, exportar=False):
        celda = nbf.v4.new_code_cell(texto)
        if exportar:
            celda.metadata["tags"] = ["s2-exportar"]
        celdas.append(celda)

    def bloque(nombre, explicacion):
        md("## " + nombre + "\n\n" + explicacion)
        codigo(bloques[nombre], exportar=True)

    md("""# Sesión 2 · Robustez y evaluación — entrega del 23 de septiembre

**VERSIÓN ADAPTADA · archivo de trabajo: `src/S2_Robustez_y_Evaluacion_Alumno.ipynb`.**

Se recuperan las adaptaciones del enunciado `Practica_LLM_Agente_10K.docx` a partir de `agente/interfaz.py`. El material de `Clase_2` se conserva como referencia y el baseline sigue siendo tu notebook `src/Baseline_Agente_10K.ipynb`.

Cada bloque explica las modificaciones realizadas y su propósito, con comentarios en el código. Las definiciones están visibles en el notebook y disponibles como módulo para ejecutar el holdout sin abrir Jupyter. No se sustituyen los resultados pendientes por resultados inventados.

| Requisito del enunciado | Adaptación |
| --- | --- |
| Cuatro herramientas con los mismos nombres y parámetros | Herramientas del agente final, con XBRL sin perder decimales |
| Salida estructurada con ocho campos obligatorios | Se mantienen y se añaden `datos` y `citas` para comparativas |
| Límite de herramientas y middleware propio | Ocho herramientas, diez llamadas al modelo y verificación de cifras |
| Golden propio de 20 preguntas y seis comparativas | Se carga `src/golden_set_propio.jsonl` y se valida localmente |
| Filtros, BM25+denso y reescritura | Funciones implementadas y recall@5 después de cada variante |
| Tres evaluadores | Citas, cifras XBRL y trayectoria de herramientas |
| `responder(pregunta)` y `evaluar(ruta_jsonl)` | Disponibles aquí y en `agente/interfaz.py` |
| Baseline frente a final y resultados reproducibles | Misma muestra, tablas, coste, latencia, llamadas y trazas guardadas |

**Uso:** selecciona `Python (.venv - Taller 10K)` y ejecuta desde el principio. Para evaluar los agentes establece `EJECUTAR_API = True` en la celda siguiente. No necesitas ejecutar el notebook baseline por separado. En un clon nuevo instala antes con `python -m pip install -r requirements.txt`.

**Para evitar otra pérdida:** abre esta copia de `src`. Si VS Code avisa de que el archivo cambió en disco, recarga la versión del disco antes de guardar una pestaña antigua. La recuperación también se puede repetir con `scripts/recuperar_notebook_s2.py`, que guarda siempre una copia previa.""")
    codigo('''# Configuración visible; False ejecuta únicamente la parte local.
EJECUTAR_API = False  # Cambia a True para reescritura y evaluación completa.
EJECUTAR_RETRIEVAL = True

import os
import sys
import getpass

if EJECUTAR_API and not os.environ.get("OPENROUTER_API_KEY"):
    clave = getpass.getpass("OPENROUTER_API_KEY (openrouter.ai/keys): ").strip()
    if clave:
        os.environ["OPENROUTER_API_KEY"] = clave
    del clave
if EJECUTAR_API and not os.environ.get("OPENROUTER_API_KEY"):
    raise RuntimeError("Falta la clave de OpenRouter para ejecutar la API.")
print("Python:", sys.executable)
print("API activada:", EJECUTAR_API)''')
    bloque("Configuración", "Rutas válidas desde la raíz o desde `src`, datos locales y modelo compartido. Se usa `MAX_TOKENS=4096` en ambos agentes para evitar respuestas estructuradas truncadas; se puede configurar antes de ejecutar con `MAX_TOKENS_10K`. Importar el módulo no pide una clave ni llama al LLM. El corpus se prepara con los ZIP de `dataset/` y se comprueban sus hashes y manifiestos.")
    codigo('''# Usar tus datos y tus preguntas, no los IDs of-... del ejemplo.
print("Proyecto:", RAIZ)
print("Corpus:", preparar_corpus())
print("Modelo:", MODELO, "· max_tokens:", MAX_TOKENS)
golden = leer_jsonl(RUTA_GOLDEN)
secciones, chunks = miax_s2.cargar_corpus()
print("Preguntas:", len(golden), "· secciones:", len(secciones), "· fragmentos:", len(chunks))
print(pd.Series(g["familia"] for g in golden).value_counts().to_string())''')
    md("""## Qué se recupera y qué se mide

Conservamos el corpus y el índice de clase: 48 secciones, 1749 fragmentos y 135 hechos XBRL. El troceado puede separar una cifra de su encabezado; por eso las cantidades se consultan en XBRL y las explicaciones se buscan en texto.

Las siguientes funciones completan los ejercicios de búsqueda. Se definen primero y se ejecutan en el bloque de medición, cuando ya están disponibles todos sus auxiliares. Los filtros del golden se usan sólo para medir el buscador con metadatos conocidos; en una pregunta nueva el agente tiene que inferirlos.""")
    bloque("Búsqueda densa y filtros", "Se implementan `denso_plano` y `con_filtros`. Se busca sobre todo el índice antes de descartar compañías, ejercicios e items que no coinciden. Sólo se aplican los filtros distintos de `None` y después se eligen los primeros `k` resultados.")
    bloque("Fusión BM25 y denso", "Se completa `hibrido` con Reciprocal Rank Fusion: se suman `1 / (60 + posición)` de ambas listas filtradas. No se suman puntuaciones BM25 y cosenos, porque sus escalas son diferentes. La mejora se decide con la medición, no se presupone.")
    bloque("Reescritura y medición del retrieval", "El LLM recibe sólo la consulta, sin respuestas ni anclas del golden. Se miden denso, filtros, híbrido y, con API, híbrido reescrito. El acierto exige frase literal y metadatos correctos; no depende de un chunk_id esperado. Las comparativas aportan evidencias de ambos años: son 19 anclas. Se guardan consultas, resultados y coste de reescritura.")
    md("""### Comparativas y memoria

Las preguntas comparativas requieren consultar ambos ejercicios. Según lo solicitado, el agente hace dos consultas XBRL y dos búsquedas de texto, y calcula las variaciones. Se añaden `datos` y `citas` para guardar las dos evidencias, manteniendo los ocho campos obligatorios.

Cada pregunta evaluada usa un `thread_id` nuevo para evitar contaminación entre respuestas. Para conversar intencionadamente se puede llamar a `ejecutar(pregunta, thread_id="conversacion")`. No se resume la trayectoria durante la evaluación porque necesitamos las llamadas originales.""")
    bloque("Herramientas y esquema final", "Se conservan las firmas públicas del enunciado. `search_filings` usa realmente reescritura, filtros y búsqueda híbrida. `get_xbrl_fact` conserva decimales: EPS 7.46 no se convierte en 7. El argumento interno `config` permite registrar también el coste de reescritura; LangChain no lo muestra como argumento al modelo.")
    bloque("Middleware de cifras", "Se verifican el campo `cifra`, los hechos declarados y las cantidades de la prosa contra XBRL consultado. Se admiten diferencias y crecimientos de dos años del mismo concepto. Tolerancia: 1 % relativo y comparación exacta para cero. Si hay desajuste, se informa al modelo y se usa `jump_to='model'`; sólo hay una corrección. Si vuelve a fallar, se devuelve una abstención estructurada. Se excluyen años e items; la regla para separadores ambiguos está documentada.")
    bloque("Construcción, baseline propio y observabilidad", "Se reconstruye tu agente desde las definiciones y la llamada `create_agent` de `Baseline_Agente_10K.ipynb`, usando AST para no ejecutar demostraciones, claves ni evaluaciones de ese notebook. No se usa el agente de repuesto. La versión final incorpora las herramientas mejoradas y límites de ocho llamadas a herramienta y diez al modelo; cada búsqueda añade como máximo una reescritura. Se registran tokens, latencia y coste, incluido el de reescritura. Un coste desconocido queda vacío.")
    md("""### Límites del agente

La evaluación automática no incluye la demostración opcional de aprobación humana de clase. El límite de llamadas evita bucles, pero no es un presupuesto monetario: leer una sección larga puede costar más que buscar fragmentos.

El middleware verifica consistencia numérica con hechos consultados. El evaluador comprueba además el concepto esperado por la pregunta. El extractor numérico y el juez semántico tienen limitaciones; los resultados y las trazas permiten revisar errores.""")
    bloque("Los tres evaluadores", "`cita_correcta`: existencia, frase literal completa, metadatos y evidencia recuperada, seguida de un juez LLM que comprueba el soporte semántico y guarda su motivo. Sin juez, el soporte queda pendiente. `cifra_coincide_xbrl`: cifra, unidad y ejercicio contra el parquet; en comparativas, ambos valores y sus variaciones. `uso_la_tool_correcta`: herramientas y argumentos correctos, incluyendo hechos XBRL ejecutados de ambos años.")
    bloque("Evaluar, guardar y comparar", "Se admite la ruta JSONL que exige el holdout. Se guardan preguntas, configuración y huellas, respuestas, mensajes, métricas, errores y resumen por familia. Antes del baseline se copian su notebook y su auxiliar. Ambas versiones se evalúan sobre las mismas preguntas. Cada ejecución crea una carpeta nueva, conserva fallos y no da costes desconocidos por gratuitos. Sólo se destacan mejores valores con ambas evaluaciones completas.")
    md("""## Comprobaciones locales

Esta validación comprueba el golden propio contra el corpus. No sustituye al validador oficial externo, que no figura entre los archivos proporcionados. Las pruebas automatizadas del grafo, límites y evaluadores se ejecutan con `python -m unittest discover -s tests -v`, sin API.""")
    codigo('''# Validación real de los datos y de las firmas públicas.
print(validar_golden())
assert {t.name for t in HERRAMIENTAS} == NOMBRES_TOOLS
assert list(search_filings.args) == ["query", "ticker", "fiscal_year", "item", "k"]
assert "7.46 USD/shares" in get_xbrl_fact.invoke({
    "ticker": "AAPL", "fiscal_year": 2025, "concept": "EarningsPerShareDiluted"})
assert extraer_cifras("FY2025: 7,46 USD por acción") == [(7.46, False)]
print("Datos y contrato comprobados.")''')
    md("""## Exportar las definiciones

`agente/interfaz.py` ya contiene estas funciones. Si cambias código del notebook, **guárdalo primero** y ejecuta esta celda para actualizar el módulo; no edites ambas copias por separado. Se exportan sólo las nueve celdas `s2-exportar`, nunca claves, demostraciones ni salidas. La comprobación impide que un notebook antiguo o incompleto borre el módulo. Si hay cambios, se guarda la versión anterior.""")
    codigo('''# Comprobar la versión guardada antes de exportar.
ruta_notebook = RAIZ / "src/S2_Robustez_y_Evaluacion_Alumno.ipynb"
cuaderno = json.loads(ruta_notebook.read_text(encoding="utf-8"))
exportables = [c for c in cuaderno["cells"]
               if "s2-exportar" in c.get("metadata", {}).get("tags", [])]
if len(exportables) != 9:
    raise RuntimeError("El archivo guardado no es la versión adaptada completa. No se sobrescribe el módulo.")
codigo_exportado = "\\n\\n".join("".join(c["source"]).rstrip() for c in exportables)
arbol = ast.parse(codigo_exportado)
definidas = {n.name for n in arbol.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
necesarias = {"responder", "evaluar", "comparar", "cargar_baseline_propio", "reescribir",
              "hibrido", "con_filtros", "verificar_cifras_contra_xbrl",
              "cita_correcta", "cifra_coincide_xbrl", "uso_la_tool_correcta"}
if not necesarias.issubset(definidas):
    raise RuntimeError("Faltan definiciones. No se sobrescribe el módulo.")
destino = RAIZ / "agente/interfaz.py"
anterior = destino.read_text(encoding="utf-8")
if not anterior.endswith(codigo_exportado + "\\n"):
    copia = nueva_salida("exportacion")
    copia.mkdir(parents=True)
    (copia / "interfaz_anterior.py").write_text(anterior, encoding="utf-8")
    destino.write_text('"""Definiciones exportadas del notebook S2."""\\n\\n'
                       + codigo_exportado + "\\n", encoding="utf-8")
    print("Módulo actualizado; copia previa:", copia)
else:
    print("Notebook y módulo ya están sincronizados.")''')
    md("""## Medición real del retrieval

Se guarda `recall.csv`, el detalle por evidencia y las consultas reescritas. Los filtros se toman del golden: es una medición aislada del buscador. Si el modelo de embeddings `BAAI/bge-small-en-v1.5` no está en caché, la primera ejecución necesita descargarlo.

Con `EJECUTAR_API=False` se miden las primeras tres variantes. La reescritura no se registra como medida hasta que se ejecuta el LLM.""")
    codigo('''# Métricas calculadas a partir del corpus, sin valores fijos.
if EJECUTAR_RETRIEVAL:
    tabla_retrieval = medir_retrieval(RUTA_GOLDEN, usar_llm=EJECUTAR_API)
    print(tabla_retrieval.groupby("variante", sort=False).agg(
        recall_at_5=("acierto", "mean"), evidencias=("acierto", "size")).to_string())
    print("Guardado en:", tabla_retrieval.attrs["salida"])
else:
    print("Medición de retrieval desactivada.")''')
    md("""## Comparación baseline frente a final

Se ejecutan las mismas 20 preguntas propias. La comparación guarda aciertos por familia, coste, latencia y llamadas. `recall_trayectoria` agrega los fragmentos de las búsquedas reales del agente; no equivale al recall@5 de una única búsqueda del bloque anterior. El coste del juez se guarda aparte del coste del agente.

La evaluación con API puede tardar por las varias llamadas y el límite de frecuencia. Si falta crédito o hay limitación de peticiones, se guardan los avances y las preguntas no intentadas. Una evaluación incompleta no demuestra que una versión sea mejor.""")
    codigo('''# Usa tu baseline y tu golden, y guarda resultados sin sobrescribir los anteriores.
if EJECUTAR_API:
    comparacion = comparar(RUTA_GOLDEN)
    print(comparacion.round(4).to_string(index=False))
    print("Tablas y trazas guardadas en:", comparacion.attrs["salida"])
else:
    print("Pendiente: activa EJECUTAR_API para medir reescritura y evaluar baseline/final.")''')
    md("""## Entrega del 23 y holdout del 24

El enunciado pide GitHub e informe PDF el **23 de septiembre a las 23:59**. Este notebook contiene la implementación y el código que genera las tablas; todavía hay que ejecutar la parte con API, revisar resultados y redactar el informe. Explica qué mejoró, qué no mejoró y el coste. La defensa del 24 dura ocho minutos más preguntas e incorpora diez preguntas ciegas.

Desde la raíz del repositorio, con `.venv` y `OPENROUTER_API_KEY` configurados:

```python
from agente.interfaz import responder, evaluar
respuesta = responder("¿Cuáles fueron los ingresos de NVIDIA en FY2025?")
tabla = evaluar("holdout.jsonl")
print(tabla)
print(tabla.attrs["salida"])
```

`responder` devuelve la respuesta estructurada; `evaluar` conserva también los mensajes para verificar la trayectoria. Se crean carpetas nuevas en `resultados/s2/`. No guardes claves en el repositorio.""")

    # Conservar los apuntes añadidos por la usuaria sin ejecutar expresiones incompletas.
    apuntes = [c.source for c in actual.cells if c.cell_type == "code"
               and ("intento = indice.search" in c.source
                    or c.source.strip().startswith(("int(meta.iloc[intento", "meta.iloc[intento")))]
    if apuntes:
        md("## Apuntes de clase conservados\n\nSe conservan como referencia los apuntes que había en el archivo. No se ejecutan: dependen de variables de una demostración de clase y una línea intenta convertir el texto de un informe a entero. La búsqueda ejecutable está implementada más arriba.\n\n"
           + "\n\n".join("```python\n" + s + "\n```" for s in apuntes))
    md("**Copia previa a esta recuperación:** `" + copia.relative_to(RAIZ).as_posix()
       + "/notebook_antes.ipynb`. La implementación conservada se guardó en esa misma carpeta.")

    notebook = nbf.v4.new_notebook(cells=celdas)
    notebook.metadata["kernelspec"] = {"display_name": "Python (.venv - Taller 10K)",
                                       "language": "python", "name": "taller-10k"}
    notebook.metadata["language_info"] = {"name": "python", "version": "3.11"}
    notebook.metadata["s2_adaptado"] = {"version": 1, "recuperacion": marca,
        "modulo_sha256": hashlib.sha256(MODULO.read_bytes()).hexdigest()}
    nbf.validate(notebook)
    for c in notebook.cells:
        if c.cell_type == "code":
            ast.parse(c.source)
    nbf.write(notebook, RUTA)
    (copia / "notebook_recuperado.ipynb").write_bytes(RUTA.read_bytes())
    print("Notebook recuperado:", RUTA)
    print("Copia de seguridad:", copia)
    print("Celdas:", len(celdas), "· bloques exportables:", len(bloques))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forzar", action="store_true", help="Reconstruir aunque ya esté adaptado; guarda copia previa")
    recuperar(parser.parse_args().forzar)
