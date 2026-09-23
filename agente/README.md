# Agente de la sesión 2

El código se explica en `src/S2_Robustez_y_Evaluacion_Alumno.ipynb`. Las celdas
marcadas `s2-exportar` generan `agente/interfaz.py`. Si cambias una definición,
guarda el notebook y ejecuta su celda de exportación; no edites las dos copias
por separado. Los originales de `Clase_2` y el baseline se conservan.

Si el notebook vuelve a sobrescribirse con la copia de clase, puedes recuperarlo
desde el módulo conservado:

```powershell
.\.venv\Scripts\python scripts/recuperar_notebook_s2.py
```

El script guarda primero una copia en `resultados/s2/recuperacion/` y conserva
los apuntes de clase como referencia. Si detecta que el notebook ya está
adaptado, se detiene para evitar perder cambios posteriores. La prueba
`test_notebook_adaptado_y_sincronizado` detecta una sustitución por la versión
antigua. Abre la copia de **`src`** y recarga la versión del disco si VS Code
avisa de cambios externos antes de guardar una pestaña antigua.

## Ejecutar

Desde la raíz del repositorio, con Python 3.11:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m unittest discover -s tests -v
```

Los ZIP de clase deben estar en `dataset/`, o el corpus completo en `corpus/`.
No se descargan los informes de EDGAR. La primera carga de embeddings puede
descargar `BAAI/bge-small-en-v1.5` si todavía no está en caché.

En el notebook selecciona `.venv`, cambia `EJECUTAR_API` a `True` y ejecuta las
celdas desde el principio. La clave se solicita con `getpass`. Para ejecutar
desde otro programa, define `OPENROUTER_API_KEY` en su entorno:

```python
from agente.interfaz import responder, evaluar, comparar, medir_retrieval

respuesta = responder("¿Cuáles fueron los ingresos de NVIDIA en FY2025?")
tabla = evaluar("holdout.jsonl")
print(tabla.attrs["salida"])

# Comparación completa sobre las mismas 20 preguntas:
comparacion = comparar()
retrieval = medir_retrieval(usar_llm=True)
```

Estas llamadas usan la API. El modo local del notebook no llama al LLM.
`MODELO_10K` y `MAX_TOKENS_10K` permiten configurar modelo y límite antes de
importar; los valores por defecto coinciden con el baseline: Gemini 3.8 Flash
y 1024 tokens. Si se trunca la salida estructurada, aumenta el límite y vuelve
a medir **ambos** agentes con la misma configuración.

## Qué se guarda

Cada ejecución crea una carpeta nueva en `resultados/s2/`. La evaluación guarda
las preguntas, configuración y hashes, respuestas y mensajes, métricas por
pregunta y resumen por familia. La comparación añade `comparacion.csv` y
`comparacion.md`, resaltando mejores valores sólo con evaluaciones completas.
Antes de ejecutar el baseline se guardan su notebook y `miax_s1.py`.

El baseline se construye a partir de las definiciones de
`src/Baseline_Agente_10K.ipynb`; no se usa el agente de repuesto del profesor.
La extracción mediante AST evita ejecutar ejemplos, solicitar claves o lanzar
evaluaciones al cargarlo. El agente final tiene las herramientas mejoradas.

Los errores de API y las preguntas no intentadas permanecen en la tabla. Los
aciertos se dividen entre todas las preguntas; una tabla incompleta no debe
interpretarse como una comparación de calidad. El coste desconocido queda
vacío y la tabla informa cuántos costes se conocen. El coste del juez de citas
se guarda separado del coste del agente. Se conservan las llamadas de
reescritura en el cálculo del coste del agente.

## Cómo interpretar las métricas

- El recall@5 aislado usa filtros conocidos del golden y se mide contra frases
  literales, con compañía, ejercicio e item correctos. Las comparativas aportan
  dos anclas; son 19 evidencias en el conjunto propio.
- `recall_trayectoria` usa los fragmentos recuperados por las búsquedas reales
  del agente. Puede agregar varias búsquedas y no equivale al recall@5 de una
  sola consulta.
- La tolerancia numérica es 1 % relativo y exige igualdad para cero. Se
  comprueban unidades y conceptos. En comparativas se revisan valores y cambios.
- El extractor reconoce formatos numéricos comunes; los separadores ambiguos
  y números contextuales siguen siendo una limitación. El prompt pide cifras
  sin separadores de miles. El middleware verifica consistencia con los hechos
  consultados, mientras el evaluador verifica el concepto esperado.
- El juez de citas es una evaluación automática, no una garantía de verdad.
  Se guarda su explicación para revisar desacuerdos. Sin juez, el soporte
  semántico queda pendiente y no se cuenta como aprobado.

La ejecución local comprueba implementación y retrieval. Para completar el
informe hay que ejecutar la reescritura y ambos agentes con API, revisar los
resultados y preparar el PDF. No hay resultados finales del LLM simulados.
