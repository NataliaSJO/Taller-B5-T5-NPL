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
### Configuración por entorno

Todo se fija antes de importar el módulo. El agente y la reescritura van por
defecto a Gemini 3.8 Flash y el juez a Claude Opus 5.5, de otra familia para que el
agente no se juzgue a sí mismo. El juez ya no hereda el modelo auxiliar: cambiar
`MODELO_AUX_10K` no cambia el juez.

| Variable | Por defecto | Para qué |
| --- | --- | --- |
| `MODELO_10K` | `openrouter:google/gemini-3.8-flash` | cerebro del agente |
| `MODELO_AUX_10K` | `openrouter:google/gemini-3.8-flash` | reescritura de consultas |
| `MODELO_JUEZ_10K` | `openrouter:anthropic/claude-opus-5.5` | juez de citas, de otra familia que el agente |
| `MAX_TOKENS_10K` | 4096 | salida del agente |
| `MAX_TOKENS_AUX_10K` / `MAX_TOKENS_JUEZ_10K` | 1024 / 1024 | salida de las tareas cortas (incluye el razonamiento del modelo) |
| `LIMITE_SEGUNDOS_10K` | 150 | corte por pregunta |
| `USAR_ENCABEZADOS_10K` | 1 | tercera señal de ranking con la etiqueta derivada |

La reescritura usa el mismo modelo que el agente, con un tope de 1024 tokens porque
Gemini 3.8 Flash razona antes de contestar (con 128 las consultas salían cortadas).
El juez se configura aparte a propósito: si fuera el mismo modelo que el agente, se
estaría midiendo con un juez que tiende a favorecer sus propias salidas. Si se trunca
la salida estructurada, aumenta `MAX_TOKENS_10K` y vuelve a medir **ambos** agentes
con la misma configuración.

Para repetir la comparación del 21-sep (55 % -> 95 %), que usó Gemini 3.5 Flash Lite
para reescribir **y** para juzgar, no basta con `MODELO_AUX_10K`: hay que fijar las
dos variables, y para ser exactos también los topes de entonces:

```bash
export MODELO_AUX_10K=openrouter:google/gemini-3.5-flash-lite
export MODELO_JUEZ_10K=openrouter:google/gemini-3.5-flash-lite
export MAX_TOKENS_AUX_10K=128 MAX_TOKENS_JUEZ_10K=256
```

Con solo `MODELO_AUX_10K` se mediría con el juez Opus, mucho más exigente (con él,
el baseline baja del 55 al 30 %). Hay una diferencia que no se deshace por entorno:
el juez pide ahora la salida con `json_schema` en lugar de `function_calling`,
porque Claude Opus 5.5 rechaza lo segundo; con Flash Lite funcionan los dos.

## Etiquetas derivadas del corpus

El corpus entregado ya trae compañía, ejercicio e item, y esos filtros ya están
explotados. Lo que faltaba era la estructura interna de cada sección, así que se
deriva una etiqueta más —el encabezado de subsección vigente en cada fragmento—
en `corpus/derivado/etiquetas.parquet`, con su sello SHA-256 del corpus al lado.

- Se construye sola la primera vez y se regenera si cambia `chunks.jsonl`, de
  modo que un clon limpio funciona sin pasos manuales.
- **No se toca `chunks.jsonl`**: su hash se verifica contra `MANIFEST.md` y los
  `chunk_id` tienen que seguir siendo los mismos para que las citas se verifiquen.
- Se usa en dos sitios: como tercera lista del RRF (`hibrido(..., encabezados=True)`)
  y en la cabecera de cada fragmento que ve el modelo.
- Se mide como variante propia (`hibrido_enc`, `hibrido_enc_reescrito`): con la
  consulta en español **empeora**; con la consulta reescrita al inglés es la mejor
  a k=3 y k=5. Ambos números están en `recall_por_k.csv`.

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
  Sobre magnitudes de 1e11 ese 1 % son ±1.000 millones: es tolerancia de
  redondeo, no de exactitud.
- `cita` es el evaluador del enunciado: la cita existe, es literal, se mostró por
  una herramienta y el juez confirma que respalda lo que se afirma. `cita_ancla`
  es la medida estricta adicional: la cita cae sobre la frase exacta del golden.
  Se reportan las dos porque miden cosas distintas —la segunda es recuperación—
  y confundirlas penalizaba respuestas correctas que citaban otro pasaje válido.
- `coste_usd` es lo que informa OpenRouter y queda vacío cuando una llamada falla;
  `coste_estimado_usd` lo rellena con la tarifa publicada. Sin esa columna la
  media está sesgada a la baja, porque las llamadas que se pierden son las caras.
- Los tres avisos de la ejecución medida eran **falsos positivos**: el extractor
  confundía «aproximadamente» con «April» y leía un número fantasma en
  «128.528 mil millones». Está corregido y con prueba, pero **después** de medir:
  el coste de la tabla incluye tres correcciones evitables, así que es una cota
  superior. Las tres respuestas fueron correctas igualmente.
- `avisos_guardrail` cuenta las veces que el middleware obligó a corregir. Menos
  avisos sólo es mejor si no se pierden errores reales: el borrador que provocó
  cada aviso se guarda en `respuestas.jsonl` para poder medir su precisión.
- El extractor reconoce formatos numéricos comunes; los separadores ambiguos
  siguen siendo una limitación. El prompt pide cifras sin separadores de miles.
  Ya no se leen como cifras los días de una fecha, las viñetas numeradas ni los
  años: eran el grueso de los avisos falsos, y cada aviso falso costaba un turno
  de modelo entero.
- El middleware acepta un número si es un hecho XBRL consultado, una derivada
  declarada (variación interanual del mismo concepto, márgenes entre hechos del
  mismo ejercicio) o si aparece **literalmente** en el texto recuperado. Esto
  último exige coincidencia exacta, no tolerancia: citar un dato del informe no
  es inventarlo, pero tampoco puede ser una puerta abierta.
- Antes de verificar cifras se repara la cita: los 10-K traen numeración de
  página y «Table of Contents» dentro del párrafo, el modelo cose por encima y la
  cita deja de ser literal. Se recorta al tramo que sí existe en el fragmento, sin
  pedir nada al modelo y sin coste.
- El juez de citas es una evaluación automática, no una garantía de verdad.
  Se guarda su explicación para revisar desacuerdos. Sin juez, el soporte
  semántico queda pendiente y no se cuenta como aprobado.

La ejecución local comprueba implementación y retrieval. Para completar el
informe hay que ejecutar la reescritura y ambos agentes con API, revisar los
resultados y preparar el PDF. No hay resultados finales del LLM simulados.
