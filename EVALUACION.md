# Ejecutar y revisar la evaluación

Usa Python 3.11 y el entorno `.venv` del proyecto. La implementación visible está
en `src/S2_Robustez_y_Evaluacion_Alumno.ipynb`; sus nueve celdas exportables se
sincronizan con `agente/interfaz.py` mediante la celda «Exportar las definiciones».

## Comprobaciones sin API

Desde la raíz, en PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

La validación local comprueba las 20 preguntas, las seis comparativas y las
19 evidencias contra el corpus. No sustituye al validador oficial del aula.

## Evaluación real con entrada privada de clave

```powershell
.\.venv\Scripts\python.exe scripts\ejecutar_evaluacion_s2.py --salida resultados\s2\mi_ejecucion
```

Elige una carpeta nueva cada vez. Si `OPENROUTER_API_KEY` no está definida,
aparece una ventana con entrada oculta. La clave se mantiene en memoria, no
se escribe en los resultados. La evaluación consume crédito de OpenRouter.

El proceso ejecuta ambas versiones sobre el mismo golden y guarda:

- `estado.json`: iniciando, esperando_clave, evaluando, completa, incompleta o error.
- `ejecucion.log`: progreso y diagnóstico.
- `comparacion/comparacion.csv` y `.md`: tabla general.
- `comparacion/baseline/` y `comparacion/final/`: preguntas, configuración,
  métricas, resumen por familia, respuestas y trazas.

La salida es completa cuando ambas versiones tienen 20 preguntas evaluadas.
Llegar al mensaje `20/20` no garantiza que estén evaluadas: consulta `estado`.
Una respuesta incorrecta evaluada no equivale a un error de ejecución.

El límite de gasto restante de la clave no es el saldo de la cuenta. Ante un
402, consulta el detalle del error. Los rechazos con causa
`openrouter_in_flight_budget` se reintentan hasta tres veces respetando
`Retry-After` numérico (máximo 120 segundos por espera). Los fallos de saldo
o límite permanente no se reintentan automáticamente. Esta política cubre
las llamadas síncronas que utiliza el proyecto.

Cada llamada tiene un tiempo de espera de 180 segundos y cada **pregunta** un
corte de 150 s (`LIMITE_SEGUNDOS_10K`): una pregunta colgada llegó a consumir 62
minutos y se llevó por delante la evaluación entera. Al agotarse, esa pregunta
se cierra con una abstención estructurada y las demás continúan. Un fallo de
conexión se reintenta una vez, conservando como desconocido el coste informado
de la solicitud sin respuesta; junto a él se guarda el coste estimado por tarifa.
Si el modelo devuelve una reescritura vacía, se busca con la consulta original;
no se inventa una consulta usando respuestas del golden. Ambos agentes usan el
mismo máximo de 4096 tokens por llamada. La reescritura usa el modelo auxiliar (por
defecto, el mismo Gemini 3.8 Flash del agente) y el juez tiene su propia variable (por
defecto, Claude Opus 5.5, de otra familia); las dos tareas tienen un tope de 1024
tokens. La comparación del 21-sep usó Flash Lite en ambas: para repetirla hay que
fijar `MODELO_AUX_10K` y `MODELO_JUEZ_10K` (ver `agente/README.md`).

## Desde el notebook

Selecciona `.venv`, configura `EJECUTAR_API=True` y ejecuta las celdas en orden.
`EJECUTAR_RETRIEVAL=True` mide las cuatro variantes del buscador; si conservas
una medición válida y sólo repites la comparación, puedes usar `False`.
Guarda antes de ejecutar «Exportar las definiciones». No hace falta ejecutar
las demostraciones del notebook baseline.

Una respuesta vacía del juez se reintenta una vez; si persiste, se guarda el
diagnóstico y queda como error, nunca como acierto. Las preguntas numéricas
sin ancla textual no requieren juicio semántico de citas opcionales.

## Controles que el golden propio no cubre

Dos comprobaciones que el enunciado menciona y que las 20 preguntas propias no
miden. Se ejecutan por separado y son baratas:

```powershell
python scripts\control_memorizacion.py
python scripts\prueba_huecos_xbrl.py
```

El primero repite las preguntas **sin herramientas ni corpus**: mide cuánto del
acierto vendría del preentrenamiento del modelo en vez del 10-K, que es lo que
el enunciado llama acertar por el camino equivocado. El segundo pregunta por
conceptos que una compañía no reporta —Amazon no publica GrossProfit,
Liabilities ni ResearchAndDevelopmentExpense— donde la respuesta correcta es
decir que no está en el corpus.

## Generar el informe

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-informe.txt
.\.venv\Scripts\python.exe scripts\generar_informe_s2.py --comparacion resultados\s2\mi_ejecucion\comparacion --retrieval resultados\s2\retrieval\20260921T114707Z_78cf7608 --salida resultados\s2\mi_ejecucion\informe
```

Si se ha vuelto a medir sólo una de las dos mitades, `escribir_comparacion(base,
final, carpeta)` reconstruye `comparacion.csv` y `comparacion.md` a partir de las
dos tablas ya guardadas, que es lo que el informe necesita.

El resumen del repositorio se regenera con:

```powershell
python scripts\generar_resumen.py --comparacion <carpeta>\comparacion ^
  --retrieval resultados\s2
etrieval\<marca> ^
  --control resultados\s2\control_memorizacion\<marca> ^
  --huecos resultados\s2\huecos_xbrl\<antes> resultados\s2\huecos_xbrl\<despues> ^
  --repeticion <otra carpeta de comparacion con el mismo baseline>
```

`--repeticion` añade la tabla de varianza: el mismo agente medido dos veces.
Sin ella no se sabe cuánto de una diferencia es ruido.

Se generan `informe.md` e `informe.pdf` con tablas, ejemplos de respuestas,
criterios, incidencias y límites. Si cambias el buscador o modelo, vuelve a
medir retrieval y utiliza la carpeta nueva en `--retrieval`. El informe marca
como provisional toda comparación incompleta.

Revisa aciertos por familia y las trazas de fallos, no sólo el promedio.
El coste informado permanece vacío cuando OpenRouter no lo devuelve, y al lado
queda el estimado por tarifa; el juez se registra aparte.

Con n = 20 los intervalos de confianza rondan ±20 puntos y dos ejecuciones del
mismo código dieron 0,30 y 0,35 de acierto en el baseline. Compara por pregunta
(las filas están en `metricas.csv`), no sólo las medias.
La medición aislada del buscador utiliza filtros del golden y no equivale al
recall de todas las búsquedas efectuadas por el agente: es una cota superior,
porque en producción esos filtros los infiere el agente a partir de la pregunta.

`recall.csv` conserva el recall@5 por variante y `recall_por_k.csv` añade la
curva a k = 1, 3, 5 y 10 para las ocho casillas de la matriz. Sin las dos
mitades de la matriz no se puede atribuir la mejora a la fusión o a la
reescritura.

## Preguntas ciegas

Con la clave configurada y desde la raíz:

```python
from agente.interfaz import responder, evaluar

respuesta = responder("¿Cuáles fueron los ingresos de NVIDIA en FY2025?")
tabla = evaluar("holdout.jsonl")
print(tabla)
print(tabla.attrs["salida"])
```

El holdout se ejecuta cuando se entregue el archivo real. No se sustituyen las
preguntas ciegas por el golden ni se inventa su resultado. La publicación en
GitHub y la entrega en el aula son pasos separados de generar los archivos.
