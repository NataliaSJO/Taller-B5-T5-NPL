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

Cada llamada tiene un tiempo de espera de 180 segundos. Un fallo de conexión
se reintenta una vez, conservando como desconocido el coste de la solicitud
sin respuesta. Si el modelo devuelve una reescritura vacía, se busca con la
consulta original; no se inventa una consulta usando respuestas del golden.
Ambos agentes usan el mismo máximo de 4096 tokens por llamada.

## Desde el notebook

Selecciona `.venv`, configura `EJECUTAR_API=True` y ejecuta las celdas en orden.
`EJECUTAR_RETRIEVAL=True` mide las cuatro variantes del buscador; si conservas
una medición válida y sólo repites la comparación, puedes usar `False`.
Guarda antes de ejecutar «Exportar las definiciones». No hace falta ejecutar
las demostraciones del notebook baseline.

Una respuesta vacía del juez se reintenta una vez; si persiste, se guarda el
diagnóstico y queda como error, nunca como acierto. Las preguntas numéricas
sin ancla textual no requieren juicio semántico de citas opcionales.

## Generar el informe

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-informe.txt
.\.venv\Scripts\python.exe scripts\generar_informe_s2.py --comparacion resultados\s2\mi_ejecucion\comparacion --retrieval resultados\s2\retrieval\20260921T114707Z_78cf7608 --salida resultados\s2\mi_ejecucion\informe
```

Se generan `informe.md` e `informe.pdf` con tablas, ejemplos de respuestas,
criterios, incidencias y límites. Si cambias el buscador o modelo, vuelve a
medir retrieval y utiliza la carpeta nueva en `--retrieval`. El informe marca
como provisional toda comparación incompleta.

Revisa aciertos por familia y las trazas de fallos, no sólo el promedio.
Los costes desconocidos permanecen vacíos; el juez se registra aparte.
La medición aislada del buscador utiliza filtros del golden y no equivale al
recall de todas las búsquedas efectuadas por el agente.

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
