# Entorno de Python

El proyecto utiliza Python 3.11 y un entorno virtual local en `.venv`.
Las dependencias se declaran en `requirements.txt`.

Para reconstruir el entorno desde la raiz del proyecto, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip --isolated install --index-url https://pypi.org/simple -r requirements.txt
```

`--isolated` evita que la configuracion de pip del equipo impida consultar
PyPI, como ocurria con `PIP_NO_INDEX` en este entorno.

Para activar el entorno en una terminal:

```powershell
.\.venv\Scripts\Activate.ps1
```

En VS Code, abre `src/Baseline_Agente_10K.ipynb` y selecciona como kernel
el interprete `.venv\Scripts\python.exe`. No hace falta activar una terminal
para usar ese kernel. Ejecuta las celdas del notebook en orden.

El kernel registrado se llama `Python (.venv - Taller 10K)`. Comprueba que
su ruta corresponde a `Taller-B5-T5-NPL`, no a otro taller. La primera celda
muestra un aviso si se esta usando un entorno diferente, sin detenerse
solo por la ruta del interprete. Tras cambiar
el kernel, ejecuta las celdas desde el principio; las variables y la clave
introducida en el kernel anterior no se transfieren al nuevo.

La clave `OPENROUTER_API_KEY` se solicita de forma oculta en el notebook
si no esta definida en el entorno. No se incluye en los archivos del proyecto.

Para comprobar las dependencias instaladas:

```powershell
.\.venv\Scripts\python.exe -m pip check
```

## Evaluar el agente del notebook

La ultima celda de `src/Baseline_Agente_10K.ipynb` valida las 20 preguntas
de `src/golden_set_propio.jsonl` y ejecuta:

```python
resultados_baseline = evaluar(RUTA_EVALUACION, agente_evaluacion=agente)
```

Se utiliza el objeto `agente` creado en esa misma sesion. La evaluacion
no crea otro modelo ni solicita otra clave. Si las funciones ya estaban
cargadas antes de modificarlas, vuelve a ejecutar las celdas que definen
`ejecutar` y `evaluar` antes de ejecutar la ultima celda.

Cada ejecucion guarda una carpeta distinta en `resultados/baseline/`, con
respuestas, trazas, metricas, resumen por familia y una copia del codigo
y las preguntas utilizados. `coste_reportado_usd` recoge el coste que
devuelve OpenRouter; `coste_usd` usa ese valor o, si falta, la estimacion
con los precios configurados. Un coste desconocido permanece vacio.
