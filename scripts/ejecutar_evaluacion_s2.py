"""Ejecuta la comparación real; solicita la clave en una ventana privada."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import time
import importlib
import urllib.request

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[1]
load_dotenv(RAIZ / ".env")
sys.path.insert(0, str(RAIZ))


def pedir_clave():
    import tkinter as tk
    from tkinter import simpledialog
    ventana = tk.Tk()
    ventana.withdraw()
    ventana.attributes("-topmost", True)
    try:
        return simpledialog.askstring(
            "Evaluación del taller", "Introduce OPENROUTER_API_KEY.\n"
            "Se usará sólo en memoria para evaluar los dos agentes.",
            show="*", parent=ventana)
    finally:
        ventana.destroy()


class Registro:
    def __init__(self, archivo):
        self.archivo = archivo

    def write(self, texto):
        self.archivo.write(texto)
        self.archivo.flush()
        return len(texto)

    def flush(self):
        self.archivo.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, required=True)
    parser.add_argument("--mantener-sesion", action="store_true",
                        help="Mantiene la clave en memoria hasta 30 minutos para reintentos coordinados")
    args = parser.parse_args()
    args.salida.mkdir(parents=True, exist_ok=False)
    with (args.salida / "ejecucion.log").open("w", encoding="utf-8") as log:
        sys.stdout = sys.stderr = Registro(log)
        estado = {"estado": "iniciando"}
        ruta_estado = args.salida / "estado.json"

        def guardar():
            ruta_estado.write_text(json.dumps(estado, ensure_ascii=False, indent=2), encoding="utf-8")

        guardar()
        try:
            if not os.environ.get("OPENROUTER_API_KEY"):
                estado["estado"] = "esperando_clave"
                guardar()
                clave = pedir_clave()
                if not clave or not clave.strip():
                    raise RuntimeError("Entrada de clave cancelada")
                os.environ["OPENROUTER_API_KEY"] = clave.strip()
                del clave
            estado["estado"] = "comprobando_api"
            guardar()
            peticion = urllib.request.Request(
                "https://openrouter.ai/api/v1/key",
                headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]})
            with urllib.request.urlopen(peticion, timeout=30) as respuesta:
                datos = json.load(respuesta)["data"]
            print("Autenticación OpenRouter: correcta.")
            print("Crédito restante del límite de la clave:", datos.get("limit_remaining"))
            from agente import interfaz as s
            print("Validación local:", s.validar_golden())
            print("Modelo:", s.MODELO, "max_tokens:", s.MAX_TOKENS)
            estado["estado"] = "evaluando"
            guardar()
            tabla = s.comparar(salida=args.salida / "comparacion")
            print(tabla.to_string(index=False))
            estado.update(estado="completa" if all(tabla.evaluadas == tabla.preguntas) else "incompleta",
                          comparacion=tabla.attrs["salida"])
            guardar()
            if args.mantener_sesion:
                print("Sesión disponible en memoria para finalizar o repetir la verificación.")
                limite = time.monotonic() + 1800
                control = args.salida / "control.json"
                ultimo = None
                while time.monotonic() < limite:
                    if control.exists():
                        orden = json.loads(control.read_text(encoding="utf-8"))
                        if orden.get("accion") == "terminar":
                            break
                        if orden.get("accion") == "comparar" and orden.get("id") != ultimo:
                            ultimo = orden["id"]
                            s = importlib.reload(s)
                            estado["estado"] = "evaluando"
                            guardar()
                            tabla = s.comparar(salida=args.salida / ("comparacion_" + str(ultimo)))
                            print(tabla.to_string(index=False))
                            estado.update(estado="completa" if all(tabla.evaluadas == tabla.preguntas) else "incompleta",
                                          comparacion=tabla.attrs["salida"])
                            guardar()
                    time.sleep(2)
        except Exception as exc:
            estado.update(estado="error", error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            os.environ.pop("OPENROUTER_API_KEY", None)
            guardar()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()
