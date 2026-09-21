"""Pruebas locales: python -m unittest discover -s tests -v (sin API)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import httpx

from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from agente import interfaz as s


def respuesta(cifra=364980000000.0):
    return s.RespuestaFinanciera(respuesta=f"Los activos fueron {cifra} USD.", cifra=cifra,
                                unidad="USD", ticker="AAPL", ejercicio=2024, fuente="xbrl")


def traza(r=None, year=2024, concept="Assets"):
    args = {"ticker": "AAPL", "fiscal_year": year, "concept": concept}
    return {"structured_response": r or respuesta(), "messages": [
        HumanMessage(content="¿Cuántos activos tuvo Apple en FY2024?"),
        AIMessage(content="", tool_calls=[{"name": "get_xbrl_fact", "args": args, "id": "t1"}]),
        ToolMessage(content=s.get_xbrl_fact.invoke(args), tool_call_id="t1", name="get_xbrl_fact"),
    ]}


class ModeloGuion(BaseChatModel):
    """Modelo local que permite probar el grafo real, sin simular el middleware."""
    respuestas: list
    _posicion: int = PrivateAttr(default=0)

    @property
    def _llm_type(self):
        return "guion-local"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        r = self.respuestas[min(self._posicion, len(self.respuestas) - 1)]
        self._posicion += 1
        return ChatResult(generations=[ChatGeneration(message=r)])


def salida_modelo(r, identificador):
    return AIMessage(content="", tool_calls=[{
        "id": identificador, "name": "RespuestaFinanciera", "args": r.model_dump()}])


class PruebasS2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden = s.leer_jsonl(s.RUTA_GOLDEN)

    def test_notebook_adaptado_y_sincronizado(self):
        """Detecta si una copia antigua vuelve a sobrescribir el notebook de entrega."""
        ruta = s.RAIZ / "src/S2_Robustez_y_Evaluacion_Alumno.ipynb"
        notebook = json.loads(ruta.read_text(encoding="utf-8"))
        exportables = [c for c in notebook["cells"]
                      if "s2-exportar" in c.get("metadata", {}).get("tags", [])]
        self.assertEqual(len(exportables), 9, "Se ha perdido la versión adaptada del notebook")
        codigo = "\n\n".join("".join(c["source"]).rstrip() for c in exportables)
        self.assertTrue((s.RAIZ / "agente/interfaz.py").read_text(encoding="utf-8").endswith(codigo + "\n"))
        self.assertNotIn("TODO (alumno)", codigo)
        self.assertEqual(notebook["metadata"]["kernelspec"]["language"], "python")

    def test_juez_reintenta_respuesta_vacia(self):
        juez = Mock()
        juez.invoke.side_effect = [
            {"parsed": None, "raw": AIMessage(content=""), "parsing_error": None},
            {"parsed": s.JuicioCita(respalda=True, motivo="Evidencia suficiente"),
             "raw": AIMessage(content=""), "parsing_error": None},
        ]
        with patch.object(s, "modelo") as modelo:
            modelo.return_value.with_structured_output.return_value = juez
            juicio = s.juzgar_cita(self.golden[7], traza())
        self.assertTrue(juicio["respalda"])
        self.assertEqual(len(juicio["intentos"]), 2)

    def test_reintento_solo_reserva_temporal(self):
        exc = RuntimeError("402")
        exc.raw_response = httpx.Response(402, headers={"Retry-After": "1"}, json={
            "error": {"metadata": {"limit_source": "openrouter_in_flight_budget"}}})
        modelo = s.ModeloOpenRouter(model="prueba", api_key="prueba", client=Mock())
        esperado = ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])
        with patch.object(s.ChatOpenRouter, "_generate", side_effect=[exc, esperado]) as generar, patch.object(s.time, "sleep") as dormir:
            self.assertIs(modelo._generate([]), esperado)
            self.assertEqual(generar.call_count, 2)
            dormir.assert_called_once_with(1.0)
        exc.raw_response = httpx.Response(402, json={
            "error": {"metadata": {"limit_source": "openrouter_credits"}}})
        with patch.object(s.ChatOpenRouter, "_generate", side_effect=exc) as generar:
            with self.assertRaises(RuntimeError):
                modelo._generate([])
            generar.assert_called_once()

    def test_reescritura_vacia_conserva_consulta(self):
        with patch.object(s, "modelo") as modelo:
            modelo.return_value.invoke.return_value = AIMessage(content="")
            self.assertEqual(s.reescribir("Apple risks FY2025"), "Apple risks FY2025")
            modelo.return_value.invoke.assert_called_once()

    def test_reintento_red_no_oculta_coste_desconocido(self):
        modelo = s.ModeloOpenRouter(model="prueba", api_key="prueba", client=Mock())
        esperado = ChatResult(generations=[ChatGeneration(message=AIMessage(
            content="ok", response_metadata={"cost": 0.01}))])
        with patch.object(s.ChatOpenRouter, "_generate", side_effect=[httpx.ReadTimeout("timeout"), esperado]), patch.object(s.time, "sleep"):
            resultado = modelo._generate([])
        from langchain_core.outputs import LLMResult
        registro = s.RegistroLLM()
        registro.on_llm_end(LLMResult(generations=[resultado.generations]))
        self.assertIsNone(registro.resumen()["coste_usd"])
        self.assertEqual(registro.resumen()["llamadas_modelo"], 2)

    def test_juez_vacio_no_se_convierte_en_acierto(self):
        juez = Mock()
        juez.invoke.return_value = {"parsed": None, "raw": AIMessage(content=""),
                                   "parsing_error": None}
        resultado = traza()
        with patch.object(s, "modelo") as modelo:
            modelo.return_value.with_structured_output.return_value = juez
            with self.assertRaisesRegex(RuntimeError, "dos intentos"):
                s.juzgar_cita(self.golden[7], resultado)
        self.assertEqual(len(resultado["diagnostico_juez"]["intentos"]), 2)

    def test_numerica_no_necesita_juez_de_citas(self):
        resultado = traza()
        resultado["structured_response"].cita = "Cita opcional"
        resultado["structured_response"].chunk_id = "no-existe"
        with tempfile.TemporaryDirectory(dir=s.RAIZ) as directorio:
            ruta = Path(directorio) / "pregunta.jsonl"
            ruta.write_text(json.dumps(self.golden[0]), encoding="utf-8")
            with patch.object(s, "juzgar_cita") as juez:
                tabla = s.evaluar(ruta, funcion_responder=lambda _: resultado,
                                  salida=Path(directorio) / "salida")
                juez.assert_not_called()
            self.assertEqual(tabla.estado.tolist(), ["evaluado"])
            self.assertTrue(tabla.acierto.iloc[0])

    def test_contrato_y_decimales(self):
        self.assertEqual(set(t.name for t in s.HERRAMIENTAS), s.NOMBRES_TOOLS)
        self.assertEqual(list(s.search_filings.args), ["query", "ticker", "fiscal_year", "item", "k"])
        self.assertIn("7.46 USD/shares", s.get_xbrl_fact.invoke({
            "ticker": "AAPL", "fiscal_year": 2025, "concept": "EarningsPerShareDiluted"}))

    def test_golden_real(self):
        self.assertEqual(s.validar_golden()["anclas"], 19)

    def test_extraer_cifras(self):
        self.assertEqual(s.extraer_cifras("FY2025 Item 7A 10-K: 7,46 USD; -2.5 millones; 20,40 %"),
                         [(7.46, False), (-2500000, False), (20.4, True)])
        self.assertEqual(s.extraer_cifras("139.514.000.000 USD y 139,514,000,000 USD"),
                         [(139514000000, False), (139514000000, False)])

    def test_numero_y_ejercicio(self):
        self.assertTrue(s.cifra_coincide_xbrl(self.golden[0], traza()))
        self.assertTrue(s.uso_la_tool_correcta(self.golden[0], traza()))
        self.assertFalse(s.uso_la_tool_correcta(self.golden[0], traza(year=2025)))
        self.assertEqual(s.desajustes_cifras(traza()), [])
        self.assertTrue(s.desajustes_cifras(traza(respuesta(123))))
        r = respuesta()
        r.respuesta = "El importe fue 123 USD."
        self.assertTrue(s.desajustes_cifras(traza(r)))

    def test_eps_unidad_equivalente(self):
        item = self.golden[6]
        r = s.RespuestaFinanciera(respuesta="7.46 USD por acción", cifra=7.46,
            unidad="USD/acción", ticker="AAPL", ejercicio=2025, fuente="xbrl")
        self.assertTrue(s.cifra_coincide_xbrl(item, {"structured_response": r}))

    def test_cita_real_no_implica_soporte(self):
        item = self.golden[7]
        f = s.datos()[1][item["chunk_id_esperado"]]
        r = {"structured_response": {"respuesta": "Una afirmación incorrecta",
             "cita": item["ancla_texto"], "chunk_id": f["chunk_id"]},
             "messages": [ToolMessage(content=f["texto"], name="search_filings", tool_call_id="t")]}
        self.assertIsNone(s.cita_correcta(item, r))
        r["juicio_cita"] = {"respalda": False}
        self.assertFalse(s.cita_correcta(item, r))
        r["structured_response"]["cita"] += " INVENTADO"
        r["juicio_cita"] = {"respalda": True}
        self.assertFalse(s.cita_correcta(item, r))

    def test_comparativa_necesita_ambos_anos(self):
        g = self.golden[-1]
        r = {"structured_response": {"respuesta": g["respuesta_esperada"],
             "cifra": g["cifra_esperada"], "unidad": g["unidad"],
             "ticker": g["ticker"], "ejercicio": g["fiscal_year"]}}
        self.assertTrue(s.cifra_coincide_xbrl(g, r))
        r["structured_response"]["respuesta"] = "FY2025: 139514000000 USD."
        self.assertFalse(s.cifra_coincide_xbrl(g, r))

    def test_middleware_corrige_en_grafo(self):
        primera = traza()["messages"][1]
        modelo = ModeloGuion(respuestas=[primera, salida_modelo(respuesta(123), "r1"),
                                        salida_modelo(respuesta(), "r2")])
        agente = s.create_agent(model=modelo, tools=[s.get_xbrl_fact],
            response_format=ToolStrategy(s.RespuestaFinanciera),
            middleware=[s.verificar_cifras_contra_xbrl])
        r = agente.invoke({"messages": [HumanMessage(content="Activos Apple 2024")]})
        self.assertEqual(r["structured_response"].cifra, 364980000000)
        self.assertEqual(modelo._posicion, 3)
        self.assertEqual(sum(s.MARCA in str(m.content) for m in r["messages"]), 1)

    def test_middleware_se_abstiene_si_insiste(self):
        modelo = ModeloGuion(respuestas=[salida_modelo(respuesta(123), "r1"),
                                        salida_modelo(respuesta(456), "r2")])
        agente = s.create_agent(model=modelo, tools=[s.get_xbrl_fact],
            response_format=ToolStrategy(s.RespuestaFinanciera),
            middleware=[s.verificar_cifras_contra_xbrl])
        r = agente.invoke({"messages": [HumanMessage(content="Activos Apple 2024")]})
        self.assertEqual(r["structured_response"].fuente, "ninguna")
        self.assertIsNone(r["structured_response"].cifra)
        self.assertEqual(modelo._posicion, 2)

    def test_carga_baseline_propio_sin_demos(self):
        with patch.object(s, "modelo", return_value=ModeloGuion(respuestas=[])), patch.object(s, "create_agent", side_effect=lambda **kw: kw):
            base = s.cargar_baseline_propio()
        self.assertEqual({t.name for t in base["tools"]}, s.NOMBRES_TOOLS)
        # El código original redondea EPS; el final lo corrige sin cambiar el baseline.
        herramienta = next(t for t in base["tools"] if t.name == "get_xbrl_fact")
        self.assertIn("= 7 USD/shares", herramienta.invoke({
            "ticker": "AAPL", "fiscal_year": 2025, "concept": "EarningsPerShareDiluted"}))

    def test_agente_final_limita_a_ocho_herramientas(self):
        peticiones = [{"name": "get_xbrl_fact", "id": f"t{i}", "args": {
            "ticker": "AAPL", "fiscal_year": 2024, "concept": "Assets"}} for i in range(9)]
        modelo = ModeloGuion(respuestas=[AIMessage(content="", tool_calls=peticiones),
                                        salida_modelo(respuesta(), "final")])
        s.construir_agente.cache_clear()
        try:
            with patch.object(s, "modelo", return_value=modelo):
                agente = s.construir_agente()
            r = agente.invoke({"messages": [HumanMessage(content="Activos Apple 2024")]},
                              config={"configurable": {"thread_id": "test-limite"}})
            herramientas = [m for m in r["messages"] if isinstance(m, ToolMessage)
                            and m.name == "get_xbrl_fact"]
            self.assertEqual(sum(m.status == "success" for m in herramientas), 8)
            self.assertEqual(sum(m.status == "error" for m in herramientas), 1)
        finally:
            s.construir_agente.cache_clear()

    def test_agente_final_limita_modelo_a_diez(self):
        modelo = ModeloGuion(respuestas=[AIMessage(content="", tool_calls=[{
            "name": "get_xbrl_fact", "id": f"t{i}", "args": {
                "ticker": "AMZN", "fiscal_year": 2025, "concept": "GrossProfit"}}])
            for i in range(12)])
        s.construir_agente.cache_clear()
        try:
            with patch.object(s, "modelo", return_value=modelo):
                agente = s.construir_agente()
            r = agente.invoke({"messages": [HumanMessage(content="Margen bruto Amazon")]},
                config={"configurable": {"thread_id": "test-bucle"}, "recursion_limit": 100})
            self.assertEqual(modelo._posicion, 10)
            self.assertIsNone(r.get("structured_response"))
        finally:
            s.construir_agente.cache_clear()

    def test_evaluacion_guarda_errores_y_no_fabrica_coste(self):
        def falla(pregunta):
            raise RuntimeError("402: insufficient credits")
        with tempfile.TemporaryDirectory(dir=s.RAIZ) as directorio:
            ruta = Path(directorio) / "preguntas.jsonl"
            ruta.write_text("\n".join(json.dumps(g) for g in self.golden[:2]), encoding="utf-8")
            tabla = s.evaluar(ruta, funcion_responder=falla, salida=Path(directorio) / "salida", juzgar=False)
            self.assertEqual(tabla.estado.tolist(), ["error", "no_intentado"])
            self.assertEqual(int(tabla.acierto.sum()), 0)
            self.assertTrue(tabla.coste_usd.isna().all())
            self.assertTrue((Path(tabla.attrs["salida"]) / "respuestas.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
