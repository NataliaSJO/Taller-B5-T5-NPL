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

    def test_extraer_cifras_ignora_fechas_y_vinetas(self):
        # Los dos falsos positivos medidos: el dia de la fecha de cierre y la lista.
        self.assertEqual(s.extraer_cifras("Cerro el 28 de septiembre de 2024 con 100 millones USD."),
                         [(100000000.0, False)])
        self.assertEqual(s.extraer_cifras("Riesgos:\n1. Competencia.\n2. Regulacion."), [])
        self.assertEqual(s.extraer_cifras("September 28, 2024 fue el cierre."), [])
        self.assertEqual(s.extraer_cifras("El margen fue del 49,0 %."), [(49.0, True)])
        # "aproximadamente" empieza por "apr": no puede confundirse con April.
        leidas = s.extraer_cifras("128528000000 USD (aproximadamente 128.528 mil millones)")
        self.assertEqual(len(leidas), 2)
        for valor, es_porcentaje in leidas:  # la escala hace decimal al separador
            self.assertFalse(es_porcentaje)
            self.assertAlmostEqual(valor, 128528000000.0, delta=1)
        self.assertEqual(s.extraer_cifras("El cierre fue en April 2025 y no hay cifra."), [])

    def test_numero_citado_del_texto_no_es_invento(self):
        r = s.RespuestaFinanciera(respuesta="Data Center crecio un 142 % segun el informe.",
                                  fuente="texto")
        base = traza(r)
        self.assertTrue(s.desajustes_cifras(base))  # sin contexto sigue siendo un invento
        con_texto = {**base, "messages": base["messages"] + [ToolMessage(
            content="[NVDA-2025-7-0007] Data Center revenue for fiscal year 2025 was up 142%.",
            tool_call_id="t2", name="search_filings")]}
        self.assertEqual(s.desajustes_cifras(con_texto), [])

    def test_margen_entre_dos_hechos_del_mismo_ejercicio(self):
        hechos = {("AAPL", 2024, "GrossProfit"): (180683000000.0, "USD"),
                  ("AAPL", 2024, "Revenues"): (391035000000.0, "USD")}
        _, porcentajes = s.valores_admisibles(hechos)
        self.assertTrue(any(abs(p - 46.2) < 0.5 for p in porcentajes))

    def test_cita_reparada_sobre_ruido_de_pagina(self):
        # Caso real de propio-011: la cita cosia por encima de "10. Table of Contents".
        fragmento = s.datos()[1]["GOOGL-2024-1A-0001"]
        rota = ("We generate a significant portion of our revenues from advertising. "
                "Reduced spending by advertisers, a loss of partners, or new and existing "
                "technologies that block ads online and/or affect our ability to personalize "
                "ads could harm our business. We generated more than 75% of total revenues "
                "from online advertising in 2024. Many of our advertisers, companies that "
                "distribute our products and services, digital publishers, and content "
                "providers can terminate their contracts with us at any time. These partners "
                "may not continue to do business with us if we do not create more value (such "
                "as increased numbers of users or customers, new sales leads, increased brand "
                "awareness, or more effective monetization) than their available alternatives.")
        self.assertNotIn(s.miax_s2.normalizar(rota), s.miax_s2.normalizar(fragmento["texto"]))
        r = s.RespuestaFinanciera(respuesta="Depende de la publicidad.", fuente="texto",
                                  ticker="GOOGL", ejercicio=2024,
                                  cita=rota, chunk_id=fragmento["chunk_id"])
        resultado = {"structured_response": r, "messages": [ToolMessage(
            content=s.formatear([{**fragmento, "puntuacion": 0.1}]),
            tool_call_id="t1", name="search_filings")]}
        corregida, cambios = s.reparar_citas(resultado)
        self.assertTrue(cambios)
        self.assertIn(s.miax_s2.normalizar(corregida.cita),
                      s.miax_s2.normalizar(fragmento["texto"]))

    def test_cita_valida_no_exige_el_ancla_del_golden(self):
        item = next(g for g in self.golden if g["id"] == "propio-008")
        otro = next(f for f in s.datos()[1].values()
                    if f["ticker"] == item["ticker"] and f["fiscal_year"] == item["fiscal_year"]
                    and f["item"] == item["item_esperado"]
                    and s.miax_s2.normalizar(item["ancla_texto"]) not in s.miax_s2.normalizar(f["texto"]))
        frase = [x for x in otro["texto"].split(". ") if len(x) > 80][0] + "."
        r = {"respuesta": "Depende de proveedores externos.", "fuente": "texto",
             "cita": frase, "chunk_id": otro["chunk_id"]}
        resultado = {"structured_response": r, "juicio_cita": {"respalda": True},
                     "messages": [ToolMessage(content=s.formatear([{**otro, "puntuacion": 0.1}]),
                                              tool_call_id="t1", name="search_filings")]}
        self.assertTrue(s.cita_correcta(item, resultado))   # criterio del enunciado
        self.assertFalse(s.cita_ancla(item, resultado))     # medida estricta, informativa

    def test_etiquetas_cubren_el_corpus(self):
        etiquetas = s.etiquetas()
        self.assertEqual(len(etiquetas), len(s.datos()[1]))
        self.assertEqual(etiquetas["NVDA-2025-7-0001"], "Demand and Supply")

    def test_limite_de_tiempo_cierra_la_pregunta(self):
        # Una pregunta colgada no debe llevarse por delante el resto de la evaluación.
        modelo = ModeloGuion(respuestas=[salida_modelo(respuesta(), "r1")])
        agente = s.create_agent(model=modelo, tools=[s.get_xbrl_fact],
            response_format=ToolStrategy(s.RespuestaFinanciera),
            middleware=[s.limite_de_tiempo])
        with patch.object(s, "_INICIO_PREGUNTA", s.time.perf_counter() - s.LIMITE_SEGUNDOS - 1):
            r = agente.invoke({"messages": [HumanMessage(content="Activos Apple 2024")]})
        self.assertEqual(r["structured_response"].fuente, "ninguna")
        self.assertEqual(modelo._posicion, 0)  # ni siquiera se llega a llamar al modelo

    def test_coste_estimado_rellena_el_desconocido(self):
        registro = s.RegistroLLM()
        registro.llamadas = [{"coste_usd": None, "modelo": "google/gemini-3.8-flash",
                              "tokens": {"input_tokens": 1000, "output_tokens": 1000}}]
        resumen = registro.resumen()
        self.assertIsNone(resumen["coste_usd"])
        self.assertAlmostEqual(resumen["coste_estimado_usd"], (0.75 + 3.75) / 1e3, places=6)

    def test_reintento_de_firma_de_pensamiento(self):
        # Gemini responde 400 si los bloques de razonamiento no vuelven intactos.
        exc = RuntimeError("Provider returned error")
        exc.raw_response = httpx.Response(400, json={"error": {"metadata": {
            "raw": "Gemini models require OpenRouter reasoning details to be preserved. "
                   "Upstream error: Corrupted thought signature."}}})
        modelo = s.ModeloOpenRouter(model="prueba", api_key="prueba", client=Mock())
        esperado = ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])
        mensajes = [AIMessage(content="", additional_kwargs={"reasoning_details": [{"x": 1}]})]
        vistos = []

        def responder(messages, **kwargs):
            vistos.append(messages)
            if len(vistos) == 1:
                raise exc
            return esperado

        with patch.object(s.ChatOpenRouter, "_generate", side_effect=responder):
            self.assertIs(modelo._generate(mensajes), esperado)
        self.assertEqual(len(vistos), 2)
        self.assertIn("reasoning_details", vistos[0][0].additional_kwargs)
        self.assertNotIn("reasoning_details", vistos[1][0].additional_kwargs)

    def test_busqueda_reintenta_sin_el_item(self):
        # Un item mal inferido devolvia vacio y gastaba un turno entero del modelo.
        with patch.object(s, "reescribir", side_effect=lambda q, c=None: q):
            salida = s.search_filings.invoke({
                "query": "foreign exchange risk hedging", "ticker": "AAPL",
                "fiscal_year": 2024, "item": "1A", "k": 3})
        self.assertIn("AAPL-2024-1A", salida)
        with patch.object(s, "reescribir", side_effect=lambda q, c=None: q),              patch.object(s, "hibrido", side_effect=[[], [{"chunk_id": "AAPL-2024-7A-0001",
                 "ticker": "AAPL", "fiscal_year": 2024, "item": "7A", "texto": "t",
                 "puntuacion": 0.1}]]):
            salida = s.search_filings.invoke({
                "query": "x", "ticker": "AAPL", "fiscal_year": 2024, "item": "1A", "k": 3})
        self.assertIn("Sin resultados en el Item 1A", salida)
        self.assertIn("AAPL-2024-7A-0001", salida)

    def test_prueba_pareada_detecta_empate(self):
        import pandas as pd
        a = pd.DataFrame({"id": ["a", "b", "c"], "acierto": [True, False, True]})
        b = pd.DataFrame({"id": ["a", "b", "c"], "acierto": [True, True, False]})
        r = s.prueba_pareada(a, b)
        self.assertEqual((r["solo_acierta_final"], r["solo_acierta_baseline"]), (1, 1))
        self.assertEqual(r["p_valor_mcnemar"], 1.0)

    def test_respuesta_en_prosa_se_reconduce_al_esquema(self):
        # propio-011 se perdio asi: el modelo contesto en prosa y el grafo acabo sin esquema.
        modelo = ModeloGuion(respuestas=[AIMessage(content="La respuesta es que si."),
                                        salida_modelo(respuesta(), "r1")])
        agente = s.create_agent(model=modelo, tools=[s.get_xbrl_fact],
            response_format=ToolStrategy(s.RespuestaFinanciera),
            middleware=[s.exigir_respuesta_estructurada])
        r = agente.invoke({"messages": [HumanMessage(content="Activos Apple 2024")]})
        self.assertEqual(r["structured_response"].cifra, 364980000000)
        self.assertEqual(modelo._posicion, 2)
        self.assertEqual(sum(s.AVISO_ESQUEMA in str(m.content) for m in r["messages"]), 1)

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

    def test_agente_final_limita_el_numero_de_llamadas_al_modelo(self):
        modelo = ModeloGuion(respuestas=[AIMessage(content="", tool_calls=[{
            "name": "get_xbrl_fact", "id": f"t{i}", "args": {
                "ticker": "AMZN", "fiscal_year": 2025, "concept": "GrossProfit"}}])
            for i in range(s.LIMITE_MODELO + 2)])
        s.construir_agente.cache_clear()
        try:
            with patch.object(s, "modelo", return_value=modelo):
                agente = s.construir_agente()
            r = agente.invoke({"messages": [HumanMessage(content="Margen bruto Amazon")]},
                config={"configurable": {"thread_id": "test-bucle"}, "recursion_limit": 100})
            self.assertEqual(modelo._posicion, s.LIMITE_MODELO)
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
