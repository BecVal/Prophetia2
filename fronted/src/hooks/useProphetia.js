import { useState, useEffect } from 'react';

export function useProphetia() {
  const [partidos, setPartidos] = useState([]);
  const [datosPrediccion, setDatosPrediccion] = useState(null);
  const [cargando, setCargando] = useState(false);
  const [ejecutandoScript, setEjecutandoScript] = useState(null);
  const [modelStats, setModelStats] = useState(null);

  const cargarDatosGlobales = () => {
    fetch('http://127.0.0.1:8000/api/partidos')
      .then(res => res.json())
      .then(data => setPartidos(data))
      .catch(err => console.error("Error cargando partidos:", err));

    fetch('http://127.0.0.1:8000/api/model-stats')
      .then(res => res.json())
      .then(data => setModelStats(data))
      .catch(err => console.error("Error stats:", err));
  };

  useEffect(() => {
    cargarDatosGlobales();
  }, []);

  const buscarPrediccion = async (id) => {
    if (!id) return setDatosPrediccion(null);
    setCargando(true);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/prediccion/${id}`);
      const data = await res.json();
      setDatosPrediccion(data);
    } catch (err) {
      console.error("Error buscando predicción:", err);
    } finally {
      setCargando(false);
    }
  };

  const ejecutarPipeline = async (endpoint, nombreScript) => {
    setEjecutandoScript(nombreScript);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/run/${endpoint}`, { method: 'POST' });
      if (!res.ok) throw new Error("Fallo en la ejecución");
      alert(`✅ ${nombreScript} ejecutado con éxito.`);
      if (endpoint === 'train' || endpoint === 'features') cargarDatosGlobales();
    } catch (err) {
      alert(`❌ Error al ejecutar ${nombreScript}. Revisa la consola de Python.`);
    } finally {
      setEjecutandoScript(null);
    }
  };

  return { partidos, datosPrediccion, buscarPrediccion, cargando, ejecutarPipeline, ejecutandoScript, modelStats };
}