import { useState, useEffect } from 'react';

export function useProphetia() {
  const [partidos, setPartidos] = useState([]);
  const [datosPrediccion, setDatosPrediccion] = useState(null);
  const [cargando, setCargando] = useState(false);
  const [ejecutandoScript, setEjecutandoScript] = useState(null);
  const [modelStats, setModelStats] = useState(null);
  const [appStatus, setAppStatus] = useState('loading'); // 'loading' | 'ready' | 'empty_dataset' | 'untrained'

  const cargarDatosGlobales = async () => {
    try {
      // 1. Verificar Partidos (Dataset)
      const resPartidos = await fetch('http://127.0.0.1:8000/api/partidos');
      const dataPartidos = await resPartidos.json();

      if (dataPartidos.status === 'empty_dataset') {
        setPartidos([]);
        setAppStatus('empty_dataset');
        return; // Detenemos aquí si no hay datos
      } else {
        setPartidos(dataPartidos);
      }

      // 2. Verificar Modelo
      const resStats = await fetch('http://127.0.0.1:8000/api/model-stats');
      const dataStats = await resStats.json();

      if (dataStats.status === 'untrained') {
        setModelStats(null);
        setAppStatus('untrained');
      } else {
        setModelStats(dataStats);
        setAppStatus('ready');
      }
    } catch (err) {
      console.error("Error cargando estado global:", err);
    }
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
      
      if (data.status === 'untrained' || data.status === 'empty_dataset') {
        setAppStatus(data.status);
      } else {
        setDatosPrediccion(data);
      }
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
      
      // Tras terminar el script, recargamos el estado global
      await cargarDatosGlobales();
      
    } catch (err) {
      alert(`❌ Error al ejecutar ${nombreScript}. Revisa la consola del backend.`);
    } finally {
      setEjecutandoScript(null);
    }
  };

  return { partidos, datosPrediccion, buscarPrediccion, cargando, ejecutarPipeline, ejecutandoScript, modelStats, appStatus };
}