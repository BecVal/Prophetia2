// src/App.jsx
import { useState, useMemo, useEffect } from 'react';
import { useProphetia } from './hooks/useProphetia';
import Sidebar from './components/Sidebar';
import MatchDashboard from './components/MatchDashboard';
import ModelStatsPanel from './components/ModelStatsPanel';
import Header from './components/Header';
import { X, Swords, AlertTriangle, Info } from 'lucide-react';

export default function App() {
  const { partidos, datosPrediccion, buscarPrediccion, cargando, ejecutarPipeline, ejecutandoScript, modelStats } = useProphetia();
  const [menuAbierto, setMenuAbierto] = useState(false);
  const [modalH2H, setModalH2H] = useState(false);

  // CONTROL ROBUSTO DEL DARK MODE
  const [isDarkMode, setIsDarkMode] = useState(() => {
    if (typeof window !== 'undefined') {
      const savedTheme = localStorage.getItem('theme');
      if (savedTheme) return savedTheme === 'dark';
      return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    return true;
  });

  useEffect(() => {
    const root = window.document.documentElement;
    if (isDarkMode) {
      root.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      root.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [isDarkMode]);

  // ESTADOS DEL EMBUDO DE SELECCIÓN
  const [ligaSel, setLigaSel] = useState('');
  const [equipo1, setEquipo1] = useState('');
  const [equipo2, setEquipo2] = useState('');

  // Lógica de Ligas
  const ligasUnicas = useMemo(() => [...new Set(partidos.map(p => p.liga))], [partidos]);

  // Estadísticas Contextuales del Paso 1
  const statsContextuales = useMemo(() => {
    if (!ligaSel) return null;
    const partidosLiga = partidos.filter(p => p.liga === ligaSel);
    const equipos = new Set();
    const seasons = new Set();
    
    partidosLiga.forEach(p => {
      equipos.add(p.local);
      equipos.add(p.visita);
      seasons.add(p.fecha.substring(0, 4));
    });
    
    const sortedSeasons = [...seasons].sort();
    const seasonRange = sortedSeasons.length > 1 
      ? `${sortedSeasons[0]} - ${sortedSeasons[sortedSeasons.length - 1]}` 
      : sortedSeasons[0] || 'N/A';

    return {
      totalEquipos: equipos.size,
      totalPartidos: partidosLiga.length,
      temporadas: seasonRange
    };
  }, [ligaSel, partidos]);

  // Lógica del Paso 2: Equipos de la liga seleccionada
  const equiposEnLiga = useMemo(() => {
    if (!ligaSel) return [];
    const equipos = new Set();
    partidos.filter(p => p.liga === ligaSel).forEach(p => {
      equipos.add(p.local);
      equipos.add(p.visita);
    });
    return [...equipos].sort();
  }, [ligaSel, partidos]);

  // Lógica del Paso 3: Filtrado inteligente de rivales con historial válido
  const rivalesValidos = useMemo(() => {
    if (!equipo1 || !ligaSel) return [];
    const rivales = new Set();
    partidos.filter(p => p.liga === ligaSel && (p.local === equipo1 || p.visita === equipo1)).forEach(p => {
      if (p.local !== equipo1) rivales.add(p.local);
      if (p.visita !== equipo1) rivales.add(p.visita);
    });
    return [...rivales].sort();
  }, [equipo1, ligaSel, partidos]);

  // Extraer el partido más reciente de la selección final
  const historialH2H = useMemo(() => {
    if (!equipo1 || !equipo2) return [];
    return partidos.filter(p => 
      (p.local === equipo1 && p.visita === equipo2) || 
      (p.local === equipo2 && p.visita === equipo1)
    ).sort((a, b) => new Date(b.fecha) - new Date(a.fecha));
  }, [equipo1, equipo2, partidos]);

  const analizarEnfrentamiento = () => {
    if (historialH2H.length > 0) {
      buscarPrediccion(historialH2H[0].id); // Manda el ID del partido más reciente al modelo
      setModalH2H(false);
    }
  };

  const infoExtraPartido = useMemo(() => {
    if (!datosPrediccion) return null;
    return partidos.find(p => p.id === datosPrediccion.id);
  }, [datosPrediccion, partidos]);

  return (
    <div className="flex h-screen font-sans overflow-hidden transition-colors duration-300 bg-gray-50 dark:bg-pro-bg text-gray-900 dark:text-gray-200">
      
      {menuAbierto && (
        <div 
          className="fixed inset-0 bg-gray-900/60 dark:bg-black/60 z-20 md:hidden backdrop-blur-sm transition-opacity"
          onClick={() => setMenuAbierto(false)}
        />
      )}

      <div className={`fixed md:relative z-30 h-full transition-transform duration-300 ease-in-out ${menuAbierto ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}`}>
        <Sidebar 
          ejecutarPipeline={ejecutarPipeline} 
          ejecutandoScript={ejecutandoScript} 
          stats={modelStats}
          totalPartidos={partidos.length}
        />
      </div>
      
      <main className="flex-1 flex flex-col min-w-0 w-full relative">
        <Header 
          menuAbierto={menuAbierto} 
          setMenuAbierto={setMenuAbierto} 
          setModalH2H={setModalH2H}
          isDarkMode={isDarkMode}
          toggleDarkMode={() => setIsDarkMode(!isDarkMode)}
        />

        <div className="flex-1 overflow-y-auto p-4 md:p-6 custom-scrollbar">
          <div className="max-w-7xl mx-auto pb-10 space-y-4">
            
            {/* El Dashboard maneja el Onboarding Visual si datosPrediccion es null */}
            <MatchDashboard 
               datos={datosPrediccion} 
               infoExtra={infoExtraPartido} 
               cargando={cargando} 
            />
            
            {!cargando && <ModelStatsPanel stats={modelStats} />}
          </div>
        </div>

        {/* MODAL WIZARD: SELECCIÓN DE ENFRENTAMIENTO */}
        {modalH2H && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-gray-900/60 dark:bg-black/80 backdrop-blur-sm animate-fade-in-up">
            <div className="bg-white dark:bg-pro-card border border-gray-200 dark:border-pro-border w-full max-w-2xl rounded-3xl p-6 md:p-8 shadow-2xl relative transition-colors duration-300 flex flex-col max-h-[90vh]">
              
              <button onClick={() => setModalH2H(false)} className="absolute top-6 right-6 text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors">
                <X size={24} />
              </button>
              
              <h2 className="text-xl md:text-2xl font-black text-gray-900 dark:text-white mb-2 flex items-center gap-3 tracking-tight">
                <div className="p-2 bg-blue-100 dark:bg-pro-accent/10 rounded-xl"><Swords className="text-blue-600 dark:text-pro-accent" size={20} /></div> 
                Configurar Predicción
              </h2>
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">Sigue los pasos para inicializar el análisis XGBoost.</p>

              <div className="space-y-6 overflow-y-auto custom-scrollbar pr-2 pb-2">
                
                {/* PASO 1 */}
                <div className="bg-gray-50 dark:bg-[#0B1020] border border-gray-200 dark:border-white/5 rounded-2xl p-5 transition-colors">
                  <label className="flex items-center gap-2 text-sm font-bold text-gray-700 dark:text-gray-300 mb-3">
                    <span className="bg-blue-600 dark:bg-pro-accent text-white w-5 h-5 rounded-full flex items-center justify-center text-xs">1</span>
                    Seleccionar Liga
                  </label>
                  <select 
                    className="w-full bg-white dark:bg-pro-card border border-gray-300 dark:border-pro-border text-gray-900 dark:text-white rounded-xl px-4 py-3 focus:ring-2 focus:ring-blue-500 dark:focus:ring-pro-accent outline-none appearance-none font-medium text-sm transition-colors cursor-pointer shadow-sm"
                    value={ligaSel}
                    onChange={(e) => { setLigaSel(e.target.value); setEquipo1(''); setEquipo2(''); }}
                  >
                    <option value="">-- Buscar competición disponible --</option>
                    {ligasUnicas.map(l => <option key={l} value={l}>{l}</option>)}
                  </select>

                  {/* Estadísticas Contextuales */}
                  {statsContextuales && (
                    <div className="mt-4 grid grid-cols-3 gap-3 border-t border-gray-200 dark:border-white/10 pt-4 animate-fade-in-up">
                      <div className="text-center">
                        <p className="text-[10px] text-gray-500 uppercase tracking-widest font-bold">Equipos</p>
                        <p className="text-lg font-black text-gray-900 dark:text-white">{statsContextuales.totalEquipos}</p>
                      </div>
                      <div className="text-center border-l border-gray-200 dark:border-white/10">
                        <p className="text-[10px] text-gray-500 uppercase tracking-widest font-bold">Partidos</p>
                        <p className="text-lg font-black text-gray-900 dark:text-white">{statsContextuales.totalPartidos}</p>
                      </div>
                      <div className="text-center border-l border-gray-200 dark:border-white/10">
                        <p className="text-[10px] text-gray-500 uppercase tracking-widest font-bold">Temporadas</p>
                        <p className="text-sm font-black text-blue-600 dark:text-pro-accent mt-1">{statsContextuales.temporadas}</p>
                      </div>
                    </div>
                  )}
                </div>

                {/* PASOS 2 Y 3 */}
                <div className={`transition-opacity duration-300 ${ligaSel ? 'opacity-100' : 'opacity-40 pointer-events-none'}`}>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                    
                    {/* PASO 2 */}
                    <div className="bg-blue-50/50 dark:bg-pro-local/5 border border-blue-100 dark:border-pro-local/20 rounded-2xl p-5 transition-colors">
                      <label className="flex items-center gap-2 text-sm font-bold text-blue-700 dark:text-pro-local mb-3">
                        <span className="bg-blue-600 dark:bg-pro-local text-white w-5 h-5 rounded-full flex items-center justify-center text-xs">2</span>
                        Equipo Local
                      </label>
                      <select 
                        className="w-full bg-white dark:bg-pro-card border border-blue-200 dark:border-pro-local/30 text-gray-900 dark:text-white rounded-xl px-4 py-3 focus:ring-2 focus:ring-blue-500 dark:focus:ring-pro-local outline-none appearance-none font-medium text-sm cursor-pointer shadow-sm transition-colors"
                        value={equipo1} onChange={(e) => { setEquipo1(e.target.value); setEquipo2(''); }}
                        disabled={!ligaSel}
                      >
                        <option value="">-- Buscar equipo --</option>
                        {equiposEnLiga.map(eq => <option key={eq} value={eq}>{eq}</option>)}
                      </select>
                    </div>

                    {/* PASO 3 */}
                    <div className="bg-rose-50/50 dark:bg-pro-away/5 border border-rose-100 dark:border-pro-away/20 rounded-2xl p-5 transition-colors">
                      <label className="flex items-center gap-2 text-sm font-bold text-rose-700 dark:text-pro-away mb-3">
                        <span className="bg-rose-600 dark:bg-pro-away text-white w-5 h-5 rounded-full flex items-center justify-center text-xs">3</span>
                        Equipo Visitante
                      </label>
                      <select 
                        className="w-full bg-white dark:bg-pro-card border border-rose-200 dark:border-pro-away/30 text-gray-900 dark:text-white rounded-xl px-4 py-3 focus:ring-2 focus:ring-rose-500 dark:focus:ring-pro-away outline-none appearance-none font-medium text-sm cursor-pointer shadow-sm transition-colors disabled:opacity-50"
                        value={equipo2} onChange={(e) => setEquipo2(e.target.value)}
                        disabled={!equipo1}
                      >
                        <option value="">-- Buscar rival compatible --</option>
                        {rivalesValidos.map(eq => <option key={eq} value={eq}>{eq}</option>)}
                      </select>
                    </div>

                  </div>
                  
                  {/* Mensaje de Ayuda de UX */}
                  <div className="mt-3 flex items-start gap-2 px-2 animate-fade-in-up">
                    <Info size={14} className="text-gray-400 dark:text-gray-500 flex-shrink-0 mt-0.5" />
                    <p className="text-[11px] text-gray-500 dark:text-gray-400 leading-tight">
                      Si el equipo que buscas no aparece en la lista de visitantes, no existen registros suficientes en la base de datos para generar una predicción confiable.
                    </p>
                  </div>
                </div>

                {/* PASO 4 */}
                <div className={`pt-2 transition-opacity duration-300 ${equipo1 && equipo2 ? 'opacity-100' : 'opacity-40 pointer-events-none'}`}>
                  {historialH2H.length === 0 && equipo1 && equipo2 ? (
                    <div className="flex items-center gap-3 bg-rose-50 dark:bg-pro-away/10 text-rose-700 dark:text-pro-away p-4 rounded-xl border border-rose-200 dark:border-pro-away/20">
                      <AlertTriangle size={20} className="flex-shrink-0" />
                      <p className="text-sm font-medium">No hay historial directo entre estos equipos.</p>
                    </div>
                  ) : (
                    <button 
                      onClick={analizarEnfrentamiento}
                      disabled={!equipo1 || !equipo2}
                      className="w-full flex items-center justify-center gap-2 bg-gray-900 hover:bg-black dark:bg-gradient-to-r dark:from-pro-accent dark:to-blue-600 dark:hover:from-blue-500 dark:hover:to-indigo-500 text-white font-bold py-4 rounded-xl transition-all shadow-lg hover:shadow-xl text-sm"
                    >
                      <span className="bg-white/20 text-white w-5 h-5 rounded-full flex items-center justify-center text-[10px] mr-1">4</span>
                      Generar Predicción XGBoost
                    </button>
                  )}
                </div>

              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}