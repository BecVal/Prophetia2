import { ShieldAlert, ActivitySquare, Zap, TrendingUp, CalendarClock, Activity, Crosshair, Swords, BarChart2, Database, GitCommit, BrainCircuit } from 'lucide-react';
import TeamBadge from './TeamBadge';

// --- COMPONENTES MODULARES (StatBar y KPICard se mantienen igual) ---
const StatBar = ({ label, leftValue = 0, rightValue = 0, isPercentage = false, inverseColors = false }) => {
  const total = Number(leftValue) + Number(rightValue) || 1;
  const leftPct = (Number(leftValue) / total) * 100;
  const rightPct = (Number(rightValue) / total) * 100;

  const leftColor = inverseColors ? "bg-rose-500 dark:bg-pro-away" : "bg-blue-600 dark:bg-pro-local";
  const rightColor = inverseColors ? "bg-blue-600 dark:bg-pro-local" : "bg-rose-500 dark:bg-pro-away";
  const textLeftColor = inverseColors ? "text-rose-600 dark:text-pro-away" : "text-blue-700 dark:text-pro-local";
  const textRightColor = inverseColors ? "text-blue-700 dark:text-pro-local" : "text-rose-600 dark:text-pro-away";

  return (
    <div className="mb-4 group">
      <div className="flex justify-between items-end mb-1.5">
        <span className={`font-bold text-base ${textLeftColor}`}>{leftValue}{isPercentage ? '%' : ''}</span>
        <span className="text-gray-500 dark:text-gray-400 uppercase tracking-widest text-[9px] font-bold">{label}</span>
        <span className={`font-bold text-base ${textRightColor}`}>{rightValue}{isPercentage ? '%' : ''}</span>
      </div>
      <div className="flex w-full h-2 bg-gray-200 dark:bg-black/40 rounded-full overflow-hidden shadow-inner">
        <div className={`${leftColor} transition-all duration-1000 ease-out`} style={{ width: `${leftPct}%` }}></div>
        <div className={`${rightColor} transition-all duration-1000 ease-out`} style={{ width: `${rightPct}%` }}></div>
      </div>
    </div>
  );
};

const KPICard = ({ title, value, subtext, icon: Icon }) => (
  <div className="bg-white dark:bg-pro-card border border-gray-200 dark:border-pro-border rounded-xl p-4 flex flex-col justify-between hover:border-blue-400 dark:hover:border-pro-accent/50 transition-colors shadow-sm">
    <div className="flex justify-between items-start mb-1">
      <span className="text-[9px] font-bold text-gray-500 dark:text-gray-400 uppercase tracking-widest">{title}</span>
      {Icon && <Icon size={14} className="text-blue-600 dark:text-pro-accent" />}
    </div>
    <div className="flex items-end justify-between mt-1">
      <span className="text-xl font-black text-gray-900 dark:text-white font-mono">{value}</span>
      {subtext && <span className="text-[10px] font-bold text-gray-400">{subtext}</span>}
    </div>
  </div>
);


export default function MatchDashboard({ datos, infoExtra, cargando, appStatus }) {
  
  // ESTADO 1: CARGANDO PREDICCIÓN
  if (cargando) {
    return (
      <div className="flex-1 flex justify-center items-center h-[60vh] animate-fade-in-up">
        <div className="text-blue-600 dark:text-pro-accent flex flex-col items-center gap-6">
          <div className="relative">
            <div className="absolute inset-0 border-4 border-blue-500/30 dark:border-pro-accent/30 rounded-full animate-ping"></div>
            <ActivitySquare size={48} className="animate-spin relative z-10" />
          </div>
          <p className="text-xs font-bold tracking-widest uppercase bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-pro-accent dark:to-blue-500 bg-clip-text text-transparent">
            Procesando Nodos XGBoost...
          </p>
        </div>
      </div>
    );
  }

  // ESTADO 2: SISTEMA SIN INICIALIZAR (Falta Dataset o Modelo)
  if (appStatus === 'empty_dataset' || appStatus === 'untrained') {
    return (
      <div className="flex-1 flex flex-col justify-center items-center h-[70vh] animate-fade-in-up px-4">
        <div className="w-16 h-16 bg-rose-50 dark:bg-rose-500/10 text-rose-600 dark:text-rose-400 rounded-2xl flex items-center justify-center mb-6 shadow-sm border border-rose-100 dark:border-rose-500/20">
          <ShieldAlert size={32} />
        </div>
        <h2 className="text-2xl md:text-3xl font-black text-gray-900 dark:text-white mb-2 tracking-tight">Inicialización Requerida</h2>
        <p className="text-gray-500 dark:text-gray-400 text-sm mb-10 text-center max-w-md">
          {appStatus === 'empty_dataset' 
            ? "El dataset procesado no existe. Ejecuta los pasos del Pipeline ML en el panel lateral para prepararlo." 
            : "El dataset está listo, pero el modelo XGBoost no ha sido entrenado. Ejecuta el paso final del Pipeline."}
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 w-full max-w-5xl">
           <div className={`p-5 rounded-2xl border ${appStatus === 'empty_dataset' ? 'bg-blue-50 dark:bg-pro-accent/10 border-blue-200 dark:border-pro-accent/30' : 'bg-white dark:bg-pro-card border-gray-200 dark:border-pro-border opacity-50'}`}>
              <Database size={20} className="text-blue-600 dark:text-pro-accent mb-3" />
              <h3 className="font-bold text-gray-900 dark:text-white text-sm mb-1">1. Ingesta</h3>
              <p className="text-xs text-gray-500 dark:text-gray-400">Descarga datos crudos.</p>
           </div>
           <div className={`p-5 rounded-2xl border ${appStatus === 'empty_dataset' ? 'bg-white dark:bg-pro-card border-gray-200 dark:border-pro-border' : 'bg-white dark:bg-pro-card border-gray-200 dark:border-pro-border opacity-50'}`}>
              <GitCommit size={20} className="text-gray-700 dark:text-gray-300 mb-3" />
              <h3 className="font-bold text-gray-900 dark:text-white text-sm mb-1">2. Adapter</h3>
              <p className="text-xs text-gray-500 dark:text-gray-400">Estandariza columnas.</p>
           </div>
           <div className={`p-5 rounded-2xl border ${appStatus === 'empty_dataset' ? 'bg-white dark:bg-pro-card border-gray-200 dark:border-pro-border' : 'bg-white dark:bg-pro-card border-gray-200 dark:border-pro-border opacity-50'}`}>
              <Activity size={20} className="text-gray-700 dark:text-gray-300 mb-3" />
              <h3 className="font-bold text-gray-900 dark:text-white text-sm mb-1">3. Features</h3>
              <p className="text-xs text-gray-500 dark:text-gray-400">Genera métricas xG/ELO.</p>
           </div>
           <div className={`p-5 rounded-2xl border ${appStatus === 'untrained' ? 'bg-blue-50 dark:bg-pro-accent/10 border-blue-200 dark:border-pro-accent/30 shadow-md' : 'bg-white dark:bg-pro-card border-gray-200 dark:border-pro-border opacity-50'}`}>
              <BrainCircuit size={20} className={`${appStatus === 'untrained' ? 'text-blue-600 dark:text-pro-accent' : 'text-gray-700 dark:text-gray-300'} mb-3`} />
              <h3 className="font-bold text-gray-900 dark:text-white text-sm mb-1">4. Entrenar Modelo</h3>
              <p className="text-xs text-gray-500 dark:text-gray-400">Genera archivo .pkl.</p>
           </div>
        </div>
      </div>
    );
  }

  // ESTADO 3: ONBOARDING VISUAL (Listo, pero sin partido seleccionado)
  if (!datos) {
    return (
      <div className="flex-1 flex flex-col justify-center items-center h-[60vh] animate-fade-in-up">
        <h2 className="text-2xl md:text-3xl font-black text-gray-900 dark:text-white mb-2 tracking-tight">Bienvenido a Prophetia</h2>
        <p className="text-gray-500 dark:text-gray-400 text-sm mb-12 text-center max-w-md">El motor predictivo está listo. Utiliza el botón superior para configurar un enfrentamiento.</p>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-4xl px-4">
           <div className="bg-white dark:bg-pro-card border border-gray-200 dark:border-pro-border p-6 rounded-2xl shadow-sm text-center relative overflow-hidden group">
              <div className="w-12 h-12 bg-blue-50 dark:bg-pro-accent/10 rounded-full flex items-center justify-center mx-auto mb-4 text-blue-600 dark:text-pro-accent group-hover:scale-110 transition-transform"><Swords size={20} /></div>
              <h3 className="font-bold text-gray-900 dark:text-white mb-2">1. Seleccionar Partido</h3>
              <p className="text-xs text-gray-500 dark:text-gray-400">Filtra por competición, local y visitante.</p>
           </div>
           <div className="bg-white dark:bg-pro-card border border-gray-200 dark:border-pro-border p-6 rounded-2xl shadow-sm text-center relative overflow-hidden group">
              <div className="w-12 h-12 bg-indigo-50 dark:bg-indigo-500/10 rounded-full flex items-center justify-center mx-auto mb-4 text-indigo-600 dark:text-indigo-400 group-hover:scale-110 transition-transform"><Zap size={20} /></div>
              <h3 className="font-bold text-gray-900 dark:text-white mb-2">2. ML Engine</h3>
              <p className="text-xs text-gray-500 dark:text-gray-400">XGBoost procesará el historial al instante.</p>
           </div>
           <div className="bg-white dark:bg-pro-card border border-gray-200 dark:border-pro-border p-6 rounded-2xl shadow-sm text-center relative overflow-hidden group">
              <div className="w-12 h-12 bg-rose-50 dark:bg-pro-away/10 rounded-full flex items-center justify-center mx-auto mb-4 text-rose-600 dark:text-pro-away group-hover:scale-110 transition-transform"><BarChart2 size={20} /></div>
              <h3 className="font-bold text-gray-900 dark:text-white mb-2">3. Insights Activos</h3>
              <p className="text-xs text-gray-500 dark:text-gray-400">Obtén métricas de xG, posesión y ventaja táctica.</p>
           </div>
        </div>
      </div>
    );
  }

  const totalXG = (Number(datos.metricas?.xG?.local || 0) + Number(datos.metricas?.xG?.visitante || 0)).toFixed(2);
  const totalTiros = Number(datos.metricas?.tiros?.local || 0) + Number(datos.metricas?.tiros?.visitante || 0);

  // ESTADO 4: DASHBOARD ACTIVO (Se mantiene el código del dashboard original aquí)
  return (
    <div className="space-y-4 pb-2 animate-fade-in-up">
      
      {/* BARRA CONTEXTUAL SUPERIOR */}
      <div className="bg-white dark:bg-pro-card border border-gray-200 dark:border-pro-border rounded-xl p-3 flex flex-wrap justify-between items-center shadow-sm text-xs md:text-sm">
         <div className="flex flex-wrap items-center gap-3 text-gray-600 dark:text-gray-300">
            <span className="font-bold text-gray-900 dark:text-white flex items-center gap-2"><Swords size={14}/> {infoExtra?.local} vs {infoExtra?.visita}</span>
            <span className="hidden md:block w-px h-4 bg-gray-300 dark:bg-white/10"></span>
            <span className="font-medium text-gray-500">{infoExtra?.liga || 'Liga N/A'}</span>
            <span className="hidden md:block w-px h-4 bg-gray-300 dark:bg-white/10"></span>
            <span className="font-medium text-gray-500">{infoExtra?.fecha || 'Fecha N/A'}</span>
         </div>
         <div className="flex items-center gap-2 mt-2 md:mt-0 font-mono text-[10px] md:text-xs text-blue-600 dark:text-pro-accent font-bold bg-blue-50 dark:bg-pro-accent/10 px-2 py-1 rounded">
            <BrainCircuit size={12}/> XGBoost Classifier Pro
         </div>
      </div>

      {/* HERO CARD COMPACTADO */}
      <div className="bg-white dark:bg-pro-card border border-gray-200 dark:border-pro-border rounded-2xl p-6 shadow-sm relative overflow-hidden transition-colors">
        
        <div className="absolute top-0 left-0 w-full h-full pointer-events-none opacity-20 dark:opacity-100">
          <div className="absolute -top-32 -left-32 w-64 h-64 bg-blue-400 dark:bg-pro-local/10 rounded-full blur-3xl"></div>
          <div className="absolute -bottom-32 -right-32 w-64 h-64 bg-rose-400 dark:bg-pro-away/10 rounded-full blur-3xl"></div>
        </div>

        <div className="flex justify-between items-center text-center relative z-10 mb-6">
          <div className="flex-1 flex flex-col items-center">
            <TeamBadge teamName={datos.equipos.local} size="w-16 h-16 md:w-20 md:h-20" colorShadow="rgba(37, 99, 235, 0.2)" />
            <h3 className="text-base md:text-xl font-bold text-gray-900 dark:text-white tracking-tight leading-none mt-2">{datos.equipos.local}</h3>
            {datos.marcador_real?.local !== undefined && (
               <p className="text-3xl font-black text-blue-700 dark:text-pro-local font-mono mt-1">{datos.marcador_real.local}</p>
            )}
          </div>

          <div className="px-4 flex flex-col items-center">
            <span className="text-gray-300 dark:text-gray-600 font-black text-lg italic tracking-widest">VS</span>
          </div>

          <div className="flex-1 flex flex-col items-center">
            <TeamBadge teamName={datos.equipos.visitante} size="w-16 h-16 md:w-20 md:h-20" colorShadow="rgba(225, 29, 72, 0.2)" />
            <h3 className="text-base md:text-xl font-bold text-gray-900 dark:text-white tracking-tight leading-none mt-2">{datos.equipos.visitante}</h3>
            {datos.marcador_real?.visitante !== undefined && (
               <p className="text-3xl font-black text-rose-600 dark:text-pro-away font-mono mt-1">{datos.marcador_real.visitante}</p>
            )}
          </div>
        </div>

        {/* PROBABILIDADES */}
        <div className="relative z-10 bg-gray-50 dark:bg-[#0B1020] border border-gray-200 dark:border-pro-border rounded-xl p-4 shadow-inner">
          <div className="flex w-full h-3.5 rounded-full overflow-hidden shadow-inner bg-gray-300 dark:bg-gray-800">
            <div className="bg-blue-600 dark:bg-pro-local transition-all duration-1000 ease-out" style={{ width: `${datos.probabilidades.local}%` }}></div>
            <div className="bg-gray-400 dark:bg-gray-500 transition-all duration-1000 ease-out" style={{ width: `${datos.probabilidades.empate}%` }}></div>
            <div className="bg-rose-500 dark:bg-pro-away transition-all duration-1000 ease-out" style={{ width: `${datos.probabilidades.visitante}%` }}></div>
          </div>
          
          <div className="flex justify-between mt-2 text-xs font-black font-mono">
            <span className="text-blue-700 dark:text-pro-local">LOCAL {datos.probabilidades.local}%</span>
            <span className="text-gray-600 dark:text-gray-400">EMPATE {datos.probabilidades.empate}%</span>
            <span className="text-rose-600 dark:text-pro-away">VISITA {datos.probabilidades.visitante}%</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KPICard title="Total xG" value={totalXG} subtext="Goles Esp." icon={Crosshair} />
        <KPICard title="Total Tiros" value={totalTiros} subtext="Ambos eq." icon={Activity} />
        <KPICard title="Posesión Loc" value={`${datos.metricas?.posesion?.local || 0}%`} subtext="Control" icon={ActivitySquare} />
        <KPICard title="Faltas Tot" value={(Number(datos.metricas?.faltas?.local || 0) + Number(datos.metricas?.faltas?.visitante || 0))} subtext="Agresividad" icon={ShieldAlert} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        
        <div className="lg:col-span-1 bg-white dark:bg-pro-card border border-gray-200 dark:border-pro-border rounded-2xl p-5 shadow-sm flex flex-col gap-4">
          <h2 className="text-[10px] uppercase tracking-widest text-gray-500 font-bold border-b border-gray-100 dark:border-white/5 pb-2">Insights ML</h2>
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-50 dark:bg-pro-local/10 rounded-lg text-blue-600 dark:text-pro-local"><TrendingUp size={16} /></div>
            <div>
              <p className="text-[9px] text-gray-500 uppercase tracking-widest font-bold">ELO Local</p>
              <p className="text-base font-black text-gray-900 dark:text-white leading-none">{datos.insights?.elo_local || "N/A"} <span className="text-[10px] font-medium text-gray-400">pts</span></p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="p-2 bg-indigo-50 dark:bg-pro-accent/10 rounded-lg text-indigo-600 dark:text-pro-accent"><Zap size={16} /></div>
            <div>
              <p className="text-[9px] text-gray-500 uppercase tracking-widest font-bold">Ataque Relativo</p>
              <p className="text-base font-black text-gray-900 dark:text-white leading-none">{datos.insights?.poder_ataque_relativo || "N/A"}<span className="text-indigo-600 dark:text-pro-accent ml-0.5">x</span></p>
            </div>
          </div>
        </div>

        <div className="lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-pro-card border border-gray-200 dark:border-pro-border rounded-2xl p-5 shadow-sm">
            <h2 className="text-gray-900 dark:text-white font-bold text-xs mb-4 flex items-center gap-2 uppercase"><ActivitySquare size={14} className="text-blue-600 dark:text-pro-local" /> Creación</h2>
            <StatBar label="Goles Esperados (xG)" leftValue={datos.metricas?.xG?.local ?? "0.0"} rightValue={datos.metricas?.xG?.visitante ?? "0.0"} />
            <StatBar label="Tiros a Puerta" leftValue={datos.metricas?.tiros?.local ?? "0"} rightValue={datos.metricas?.tiros?.visitante ?? "0"} />
          </div>

          <div className="bg-white dark:bg-pro-card border border-gray-200 dark:border-pro-border rounded-2xl p-5 shadow-sm">
            <h2 className="text-gray-900 dark:text-white font-bold text-xs mb-4 flex items-center gap-2 uppercase"><ShieldAlert size={14} className="text-rose-600 dark:text-pro-away" /> Disciplina</h2>
            <StatBar label="Tarjetas Amarillas" leftValue={datos.metricas?.tarjetas_amarillas?.local ?? "0"} rightValue={datos.metricas?.tarjetas_amarillas?.visitante ?? "0"} inverseColors={true} />
            <StatBar label="Faltas Cometidas" leftValue={datos.metricas?.faltas?.local ?? "0"} rightValue={datos.metricas?.faltas?.visitante ?? "0"} inverseColors={true} />
          </div>
        </div>

      </div>
    </div>
  );
}