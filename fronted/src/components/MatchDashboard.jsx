import { ShieldAlert, ActivitySquare, Zap, TrendingUp, CalendarClock, Activity, Crosshair } from 'lucide-react';
import TeamBadge from './TeamBadge';

const StatBar = ({ label, leftValue = 0, rightValue = 0, isPercentage = false, inverseColors = false }) => {
  const total = Number(leftValue) + Number(rightValue) || 1;
  const leftPct = (Number(leftValue) / total) * 100;
  const rightPct = (Number(rightValue) / total) * 100;

  const leftColor = inverseColors ? "bg-rose-500 dark:bg-pro-away" : "bg-blue-600 dark:bg-pro-local";
  const rightColor = inverseColors ? "bg-blue-600 dark:bg-pro-local" : "bg-rose-500 dark:bg-pro-away";
  const textLeftColor = inverseColors ? "text-rose-600 dark:text-pro-away" : "text-blue-700 dark:text-pro-local";
  const textRightColor = inverseColors ? "text-blue-700 dark:text-pro-local" : "text-rose-600 dark:text-pro-away";

  return (
    <div className="mb-5 group">
      <div className="flex justify-between items-end mb-2">
        <span className={`font-bold text-lg ${textLeftColor} transition-transform group-hover:scale-110`}>
          {leftValue}{isPercentage ? '%' : ''}
        </span>
        <span className="text-gray-500 dark:text-gray-400 uppercase tracking-widest text-[10px] font-bold group-hover:text-gray-900 dark:group-hover:text-gray-200 transition-colors">{label}</span>
        <span className={`font-bold text-lg ${textRightColor} transition-transform group-hover:scale-110`}>
          {rightValue}{isPercentage ? '%' : ''}
        </span>
      </div>
      <div className="flex w-full h-2.5 bg-gray-200 dark:bg-black/40 rounded-full overflow-hidden border border-gray-300 dark:border-white/5 shadow-inner">
        <div className={`${leftColor} transition-all duration-1000 ease-out`} style={{ width: `${leftPct}%` }}></div>
        <div className={`${rightColor} transition-all duration-1000 ease-out`} style={{ width: `${rightPct}%` }}></div>
      </div>
    </div>
  );
};

const KPICard = ({ title, value, subtext, icon: Icon }) => (
  <div className="bg-white dark:bg-pro-card border border-gray-200 dark:border-pro-border rounded-xl p-4 flex flex-col justify-between hover:border-blue-400 dark:hover:border-pro-accent/50 transition-colors shadow-sm group">
    <div className="flex justify-between items-start mb-2">
      <span className="text-[10px] font-bold text-gray-500 dark:text-gray-400 uppercase tracking-widest group-hover:text-gray-800 dark:group-hover:text-gray-300 transition-colors">{title}</span>
      {Icon && <Icon size={14} className="text-blue-600 dark:text-pro-accent transition-transform group-hover:scale-110" />}
    </div>
    <div className="flex items-end justify-between mt-1">
      <span className="text-2xl font-black text-gray-900 dark:text-white font-mono">{value}</span>
      {subtext && <span className="text-xs font-bold text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300 transition-colors">{subtext}</span>}
    </div>
  </div>
);

export default function MatchDashboard({ datos, cargando }) {
  if (cargando) {
    return (
      <div className="flex-1 flex justify-center items-center h-[60vh] animate-fade-in-up">
        <div className="text-blue-600 dark:text-pro-accent flex flex-col items-center gap-6">
          <div className="relative">
            <div className="absolute inset-0 border-4 border-blue-500/30 dark:border-pro-accent/30 rounded-full animate-ping"></div>
            <ActivitySquare size={56} className="animate-spin relative z-10" />
          </div>
          <p className="text-sm font-bold tracking-widest uppercase bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-pro-accent dark:to-blue-500 bg-clip-text text-transparent">
            Procesando Modelo XGBoost...
          </p>
        </div>
      </div>
    );
  }

  if (!datos) {
    return (
      <div className="flex-1 flex justify-center items-center h-[60vh] text-gray-400 flex-col gap-4 animate-fade-in-up">
        <ShieldAlert size={80} className="opacity-20 mb-4 text-gray-400 dark:text-gray-500" />
        <h2 className="text-2xl font-bold text-gray-700 dark:text-white/50 tracking-wide">Dashboard Inactivo</h2>
        <p className="text-sm text-center max-w-md text-gray-500">Utiliza el configurador superior para seleccionar un enfrentamiento histórico. El pipeline de ML calculará las estadísticas al instante.</p>
      </div>
    );
  }

  const totalXG = (Number(datos.metricas?.xG?.local || 0) + Number(datos.metricas?.xG?.visitante || 0)).toFixed(2);
  const totalTiros = Number(datos.metricas?.tiros?.local || 0) + Number(datos.metricas?.tiros?.visitante || 0);

  return (
    <div className="space-y-6 pb-2">
      
      <div className="bg-white dark:bg-pro-card border border-gray-200 dark:border-pro-border rounded-2xl md:rounded-[2rem] p-6 md:p-10 shadow-sm relative overflow-hidden animate-fade-in-up transition-colors duration-300">
        
        <div className="absolute top-0 left-0 w-full h-full pointer-events-none opacity-20 dark:opacity-100">
          <div className="absolute -top-32 -left-32 w-96 h-96 bg-blue-400 dark:bg-pro-local/10 rounded-full blur-3xl"></div>
          <div className="absolute -bottom-32 -right-32 w-96 h-96 bg-rose-400 dark:bg-pro-away/10 rounded-full blur-3xl"></div>
        </div>

        <div className="flex justify-between items-center text-center relative z-10 mb-10">
          <div className="flex-1 flex flex-col items-center group cursor-default">
            <TeamBadge teamName={datos.equipos.local} colorShadow="rgba(37, 99, 235, 0.2)" />
            <h3 className="text-lg md:text-2xl font-bold text-gray-900 dark:text-white tracking-tight">{datos.equipos.local}</h3>
            {datos.marcador_real?.local !== undefined && (
               <p className="text-3xl md:text-5xl font-black text-blue-700 dark:text-pro-local mt-2 font-mono tracking-tighter">{datos.marcador_real.local}</p>
            )}
          </div>

          <div className="px-2 md:px-8 flex flex-col items-center">
            <span className="px-3 py-1 bg-gray-100 dark:bg-white/5 text-gray-700 dark:text-gray-400 text-[10px] font-bold rounded-full mb-4 uppercase tracking-widest border border-gray-300 dark:border-white/10 shadow-sm">
              Histórico H2H
            </span>
            <span className="text-gray-400 dark:text-gray-600 font-black text-xl md:text-2xl italic tracking-widest">VS</span>
          </div>

          <div className="flex-1 flex flex-col items-center group cursor-default">
            <TeamBadge teamName={datos.equipos.visitante} colorShadow="rgba(225, 29, 72, 0.2)" />
            <h3 className="text-lg md:text-2xl font-bold text-gray-900 dark:text-white tracking-tight">{datos.equipos.visitante}</h3>
            {datos.marcador_real?.visitante !== undefined && (
               <p className="text-3xl md:text-5xl font-black text-rose-600 dark:text-pro-away mt-2 font-mono tracking-tighter">{datos.marcador_real.visitante}</p>
            )}
          </div>
        </div>

        <div className="relative z-10 bg-gray-100 dark:bg-[#0B1020] border border-gray-300 dark:border-pro-border rounded-xl p-5 shadow-inner">
          <div className="flex justify-between items-end mb-4">
            <h3 className="text-xs uppercase tracking-widest text-gray-700 dark:text-gray-400 font-bold flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-blue-600 dark:bg-pro-accent animate-pulse"></div>
              Predicción XGBoost
            </h3>
            <span className="text-[10px] text-gray-500 dark:text-gray-400 font-mono flex items-center gap-1">
              <Zap size={12} className="text-amber-500 dark:text-yellow-400" /> ML Engine Active
            </span>
          </div>
          
          <div className="flex w-full h-4 rounded-full overflow-hidden shadow-inner bg-gray-300 dark:bg-gray-800">
            <div className="bg-blue-600 dark:bg-pro-local transition-all duration-1000 ease-out flex items-center justify-center text-[10px] font-bold text-white dark:text-black" style={{ width: `${datos.probabilidades.local}%` }}></div>
            <div className="bg-gray-400 dark:bg-gray-500 transition-all duration-1000 ease-out" style={{ width: `${datos.probabilidades.empate}%` }}></div>
            <div className="bg-rose-500 dark:bg-pro-away transition-all duration-1000 ease-out" style={{ width: `${datos.probabilidades.visitante}%` }}></div>
          </div>
          
          <div className="flex justify-between mt-3 text-sm font-black font-mono">
            <span className="text-blue-700 dark:text-pro-local flex flex-col md:flex-row items-center gap-1">
              <span className="text-[10px] uppercase text-gray-600 dark:text-gray-500 md:mr-2">Local</span> 
              {datos.probabilidades.local}%
            </span>
            <span className="text-gray-700 dark:text-gray-400 flex flex-col md:flex-row items-center gap-1">
              <span className="text-[10px] uppercase text-gray-500 dark:text-gray-400 md:mr-2">Empate</span> 
              {datos.probabilidades.empate}%
            </span>
            <span className="text-rose-700 dark:text-pro-away flex flex-col md:flex-row items-center gap-1">
              <span className="text-[10px] uppercase text-gray-600 dark:text-gray-500 md:mr-2">Visita</span> 
              {datos.probabilidades.visitante}%
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 animate-fade-in-up" style={{animationDelay: '0.1s'}}>
        <KPICard title="Total xG Partido" value={totalXG} subtext="Goles Esp." icon={Crosshair} />
        <KPICard title="Total Tiros" value={totalTiros} subtext="Local + Visita" icon={Activity} />
        <KPICard title="Posesión Local" value={`${datos.metricas?.posesion?.local || 0}%`} subtext="Control" icon={ActivitySquare} />
        <KPICard title="Faltas Totales" value={(Number(datos.metricas?.faltas?.local || 0) + Number(datos.metricas?.faltas?.visitante || 0))} subtext="Agresividad" icon={ShieldAlert} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-fade-in-up" style={{animationDelay: '0.2s'}}>
        
        <div className="lg:col-span-1 bg-white dark:bg-gradient-to-br dark:from-[#161B22] dark:to-[#0D1117] border border-gray-200 dark:border-white/5 rounded-2xl p-6 shadow-sm flex flex-col gap-6 transition-colors group/panel hover:border-gray-300 dark:hover:border-pro-accent/30">
          <h2 className="text-xs uppercase tracking-widest text-gray-600 dark:text-gray-400 font-bold flex items-center gap-2 border-b border-gray-100 dark:border-white/10 pb-3">
            <TrendingUp size={14} className="text-blue-600 dark:text-pro-accent" /> Key Insights ML
          </h2>

          <div className="flex items-center gap-4 group">
            <div className="p-3 bg-blue-50 dark:bg-pro-local/10 rounded-xl text-blue-600 dark:text-pro-local transition-colors group-hover:bg-blue-100 dark:group-hover:bg-pro-local/20"><TrendingUp size={20} /></div>
            <div>
              <p className="text-[10px] text-gray-500 uppercase tracking-widest font-bold mb-1 group-hover:text-gray-800 dark:group-hover:text-gray-300 transition-colors">Rating ELO Local</p>
              <p className="text-xl font-black text-gray-900 dark:text-white tracking-tight">
                {datos.insights?.elo_local || "N/A"} <span className="text-xs font-medium text-gray-500 dark:text-gray-400">pts</span>
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4 group">
            <div className="p-3 bg-indigo-50 dark:bg-pro-accent/10 rounded-xl text-indigo-600 dark:text-pro-accent transition-colors group-hover:bg-indigo-100 dark:group-hover:bg-pro-accent/20"><Zap size={20} /></div>
            <div>
              <p className="text-[10px] text-gray-500 uppercase tracking-widest font-bold mb-1 group-hover:text-gray-800 dark:group-hover:text-gray-300 transition-colors">Fuerza Ataque Rel.</p>
              <p className="text-xl font-black text-gray-900 dark:text-white tracking-tight">
                {datos.insights?.poder_ataque_relativo || "N/A"}<span className="text-indigo-600 dark:text-pro-accent ml-1">x</span>
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4 group">
            <div className="p-3 bg-rose-50 dark:bg-pro-away/10 rounded-xl text-rose-600 dark:text-pro-away transition-colors group-hover:bg-rose-100 dark:group-hover:bg-pro-away/20"><CalendarClock size={20} /></div>
            <div>
              <p className="text-[10px] text-gray-500 uppercase tracking-widest font-bold mb-1 group-hover:text-gray-800 dark:group-hover:text-gray-300 transition-colors">Fatiga Local</p>
              <p className="text-xl font-black text-gray-900 dark:text-white tracking-tight">
                {datos.insights?.descanso_local || 0} <span className="text-xs font-medium text-gray-500 dark:text-gray-400">días rest.</span>
              </p>
            </div>
          </div>
        </div>

        <div className="lg:col-span-2 flex flex-col gap-6">
          
          <div className="bg-white dark:bg-pro-card border border-gray-200 dark:border-pro-border rounded-2xl p-6 shadow-sm transition-colors hover:border-gray-300 dark:hover:border-pro-accent/30">
            <h2 className="text-gray-900 dark:text-white font-bold text-sm mb-6 flex items-center gap-2 uppercase tracking-wide">
              <ActivitySquare size={16} className="text-blue-600 dark:text-pro-local" /> Peligro y Creación
            </h2>
            <StatBar label="Goles Esperados (xG)" leftValue={datos.metricas?.xG?.local ?? "0.0"} rightValue={datos.metricas?.xG?.visitante ?? "0.0"} />
            <StatBar label="Tiros a Puerta" leftValue={datos.metricas?.tiros?.local ?? "0"} rightValue={datos.metricas?.tiros?.visitante ?? "0"} />
            <StatBar label="Posesión" leftValue={datos.metricas?.posesion?.local ?? "50"} rightValue={datos.metricas?.posesion?.visitante ?? "50"} isPercentage={true} />
            <StatBar label="Precisión de Pases" leftValue={datos.metricas?.pases_precisos?.local ?? "0"} rightValue={datos.metricas?.pases_precisos?.visitante ?? "0"} isPercentage={true} />
          </div>

          <div className="bg-white dark:bg-pro-card border border-gray-200 dark:border-pro-border rounded-2xl p-6 shadow-sm transition-colors hover:border-gray-300 dark:hover:border-pro-accent/30">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-gray-900 dark:text-white font-bold text-sm flex items-center gap-2 uppercase tracking-wide">
                <ShieldAlert size={16} className="text-rose-600 dark:text-pro-away" /> Disciplina
              </h2>
            </div>
            <StatBar label="Tarjetas Amarillas" leftValue={datos.metricas?.tarjetas_amarillas?.local ?? "0"} rightValue={datos.metricas?.tarjetas_amarillas?.visitante ?? "0"} inverseColors={true} />
            <StatBar label="Faltas Cometidas" leftValue={datos.metricas?.faltas?.local ?? "0"} rightValue={datos.metricas?.faltas?.visitante ?? "0"} inverseColors={true} />
          </div>

        </div>
      </div>
    </div>
  );
}