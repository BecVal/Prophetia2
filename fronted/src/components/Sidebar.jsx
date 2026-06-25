import { useState, useEffect } from 'react';
import { Database, Activity, GitCommit, BrainCircuit, Loader2, TrendingUp, BarChart2, ChevronLeft, ChevronRight, CheckCircle2 } from 'lucide-react';

export default function Sidebar({ ejecutarPipeline, ejecutandoScript, stats, totalPartidos }) {
  const [isCollapsed, setIsCollapsed] = useState(() => localStorage.getItem('prophetia_sidebar') === 'true');
  const [pipelineStates, setPipelineStates] = useState({});
  const [prevScript, setPrevScript] = useState(null);

  // Guardar estado
  useEffect(() => {
    localStorage.setItem('prophetia_sidebar', isCollapsed);
  }, [isCollapsed]);

  // Detector visual de Estados del Pipeline
  useEffect(() => {
    if (ejecutandoScript) {
      setPipelineStates(prev => ({ ...prev, [ejecutandoScript]: 'running' }));
      setPrevScript(ejecutandoScript);
    } else if (prevScript) {
      setPipelineStates(prev => ({ ...prev, [prevScript]: 'completed' }));
      setPrevScript(null);
    }
  }, [ejecutandoScript]);

  const RenderButton = ({ label, icon: Icon, endpoint }) => {
    const status = pipelineStates[label] || 'pending';
    
    return (
      <button 
        onClick={() => ejecutarPipeline(endpoint, label)}
        disabled={ejecutandoScript !== null}
        className={`group relative flex items-center justify-between w-full px-3 py-2.5 rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed border ${
          status === 'running' ? 'bg-blue-50 dark:bg-pro-accent/10 border-blue-200 dark:border-pro-accent/20' : 'border-transparent hover:bg-gray-100 dark:hover:bg-white/5'
        }`}
      >
        <div className={`flex items-center gap-3 ${isCollapsed ? 'mx-auto' : ''}`}>
          {status === 'running' ? (
             <Loader2 size={18} className="animate-spin text-blue-600 dark:text-pro-accent" />
          ) : status === 'completed' ? (
             <CheckCircle2 size={18} className="text-green-500 dark:text-pro-success" />
          ) : (
             <Icon size={18} className="text-gray-500 dark:text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white" />
          )}
          {!isCollapsed && (
            <span className={`text-sm font-medium tracking-wide truncate ${status === 'running' ? 'text-blue-700 dark:text-pro-accent font-bold' : 'text-gray-700 dark:text-gray-300'}`}>
              {label}
            </span>
          )}
        </div>

        {/* Badge de Estado */}
        {!isCollapsed && status !== 'pending' && (
          <span className={`text-[9px] uppercase tracking-wider font-bold px-1.5 py-0.5 rounded ${
            status === 'running' ? 'bg-blue-100 text-blue-700 dark:bg-pro-accent/20 dark:text-pro-accent' : 'bg-green-100 text-green-700 dark:bg-pro-success/20 dark:text-pro-success'
          }`}>
            {status === 'running' ? 'Exec' : 'Done'}
          </span>
        )}

        {/* Tooltip para cuando está colapsado */}
        {isCollapsed && (
          <span className="absolute left-14 hidden group-hover:block bg-gray-900 dark:bg-white text-white dark:text-gray-900 text-xs font-bold px-2 py-1 rounded shadow-lg whitespace-nowrap z-50">
            {label}
          </span>
        )}
      </button>
    );
  };

  return (
    <aside className={`${isCollapsed ? 'w-[72px]' : 'w-[240px]'} relative bg-white dark:bg-pro-card border-r border-gray-200 dark:border-pro-border h-screen flex flex-col transition-all duration-300 z-40`}>
      
      <button 
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="absolute -right-3 top-6 bg-white dark:bg-pro-card border border-gray-200 dark:border-pro-border text-gray-500 hover:text-blue-600 dark:hover:text-pro-accent rounded-full p-1 z-50 shadow-sm transition-colors"
      >
        {isCollapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
      </button>

      {/* Header y Logo */}
      <div className={`p-5 border-b border-gray-200 dark:border-pro-border flex items-center ${isCollapsed ? 'justify-center' : 'gap-3'}`}>
        <img src="/Logo.png" alt="Prophetia" className={`h-auto object-contain flex-shrink-0 drop-shadow-md transition-all duration-300 ${isCollapsed ? 'w-8' : 'w-10'}`} />
        {!isCollapsed && (
          <div className="overflow-hidden whitespace-nowrap animate-fade-in-up">
            <h1 className="font-bold text-lg tracking-tight text-gray-900 dark:text-white leading-none">PROPHETIA</h1>
            <p className="text-[9px] uppercase tracking-widest text-blue-600 dark:text-pro-accent font-bold mt-0.5">SaaS Intelligence</p>
          </div>
        )}
      </div>

      {/* Panel de Estado de Sistema */}
      <div className={`px-4 py-4 border-b border-gray-200 dark:border-pro-border bg-gray-50 dark:bg-[#0B1020] transition-all overflow-hidden ${isCollapsed ? 'h-0 p-0 border-0 opacity-0' : 'opacity-100'}`}>
         <div className="flex items-center justify-between mb-3">
           <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Estado del Sistema</span>
           <span className="flex items-center gap-1 text-[10px] text-green-600 dark:text-pro-success font-bold"><span className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"></span> Online</span>
         </div>
         <div className="grid grid-cols-2 gap-2 text-center">
            <div className="bg-white dark:bg-pro-card rounded border border-gray-200 dark:border-white/5 p-1.5">
              <p className="text-[10px] text-gray-500">Precisión</p>
              <p className="text-xs font-bold text-blue-600 dark:text-pro-local">{stats?.accuracy_global || 'N/A'}</p>
            </div>
            <div className="bg-white dark:bg-pro-card rounded border border-gray-200 dark:border-white/5 p-1.5">
              <p className="text-[10px] text-gray-500">Partidos</p>
              <p className="text-xs font-bold text-gray-900 dark:text-white">{totalPartidos || 0}</p>
            </div>
         </div>
      </div>
      
      <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-6 custom-scrollbar overflow-x-hidden">
        
        <div>
          {!isCollapsed && <h2 className="text-[10px] font-bold text-gray-400 dark:text-gray-500 uppercase tracking-widest mb-2 px-2">Market</h2>}
          <ul className="space-y-1">
            <li>
              <button className={`group relative w-full flex items-center px-3 py-2.5 rounded-lg transition-all bg-blue-50 dark:bg-pro-accent/10 text-blue-700 dark:text-pro-accent font-semibold ${isCollapsed ? 'justify-center' : 'gap-3'}`}>
                <TrendingUp size={18} className="flex-shrink-0" />
                {!isCollapsed && <span className="text-sm truncate">Explorar Partidos</span>}
                {isCollapsed && <span className="absolute left-14 hidden group-hover:block bg-gray-900 dark:bg-white text-white dark:text-gray-900 text-xs font-bold px-2 py-1 rounded shadow-lg whitespace-nowrap z-50">Explorar Partidos</span>}
              </button>
            </li>
          </ul>
        </div>

        <div>
          {!isCollapsed && <h2 className="text-[10px] font-bold text-gray-400 dark:text-gray-500 uppercase tracking-widest mb-2 px-2">Pipeline ML</h2>}
          <ul className="space-y-1">
            <RenderButton label="Ingesta StatsBomb" icon={Database} endpoint="ingestion" />
            <RenderButton label="Data Adapter" icon={GitCommit} endpoint="adapter" />
            <RenderButton label="Feature Engineering" icon={Activity} endpoint="features" />
            <RenderButton label="Entrenar Modelo" icon={BrainCircuit} endpoint="train" />
          </ul>
        </div>

      </nav>
    </aside>
  );
}