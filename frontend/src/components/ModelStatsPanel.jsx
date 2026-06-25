import { useState } from 'react';
import { Activity, Database, Target, Brain, PieChart } from 'lucide-react';

export default function ModelStatsPanel({ stats }) {
  const [topLimit, setTopLimit] = useState(10);

  if (!stats) return null;

  return (
    <div className="mt-8 grid grid-cols-1 lg:grid-cols-3 gap-6 animate-fade-in-up" style={{animationDelay: '0.3s'}}>
      
      <div className="bg-white dark:bg-pro-card border border-gray-200 dark:border-pro-border rounded-2xl p-6 md:p-8 shadow-sm relative overflow-hidden transition-colors">
        <h2 className="text-gray-600 dark:text-gray-400 text-xs font-bold tracking-widest uppercase mb-8 flex items-center gap-2">
          <Database size={14} className="text-blue-600 dark:text-pro-accent" /> Database Info
        </h2>
        
        <div className="space-y-5">
          <div className="flex justify-between items-center border-b border-gray-100 dark:border-white/5 pb-3">
            <span className="text-sm text-gray-700 dark:text-gray-400 font-medium">Total Variables</span>
            <span className="text-2xl font-black text-blue-600 dark:text-pro-accent tracking-tight">{stats.total_variables}</span>
          </div>
          <div className="flex justify-between items-center border-b border-gray-100 dark:border-white/5 pb-3">
            <span className="text-sm text-gray-700 dark:text-gray-400 font-medium">StatsBomb Data</span>
            <span className="text-lg font-bold text-gray-900 dark:text-white">{stats.fuentes?.statsbomb || 0} <span className="text-xs text-gray-500 dark:text-gray-400 font-normal">pts</span></span>
          </div>
          <div className="flex justify-between items-center border-b border-gray-100 dark:border-white/5 pb-3">
            <span className="text-sm text-gray-700 dark:text-gray-400 font-medium">Football-Data.co.uk</span>
            <span className="text-lg font-bold text-gray-900 dark:text-white">{stats.fuentes?.footballdata || 0} <span className="text-xs text-gray-500 dark:text-gray-400 font-normal">pts</span></span>
          </div>
        </div>

        <h2 className="text-gray-600 dark:text-gray-400 text-xs font-bold tracking-widest uppercase mt-8 mb-6 flex items-center gap-2">
          <PieChart size={14} className="text-rose-600 dark:text-pro-away" /> Histórico Clases
        </h2>
        <div className="flex justify-between text-center bg-gray-50 dark:bg-black/30 p-5 rounded-2xl border border-gray-200 dark:border-white/5 shadow-inner">
            <div className="flex flex-col items-center gap-1">
                <span className="text-[10px] text-gray-600 dark:text-gray-500 font-bold uppercase tracking-wider">Visita (0)</span>
                <span className="text-rose-600 dark:text-pro-away font-black text-xl">{stats.clases?.[0] || 0}</span>
            </div>
            <div className="w-px h-10 bg-gray-200 dark:bg-white/10"></div>
            <div className="flex flex-col items-center gap-1">
                <span className="text-[10px] text-gray-600 dark:text-gray-500 font-bold uppercase tracking-wider">Empate (1)</span>
                <span className="text-gray-700 dark:text-gray-400 font-black text-xl">{stats.clases?.[1] || 0}</span>
            </div>
            <div className="w-px h-10 bg-gray-200 dark:bg-white/10"></div>
            <div className="flex flex-col items-center gap-1">
                <span className="text-[10px] text-gray-600 dark:text-gray-500 font-bold uppercase tracking-wider">Local (2)</span>
                <span className="text-blue-600 dark:text-pro-local font-black text-xl">{stats.clases?.[2] || 0}</span>
            </div>
        </div>
      </div>

      <div className="bg-white dark:bg-pro-card border border-gray-200 dark:border-pro-border rounded-2xl p-6 md:p-8 shadow-sm lg:col-span-2 flex flex-col transition-colors">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-8 gap-4">
            
            <div>
              <h2 className="text-gray-900 dark:text-white font-bold text-lg flex items-center gap-2">
                <Brain size={18} className="text-blue-600 dark:text-pro-local" /> Diagnóstico del Modelo
              </h2>
              <div className="flex gap-4 mt-2">
                  <span className="text-xs font-medium text-gray-600 dark:text-gray-500 flex items-center gap-1">
                      <Target size={12} className="text-blue-600 dark:text-pro-local"/> Acc: <strong className="text-gray-900 dark:text-gray-200">{stats.accuracy_global}</strong>
                  </span>
                  <span className="text-xs font-medium text-gray-600 dark:text-gray-500 flex items-center gap-1">
                      <Activity size={12} className="text-rose-600 dark:text-pro-away"/> Loss: <strong className="text-gray-900 dark:text-gray-200">{stats.log_loss}</strong>
                  </span>
              </div>
            </div>

            <div className="flex bg-gray-100 dark:bg-pro-bg rounded-lg p-1 border border-gray-200 dark:border-pro-border">
              <button 
                onClick={() => setTopLimit(10)}
                className={`px-3 py-1.5 text-xs font-bold rounded-md transition-all ${topLimit === 10 ? 'bg-white dark:bg-pro-card text-gray-900 dark:text-white shadow-sm' : 'text-gray-600 dark:text-gray-500 hover:text-gray-900 dark:hover:text-gray-300'}`}
              >
                Top 10
              </button>
              <button 
                onClick={() => setTopLimit(20)}
                className={`px-3 py-1.5 text-xs font-bold rounded-md transition-all ${topLimit === 20 ? 'bg-white dark:bg-pro-card text-gray-900 dark:text-white shadow-sm' : 'text-gray-600 dark:text-gray-500 hover:text-gray-900 dark:hover:text-gray-300'}`}
              >
                Top 20
              </button>
            </div>
        </div>

        <div className="space-y-4 pr-2 flex-1 overflow-y-auto custom-scrollbar">
          {stats.feature_importance?.slice(0, topLimit).map((feature, idx) => {
            const maxWeight = stats.feature_importance[0]?.peso || 1;
            const percentage = (feature.peso / maxWeight) * 100;

            return (
              <div key={feature.nombre} className="flex items-center gap-4 group">
                <span className="w-5 text-[10px] font-bold text-gray-500 dark:text-gray-400">{idx + 1}.</span>
                <span className="text-xs font-mono font-medium text-gray-700 dark:text-gray-400 w-40 truncate group-hover:text-blue-600 dark:group-hover:text-pro-accent transition-colors duration-300">
                  {feature.nombre}
                </span>
                <div className="flex-1 bg-gray-200 dark:bg-black/40 rounded-full h-2.5 relative overflow-hidden shadow-inner">
                  <div 
                    className="absolute top-0 left-0 h-full bg-gray-500 dark:bg-pro-accent rounded-full transition-all duration-1000 ease-out group-hover:bg-blue-500 dark:group-hover:bg-pro-accent"
                    style={{ width: `${percentage}%` }}
                  ></div>
                </div>
                <span className="text-xs text-gray-800 dark:text-gray-300 font-mono font-bold w-12 text-right">
                  {feature.peso.toFixed(3)}
                </span>
              </div>
            );
          })}
        </div>
      </div>

    </div>
  );
}