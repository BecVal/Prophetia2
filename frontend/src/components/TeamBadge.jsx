import { Shield } from 'lucide-react';
import { useTeamLogo } from '../hooks/useTeamLogo';

export default function TeamBadge({ teamName, size = "w-20 h-20 md:w-28 md:h-28", colorShadow = "rgba(255,255,255,0.2)" }) {
  const { logo, loading, error } = useTeamLogo(teamName);

  const getInitials = (name) => {
    if (!name) return "??";
    const words = name.split(" ");
    if (words.length >= 2) return (words[0][0] + words[1][0]).toUpperCase();
    return name.substring(0, 2).toUpperCase();
  };

  if (loading) {
    return (
      <div className={`${size} relative z-10 bg-gray-200 dark:bg-black/40 rounded-full flex items-center justify-center border border-gray-300 dark:border-white/5 mb-4 animate-pulse shadow-sm`}>
        <Shield size={size.includes('28') ? 40 : 24} className="text-gray-400 dark:text-gray-600 opacity-50" />
      </div>
    );
  }

  if (error || !logo) {
    return (
      <div 
        className={`${size} relative z-10 bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-800 dark:to-gray-900 rounded-full flex items-center justify-center border-2 border-gray-300 dark:border-white/10 mb-4 shadow-lg transition-transform duration-500 hover:scale-110 hover:-translate-y-1 group-hover:shadow-md`}
      >
        <span className="font-black text-2xl md:text-3xl tracking-widest text-gray-600 dark:text-gray-400 opacity-80">
          {getInitials(teamName)}
        </span>
      </div>
    );
  }

  return (
    <img 
      src={logo} 
      alt={`Escudo oficial de ${teamName}`} 
      loading="lazy"
      className={`${size} relative z-10 drop-shadow-xl object-contain mb-4 transition-all duration-500 group-hover:scale-110 group-hover:-translate-y-1 animate-fade-in-up`} 
      style={{ filter: `drop-shadow(0 0 15px ${colorShadow})` }}
    />
  );
}