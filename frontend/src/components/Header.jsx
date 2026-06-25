import { Sun, Moon, Globe, ChevronDown, Menu, X, Swords } from 'lucide-react';

export default function Header({ 
  menuAbierto, 
  setMenuAbierto, 
  setModalH2H, 
  isDarkMode, 
  toggleDarkMode 
}) {
  return (
    <header className="h-20 border-b border-gray-200 dark:border-pro-border bg-white/80 dark:bg-pro-bg/80 backdrop-blur-md flex items-center px-6 md:px-10 justify-between z-40 sticky top-0 transition-colors duration-300">
      
      <button 
        className="md:hidden p-2 text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors" 
        onClick={() => setMenuAbierto(!menuAbierto)}
      >
        {menuAbierto ? <X size={24} /> : <Menu size={24} />}
      </button>

      <div className="flex-1 flex justify-center md:justify-start">
        <button 
          onClick={() => setModalH2H(true)}
          className="group relative flex items-center gap-3 bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-pro-accent dark:to-blue-600 hover:from-blue-700 hover:to-indigo-700 text-white px-6 py-2.5 md:py-3 rounded-full font-bold shadow-lg hover:shadow-xl transition-all"
        >
          <Swords size={20} className="group-hover:rotate-12 transition-transform" />
          <span className="text-sm md:text-base hidden sm:inline">Configurar Enfrentamiento</span>
          <span className="text-sm sm:hidden">H2H</span>
        </button>
      </div>

      <div className="flex items-center gap-4 md:gap-6">
        <div className="hidden md:flex items-center gap-2 text-sm font-medium text-gray-500 dark:text-gray-400 cursor-pointer hover:text-gray-900 dark:hover:text-pro-accent transition-colors">
          <Globe size={18} />
          <span>ES</span>
          <ChevronDown size={14} />
        </div>

        <button 
          onClick={toggleDarkMode}
          className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-white/5 transition-colors text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
        >
          {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
        </button>

        <div className="flex items-center gap-3 border-l border-gray-200 dark:border-pro-border pl-4 md:pl-6 cursor-pointer group">
          <div className="w-9 h-9 rounded-full bg-gradient-to-tr from-blue-600 to-purple-600 dark:from-pro-accent dark:to-purple-600 flex items-center justify-center text-white font-semibold shadow-md transition-transform group-hover:scale-105">
            JL
          </div>
          <div className="hidden md:block">
            <p className="text-sm font-bold leading-none text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-pro-accent transition-colors">Admin</p>
            <p className="text-xs text-green-600 dark:text-pro-success mt-1 flex items-center gap-1 font-medium">
              <span className="w-2 h-2 rounded-full bg-green-500 dark:bg-pro-success block"></span>
              Online
            </p>
          </div>
        </div>
      </div>
    </header>
  );
}