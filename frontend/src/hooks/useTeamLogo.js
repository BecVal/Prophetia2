// src/hooks/useTeamLogo.js
import { useState, useEffect } from 'react';

// Obtiene la API Key de las variables de entorno de Vite
const API_FOOTBALL_KEY = import.meta.env.VITE_API_FOOTBALL_KEY || '';

export function useTeamLogo(teamName) {
  const [logo, setLogo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!teamName) {
      setLoading(false);
      return;
    }

    const fetchLogo = async () => {
      setLoading(true);
      setError(false);
      
      // Sanitizamos el nombre del equipo para usarlo como llave de caché
      const cacheKey = `prophetia_logo_${teamName.toLowerCase().replace(/\s+/g, '_')}`;
      
      // NIVEL 3: Caché (LocalStorage)
      const cachedLogo = localStorage.getItem(cacheKey);
      if (cachedLogo) {
        setLogo(cachedLogo);
        setLoading(false);
        return;
      }

      try {
        // NIVEL 1: API-Football (Principal)
        if (API_FOOTBALL_KEY) {
          const resApiFootball = await fetch(`https://v3.football.api-sports.io/teams?search=${encodeURIComponent(teamName)}`, {
            headers: {
              'x-apisports-key': API_FOOTBALL_KEY
            }
          });
          const dataApiFootball = await resApiFootball.json();
          
          if (dataApiFootball.response && dataApiFootball.response.length > 0) {
            const fetchedLogo = dataApiFootball.response[0].team.logo;
            setLogo(fetchedLogo);
            localStorage.setItem(cacheKey, fetchedLogo);
            setLoading(false);
            return;
          }
        }
        
        // Si no hay API Key o no encontró resultados, forzamos el error para ir al Fallback
        throw new Error("No encontrado en API-Football o API Key no configurada");
        
      } catch (err) {
        
        // NIVEL 2: TheSportsDB (Fallback)
        try {
          const resSportsDb = await fetch(`https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t=${encodeURIComponent(teamName)}`);
          const dataSportsDb = await resSportsDb.json();

          if (dataSportsDb.teams && dataSportsDb.teams[0].strTeamBadge) {
            const fetchedLogo = dataSportsDb.teams[0].strTeamBadge;
            setLogo(fetchedLogo);
            localStorage.setItem(cacheKey, fetchedLogo);
          } else {
            setError(true); // Ningún proveedor encontró el logo
          }
        } catch (fallbackErr) {
          console.error("Error crítico buscando logos en Prophetia:", fallbackErr);
          setError(true);
        }

      } finally {
        setLoading(false);
      }
    };

    fetchLogo();
  }, [teamName]);

  return { logo, loading, error };
}