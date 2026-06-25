/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class', // CRÍTICO: Debe ser 'class' para que el botón funcione
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'Manrope', 'sans-serif'],
      },
      colors: {
        pro: {
          bg: '#0B1020',
          card: '#121826',
          border: 'rgba(255,255,255,0.08)',
          local: '#00D4FF',
          away: '#FF4D6D',
          accent: '#4F7CFF',
          success: '#22C55E',
          warning: '#F59E0B'
        }
      }
    },
  },
  plugins: [],
}