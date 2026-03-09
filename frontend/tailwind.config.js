/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        industrial: {
          900: '#121417', 
          800: '#1c1f24', 
          700: '#2a2e35', 
          cyan: '#00f2ff', 
          green: '#4ade80', 
          amber: '#fbbf24', 
        }
      }
    },
  },
  plugins: [],
}