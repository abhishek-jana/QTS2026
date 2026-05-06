/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      gridTemplateCols: {
        '64': 'repeat(64, minmax(0, 1fr))',
      }
    },
  },
  plugins: [],
}
