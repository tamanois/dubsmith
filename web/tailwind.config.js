/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#080b12",
          900: "#0d1220",
          850: "#111827",
          800: "#161f31",
          700: "#1f2a3d",
        },
        accent: {
          400: "#8b9cff",
          500: "#6d7ff5",
          600: "#5566e8",
        },
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(139,156,255,0.18), 0 18px 50px rgba(0,0,0,0.35)",
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "sans-serif",
        ],
      },
    },
  },
  plugins: [],
};
