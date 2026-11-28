import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
// Bundled (not CDN) so screenshots render identically on Windows, macOS and a Colab VM.
// Both faces include Greek, which the interface needs for ΔΔG and σ.
import '@fontsource-variable/ibm-plex-sans'
import '@fontsource-variable/jetbrains-mono'
import App from './App'

const container = document.getElementById('root')!
createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>
)
