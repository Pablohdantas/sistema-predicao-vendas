import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import AppRotas from './AppRotas'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AppRotas />
  </StrictMode>,
)
