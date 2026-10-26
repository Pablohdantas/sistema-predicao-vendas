import { useState } from 'react'
import { BrowserRouter, Navigate, NavLink, Outlet, Route, Routes } from 'react-router-dom'
import {
  Alert,
  Box,
  CircularProgress,
  CssBaseline,
  Divider,
  Drawer,
  IconButton,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  ThemeProvider,
  Toolbar,
  Typography,
} from '@mui/material'
import { BarChart3, CalendarDays, Database, History, Menu } from 'lucide-react'
import { ExecucaoProvider } from './context/ExecucaoContext'
import { useExecucao } from './context/useExecucao'
import Entrada from './pages/Entrada'
import Treinamento from './pages/Treinamento'
import Dashboard from './pages/Dashboard'
import Previsoes from './pages/Previsoes'
import Historico from './pages/Historico'
import { cores, tema } from './theme/tema'
import './App.css'
import './fluxo.css'
import './dark-legacy.css'
import logo from './assets/logo.svg'

const LARGURA_MENU = 228

function Carregando() {
  return (
    <Box role="status" sx={{ minHeight: 240, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 1.5 }}>
      <CircularProgress size={19} color="primary" />
      <Typography sx={{ fontSize: 13, color: 'text.secondary' }}>
        Verificando o histórico de vendas...
      </Typography>
    </Box>
  )
}

function ProtegerDataset() {
  const { carregando, execucaoId } = useExecucao()
  if (carregando) return <Carregando />
  return execucaoId ? <Outlet /> : <Navigate to="/entrada" replace />
}

function ProtegerModelo() {
  const { carregando, execucaoId, statusSemanal } = useExecucao()
  if (carregando) return <Carregando />
  if (!execucaoId) return <Navigate to="/entrada" replace />
  return statusSemanal?.status === 'concluido'
    ? <Outlet />
    : <Navigate to="/treinamento" replace />
}

function Inicio() {
  const { carregando, execucaoId, statusSemanal } = useExecucao()
  if (carregando) return <Carregando />
  if (!execucaoId) return <Navigate to="/entrada" replace />
  return <Navigate to={statusSemanal?.status === 'concluido' ? '/dashboard' : '/treinamento'} replace />
}

function Estrutura() {
  const { execucaoId, statusSemanal, erro } = useExecucao()
  const [menuAberto, setMenuAberto] = useState(false)
  const modeloDisponivel = Boolean(execucaoId) && statusSemanal?.status === 'concluido'

  const links = [
    { to: '/entrada', label: 'Dados', Icone: Database },
    ...(modeloDisponivel ? [
      { to: '/dashboard', label: 'Visão geral', Icone: BarChart3 },
      { to: '/previsoes', label: 'Previsões', Icone: CalendarDays },
      { to: '/historico', label: 'Atualizar vendas', Icone: History },
    ] : []),
  ]

  const navegacao = (
    <Box component="nav" aria-label="Navegação principal" sx={{ height: '100%', display: 'flex', flexDirection: 'column', bgcolor: cores.sidebar, color: cores.texto }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.4, px: 2.4, pt: 3, pb: 2.7 }}>
        <Box sx={{ width: 33, height: 33, display: 'grid', placeItems: 'center', border: `1px solid ${cores.borda}`, borderRadius: 1.5, color: cores.turquesa }}>
          <img src={logo} alt="Logo Previsão de Vendas" width={19}/>
        </Box>
        <Box sx={{ minWidth: 0 }}>
          <Typography sx={{ fontSize: 13, fontWeight: 650, lineHeight: 1.35, color: cores.texto }}>
            Previsão de Vendas
          </Typography>

        </Box>
      </Box>

      <Divider sx={{ borderColor: cores.borda, mx: 2.5 }} />
      <Typography sx={{ px: 3, pt: 3.3, pb: 1, fontSize: 10, letterSpacing: '.13em', color: cores.textoDiscreto, fontWeight: 600 }}>
        NAVEGAÇÃO
      </Typography>

      <List sx={{ px: 1.5, py: 0 }}>
        {links.map(({ to, label, Icone }) => (
          <ListItemButton
            key={to}
            component={NavLink}
            to={to}
            onClick={() => setMenuAberto(false)}
            sx={{
              position: 'relative',
              minHeight: 41,
              mb: 0.5,
              px: 1.5,
              py: 0.9,
              borderRadius: 1.5,
              color: cores.textoSecundario,
              transition: 'background-color .15s ease',
              '& .MuiListItemIcon-root': { color: 'inherit' },
              '&:hover': { bgcolor: cores.superficieElevada, color: cores.texto },
              '&.active': { bgcolor: cores.superficieElevada, color: cores.texto },
              '&.active .MuiListItemIcon-root': { color: cores.turquesa },
              '&.active::before': {
                content: '""',
                position: 'absolute',
                left: 0,
                top: 9,
                bottom: 9,
                width: 2,
                borderRadius: 2,
                bgcolor: cores.turquesa,
              },
            }}
          >
            <ListItemIcon sx={{ minWidth: 34 }}>
              <Icone size={17} strokeWidth={1.8} />
            </ListItemIcon>
            <ListItemText primary={label} slotProps={{ primary: { sx: { fontSize: 12.5, fontWeight: 500 } } }} />
          </ListItemButton>
        ))}
      </List>

      <Box sx={{ flex: 1 }} />

    </Box>
  )

  return (
    <Box className="mui-dark-shell" sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default', color: 'text.primary' }}>
      <Drawer
        variant="permanent"
        sx={{
          width: LARGURA_MENU,
          flexShrink: 0,
          display: { xs: 'none', md: 'block' },
          '& .MuiDrawer-paper': { width: LARGURA_MENU, bgcolor: cores.sidebar, borderRight: `1px solid ${cores.borda}`, boxSizing: 'border-box' },
        }}
      >
        {navegacao}
      </Drawer>
      <Drawer
        variant="temporary"
        open={menuAberto}
        onClose={() => setMenuAberto(false)}
        sx={{ display: { xs: 'block', md: 'none' }, '& .MuiDrawer-paper': { width: LARGURA_MENU, bgcolor: cores.sidebar } }}
      >
        {navegacao}
      </Drawer>

      <Box component="main" sx={{ minWidth: 0, flex: 1 }}>
        <Toolbar sx={{ display: { xs: 'flex', md: 'none' }, bgcolor: cores.sidebar, color: cores.texto, gap: 1, minHeight: 58 }}>
          <IconButton onClick={() => setMenuAberto(true)} aria-label="Abrir navegação" edge="start" sx={{ color: cores.texto }}>
            <Menu size={21} />
          </IconButton>
          <Typography sx={{ fontWeight: 650, fontSize: 13 }}>Previsão de Vendas</Typography>
        </Toolbar>
        <Box sx={{ maxWidth: 1410, mx: 'auto', px: { xs: 2, sm: 3, lg: 4 }, pt: { xs: 3, md: 4 }, pb: 4 }}>
          {erro && <Alert severity="error" sx={{ mb: 3 }}>{erro}</Alert>}
          <Outlet />
        </Box>
      </Box>
    </Box>
  )
}

export default function AppRotas() {
  return (
    <ThemeProvider theme={tema}>
      <CssBaseline />
      <BrowserRouter>
        <ExecucaoProvider>
          <Routes>
            <Route element={<Estrutura />}>
              <Route index element={<Inicio />} />
              <Route path="entrada" element={<Entrada />} />
              <Route element={<ProtegerDataset />}>
                <Route path="treinamento" element={<Treinamento />} />
              </Route>
              <Route element={<ProtegerModelo />}>
                <Route path="dashboard" element={<Dashboard />} />
                <Route path="previsoes" element={<Previsoes />} />
                <Route path="historico" element={<Historico />} />
              </Route>
              <Route path="*" element={<Inicio />} />
            </Route>
          </Routes>
        </ExecucaoProvider>
      </BrowserRouter>
    </ThemeProvider>
  )
}