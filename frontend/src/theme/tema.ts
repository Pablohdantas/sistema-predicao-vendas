import { createTheme } from '@mui/material/styles'

// Marco 09: identidade dark minimalista — azul clássico, turquesa e dourado.
// Exportações nomeadas preservadas para AppRotas.tsx e Dashboard.tsx.
export const cores = {
  fundo: '#0A0F18',
  sidebar: '#0D1522',
  superficie: '#111A28',
  superficieElevada: '#162235',
  azulClassico: '#24476A',
  azulApoio: '#5593CF',
  turquesa: '#20B8A6',
  turquesaHover: '#18A393',
  dourado: '#CDA65B',
  texto: '#E8EEF6',
  textoSecundario: '#98A8BC',
  textoDiscreto: '#75869C',
  borda: '#263346',
} as const

export const tema = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: cores.turquesa, contrastText: '#08191A' },
    secondary: { main: cores.azulApoio },
    warning: { main: cores.dourado },
    background: { default: cores.fundo, paper: cores.superficie },
    text: { primary: cores.texto, secondary: cores.textoSecundario },
    divider: cores.borda,
    success: { main: '#4AC99A' },
    error: { main: '#EF7878' },
    info: { main: cores.azulApoio },
  },
  shape: { borderRadius: 8 },
  typography: {
    fontFamily: 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    h4: { fontSize: '1.48rem', fontWeight: 650, letterSpacing: '-0.035em' },
    h6: { fontSize: '0.95rem', fontWeight: 600, letterSpacing: '-0.012em' },
    body1: { fontSize: '0.92rem' },
    body2: { fontSize: '0.83rem' },
    button: { textTransform: 'none', fontWeight: 600, letterSpacing: 0 },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        ':root': { colorScheme: 'dark' },
        body: { backgroundColor: cores.fundo, color: cores.texto },
        '*::-webkit-scrollbar': { width: 9, height: 9 },
        '*::-webkit-scrollbar-track': { backgroundColor: cores.fundo },
        '*::-webkit-scrollbar-thumb': { backgroundColor: cores.borda, borderRadius: 8 },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: { backgroundImage: 'none', backgroundColor: cores.superficie, boxShadow: 'none' },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: { backgroundImage: 'none', backgroundColor: cores.superficie, border: `1px solid ${cores.borda}`, boxShadow: 'none' },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: { backgroundColor: cores.sidebar, borderRight: `1px solid ${cores.borda}` },
      },
    },
    MuiButton: {
      defaultProps: { disableElevation: true },
      styleOverrides: {
        root: {
          borderRadius: 7,
          minHeight: 36,
          paddingInline: 16,
          boxShadow: 'none',
          '&:hover': { boxShadow: 'none' },
          // Em MUI atual, containedPrimary não é uma chave válida de styleOverrides.
          '&.MuiButton-containedPrimary:hover': { backgroundColor: cores.turquesaHover },
          '&.MuiButton-outlined': { borderColor: cores.borda },
          '&.MuiButton-outlined:hover': { borderColor: cores.turquesa },
        },
      },
    },
    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: 7,
          '&.Mui-selected': { backgroundColor: cores.superficieElevada },
          '&.Mui-selected:hover': { backgroundColor: cores.superficieElevada },
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: { borderBottom: `1px solid ${cores.borda}` },
        head: { color: cores.textoSecundario, fontWeight: 600 },
      },
    },
    MuiDivider: {
      styleOverrides: { root: { borderColor: cores.borda } },
    },
  },
})

// Exportação default adicional por compatibilidade com futuros componentes.
export default tema