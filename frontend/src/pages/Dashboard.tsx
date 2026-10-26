import { Link as RouterLink } from 'react-router-dom'
import {
  Box,
  Button,
  Card,
  CardContent,
  Divider,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material'
import { ArrowUpRight, CalendarDays } from 'lucide-react'
import { useExecucao } from '../context/useExecucao'
import { cores } from '../theme/tema'

const formatar = (valor: number | undefined) =>
  valor == null || !Number.isFinite(valor)
    ? '—'
    : new Intl.NumberFormat('pt-BR', { maximumFractionDigits: 2 }).format(valor)

function Indicador({ titulo, valor, descricao }: { titulo: string; valor: string; descricao: string }) {
  return (
    <Card sx={{ height: '100%', borderRadius: 1.25 }}>
      <CardContent sx={{ px: 2.25, py: 2.25, '&:last-child': { pb: 2.25 } }}>
        <Typography sx={{ color: 'text.secondary', fontSize: 11.5, mb: 1.5 }}>
          {titulo}
        </Typography>
        <Typography sx={{ fontSize: { xs: 21, lg: 25 }, fontWeight: 650, letterSpacing: '-.035em', lineHeight: 1.2, overflowWrap: 'anywhere' }}>
          {valor}
        </Typography>
        <Typography sx={{ color: 'text.secondary', fontSize: 11, mt: 1.35 }}>
          {descricao}
        </Typography>
      </CardContent>
    </Card>
  )
}

export default function Dashboard() {
  const { execucaoId, statusSemanal } = useExecucao()
  const avaliacao = statusSemanal?.avaliacao_externa
  const horizontes = Object.entries(avaliacao?.por_horizonte ?? {})
    .sort(([a], [b]) => Number(a) - Number(b))

  return (
    <Stack spacing={3}>
      <Box>
        <Box sx={{ display: 'flex', gap: 1.2, alignItems: 'center', mb: 1.1 }}>
          <p className="eyebrow">Resultados da execução</p>
        </Box>
        <Typography variant="h4" component="h1">Visão geral</Typography>
        <Typography sx={{ color: 'text.secondary', fontSize: 13, mt: 1 }}>
          Desempenho do modelo selecionado para esta execução.
        </Typography>
      </Box>

      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, minmax(0, 1fr))', xl: 'repeat(4, minmax(0, 1fr))' }, gap: 1.75 }}>
        <Indicador titulo="Modelo vencedor" valor={statusSemanal?.algoritmo_vencedor?.toUpperCase() ?? '—'} descricao="Seleção do AutoML semanal" />
        <Indicador titulo="RMSE externo" valor={formatar(avaliacao?.agregado?.rmse)} descricao="Menor valor indica menor erro" />
        <Indicador titulo="MAE externo" valor={formatar(avaliacao?.agregado?.mae)} descricao="Erro absoluto médio" />
        <Indicador titulo="Previsões avaliadas" valor={formatar(avaliacao?.agregado?.quantidade)} descricao="Conjunto de teste reservado" />
      </Box>

      <Card sx={{ borderRadius: 1.25 }}>
        <Box sx={{ px: { xs: 2.5, sm: 3 }, pt: 2.75, pb: 2.25 }}>
          <Typography variant="h6" component="h2">Resultados por horizonte</Typography>
          <Typography sx={{ color: 'text.secondary', fontSize: 12.5, mt: 0.7 }}>
            Métricas por dia de antecedência (H1–H7).
          </Typography>
        </Box>
        <Divider />
        {horizontes.length > 0 ? (
          <TableContainer>
            <Table size="small" aria-label="Resultados por horizonte">
              <TableHead>
                <TableRow sx={{ bgcolor: cores.sidebar }}>
                  {['Horizonte', 'RMSE', 'MAE', 'Registros'].map((titulo) => (
                    <TableCell key={titulo} sx={{ fontSize: 11.5, fontWeight: 650, py: 1.5, color: 'text.secondary' }}>
                      {titulo}
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {horizontes.map(([dia, metrica]) => (
                  <TableRow key={dia} sx={{ '&:last-child td': { borderBottom: 0 }, '&:hover': { bgcolor: cores.superficieElevada } }}>
                    <TableCell sx={{ fontSize: 12.5, fontWeight: 600, py: 1.65 }}>Dia {dia}</TableCell>
                    <TableCell sx={{ fontSize: 12.5, fontVariantNumeric: 'tabular-nums' }}>{formatar(metrica.rmse)}</TableCell>
                    <TableCell sx={{ fontSize: 12.5, fontVariantNumeric: 'tabular-nums' }}>{formatar(metrica.mae)}</TableCell>
                    <TableCell sx={{ fontSize: 12.5, fontVariantNumeric: 'tabular-nums' }}>{formatar(metrica.quantidade)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Typography sx={{ px: 3, py: 3, fontSize: 12.5, color: 'text.secondary' }}>
            Ainda não há métricas externas para esta execução.
          </Typography>
        )}
      </Card>

      <Card sx={{ borderRadius: 1.25 }}>
        <Box sx={{ px: { xs: 2.5, sm: 3 }, py: 2.5, display: 'flex', alignItems: { xs: 'flex-start', sm: 'center' }, flexDirection: { xs: 'column', sm: 'row' }, justifyContent: 'space-between', gap: 2 }}>
          <Box sx={{ minWidth: 0 }}>
            <Typography variant="h6" component="h2">Próximos sete dias</Typography>
            <Typography sx={{ color: 'text.secondary', fontSize: 12.5, mt: 0.5 }}>Acesse as previsões geradas para sua execução.</Typography>
            <Typography sx={{ color: 'text.secondary', opacity: 0.75, fontSize: 10.5, mt: 1, overflowWrap: 'anywhere' }}>
              Execução: {execucaoId || '—'}
            </Typography>
          </Box>
          <Button component={RouterLink} to="/previsoes" variant="contained" color="primary" startIcon={<CalendarDays size={16} />} endIcon={<ArrowUpRight size={15} />} sx={{ flexShrink: 0, fontSize: 12.5 }}>
            Ver previsões
          </Button>
        </Box>
      </Card>
    </Stack>
  )
}