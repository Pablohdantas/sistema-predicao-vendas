import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { AlertTriangle, BrainCircuit, CheckCircle2, LoaderCircle } from 'lucide-react'
import { api, obterMensagemErro } from '../services/api'
import { useExecucao } from '../context/ExecucaoContext'

const CHAVE_INICIO_AUTOMATICO = 'automl_treinamento_pendente'

// Parâmetros internos do protótipo; não aparecem para o usuário.
const CONFIGURACAO_PADRAO = {
  tentativas: 3,
  tempo_por_algoritmo: 30,
}

export default function Treinamento() {
  const navegar = useNavigate()
  const { execucaoId, statusSemanal, atualizarStatus, carregando } = useExecucao()
  const [iniciando, setIniciando] = useState(false)
  const [erroLocal, setErroLocal] = useState('')
  const iniciouRef = useRef<string | null>(null)

  const status = statusSemanal?.status ?? 'nao_iniciado'
  const concluido = status === 'concluido'
  const falhou = status === 'falhou' || status === 'interrompido'

  // O usuário não precisa clicar para iniciar. A chave é criada apenas
  // após upload válido e impede treinamento involuntário de execuções antigas.
  useEffect(() => {
    if (carregando || !execucaoId || !statusSemanal) return

    if (concluido) {
      if (sessionStorage.getItem(CHAVE_INICIO_AUTOMATICO) === execucaoId) {
        sessionStorage.removeItem(CHAVE_INICIO_AUTOMATICO)
      }
      navegar('/dashboard', { replace: true })
      return
    }

    if (falhou) {
      if (sessionStorage.getItem(CHAVE_INICIO_AUTOMATICO) === execucaoId) {
        sessionStorage.removeItem(CHAVE_INICIO_AUTOMATICO)
      }
      return
    }

    const pendente = sessionStorage.getItem(CHAVE_INICIO_AUTOMATICO) === execucaoId
    if (!pendente || status !== 'nao_iniciado' || iniciouRef.current === execucaoId) return

    iniciouRef.current = execucaoId
    setIniciando(true)
    setErroLocal('')

    void (async () => {
      try {
        await api.post(`/treinamentos-semanais/${execucaoId}`, CONFIGURACAO_PADRAO)
        sessionStorage.removeItem(CHAVE_INICIO_AUTOMATICO)
        await atualizarStatus()
      } catch (erro) {
        sessionStorage.removeItem(CHAVE_INICIO_AUTOMATICO)
        setErroLocal(obterMensagemErro(erro))
        await atualizarStatus()
      } finally {
        setIniciando(false)
      }
    })()
  }, [carregando, execucaoId, statusSemanal, status, concluido, falhou, atualizarStatus, navegar])

  const processando = iniciando || status === 'em_andamento' || (
    status === 'nao_iniciado' &&
    sessionStorage.getItem(CHAVE_INICIO_AUTOMATICO) === execucaoId
  )

  return (
    <div className="fluxo-conteudo">
      <header className="page-header">
        <div>
          <p className="eyebrow">Análise automática</p>
          <h1>{falhou || erroLocal ? 'Não foi possível concluir a análise' : 'Preparando suas previsões'}</h1>
          <p className="subtitle">
            {falhou || erroLocal
              ? 'Seu histórico foi preservado. Verifique o motivo antes de iniciar uma nova análise.'
              : 'O sistema compara LightGBM e XGBoost e escolhe automaticamente o modelo de menor erro de validação.'}
          </p>
        </div>
      </header>

      <section className="panel" aria-live="polite">
        {falhou || erroLocal ? (
          <div className="fluxo-acoes">
            <AlertTriangle size={32} color="#c24a42" />
            <h2>Análise não concluída</h2>
            <p className="inline-error" role="alert">
              {statusSemanal?.erro || erroLocal || 'O treinamento foi interrompido.'}
            </p>
            <p>Os arquivos existentes não serão sobrescritos. Selecione outro dataset ou verifique os artefatos desta execução.</p>
            <Link to="/entrada" className="fluxo-link-secundario">Voltar aos dados</Link>
          </div>
        ) : processando || carregando ? (
          <div className="fluxo-acoes tela-processamento">
            <LoaderCircle size={38} className="spin" aria-hidden="true" />
            <h2>Analisando seu histórico de vendas</h2>
            <p>Estamos preparando os dados, treinando os algoritmos e avaliando as previsões.</p>
            <p className="execution-id">Execução: {execucaoId}</p>
            <p>Você pode permanecer nesta página. Ao concluir, os resultados serão abertos automaticamente.</p>
          </div>
        ) : concluido ? (
          <div className="fluxo-acoes">
            <CheckCircle2 size={34} />
            <h2>Treinamento concluído</h2>
            <p>Carregando o painel de resultados...</p>
          </div>
        ) : (
          <div className="fluxo-acoes">
            <BrainCircuit size={34} />
            <h2>Execução ainda não treinada</h2>
            <p>Essa execução foi selecionada do histórico e não possui treinamento semanal. O início automático ocorre após um novo upload válido.</p>
            <Link to="/entrada" className="fluxo-link-secundario">Voltar aos dados</Link>
          </div>
        )}
      </section>
    </div>
  )
}
