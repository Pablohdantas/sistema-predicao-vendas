import { useState } from 'react'
import type { ChangeEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { CloudUpload, LoaderCircle } from 'lucide-react'
import GuiaTemplate from '../components/GuiaTemplate'
import SeletorExecucao from '../components/SeletorExecucao'
import { api, obterMensagemErro } from '../services/api'
import type { DatasetResponse } from '../services/api'
import { useExecucao } from '../context/ExecucaoContext'

const CHAVE_INICIO_AUTOMATICO = 'automl_treinamento_pendente'

export default function Entrada() {
  const navegar = useNavigate()
  const { execucaoId, selecionarExecucao, limparSelecao } = useExecucao()
  const [arquivo, setArquivo] = useState<File | null>(null)
  const [enviando, setEnviando] = useState(false)
  const [erro, setErro] = useState('')
  const [avisos, setAvisos] = useState<string[]>([])
  const [versaoLista, setVersaoLista] = useState(0)

  const selecionarArquivo = (evento: ChangeEvent<HTMLInputElement>) => {
    setArquivo(evento.target.files?.[0] ?? null)
    setErro('')
    setAvisos([])
  }

  async function selecionarAnterior(id: string) {
    if (!id) {
      limparSelecao()
      return
    }
    try {
      const valido = await selecionarExecucao(id)
      if (valido) navegar('/')
      else setErro('Não foi possível abrir o histórico selecionado.')
    } catch (falha) {
      setErro(obterMensagemErro(falha))
    }
  }

  async function enviar() {
    if (!arquivo || enviando) return
    setEnviando(true)
    setErro('')
    setAvisos([])

    const formulario = new FormData()
    formulario.append('arquivo', arquivo)

    try {
      const resposta = await api.post<DatasetResponse>('/datasets/upload', formulario)
      const dados = resposta.data
      setAvisos(dados.avisos ?? [])
      if (!dados.valido || !dados.execucao_id) {
        setErro(dados.erros?.join(' | ') || 'Histórico não aprovado na validação.')
        return
      }

      const valido = await selecionarExecucao(dados.execucao_id)
      if (!valido) {
        setErro('O arquivo foi enviado, mas não foi possível abrir o histórico. Atualize a lista e selecione-o novamente.')
        setVersaoLista((valor) => valor + 1)
        return
      }

      setVersaoLista((valor) => valor + 1)
      sessionStorage.setItem(CHAVE_INICIO_AUTOMATICO, dados.execucao_id)
      navegar('/treinamento', { replace: true })
    } catch (falha) {
      setErro(obterMensagemErro(falha))
    } finally {
      setEnviando(false)
    }
  }

  return (
    <div className="fluxo-conteudo">
      <header className="page-header">
        <div>
          <p className="eyebrow">Dados de entrada</p>
          <h1>Histórico de vendas</h1>
          <p className="subtitle">
            Envie suas vendas para iniciar uma análise automática ou retome um histórico anterior.
          </p>
        </div>
      </header>

      {execucaoId ? (
        <section className="panel">
          <h2>Histórico selecionado</h2>
          <p>
            Este histórico está ativo. As atualizações de vendas serão associadas a ele,
            sem criar outro treinamento automaticamente.
          </p>
          <button type="button" className="secondary-button" onClick={limparSelecao}>
            Selecionar ou enviar outro histórico
          </button>
        </section>
      ) : (
        <section className="panel" id="dataset">
          <div className="panel-heading">
            <div>
              <h2>Enviar histórico de vendas</h2>
              <p>Use loja_id, data e vendas. As demais colunas são opcionais.</p>
            </div>
            <CloudUpload size={24} className="heading-icon" />
          </div>
          <label className="upload-area">
            <CloudUpload size={28} />
            <strong>{arquivo?.name ?? 'Selecionar arquivo CSV'}</strong>
            <span>Histórico diário de vendas</span>
            <input type="file" accept=".csv,text/csv" onChange={selecionarArquivo} />
          </label>
          <button
            type="button"
            className="primary-button"
            disabled={!arquivo || enviando}
            onClick={() => void enviar()}
          >
            {enviando ? <LoaderCircle className="spin" size={18} /> : <CloudUpload size={18} />}
            {enviando ? 'Validando histórico...' : 'Enviar e analisar automaticamente'}
          </button>
          <GuiaTemplate />
        </section>
      )}

      {erro && <p className="alert alert-error" role="alert">{erro}</p>}
      {avisos.map((aviso, indice) => (
        <p className="warnings" key={indice}>{aviso}</p>
      ))}

      {/* Mantido temporariamente: oferece acesso a análises anteriores. */}
      <SeletorExecucao
        key={versaoLista}
        execucaoSelecionada={execucaoId}
        onSelecionar={(id) => void selecionarAnterior(id)}
      />
    </div>
  )
}
