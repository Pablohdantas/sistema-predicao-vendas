
import { useState } from 'react'

import {
  PERFIS_TREINAMENTO,
  parametrosValidos,
} from '../utils/configuracaoAutoML'

import type {
  ParametrosTreinamento,
  PerfilTreinamento,
} from '../utils/configuracaoAutoML'

type Props = {
  valor: ParametrosTreinamento
  onChange: (valor: ParametrosTreinamento) => void
  disabled?: boolean
}

export default function ConfiguracaoAutoML({
  valor,
  onChange,
  disabled = false,
}: Props) {
  const [perfil, setPerfil] =
    useState<PerfilTreinamento>('rapido')

  const valido = parametrosValidos(valor)

  const selecionarPerfil = (
    novoPerfil: PerfilTreinamento,
  ) => {
    setPerfil(novoPerfil)

    if (novoPerfil !== 'personalizado') {
      onChange({
        ...PERFIS_TREINAMENTO[novoPerfil],
      })
    }
  }

  return (
    <div className="automl-config">
      <label
        className="automl-config-label"
        htmlFor="perfil-automl"
      >
        Configuração do treinamento
      </label>

      <select
        id="perfil-automl"
        value={perfil}
        disabled={disabled}
        onChange={(evento) =>
          selecionarPerfil(
            evento.target.value as PerfilTreinamento,
          )
        }
      >
        <option value="rapido">
          Rápido — 2 tentativas, 60 s por algoritmo
        </option>

        <option value="ampliado">
          Ampliado — 15 tentativas, 360 s por algoritmo
        </option>

        <option value="personalizado">
          Personalizado
        </option>
      </select>

      {perfil === 'personalizado' && (
        <div className="automl-config-campos">
          <label>
            Tentativas por algoritmo (1 a 30)

            <input
              type="number"
              min={1}
              max={30}
              step={1}
              disabled={disabled}
              value={valor.tentativas}
              onChange={(evento) =>
                onChange({
                  ...valor,
                  tentativas:
                    evento.target.value === ''
                      ? 0
                      : Number(evento.target.value),
                })
              }
            />
          </label>

          <label>
            Orçamento por algoritmo, em segundos
            (1 a 1800)

            <input
              type="number"
              min={1}
              max={1800}
              step={1}
              disabled={disabled}
              value={valor.tempo_por_algoritmo}
              onChange={(evento) =>
                onChange({
                  ...valor,
                  tempo_por_algoritmo:
                    evento.target.value === ''
                      ? 0
                      : Number(evento.target.value),
                })
              }
            />
          </label>
        </div>
      )}

      {!valido && (
        <p
          className="automl-config-erro"
          role="alert"
        >
          Informe valores inteiros dentro dos limites
          permitidos.
        </p>
      )}

      <p className="automl-config-aviso">
        Todos os perfis produzem um modelo definitivo
        e uma avaliação externa.
        O orçamento em segundos é destinado à busca
        de configurações de cada algoritmo,
        não representa o tempo total exato de execução.
        Mais tentativas não garantem melhores previsões.
      </p>
    </div>
  )
}
