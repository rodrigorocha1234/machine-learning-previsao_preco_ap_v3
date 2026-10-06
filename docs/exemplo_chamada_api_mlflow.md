# API de Previsão de Preços de Apartamentos — Contrato de Integração

> **Endpoint:** `http://localhost:8080/invocations`
>
> **Modelo:** `previsao_preco_apartamento_modelo@champion`
>
> **Cidade:** Ribeirão Preto/SP

---

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Contrato de Entrada](#2-contrato-de-entrada)
3. [Exemplos de Chamada](#3-exemplos-de-chamada)
4. [Resposta Completa da API](#4-resposta-completa-da-api)
5. [Dicionário de Campos de Saída](#5-dicionário-de-campos-de-saída)
6. [Como Interpretar os Resultados](#6-como-interpretar-os-resultados)
   - [Exemplos para a equipe de negócio](#exemplos-para-a-equipe-de-negócio)
7. [Lógica de Fallback Hierárquico](#7-lógica-de-fallback-hierárquico)
8. [Resumo dos campos](#8-resumo-dos-40-campos-de-saída)
9. [Observabilidade da API](#observabilidade-da-api)

---

## 1. Visão Geral

A API retorna **muito mais do que o preço previsto**. Para cada imóvel enviado, a resposta identifica a Zona e o Bairro e o motor de regras de negócio enriquece automaticamente os resultados com:

- Preço de venda estimado pelo modelo carregado no serving, em R$
- Zona e Bairro associados a cada previsão, inclusive em chamadas com vários imóveis
- Preço previsto médio por Zona (`valor_previsto_zona`) e por Zona/Bairro (`valor_previsto_bairro`), calculado sobre o lote enviado
- **Preços de mercado de referência** em 3 níveis: Global, Zona e Bairro
- Índices comparativos e desvios percentuais em cada nível
- Simulações de desconto e faixas seguras de negociação

```
ENTRADA: Características do imóvel (6 campos)
    │
    ▼
SAÍDA: Análise completa em 3 níveis hierárquicos
    ├── Nível Global  — contexto da cidade de Ribeirão Preto/SP
    ├── Nível Zona    — contexto da zona (ex.: Zona Sul)
    └── Nível Bairro  — contexto do bairro (ex.: Jardim Botânico)
```

---

## 2. Contrato de Entrada

> **Não é necessário** fornecer `Código`, `Apartamento`, `valor_m2`, `media_valor_m2_bairro` ou `media_valor_m2_zona`. Esses campos são descartados mesmo que enviados.

| Campo | Tipo | Descrição | Exemplo |
| :--- | :---: | :--- | :--- |
| `Bairro` | `string` | Bairro em Ribeirão Preto/SP | `"Jardim Botânico"` |
| `Zona` | `string` | Zona da cidade | `"Zona Sul"` |
| `Quartos` | `long` | Número de quartos | `3` |
| `Banheiros` | `long` | Número de banheiros | `2` |
| `Vagas_Garagem` | `long` | Vagas de garagem | `2` |
| `Metragem` | `double` | Área útil privativa (m²) | `85.0` |

---

## 3. Exemplos de Chamada

### Exemplo A: cURL — formato `dataframe_records`

```bash
curl -X POST "http://localhost:8080/invocations" \
     -H "Content-Type: application/json" \
     -d '{
       "dataframe_records": [
         {
           "Bairro": "Jardim Botânico",
           "Zona": "Zona Sul",
           "Quartos": 3,
           "Banheiros": 2,
           "Vagas_Garagem": 2,
           "Metragem": 85.0
         }
       ]
     }'
```

### Exemplo B: cURL — formato `dataframe_split`

```bash
curl -X POST "http://localhost:8080/invocations" \
     -H "Content-Type: application/json" \
     -d '{
       "dataframe_split": {
         "columns": ["Bairro","Zona","Quartos","Banheiros","Vagas_Garagem","Metragem"],
         "data": [["Jardim Botânico","Zona Sul",3,2,2,85.0]]
       }
     }'
```

### Exemplo C: Python (`requests`)

```python
import requests, json

url = "http://localhost:8080/invocations"

payload = {
    "dataframe_records": [
        {
            "Bairro": "Jardim Botânico",
            "Zona": "Zona Sul",
            "Quartos": 3,
            "Banheiros": 2,
            "Vagas_Garagem": 2,
            "Metragem": 85.0,
        }
    ]
}

resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
resp.raise_for_status()
resultado = resp.json()["predictions"][0]

# Acesso estruturado por nível
print(f"Localização:         {resultado['Zona']} / {resultado['Bairro']}")
print(f"Previsão:            R$ {resultado['valor_previsto']:,.2f}")
print(f"R$/m²:               R$ {resultado['valor_m2_previsto']:,.2f}")
print(f"Previsão na Zona:    R$ {resultado['valor_previsto_zona']:,.2f}")
print(f"Previsão no Bairro:  R$ {resultado['valor_previsto_bairro']:,.2f}")
print(f"Previsão/m² na Zona: R$ {resultado['valor_m2_previsto_zona']:,.2f}")
print(f"Previsão/m² Bairro:  R$ {resultado['valor_m2_previsto_bairro']:,.2f}")
print()
print(f"--- GLOBAL (Ribeirão Preto/SP) ---")
print(f"Mediana de mercado:  R$ {resultado['global_mediana_mercado']:,.2f}")
print(f"Desvio vs. mercado:  {resultado['diferenca_perc_global']:+.1f}%")
print(f"Índice m²:           {resultado['indice_imovel_global']:.2f}x")
print()
print(f"--- ZONA SUL ---")
print(f"Mediana de mercado:  R$ {resultado['zona_mediana_mercado']:,.2f}")
print(f"Desvio vs. zona:     {resultado['diferenca_perc_zona']:+.1f}%")
print(f"Índice m²:           {resultado['indice_imovel_zona']:.2f}x")
print()
print(f"--- JARDIM BOTÂNICO ---")
print(f"Mediana de mercado:  R$ {resultado['bairro_mediana_mercado']:,.2f}")
print(f"Desvio vs. bairro:   {resultado['diferenca_perc_bairro']:+.1f}%")
print(f"Faixa segura:        R$ {resultado['bairro_faixa_segura_piso']:,.2f} – R$ {resultado['bairro_faixa_segura_teto']:,.2f}")
```

### Exemplo D: valor previsto médio por Zona e por Bairro

Envie vários imóveis na mesma chamada. A API devolve um `valor_previsto` individual e os campos `valor_previsto_zona`, `valor_previsto_bairro`, `valor_m2_previsto_zona` e `valor_m2_previsto_bairro`: médias das previsões no lote para a Zona e para a combinação Zona/Bairro, respectivamente. Os campos são repetidos para cada imóvel do grupo. Os campos anteriores `valor_previsto_medio_zona` e `valor_previsto_medio_bairro` permanecem disponíveis como aliases compatíveis:

```python
import pandas as pd
import requests

url = "http://localhost:8080/invocations"
payload = {
  "dataframe_records": [
    {"Bairro": "Jardim Botânico", "Zona": "Zona Sul", "Quartos": 3, "Banheiros": 2, "Vagas_Garagem": 2, "Metragem": 85.0},
    {"Bairro": "Jardim Botânico", "Zona": "Zona Sul", "Quartos": 2, "Banheiros": 2, "Vagas_Garagem": 1, "Metragem": 65.0},
    {"Bairro": "Centro", "Zona": "Centro", "Quartos": 2, "Banheiros": 1, "Vagas_Garagem": 1, "Metragem": 55.0},
  ]
}

resposta = requests.post(url, json=payload, timeout=30)
resposta.raise_for_status()
previsoes = pd.DataFrame(resposta.json()["predictions"])

por_zona = previsoes.groupby("Zona", as_index=False).agg(
  valor_previsto=("valor_previsto_zona", "first"),
  valor_m2_previsto=("valor_m2_previsto_zona", "first"),
  quantidade_imoveis=("valor_previsto", "size"),
)
por_bairro = previsoes.groupby(["Zona", "Bairro"], as_index=False).agg(
  valor_previsto=("valor_previsto_bairro", "first"),
  valor_m2_previsto=("valor_m2_previsto_bairro", "first"),
  quantidade_imoveis=("valor_previsto", "size"),
)

print("Previsão média por Zona")
print(por_zona.to_string(index=False))
print("\nPrevisão média por Bairro")
print(por_bairro.to_string(index=False))
```

As médias são calculadas somente sobre os imóveis enviados nessa chamada. Com apenas um imóvel em um grupo, a média coincide com o `valor_previsto` individual. Para representar toda uma Zona ou Bairro, envie à API um conjunto representativo dos imóveis que deseja resumir; ela não presume nem consulta anúncios que não estejam no lote.

---

### Exemplo numérico das médias por localização

Suponha que o modelo produza estas previsões individuais em uma única chamada (valores ilustrativos):

| Zona | Bairro | `valor_previsto` | `valor_previsto_zona` | `valor_previsto_bairro` |
| :--- | :--- | ---: | ---: | ---: |
| Zona Sul | Jardim Botânico | 600000.00 | 600000.00 | 500000.00 |
| Zona Sul | Jardim Botânico | 400000.00 | 600000.00 | 500000.00 |
| Zona Sul | Nova Aliança | 800000.00 | 600000.00 | 800000.00 |
| Centro | Centro | 300000.00 | 300000.00 | 300000.00 |

O preço previsto da Zona Sul é `(600000 + 400000 + 800000) / 3 = 600000`; o do Jardim Botânico é `(600000 + 400000) / 2 = 500000`. Bairros com o mesmo nome em zonas diferentes formam grupos separados. Alterar os imóveis enviados pode alterar essas médias, mesmo mantendo o modelo e as características de um imóvel individual.

### Disponibilização no MLflow Serving

Os campos fazem parte da saída de `EmpacotadorModelo.predict`. O serving fixa a versão na inicialização. Para um modelo já publicado, atualizar o código local não garante a atualização do artefato e da assinatura registrados. Reiniciar pode carregar módulos montados atualizados, mas a assinatura registrada deve continuar compatível. Registre uma versão com o empacotador atualizado, confira sua assinatura de saída e direcione o alias `champion` para essa versão; depois reinicie o serviço de serving para carregar o modelo atualizado.

Valide com o Exemplo D que cada item de `predictions` contém `valor_previsto_zona` e `valor_previsto_bairro`. Os aliases `valor_previsto_medio_zona` e `valor_previsto_medio_bairro` continuam disponíveis.

---

## 4. Resposta Completa da API

Os números abaixo são ilustrativos; não identificam uma versão fixa nem constituem garantia de preço.

A resposta contém **40 campos**: identificação do imóvel, previsão base, referências Global/Zona/Bairro e médias das previsões no lote. `valor_previsto` e `valor_m2_previsto` são estimativas individuais; os campos `*_zona` e `*_bairro` de previsão são as médias das previsões individuais para os grupos do lote. `zona_media_mercado` e `bairro_media_mercado` continuam sendo médias históricas de referência.

```json
{
  "predictions": [
    {
      "Zona": "Zona Sul",
      "Bairro": "Jardim Botânico",
      "valor_previsto":             631396.73,
      "valor_m2_previsto":            7428.20,

      "global_mediana_mercado":     290000.00,
      "global_media_mercado":       422828.81,
      "global_mediana_m2_mercado":    4738.10,
      "indice_imovel_global":            1.57,
      "diferenca_perc_global":          49.33,
      "global_desconto_5":          275500.00,
      "global_desconto_10":         261000.00,
      "global_desconto_15":         246500.00,
      "global_faixa_segura_piso":   261000.00,
      "global_faixa_segura_teto":   290000.00,

      "zona_mediana_mercado":       490000.00,
      "zona_media_mercado":         670717.08,
      "zona_mediana_m2_mercado":      6745.45,
      "indice_imovel_zona":              1.10,
      "diferenca_perc_zona":            -5.86,
      "zona_desconto_5":            465500.00,
      "zona_desconto_10":           441000.00,
      "zona_desconto_15":           416500.00,
      "zona_faixa_segura_piso":     441000.00,
      "zona_faixa_segura_teto":     490000.00,

      "bairro_mediana_mercado":     490000.00,
      "bairro_media_mercado":       670717.08,
      "bairro_mediana_m2_mercado":    6745.45,
      "indice_imovel_bairro":            1.10,
      "diferenca_perc_bairro":          -5.86,
      "bairro_desconto_5":          465500.00,
      "bairro_desconto_10":         441000.00,
      "bairro_desconto_15":         416500.00,
      "bairro_faixa_segura_piso":   441000.00,
      "bairro_faixa_segura_teto":   490000.00,

      "valor_previsto_zona":       631396.73,
      "valor_previsto_bairro":     631396.73,
      "valor_m2_previsto_zona":      7428.20,
      "valor_m2_previsto_bairro":    7428.20,
      "valor_previsto_medio_zona": 631396.73,
      "valor_previsto_medio_bairro": 631396.73
    }
  ]
}
```

---

## 5. Dicionário de Campos de Saída

### 5.1 Identificação do Imóvel

| Campo | Tipo | Descrição |
| :--- | :---: | :--- |
| `Zona` | `string` | Zona informada na requisição, para identificar a previsão |
| `Bairro` | `string` | Bairro informado na requisição, para identificar a previsão |

### 5.2 Previsão Base

| Campo | Tipo | Descrição |
| :--- | :---: | :--- |
| `valor_previsto` | `float` | Preço de venda estimado pelo modelo (R$) |
| `valor_m2_previsto` | `float` | `valor_previsto ÷ max(Metragem, 1.0)` — Preço por m² previsto (R$/m²) |
| `valor_previsto_zona` | `float` | Média dos `valor_previsto` dos imóveis da mesma Zona enviados no lote (R$) |
| `valor_previsto_bairro` | `float` | Média dos `valor_previsto` dos imóveis da mesma Zona e Bairro enviados no lote (R$) |
| `valor_m2_previsto_zona` | `float` | Média dos `valor_m2_previsto` dos imóveis da mesma Zona enviados no lote (R$/m²) |
| `valor_m2_previsto_bairro` | `float` | Média dos `valor_m2_previsto` dos imóveis da mesma Zona e Bairro enviados no lote (R$/m²) |
| `valor_previsto_medio_zona` | `float` | Alias compatível de `valor_previsto_zona` (R$) |
| `valor_previsto_medio_bairro` | `float` | Alias compatível de `valor_previsto_bairro` (R$) |

---

### 5.3 Nível Global — Cidade de Ribeirão Preto/SP

| Campo | Tipo | Fórmula / Significado |
| :--- | :---: | :--- |
| `global_mediana_mercado` ⭐ | `float` | **Mediana dos imóveis da base de desenvolvimento (R$)** — referência de mercado |
| `global_media_mercado` ⭐ | `float` | **Média dos imóveis da base de desenvolvimento (R$)** — usada no cálculo do desvio |
| `global_mediana_m2_mercado` ⭐ | `float` | **Mediana do preço/m² da base de desenvolvimento (R$/m²)** — base do índice |
| `indice_imovel_global` | `float` | `valor_m2_previsto ÷ global_mediana_m2_mercado` — >1 = acima da mediana |
| `diferenca_perc_global` | `float` | `(valor_previsto − global_media_mercado) ÷ global_media_mercado × 100` (%) |
| `global_desconto_5` | `float` | `global_mediana_mercado × 0.95` — desconto moderado sobre ref. global |
| `global_desconto_10` | `float` | `global_mediana_mercado × 0.90` — desconto agressivo sobre ref. global |
| `global_desconto_15` | `float` | `global_mediana_mercado × 0.85` — desconto de queima sobre ref. global |
| `global_faixa_segura_piso` | `float` | `global_mediana_mercado × 0.90` — piso seguro de negociação (cidade) |
| `global_faixa_segura_teto` | `float` | `global_mediana_mercado` — teto de referência (cidade) |

---

### 5.4 Nível Zona — ex.: Zona Sul

| Campo | Tipo | Fórmula / Significado |
| :--- | :---: | :--- |
| `zona_mediana_mercado` ⭐ | `float` | **Mediana dos imóveis da zona (R$)** — referência de mercado local |
| `zona_media_mercado` ⭐ | `float` | **Média dos imóveis da zona (R$)** — usada no cálculo do desvio |
| `zona_mediana_m2_mercado` ⭐ | `float` | **Mediana do preço/m² da zona (R$/m²)** — base do índice |
| `indice_imovel_zona` | `float` | `valor_m2_previsto ÷ zona_mediana_m2_mercado` — >1 = acima da mediana da zona |
| `diferenca_perc_zona` | `float` | `(valor_previsto − zona_media_mercado) ÷ zona_media_mercado × 100` (%) |
| `zona_desconto_5` | `float` | `zona_mediana_mercado × 0.95` |
| `zona_desconto_10` | `float` | `zona_mediana_mercado × 0.90` |
| `zona_desconto_15` | `float` | `zona_mediana_mercado × 0.85` |
| `zona_faixa_segura_piso` | `float` | `zona_mediana_mercado × 0.90` |
| `zona_faixa_segura_teto` | `float` | `zona_mediana_mercado` |

---

### 5.5 Nível Bairro — ex.: Jardim Botânico

| Campo | Tipo | Fórmula / Significado |
| :--- | :---: | :--- |
| `bairro_mediana_mercado` ⭐ | `float` | **Mediana dos imóveis do bairro (R$)** — referência de mercado local |
| `bairro_media_mercado` ⭐ | `float` | **Média dos imóveis do bairro (R$)** — usada no cálculo do desvio |
| `bairro_mediana_m2_mercado` ⭐ | `float` | **Mediana do preço/m² do bairro (R$/m²)** — base do índice |
| `indice_imovel_bairro` | `float` | `valor_m2_previsto ÷ bairro_mediana_m2_mercado` |
| `diferenca_perc_bairro` | `float` | `(valor_previsto − bairro_media_mercado) ÷ bairro_media_mercado × 100` (%) |
| `bairro_desconto_5` | `float` | `bairro_mediana_mercado × 0.95` |
| `bairro_desconto_10` | `float` | `bairro_mediana_mercado × 0.90` |
| `bairro_desconto_15` | `float` | `bairro_mediana_mercado × 0.85` |
| `bairro_faixa_segura_piso` | `float` | `bairro_mediana_mercado × 0.90` |
| `bairro_faixa_segura_teto` | `float` | `bairro_mediana_mercado` |

> ⭐ = campos de referência histórica da base usada pelo modelo; não representam todos os imóveis existentes na cidade.

---

## 6. Como Interpretar os Resultados

As comparações abaixo são descritivas da base de referência. Os rótulos de oportunidade, premium e negociação são interpretações ilustrativas, não recomendações automáticas de compra.

### Leitura rápida da análise hierárquica

```
valor_previsto = R$ 631.396,73   ← O modelo previu esse valor

GLOBAL (Ribeirão Preto):
  global_mediana_mercado = R$ 290.000,00  ← mediana da cidade
  global_media_mercado   = R$ 422.828,81  ← média da cidade
  diferenca_perc_global  = +49,33%        ← imóvel está 49% ACIMA da média da cidade
  indice_imovel_global   = 1,57x          ← m² previsto é 1,57× a mediana/m² da cidade

ZONA SUL:
  zona_mediana_mercado   = R$ 490.000,00  ← mediana na Zona Sul
  zona_media_mercado     = R$ 670.717,08  ← média na Zona Sul
  diferenca_perc_zona    = -5,86%         ← imóvel está 5,9% ABAIXO da média da zona
  indice_imovel_zona     = 1,10x          ← m² levemente acima da mediana da zona

JARDIM BOTÂNICO:
  bairro_mediana_mercado = R$ 490.000,00  ← referência retornada para o bairro
  bairro_media_mercado   = R$ 670.717,08  ← média no bairro
  diferenca_perc_bairro  = -5,86%         ← imóvel está abaixo da média do bairro
  bairro_faixa_segura_*  = R$ 441.000 – R$ 490.000  ← 90% a 100% da mediana histórica
```

### Guia de interpretação do `indice_imovel_*`

| Faixa | Interpretação |
| :--- | :--- |
| `< 0.85` | Preço/m² previsto mais de 15% abaixo da mediana histórica |
| `0.85 ≤ índice < 1.00` | Preço/m² previsto até 15% abaixo da mediana histórica |
| `1.00 ≤ índice ≤ 1.15` | Preço/m² previsto igual ou até 15% acima da mediana histórica |
| `1.15 < índice ≤ 1.30` | Preço/m² previsto mais de 15% e até 30% acima da mediana histórica |
| `> 1.30` | Preço/m² previsto mais de 30% acima da mediana histórica |

Essas faixas são uma convenção de leitura deste guia; não são classificações devolvidas pela API. O índice compara o preço/m² **previsto**, não o valor anunciado pelo proprietário.

### Guia de interpretação do `diferenca_perc_*`

| Sinal | Significado |
| :--- | :--- |
| Positivo (`+`) | Imóvel previsto ACIMA da média de mercado desse nível |
| Zero | Imóvel alinhado com a média |
| Negativo (`-`) | Preço previsto ABAIXO da média histórica desse nível |


### Exemplos para a equipe de negócio

Os cenários a seguir usam **valores fictícios para explicar os cálculos**. São independentes da resposta ilustrativa da seção 4. As referências históricas vêm da base de desenvolvimento do modelo carregado, não de uma consulta ao mercado em tempo real.

#### Caso 1 — Explicar a estimativa de um imóvel ao proprietário

Considere um apartamento no Jardim Botânico, Zona Sul, com 80 m², 3 quartos, 2 banheiros e 2 vagas. Suponha que a API devolva:

| Campo da API | Valor ilustrativo | Leitura para o atendimento |
| :--- | ---: | :--- |
| `valor_previsto` | R$ 600.000,00 | Estimativa individual para as características informadas |
| `valor_m2_previsto` | R$ 7.500,00/m² | R$ 600.000 ÷ 80 m² |
| `bairro_media_mercado` | R$ 500.000,00 | Média dos preços totais na referência histórica do bairro |
| `bairro_mediana_mercado` | R$ 480.000,00 | Valor central dos preços totais nessa referência |
| `bairro_mediana_m2_mercado` | R$ 6.000,00/m² | Valor central dos preços por m² nessa referência |
| `diferenca_perc_bairro` | 20,00 | Previsão total 20% acima da média histórica: `(600.000 ÷ 500.000 − 1) × 100` |
| `indice_imovel_bairro` | 1,25 | Previsão por m² 25% acima da mediana histórica por m²: `7.500 ÷ 6.000` |

**Como comunicar:** “Para as características informadas, o modelo estima R$ 600 mil. Esse valor total está 20% acima da média histórica do bairro; por metro quadrado, a estimativa está 25% acima da mediana histórica.”

Os percentuais diferem porque usam medidas e referências distintas: preço total versus média total; preço/m² versus mediana por m². Nenhum dos dois mede a valorização do imóvel ao longo do tempo. A base do bairro reúne imóveis com características variadas, não apenas apartamentos idênticos ao avaliado.

#### Caso 2 — Comparar o preço anunciado com a previsão

No mesmo exemplo, suponha que o proprietário anuncie R$ 660.000,00. Esse preço vem do cadastro comercial: **não é um dos seis campos de entrada nem um campo devolvido pela API**.

| Informação | Valor | Origem |
| :--- | ---: | :--- |
| Preço anunciado | R$ 660.000,00 | Cadastro comercial |
| Preço previsto | R$ 600.000,00 | `valor_previsto` |
| Diferença em reais | R$ 60.000,00 | Cálculo da equipe: anunciado − previsto |
| Diferença sobre a previsão | 10,00% | Cálculo da equipe: `(660.000 ÷ 600.000 − 1) × 100` |

**Como comunicar:** “O anúncio está R$ 60 mil, ou 10%, acima da estimativa do modelo.” Isso é diferente de `diferenca_perc_bairro`, que continua sendo 20% no caso 1 e compara a previsão com a média histórica.

Para a análise comercial, registre também características que a entrada atual não recebe, como conservação, andar e condomínio. A diferença calculada ajuda a organizar a revisão do anúncio; não determina automaticamente o preço a praticar.

#### Caso 3 — Ler as simulações de desconto corretamente

Ainda no caso 1, a mediana histórica do bairro é R$ 480.000,00. A API usa essa mediana como base dos seguintes campos:

| Campo da API | Cálculo | Resultado |
| :--- | :--- | ---: |
| `bairro_desconto_5` | R$ 480.000 × 0,95 | R$ 456.000,00 |
| `bairro_desconto_10` | R$ 480.000 × 0,90 | R$ 432.000,00 |
| `bairro_desconto_15` | R$ 480.000 × 0,85 | R$ 408.000,00 |
| `bairro_faixa_segura_piso` | R$ 480.000 × 0,90 | R$ 432.000,00 |
| `bairro_faixa_segura_teto` | R$ 480.000 × 1,00 | R$ 480.000,00 |

**Como comunicar:** “O cenário de 10% abaixo da mediana histórica do bairro corresponde a R$ 432 mil.”

Se a equipe quiser simular 10% de desconto sobre a **previsão individual**, o cálculo separado será `600.000 × 0,90 = R$ 540.000,00`. Sobre o **anúncio** do caso 2, será `660.000 × 0,90 = R$ 594.000,00`. Esses cálculos não correspondem ao campo `bairro_desconto_10`.

A faixa de R$ 432 mil a R$ 480 mil é uma regra sobre a mediana histórica. Apesar do nome dos campos, não representa a incerteza da previsão de R$ 600 mil nem um limite obrigatório para a negociação.

#### Caso 4 — Resumir uma carteira por zona e bairro

Retomando os quatro imóveis do exemplo numérico da seção 3, a carteira enviada tem previsões de R$ 600 mil e R$ 400 mil no Jardim Botânico, R$ 800 mil na Nova Aliança e R$ 300 mil no Centro.

| Recorte da carteira enviada | Quantidade | Previsão média por imóvel | Soma das previsões individuais |
| :--- | ---: | ---: | ---: |
| Zona Sul / Jardim Botânico | 2 | R$ 500.000,00 | R$ 1.000.000,00 |
| Zona Sul / Nova Aliança | 1 | R$ 800.000,00 | R$ 800.000,00 |
| Zona Sul / todos os bairros enviados | 3 | R$ 600.000,00 | R$ 1.800.000,00 |
| Centro / Centro | 1 | R$ 300.000,00 | R$ 300.000,00 |
| Carteira completa | 4 | R$ 525.000,00 | R$ 2.100.000,00 |

As linhas de zona e de carteira são subtotais e total; não devem ser somadas às linhas dos bairros. A soma é calculada pela equipe a partir de `valor_previsto` de cada imóvel; a API não devolve um campo de valor total da carteira. Essa soma resume estimativas, não receita realizada.

**Como comunicar:** “Nos três imóveis da Zona Sul enviados nesta análise, a estimativa média é R$ 600 mil. O total estimado desses três imóveis é R$ 1,8 milhão.”

`valor_previsto_zona` será R$ 600 mil nas três linhas da Zona Sul. `valor_previsto_bairro` será R$ 500 mil nas duas linhas do Jardim Botânico e R$ 800 mil na linha da Nova Aliança. O primeiro imóvel mantém sua previsão individual de R$ 600 mil; as médias não representam outros imóveis simulados.

Se a chamada contiver somente esse primeiro imóvel, suas médias de zona e bairro serão ambas R$ 600 mil. Os valores iguais são esperados nesse lote. Já `bairro_media_mercado` é uma referência histórica e, com a mesma versão carregada e a mesma localização, não muda por retirar outros imóveis da chamada.

Para comparar carteiras ao longo do tempo, mantenha claros o conjunto de imóveis e a versão do modelo. Uma mudança na média pode decorrer da composição dos imóveis enviados.

#### Caso 5 — Interpretar o preço médio por metro quadrado

Em outro lote ilustrativo, dois imóveis do mesmo bairro têm estas previsões:

| Imóvel | Área | Previsão individual | Previsão por m² |
| :--- | ---: | ---: | ---: |
| A | 50 m² | R$ 400.000,00 | R$ 8.000,00/m² |
| B | 100 m² | R$ 600.000,00 | R$ 6.000,00/m² |

`valor_m2_previsto_bairro` será `(8.000 + 6.000) ÷ 2 = R$ 7.000,00/m²`: cada imóvel recebe o mesmo peso nessa média. Dividir a soma dos preços pela soma das áreas daria `1.000.000 ÷ 150 = R$ 6.666,67/m²`, uma medida diferente, calculada separadamente.

**Como comunicar:** “A média dos preços por metro quadrado previstos para os dois imóveis enviados é R$ 7 mil.”

#### Caso 6 — Explicar uma referência de bairro ausente

Se o bairro informado não estiver na tabela histórica, mas a zona estiver, a API utiliza as referências da zona nos campos históricos do bairro. Por exemplo, com mediana da zona de R$ 480 mil, `bairro_mediana_mercado` também será R$ 480 mil e `bairro_desconto_10` será R$ 432 mil.

A previsão individual continua sendo calculada pelo modelo, e `valor_previsto_bairro` continua sendo a média das previsões do grupo enviado. O fallback das referências não transforma a previsão individual na mediana da zona.

Os 40 campos atuais não incluem um indicador explícito de fallback nem a quantidade de amostras históricas. Valores iguais entre zona e bairro, por si só, não comprovam que houve fallback; as estatísticas também podem coincidir. Para afirmar a origem da referência em um relatório, confirme a presença da localidade na base de referência do modelo.

#### Modelo de resumo para atendimento

Usando os casos 1 e 2:

> Apartamento de 80 m², 3 quartos, 2 banheiros e 2 vagas, no Jardim Botânico / Zona Sul. Estimativa do modelo: R$ 600.000,00, equivalente a R$ 7.500,00/m². O preço anunciado de R$ 660.000,00 está 10% acima dessa estimativa. A previsão está 20% acima da média histórica de preços totais do bairro. As referências refletem a base utilizada pelo modelo; as características adicionais do imóvel devem compor a análise comercial.

No registro interno, associe esse resumo à data da consulta, à versão efetivamente carregada no serving e ao identificador do imóvel no cadastro comercial. Esses dados de controle não fazem parte dos 40 campos da resposta; o alias `champion`, sozinho, não identifica permanentemente uma versão.

---

## 7. Lógica de Fallback Hierárquico

No caminho vetorizado usado pela API, uma referência de bairro ausente recebe a referência da zona; uma zona ausente recebe a global. **O limite mínimo de amostra não é aplicado nesse caminho**, embora exista um resolvedor de suficiência no método de previsão individual e estados amostrais na agregação de treino.

As estatísticas de referência vêm da base de desenvolvimento guardada no modelo. A tabela histórica de bairros é indexada pelo nome do Bairro; já as médias de previsão no lote agrupam pela combinação Zona/Bairro.

As simulações `*_desconto_5`, `*_desconto_10`, `*_desconto_15` e `*_faixa_segura_*` usam as medianas históricas de cada nível no caminho vetorizado. A expressão “faixa segura” é o nome do campo de negócio: não significa intervalo de confiança, garantia de retorno ou avaliação de risco financeiro. Os percentuais estão no código, não no YAML atual.

Os 40 campos descrevem a saída do empacotador completo com motor imobiliário. Entradas inválidas podem ser rejeitadas pelo MLflow; o fallback genérico sem motor não garante esse contrato completo. A telemetria identifica metragem inválida, mas isso não acrescenta uma rejeição de domínio à API: o denominador do preço por m² é limitado a pelo menos 1,0.

---

## 8. Resumo dos 40 Campos de Saída

| # | Campo | Nível |
| :---: | :--- | :---: |
| 1 | `Zona` | Identificação |
| 2 | `Bairro` | Identificação |
| 3 | `valor_previsto` | Base |
| 4 | `valor_m2_previsto` | Base |
| 5 | `global_mediana_mercado` | Global |
| 6 | `global_media_mercado` | Global |
| 7 | `global_mediana_m2_mercado` | Global |
| 8 | `indice_imovel_global` | Global |
| 9 | `diferenca_perc_global` | Global |
| 10 | `global_desconto_5` | Global |
| 11 | `global_desconto_10` | Global |
| 12 | `global_desconto_15` | Global |
| 13 | `global_faixa_segura_piso` | Global |
| 14 | `global_faixa_segura_teto` | Global |
| 15 | `zona_mediana_mercado` | Zona |
| 16 | `zona_media_mercado` | Zona |
| 17 | `zona_mediana_m2_mercado` | Zona |
| 18 | `indice_imovel_zona` | Zona |
| 19 | `diferenca_perc_zona` | Zona |
| 20 | `zona_desconto_5` | Zona |
| 21 | `zona_desconto_10` | Zona |
| 22 | `zona_desconto_15` | Zona |
| 23 | `zona_faixa_segura_piso` | Zona |
| 24 | `zona_faixa_segura_teto` | Zona |
| 25 | `bairro_mediana_mercado` | Bairro |
| 26 | `bairro_media_mercado` | Bairro |
| 27 | `bairro_mediana_m2_mercado` | Bairro |
| 28 | `indice_imovel_bairro` | Bairro |
| 29 | `diferenca_perc_bairro` | Bairro |
| 30 | `bairro_desconto_5` | Bairro |
| 31 | `bairro_desconto_10` | Bairro |
| 32 | `bairro_desconto_15` | Bairro |
| 33 | `bairro_faixa_segura_piso` | Bairro |
| 34 | `bairro_faixa_segura_teto` | Bairro |
| 35 | `valor_previsto_zona` | Valor previsto médio do lote por Zona |
| 36 | `valor_previsto_bairro` | Valor previsto médio do lote por Zona e Bairro |
| 37 | `valor_m2_previsto_zona` | R$/m² previsto médio do lote por Zona |
| 38 | `valor_m2_previsto_bairro` | R$/m² previsto médio do lote por Zona e Bairro |
| 39 | `valor_previsto_medio_zona` | Alias compatível de `valor_previsto_zona` |
| 40 | `valor_previsto_medio_bairro` | Alias compatível de `valor_previsto_bairro` |


## Observabilidade da API

O dashboard [Previsão Imobiliária](http://localhost:3000/d/painel-previsao-imoveis) acompanha requisições, erros, latência, entradas e preços previstos por localidade. As médias do dashboard resumem os imóveis atendidos no período selecionado; os campos `valor_previsto_zona` e `valor_previsto_bairro` desta API resumem somente o lote da chamada. Consulte [a documentação de observabilidade](observabilidade.md#interpretar-serving) para as métricas, fontes e limitações.
