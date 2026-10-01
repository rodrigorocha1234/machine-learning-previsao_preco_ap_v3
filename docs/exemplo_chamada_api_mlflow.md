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
  bairro_mediana_mercado = R$ 490.000,00  ← mediana no bairro (fallback = zona)
  bairro_media_mercado   = R$ 670.717,08  ← média no bairro
  diferenca_perc_bairro  = -5,86%         ← imóvel está abaixo da média do bairro
  bairro_faixa_segura    = R$ 441.000 – R$ 490.000  ← range saudável de negociação
```

### Guia de interpretação do `indice_imovel_*`

| Faixa | Interpretação |
| :--- | :--- |
| `< 0.85` | Imóvel muito barato em relação ao mercado desse nível |
| `0.85 – 1.00` | Imóvel abaixo da mediana (oportunidade) |
| `1.00 – 1.15` | Imóvel alinhado com o mercado |
| `1.15 – 1.30` | Imóvel acima da mediana (premium) |
| `> 1.30` | Imóvel significativamente acima do mercado |

### Guia de interpretação do `diferenca_perc_*`

| Sinal | Significado |
| :--- | :--- |
| Positivo (`+`) | Imóvel previsto ACIMA da média de mercado desse nível |
| Zero | Imóvel alinhado com a média |
| Negativo (`-`) | Imóvel previsto ABAIXO da média desse nível — possível oportunidade |

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
