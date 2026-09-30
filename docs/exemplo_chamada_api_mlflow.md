# Exemplo de Chamada da API do MLflow com Regras de Negócio Imobiliárias

Este documento detalha o contrato de integração com a API de Serving do **MLflow**, apresentando os formatos de requisição sem parâmetros vazados/indesejados e o payload de resposta enriquecido com todos os resultados da inteligência de negócio imobiliário (Ribeirão Preto/SP) **organizados por hierarquia: Global → Zona → Bairro**.

---

## 1. Visão Geral do Endpoint

- **Endpoint de Inferência:** `http://localhost:8080/invocations`
- **Método HTTP:** `POST`
- **Cabeçalho:** `Content-Type: application/json`
- **Modelo Servido:** `previsao_preco_apartamento_modelo` (Alias `@champion`)

---

## 2. Contrato de Entrada (Input Payload)

Os parâmetros de entrada foram estritamente limpos. **Não é necessário e nem permitido** fornecer `Código`, `Apartamento`, `valor_m2`, `media_valor_m2_bairro` ou `media_valor_m2_zona`. Identificadores do imóvel e métricas derivadas não entram na inferência; o modelo e o motor de regras calculam as métricas dinamicamente.

### Campos Obrigatórios

| Campo | Tipo | Descrição | Exemplo |
| :--- | :--- | :--- | :--- |
| `Bairro` | `string` | Bairro em Ribeirão Preto/SP | `"Jardim Botânico"` |
| `Zona` | `string` | Zona da cidade (`Zona Sul`, `Zona Leste`, etc.) | `"Zona Sul"` |
| `Quartos` | `long` | Quantidade de quartos | `3` |
| `Banheiros` | `long` | Quantidade de banheiros | `2` |
| `Vagas_Garagem` | `long` | Quantidade de vagas de garagem | `2` |
| `Metragem` | `double` | Área útil privativa do imóvel em m² | `85.0` |

---

## 3. Exemplos de Chamada da API

### Exemplo A: Chamada via `cURL` (Formato `dataframe_split`)

```bash
curl -X POST "http://localhost:8080/invocations" \
     -H "Content-Type: application/json" \
     -d '{
       "dataframe_split": {
         "columns": [
           "Bairro",
           "Zona",
           "Quartos",
           "Banheiros",
           "Vagas_Garagem",
           "Metragem"
         ],
         "data": [
           ["Jardim Botânico", "Zona Sul", 3, 2, 2, 85.0]
         ]
       }
     }'
```

---

### Exemplo B: Chamada via `cURL` (Formato `records` / Lista de Objetos)

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

---

### Exemplo C: Chamada em Python (`requests`)

```python
import requests

url = "http://localhost:8080/invocations"
headers = {"Content-Type": "application/json"}

payload = {
    "dataframe_split": {
        "columns": [
            "Bairro",
            "Zona",
            "Quartos",
            "Banheiros",
            "Vagas_Garagem",
            "Metragem",
        ],
        "data": [
            ["Jardim Botânico", "Zona Sul", 3, 2, 2, 85.0]
        ],
    }
}

resposta = requests.post(url, json=payload, headers=headers)
dados = resposta.json()

print("Resposta Completa da API:")
print(dados)
```

---

## 4. Contrato de Saída Enriquecido — Estrutura Hierárquica (Output Payload)

A API retorna o valor predito pelo modelo vencedor **junto com todas as variáveis analíticas organizadas em três níveis de hierarquia geográfica**: Global (cidade), Zona e Bairro.

```json
{
  "predictions": [
    {
      "valor_previsto": 631396.73,
      "valor_m2_previsto": 7428.20,

      "indice_imovel_global": 1.57,
      "diferenca_perc_global": 49.33,
      "global_desconto_5": 399750.00,
      "global_desconto_10": 378750.00,
      "global_desconto_15": 357750.00,
      "global_faixa_segura_piso": 378750.00,
      "global_faixa_segura_teto": 420000.00,

      "indice_imovel_zona": 1.10,
      "diferenca_perc_zona": -5.86,
      "zona_desconto_5": 474250.00,
      "zona_desconto_10": 449250.00,
      "zona_desconto_15": 424250.00,
      "zona_faixa_segura_piso": 449250.00,
      "zona_faixa_segura_teto": 499000.00,

      "indice_imovel_bairro": 1.10,
      "diferenca_perc_bairro": -5.86,
      "bairro_desconto_5": 568257.06,
      "bairro_desconto_10": 536687.22,
      "bairro_desconto_15": 505117.38,
      "bairro_faixa_segura_piso": 536687.22,
      "bairro_faixa_segura_teto": 598063.58
    }
  ]
}
```

> **Nota de interpretação hierárquica:**
> - Os campos `global_*` são calculados com base na **mediana de todos os imóveis de Ribeirão Preto/SP**.
> - Os campos `zona_*` são calculados com base na **mediana de imóveis da Zona** informada (ex.: Zona Sul).
> - Os campos `bairro_*` são calculados com base na **mediana de imóveis do Bairro** informado (ex.: Jardim Botânico). Caso o bairro não tenha amostra suficiente, usa-se o nível de Zona como fallback; caso a Zona também falhe, usa-se o Global.

---

## 5. Dicionário dos Campos de Saída

### 5.1 Campos Base

| Campo de Saída | Tipo | Descrição |
| :--- | :--- | :--- |
| `valor_previsto` | `float` | Valor justo de venda predito pelo modelo (R$) |
| `valor_m2_previsto` | `float` | Razão `valor_previsto / Metragem` (R$/m²) |

---

### 5.2 Nível Global (Referência: Cidade de Ribeirão Preto/SP)

| Campo de Saída | Tipo | Descrição |
| :--- | :--- | :--- |
| `indice_imovel_global` | `float` | `valor_m2_previsto / mediana_m2_cidade` — Relação com a mediana geral da cidade. > 1 indica acima da média. |
| `diferenca_perc_global` | `float` | `((valor_previsto - media_cidade) / media_cidade) * 100` — Desvio em % em relação à média global. |
| `global_desconto_5` | `float` | `mediana_global * 0.95` — Desconto moderado (5%) sobre referência global. |
| `global_desconto_10` | `float` | `mediana_global * 0.90` — Desconto agressivo (10%) sobre referência global. |
| `global_desconto_15` | `float` | `mediana_global * 0.85` — Desconto queima (15%) sobre referência global. |
| `global_faixa_segura_piso` | `float` | `mediana_global * 0.90` — Piso seguro de negociação no contexto da cidade. |
| `global_faixa_segura_teto` | `float` | `mediana_global` — Teto de referência no contexto da cidade. |

---

### 5.3 Nível Zona (Referência: Zona informada — ex.: Zona Sul)

| Campo de Saída | Tipo | Descrição |
| :--- | :--- | :--- |
| `indice_imovel_zona` | `float` | `valor_m2_previsto / mediana_m2_zona` — Relação com a mediana da Zona. |
| `diferenca_perc_zona` | `float` | `((valor_previsto - media_zona) / media_zona) * 100` — Desvio em % em relação à média da Zona. |
| `zona_desconto_5` | `float` | `mediana_zona * 0.95` — Desconto moderado (5%) sobre referência da Zona. |
| `zona_desconto_10` | `float` | `mediana_zona * 0.90` — Desconto agressivo (10%) sobre referência da Zona. |
| `zona_desconto_15` | `float` | `mediana_zona * 0.85` — Desconto queima (15%) sobre referência da Zona. |
| `zona_faixa_segura_piso` | `float` | `mediana_zona * 0.90` — Piso seguro de negociação no contexto da Zona. |
| `zona_faixa_segura_teto` | `float` | `mediana_zona` — Teto de referência no contexto da Zona. |

---

### 5.4 Nível Bairro (Referência: Bairro informado — ex.: Jardim Botânico)

| Campo de Saída | Tipo | Descrição |
| :--- | :--- | :--- |
| `indice_imovel_bairro` | `float` | `valor_m2_previsto / mediana_m2_bairro` — Relação com a mediana do Bairro. |
| `diferenca_perc_bairro` | `float` | `((valor_previsto - media_bairro) / media_bairro) * 100` — Desvio em % em relação à média do Bairro. |
| `bairro_desconto_5` | `float` | `mediana_bairro * 0.95` — Desconto moderado (5%) sobre referência do Bairro. |
| `bairro_desconto_10` | `float` | `mediana_bairro * 0.90` — Desconto agressivo (10%) sobre referência do Bairro. |
| `bairro_desconto_15` | `float` | `mediana_bairro * 0.85` — Desconto queima (15%) sobre referência do Bairro. |
| `bairro_faixa_segura_piso` | `float` | `mediana_bairro * 0.90` — Piso seguro de negociação no contexto do Bairro. |
| `bairro_faixa_segura_teto` | `float` | `mediana_bairro` — Teto de referência no contexto do Bairro. |

---

## 6. Lógica de Fallback Hierárquico

Quando um bairro ou zona não possui amostra estatística suficiente, o motor de regras de negócio aplica o seguinte fallback automático:

```
Bairro (suficiente) → Zona (suficiente) → Global (sempre disponível)
```

Isso garante que **todas as requisições sempre retornam valores válidos**, sem erros ou nulos, independentemente da raridade do bairro ou zona informados.
