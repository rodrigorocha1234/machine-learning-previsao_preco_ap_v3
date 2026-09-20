# Manual de Regras de Negócio e Governança Imobiliária

Este documento descreve detalhadamente as regras de negócio, alçadas comerciais, métricas financeiras e políticas operacionais aplicadas no sistema de precificação inteligente de imóveis (`previsao-preco-imoveis`).

---

## 1. Princípio Fundamental de Arquitetura

O sistema adota o princípio de **desacoplamento estrito** entre inteligência analítica e decisão comercial:

$$\text{Decisão de Negócio} = f(\text{Previsão de Mercado}, \text{Políticas Comerciais})$$

* **O Modelo de Machine Learning** estima o valor justo de mercado do imóvel ($\text{Valor\_Previsto}$ ou $\hat{y}$), com base nas características estruturais e localização.
* **As Regras de Negócio** determinam como essa estimativa pode ser comercializada, aplicando limites de desconto, margens de segurança e validação de alçadas operacionais.

Dessa forma, qualquer alteração nas políticas comerciais (como redução de comissões, promoções de feirões ou limites de alçada gerencial) pode ser realizada instantaneamente via configuração, **sem necessidade de retreinar os modelos de Machine Learning**.

---

## 2. Política de Concessão de Descontos e Alçadas

### 2.1. Fórmula de Precificação com Desconto

Quando um cliente ou corretor solicita negociação de valor, a aplicação do desconto segue a fórmula matemática:

$$\text{Valor\_Com\_Desconto} = \text{Valor\_Previsto} \times \left(1 - \frac{\text{Percentual\_Desconto}}{100}\right)$$

### 2.2. Faixas de Alçada Comercial

Os limites percentuais são configuráveis no arquivo [`configuracao.yaml`](file:///home/rodrigo/PycharmProjects/machine-learning-previsao_preco_ap_v3/configuracao.yaml) através da classe [`LimitesDesconto`](file:///home/rodrigo/PycharmProjects/machine-learning-previsao_preco_ap_v3/src/imobiliaria_ml/negocio_pkg/limites_desconto.py):

| Faixa de Desconto | Nível de Alçada | Descrição e Comportamento Operacional |
| :--- | :---: | :--- |
| **$< 0.0\%$** | 🚫 **Inválido** | **Desconto Proibido**: Não é permitida a cobrança de ágio forçado via desconto negativo. A requisição é rejeitada com erro de validação. |
| **$0.0\% \le d \le 5.0\%$** | 🟢 **Automático** | **Alçada do Corretor**: Desconto comercial concedido diretamente na ponta pelo sistema/corretor no momento da proposta. |
| **$5.0\% < d \le 10.0\%$** | 🟡 **Sob Aprovação** | **Alçada Gerencial**: Requer autorização da gerência comercial ou diretoria de vendas antes do fechamento da proposta. |
| **$> 10.0\%$** | 🔴 **Bloqueado** | **Desconto Rejeitado**: Ultrapassa o teto máximo permitido da imobiliária. O sistema bloqueia a transação para evitar corrosão de margem. |

### 2.3. Algoritmo do Desconto Seguro Recomendado

Para orientar os corretores sobre a margem de desconto recomendada sem comprometer a rentabilidade nem vender abaixo do valor patrimonial, o módulo [`CalculadorDescontoSeguro`](file:///home/rodrigo/PycharmProjects/machine-learning-previsao_preco_ap_v3/src/imobiliaria_ml/negocio_pkg/calculador_desconto_seguro.py) calcula dinamicamente o desconto seguro com base no Erro Absoluto Médio ($\text{MAE}$) do modelo campeão:

1. **Taxa de Incerteza do Modelo**:
   $$\text{Taxa de Erro} = \frac{\text{MAE}_{\text{holdout}}}{\bar{y}_{\text{referência}}}$$

2. **Cálculo do Desconto Seguro**:
   $$\text{Desconto Seguro (\%)} = \min\left(\text{Teto Máximo}, \max\left(0, \left(\text{Margem de Risco} + \frac{\text{Taxa de Erro}}{2}\right) \times 100\right)\right)$$

Essa métrica é registrada nos metadados de negócio do MLflow e orienta a equipe sobre a folga média recomendada para fechamento rápido de contratos.

---

## 3. Métricas Financeiras e Operacionais da Imobiliária

Além das métricas estatísticas usuais de Machine Learning ($\text{RMSE}$, $\text{R}^2$, $\text{MAPE}$), o módulo [`CalculadorMetricasImobiliaria`](file:///home/rodrigo/PycharmProjects/machine-learning-previsao_preco_ap_v3/src/imobiliaria_ml/negocio_pkg/calculador_metricas_imobiliaria.py) avalia o impacto financeiro da precificação sobre a carteira de imóveis no conjunto Holdout.

### 3.1. Índices de Cobertura Comercial (Hit Rates)

Mede o percentual de imóveis cujas previsões ficaram dentro de faixas aceitáveis de tolerância de mercado:

* **$\text{Cobertura}_{\pm 5\%}$**: Proporção de avaliações com erro relativo $\le 5\%$. Representa precificações de altíssima precisão.
* **$\text{Cobertura}_{\pm 10\%}$**: Proporção de avaliações com erro relativo $\le 10\%$. É o padrão de mercado para aceitação automática de laudos de avaliação.
* **$\text{Cobertura}_{\pm 15\%}$**: Proporção de avaliações com erro relativo $\le 15\%$. Delimita a margem de segurança contra litígios ou perdas de negociação.

### 3.2. Viés Médio de Precificação ($\text{Bias}$)

Mede a tendência sistemática do modelo de sobrevalorizar ou subvalorizar a carteira:

$$\text{Viés Médio} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)$$

* **Viés Positivo ($> 0$) — Risco de Superavaliação:**  
  O modelo tende a estimar preços acima do mercado real.  
  *Impacto de negócio:* Aumento no tempo médio de venda (**dias de vacância**), perda de clientes compradores e custo de oportunidade de estoque encalhado.
* **Viés Negativo ($< 0$) — Risco de Subavaliação:**  
  O modelo tende a precificar abaixo do mercado.  
  *Impacto de negócio:* Perda de receita de comissão para a imobiliária e prejuízo ao patrimônio do proprietário parceiro.

### 3.3. Riscos de Cauda Extremas

* **Risco de Subprecificação Severa**:
  $$\text{Risco}_{\text{sub}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left(\frac{\hat{y}_i - y_i}{y_i} < -15\%\right)$$
  Identifica a porcentagem de imóveis com risco de venda a preço vil.
* **Risco de Superprecificação Severa**:
  $$\text{Risco}_{\text{super}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left(\frac{\hat{y}_i - y_i}{y_i} > +15\%\right)$$
  Identifica o percentual de imóveis com alta probabilidade de rejeição imediata pelo mercado.

---

## 4. Segmentação Analítica do Negócio

O modelo campeão não é avaliado apenas globalmente; ele é decomposto em dimensões operacionais estratégicas para o negócio imobiliário:

### 4.1. Desempenho por Faixa de Preço

A carteira é dividida em 4 segmentos de valor venal:
1. **Popular / Econômico**: Até R$ 300.000,00
2. **Médio Padrão**: De R$ 300.000,00 a R$ 600.000,00
3. **Médio-Alto Padrão**: De R$ 600.000,00 a R$ 1.000.000,00
4. **Alto Padrão / Luxo**: Acima de R$ 1.000.000,00

> **Objetivo de Negócio**: Imóveis de alto padrão apresentam maior variância absoluta em reais, enquanto imóveis populares exigem menor erro percentual para viabilizar financiamentos bancários (Caixa / SBPE).

### 4.2. Desempenho por Zona Geográfica

As métricas são agrupadas por região (`Centro`, `Zona Sul`, `Zona Norte`, `Zona Leste`, `Zona Oeste`):
* Identifica regiões com escassez de dados ou maior volatilidade de preços.
* Permite à diretoria aplicar políticas de margem diferenciadas por região.

---

## 5. Governança e Validação no Serving de Produção

Em ambiente produtivo, o modelo é exposto via API REST ([`ModeloImobiliarioPyFunc`](file:///home/rodrigo/PycharmProjects/machine-learning-previsao_preco_ap_v3/src/imobiliaria_ml/mlflow_pkg/modelo_imobiliario_pyfunc.py)) na porta `5002` do endpoint `/invocations`.

### 5.1. Regra de Entrada e Saída da API

#### Entrada (`POST /invocations`)
```json
{
  "dataframe_records": [
    {
      "Zona": "Centro",
      "Quartos": 3,
      "Banheiros": 2,
      "Vagas": 2,
      "Metragem": 125.5,
      "Percentual_Desconto": 5.0
    }
  ]
}
```

* O campo `Percentual_Desconto` é **opcional**. Se omitido, o sistema assume `0.0%`.
* Se `Percentual_Desconto` for informado fora da faixa $[0.0, 10.0]$, o serviço rejeita a requisição antes da inferência, protegendo a empresa contra falhas operacionais.

#### Saída Consolidada
```json
[
  {
    "Valor_Previsto": 650000.00,
    "Percentual_Desconto": 5.0,
    "Valor_Com_Desconto": 617500.00
  }
]
```

---

## 6. Políticas de Monitoramento Contínuo e Drift (PSI)

A imobiliária opera com regras claras para identificar obsolescência do modelo em produção via Population Stability Index (PSI) no módulo [`MonitorDrift`](file:///home/rodrigo/PycharmProjects/machine-learning-previsao_preco_ap_v3/src/imobiliaria_ml/drift_pkg/monitor_drift.py):

| Valor do PSI | Nível de Drift | Ação de Negócio Obrigatória |
| :---: | :---: | :--- |
| **$\text{PSI} < 0.10$** | 🟢 **Estável** | Nenhuma ação requerida. Distribuição de imóveis condizente com o treinamento. |
| **$0.10 \le \text{PSI} < 0.25$** | 🟡 **Atenção** | Mudança moderada no perfil imobiliário (ex: aumento de lançamentos de estúdios ou valorização regional). Alerta gerencial emitido. |
| **$\text{PSI} \ge 0.25$** | 🔴 **Crítico** | **Retreinamento Obrigatório**. O modelo perdeu representatividade econômica e deve ser reajustado com os novos dados. |

---

## 7. Parametrização via `configuracao.yaml`

Todas as alçadas e políticas são mantidas no arquivo [`configuracao.yaml`](file:///home/rodrigo/PycharmProjects/machine-learning-previsao_preco_ap_v3/configuracao.yaml):

```yaml
negocio:
  desconto_minimo: 0.0                 # Desconto mínimo aceito (0%)
  desconto_maximo_automatico: 5.0      # Alçada automática do corretor (5%)
  desconto_maximo_com_aprovacao: 10.0  # Alçada máxima com aprovação gerencial (10%)

drift:
  psi_atencao: 0.10                   # Gatilho de monitoramento preventivo
  psi_forte: 0.25                     # Gatilho mandatório de retreino
```
