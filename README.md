# Usado multiagentes no google antigravity para construir um sistema de Previsão inteligente de preço de apartamento para a cidade de Ribeirão Preto
## 1 – Introdução

O objetivo desse projeto é orquestrar vários agentes usando o google antigravity para construção de um sistema de precificação imobiliária de apartamentos da cidade de Ribeirão Preto.

## 2 – Tecnologias usadas




- 🐍 Python
- 🐳 Docker
- 🧪 Mlflow
- 🤖 Sklearn

## 3 -  Requisitos dos Sistemas:

### 3.1 – Equipe multiagente e especificações formais (ESPECS)

O projeto foi concebido e implementado sob uma governança multiagente estrita. Cada aspecto do sistema — desde a concepção dos requisitos de negócio até a infraestrutura de contêineres — é gerido por agentes de inteligência artificial especializados com responsabilidades delimitadas, barreiras de segurança (*guardrails*) e validação cruzada contínua através de especificações formais (specs/) e regras inegociáveis de código (rules/).

#### 3.1.1 - **A Equipe de 7 Agentes Especializados (.agents/agents.md)**

Nenhum agente opera de forma generalista ou arbitrária. A segregação de funções impede sobreposição de escopo e garante auditoria completa em cada commit:

**@pmProduct Manager**  
** Lead Architect & Especificações Auditáveis**  
** Missão: Traduz as necessidades comerciais em requisitos formais auditáveis. Consolida `specs/` e `rules/`, gerencia o Approval Gate e preserva o isolamento entre Inner CV, Outer CV e Holdout.**  
** 🚫 Restrições Rígidas: Não escreve código de produção; não altera regras sem aval humano; proíbe terminantemente FastAPI (exige MLflow Serving nativo).**

**@ml_architectArquiteto de ML**  
** Contratos GoF & Tipagem Estrita**  
** Missão: Define a arquitetura do pipeline, tipagem estática (Protocols, TypeVar, Generic), padrões de projeto (Factory, Adapter, Repository, Strategy, Observer) e organização modular de pacotes em português.**  
** 🚫 Política "No-If": Proibido usar `if`, `elif` ou `match` para decisões de domínio/arquitetura. Exige polimorfismo, dispatch tables ou Chain of Responsibility.**

**@data_engineerEngenheiro de Dados**  
** Ingestão, Staging & Detecção de Drift**  
** Missão: Responsável por ingestão de fontes heterogêneas, validação de contratos de dados (Quality Gate), persistência imutável em staging SQLite, análise descritiva e testes de estabilidade (KS, PSI, Wasserstein).**  
** 🚫 Vetorização Pura: Proibido processamento linha a linha em Pandas (banidos: `iterrows`, `itertuples` e loops manuais).**

**@ml_engineerEngenheiro de ML**  
** Nested CV, Tuning & Ensembles**  
** Missão: Implementa pré-processamento anti-leakage, orquestra Nested Cross-Validation (15 folds externos × 5 internos), gerencia busca de hiperparâmetros (Grid e Random Search), auditoria de resíduos e comitê de modelos.**  
** 🚫 Zero Hardcode: Todos os hiperparâmetros e espaços de busca vêm de arquivos YAML. Obrigatório traduzir parâmetros para linguagem de negócio após treino.**

**@mlopsMLOps & Observabilidade**  
** Registry, Serving & Telemetria**  
** Missão: Gerencia o MLflow Tracking (runs pai/filho), registra artefatos sob o alias `@champion` no MLflow Registry, configura o servidor de inferência com adaptador ASGI e provê métricas OpenMetrics para Prometheus e Grafana.**  
** 🚫 Cardinalidade Controlada: Proibido uso de labels de alta cardinalidade no Prometheus; garante desacoplamento de storage de artefatos (S3/RustFS).**

**@qaQA & Security Auditor**  
** Compliance, Regras & Auditoria de PR**  
** Missão: Atua como o fiscal implacável do projeto. Audita o código contra as `specs/` e `rules/`, verificando tipagem estática com MyPy, integridade de testes com PyTest e conformidade arquitetural.**  
** 🚫 Bloqueios Fatais: Rejeita sumariamente código com uso de `Any`, decisões de domínio com `if`, leitura prematura do holdout ou ausência de testes.**

**@devopsDevOps & Infra**  
** Docker Compose, Resiliência & Healthchecks**  
** Missão: Orquestra a infraestrutura local e em nuvem via Docker Compose. Gerencia microsserviços (PostgreSQL, RustFS S3, MLflow, Prometheus, Grafana, Loki e Alloy), assegurando readiness, healthchecks e scripts reproduzíveis.**  
** 🚫 Reprodutibilidade Obrigatória: Todo o ambiente deve subir com um único comando declarativo (`docker compose up -d`) sem intervenções manuais.**

### 3.1.2 -  As 4 Especificações Formais do Sistema (specs/)

**Contratos Técnicos Auditáveis**

Cada documento em `specs/` atua como o contrato canônico do sistema, definindo escopo, premissas matemáticas e evidências de validação:

**Arquivo:** `01_pipeline_ml.md`

**Título da Especificação:** **Spec 01 — Pipeline de ML**

**Escopo Técnico & Requisitos Obrigatórios:** Orquestração sequencial de 20 etapas determinísticas. Particionamento treino/holdout (80/20) com semente 42. Nested Cross-Validation com `RepeatedKFold` externo (5×3) e `KFold` interno (5 splits). Auditoria estatística com testes de Shapiro-Wilk, Friedman, Nemenyi e Tukey.

**Garantia de Governança:** **Isolamento de Holdout & Anti-Leakage**

---

**Arquivo:** `02_modelos_tuning.md`

**Título da Especificação:** **Spec 02 — Modelos & Tuning**

**Escopo Técnico & Requisitos Obrigatórios:** Fábrica com suporte a até 10 algoritmos de regressão (Lineares, Ridge, Lasso, ElasticNet, Árvores, Random Forest, SVR, MLP, XGBoost, CatBoost). Busca adaptativa via `GridSearchCV` e `RandomizedSearchCV` parametrizada em YAML. Montagem de comitê `VotingRegressor` ponderado.

**Garantia de Governança:** **Tuning Declarativo & Rastreabilidade**

---

**Arquivo:** `03_dados_negocio.md`

**Título da Especificação:** **Spec 03 — Dados & Regras de Negócio**

**Escopo Técnico & Requisitos Obrigatórios:** Definição da variável alvo (`Valor_da_Venda`). Enriquecimento hierárquico em 3 níveis (Global, Zona, Bairro). Validador de suficiência amostral (mínimo 30 amostras na Zona e 20 no Bairro) com fallback geográfico. Corredores de segurança (-10%/+10%) e descontos escalonados (5%, 10%, 15%).

**Garantia de Governança:** **Defesa Patrimonial & Coerência**

---

**Arquivo:** `04_observabilidade.md`

**Título da Especificação:** **Spec 04 — Observabilidade & Serving**

**Escopo Técnico & Requisitos Obrigatórios:** Servidor de inferência nativo no MLflow Serving com adaptador ASGI de telemetria na porta 8080. Exposição de 56 métricas nativas do Prometheus na porta 8000. Persistência atômica `treino.prom` para painéis Grafana contínuos. Separação rigorosa entre métricas de treino e observações de produção.

**Garantia de Governança:** **Telemetria em Tempo Real (SLA < 150ms)**

###  3.1.3 - As 4 Regras Inegociáveis de Arquitetura (rules/)

**Padrões Verificados pelo Auditor @qa**

Diretrizes estritas aplicadas na base de código para manter excelência de engenharia de software e eliminar dívida técnica:

**Regra 01: Python 3.12 Estrito & Sem Any**\
`rules/01_python_rules.md`

Código com tipagem estática absoluta (dataclasses congeladas, tuplas imutáveis, Protocols e genéricos). Uso de `Any` é proibido sem autorização explícita. Exatamente uma classe principal por arquivo para alta coesão e baixo acoplamento.

**Regra 02: Política "No-If" em Regras de Domínio**\
`rules/02_no_if_rules.md`

Decisões de domínio não utilizam cascatas de `if`/`elif`/`match`. Substituídas por padrões GoF: *Strategy*, *State*, *Specification*, *Dispatch Tables* e polimorfismo, viabilizando extensibilidade sem modificação de código consolidado (OCP).

**Regra 03: Vetorização Pura em Pandas**\
`rules/03_pandas_rules.md`

Processamento de dados exclusivamente por operações colunares vetorizadas em C/NumPy. Proibição estrita de `iterrows()`, `itertuples()` ou `apply(axis=1)`, garantindo velocidade máxima e baixo consumo de memória.

**Regra 04: Governança de ML & Anti-Leakage**\
`rules/04_ml_rules.md`

Hiperparâmetros estritamente externos (YAML). Todo pré-processamento (imputação, encoding, scaling) deve ocorrer estritamente dentro do Pipeline/ColumnTransformer de cada fold. O cofre de holdout permanece bloqueado até o congelamento da versão.

### 3.2 – Requisitos funcionais:

 | ID | Requisito Funcional | Descrição Técnica & Comportamento Esperado | Módulo Responsável | 
 | --------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------- | 
 | **RF-01** | **Ingestão de Dados & Anti-Leakage** | Carregar dataset de **5.712 apartamentos** (`dados/bairro_final_v3_engineered.xlsx`) e purgar imediatamente colunas com risco de vazamento de target (`Código`, `Apartamento`, `valor_m2`, médias locais de m²). | `CarregadorExcel` | 
 | **RF-02** | **Validação de Contrato & Quality Gate** | Validar a presença da coluna alvo (`Valor`), integridade de tipos e limites físicos estritos (metragem > 0, quartos ≥ 0, banheiros ≥ 0, vagas ≥ 0), emitindo métricas de qualidade ao Prometheus. | `ValidadorContrato` | 
 | **RF-03** | **Persistência em Staging SQLite** | Persistir snapshot integral e imutável dos dados brutos validados na tabela SQLite `apartamentos_staging` para rastreabilidade, auditoria e reprodutibilidade histórica. | `RepositorioStaging` | 
 | **RF-04** | **Isolamento em Cofre de Holdout** | Particionar a base em **80% treino / 20% teste cego**, com semente determinística fixa (`42`). Bloquear a partição de 20% no `CofreHoldout`, proibindo leitura durante EDA, CV e tuning. | `CofreHoldout` | 
 | **RF-05** | **Análise Exploratória & Detecção de Drift** | Calcular estatísticas descritivas sobre os 80% de treino e quantificar estabilidade distributiva via testes **Kolmogorov-Smirnov (KS)**, **PSI** e **distância de Wasserstein** para variáveis numéricas contínuas. | `DetectorDrift` | 
 | **RF-06** | **Nested Cross-Validation (15×5 Folds)** | Avaliar modelos em **15 folds externos** (`RepeatedKFold`: 5 splits × 3 repetições) e ajustar hiperparâmetros em **5 folds internos** (`KFold`), mantendo imputação, codificação e scaling estritamente dentro de cada fold. | `AvaliadorAninhado` | 
 | **RF-07** | **Fábrica de Modelos & Tuning Adaptativo** | Suportar até **10 algoritmos de regressão** e aplicar `GridSearchCV` para modelos lineares/rápidos (**Ridge, ElasticNet**) e `RandomizedSearchCV` para ensembles complexos (**CatBoost, XGBoost, Random Forest**). | `FabricaTuning` | 
 | **RF-08** | **Auditoria Estatística Não-Paramétrica** | Extrair resíduos pareados dos folds externos e aplicar testes de normalidade (**Shapiro-Wilk**), significância de rankings (**Friedman**) e pós-testes de **Nemenyi** e **Tukey** para comprovação matemática. | `TesteFriedman` | 
 | **RF-09** | **Seleção do Campeão & Comitê VotingRegressor** | Eleger o modelo individual campeão ou construir um comitê `VotingRegressor` ponderado com os **Top K modelos** que sejam estatisticamente indistinguíveis pelo teste de Friedman (**p-valor < 0.05**). | `SeletorCampeao` | 
 | **RF-10** | **Motor Imobiliário & Enriquecimento Multinível** | Calcular referências de mercado históricas em **3 níveis (Global, Zona, Bairro)** e enriquecer as previsões com corredores de negociação, índices comparativos de m² e descontos de **5%, 10% e 15%**. | `MotorImobiliario` | 
 | **RF-11** | **Empacotamento Unificado PyFunc no MLflow** | Empacotar o pipeline Scikit-Learn ajustado e o `MotorImobiliario` em um wrapper único `mlflow.pyfunc`, registrando o modelo no MLflow Registry com tag `@champion` e assinatura de **32/40 campos**. | `ObservadorMlflow` | 
 | **RF-12** | **Serving REST de Inferência em Tempo Real** | Disponibilizar endpoint HTTP na porta **8080** (`/invocations`) compatível com JSON nos formatos `dataframe_records` e `dataframe_split`, retornando resposta enriquecida e expondo `/metrics` na porta **8000**. | `servidor_inferencia` | 

### 3.3 – Requitos não funcionais:

 | **ID** | **Requisito Não Funcional** | **Descrição Técnica & Comportamento Esperado** | 
 | ---------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | 
 | **RNF-01** | **Performance & Latência** | **SLA < 150ms**. A API de inferência deve processar requisições unitárias em menos de **80ms em média**, com **p95 < 150ms** e **p99 < 300ms**, viabilizando uso interativo em tempo real por portais e corretores. | 
 | **RNF-02** | **Determinismo & Reprodutibilidade** | **Zero aleatoriedade**. Todas as divisões de dados, validação aninhada e inicialização de modelos utilizam **sementes pseudoaleatórias fixas configuradas via YAML** (semente **42**), garantindo reprodução idêntica de métricas entre execuções. | 
 | **RNF-03** | **Governança Anti-Leakage** | **Lacre estrito**. É proibido qualquer contato do conjunto de teste ou folds de validação com os transformadores de features (`OneHotEncoder`, `RobustScaler`, `SimpleImputer`), eliminando contaminação estatística (*data leakage*). | 
 | **RNF-04** | **Imutabilidade Pré-Teste** | **Trava de segurança**. O cofre de *holdout* só pode ser destravado com chave mestra após o modelo final e o motor imobiliário serem formalmente congelados (`estado_congelado = True`), impossibilitando retreinos ou sobreajustes induzidos após o teste. | 
 | **RNF-05** | **Observabilidade** | **OpenMetrics — 56 métricas nativas**. Exposição de métricas no padrão Prometheus com persistência atômica via `PersistenciaMetricas`, permitindo que os dashboards Grafana continuem ativos mesmo após o término do pipeline de treino. | 
 | **RNF-06** | **Resiliência em Contêineres** | **Docker Compose**. Arquitetura desacoplada em microsserviços via Docker Compose, com isolamento de **PostgreSQL**, armazenamento de artefatos S3 (**RustFS**), **MLflow Tracking**, **Prometheus**, **Grafana** e **Serving REST**. | 
 | **RNF-07** | **Usabilidade & Responsividade** | **Mobile-First**. A documentação oficial e o simulador interativo operam com fluidez em **smartphones** (drawer lateral com overlay), **tablets** e **monitores widescreen**, sem quebras de layout ou transbordamento horizontal. | 
 | **RNF-08** | **Tipagem Estrita & Manutenibilidade** | **Python 3.12**. Código desenvolvido com tipagem estática rigorosa, utilizando **dataclasses congeladas**, tipos genéricos, tuplas imutáveis e verificação **MyPy**, assegurando integridade e documentação viva no próprio código. | 

### 3-4 – Regras de negócio:

 | **Regra** | **Conceito de Negócio** | **Fórmula / Definição Operacional** | **Impacto & Aplicação na Mesa de Vendas** | 
 | --------- | ------------------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | 
 | **RN-01** | **Preço Justo Algorítmico** | `valor_previsto` — estimativa do comitê de ML | **Âncora de Avaliação:** Ponto central de equilíbrio para captação e precificação inicial do apartamento. | 
 | **RN-02** | **Equalizador Unitário de Área** | `valor_m2_previsto = valor_previsto ÷ max(Metragem, 1.0)` | **Comparador Universal:** Permite equalizar unidades de metragens distintas no mesmo prédio ou bairro. | 
 | **RN-03** | **Hierarquia em 3 Níveis** | Comparações simultâneas em: **Global (Município), Zona e Bairro** | **Contextualização Completa:** Evita precificar um imóvel de alto padrão com base na média genérica da cidade. | 
 | **RN-04** | **Suficiência & Fallback Geográfico** | Bairros `< 20` amostras ou Zonas `< 30` amostras herdam a referência superior | **Proteção Estatística:** Impede que bairros com poucas transações gerem referências distorcidas ou voláteis. | 
 | **RN-05** | **Corredor de Segurança (Piso & Teto)** | `faixa_segura_piso = -10%` \| `faixa_segura_teto = +10%` | **Defesa Patrimonial:** O piso é a **“linha vermelha”** (nunca vender abaixo sem aval); o teto é a margem de vitrine nos portais. | 
 | **RN-06** | **Política Escalonada de Desconto** | **5%** (cortesia), **10%** (limite do corretor), **15%** (deságio urgente) | **Agilidade Comercial:** Estabelece a alçada de flexibilização do corretor sem necessidade de consultar a gerência. | 
 | **RN-07** | **Índice do Imóvel** | `indice_imovel_* = valor_m2_previsto ÷ mediana_m2_mercado` | **Diferencial Competitivo:** `> 1.15x` indica padrão construtivo superior/vista; `< 0.85x` indica oportunidade ou necessidade de reforma. | 
 | **RN-08** | **Desvio Percentual de Mercado** | `diferenca_perc_* = (valor_previsto − media_mercado) ÷ media × 100` | **Qualificação do Comprador:** Mostra se a unidade pertence ao padrão médio local ou se está no topo da pirâmide de renda. | 

## 4 – Interpretação para a equipe de negócio
## 🗺️ Panorama Territorial: Interpretação Prática de 1 Bairro por Zona

### 5 Regiões de Ribeirão Preto

Ribeirão Preto possui dinâmicas imobiliárias marcadamente heterogêneas. Avaliar um imóvel sem contextualizar a sua macrorregião leva a erros graves: subprecificar apartamentos de alto padrão no sul ou inflacionar unidades econômicas no norte.

Abaixo, detalhamos a **interpretação prática do retorno da API para 1 bairro representativo de cada uma das 5 Zonas** da cidade:

| Zona | Bairro Representativo | Tipologia | Valor Previsto | Previsto / m² | Mediana do Bairro | Índice m² | Faixa Segura (Piso / Teto) | Posicionamento de Mercado |
|---|---|---|---|---|---|---|---|---|
| **Zona Sul** | **Jardim Botânico** | 85 m² • 3Q • 2V | **R$ 631.396,73** | R$ 7.428,20 | R$ 626.000 (R$ 6.934/m²) | **1,07x** | R$ 568,2 mil – R$ 694,5 mil | Alto Padrão / Nobre & Lazer |
| **Zona Leste** | **Jardim Paulista** | 68 m² • 2Q • 1V | **R$ 312.800,00** | R$ 4.600,00 | R$ 300.000 (R$ 4.600/m²) | **1,00x** | R$ 281,5 mil – R$ 344,0 mil | Classe Média Consolidada & Comércio |
| **Zona Oeste** | **Sumarezinho** | 65 m² • 2Q • 1V | **R$ 275.000,00** | R$ 4.230,77 | R$ 245.000 (R$ 4.348/m²) | **0,97x** | R$ 247,5 mil – R$ 302,5 mil | Polo Universitário USP & Rentabilidade |
| **Zona Norte** | **Campos Elíseos** | 58 m² • 2Q • 1V | **R$ 238.000,00** | R$ 4.103,45 | R$ 260.000 (R$ 4.167/m²) | **0,98x** | R$ 214,2 mil – R$ 261,8 mil | Tradicional, Popular & 1º Imóvel |
| **Centro** | **Centro** | 95 m² • 3Q • 1V | **R$ 365.000,00** | R$ 3.842,11 | R$ 380.000 (R$ 3.842/m²) | **1,00x** | R$ 328,5 mil – R$ 401,5 mil | Plantas Amplas, Retrofit & Serviços a Pé |

### 💡 Nota de Governança Comercial — Faixa Segura de Negociação (RN-05)

A coluna **Faixa Segura (Piso / Teto)** é calculada estritamente sobre o **Valor Previsto da unidade avaliada**:

- **Piso = -10%:** alçada máxima para contrapropostas e pagamento à vista.
- **Teto = +10%:** valor de vitrine nos portais para margem de barganha.

**Exemplo prático do Sumarezinho:**

Para uma unidade estimada em **R$ 275.000,00**, o corredor seguro situa-se entre **R$ 247,5 mil e R$ 302,5 mil** — posicionando a estimativa do modelo no **centro exato da negociação**, sem conflito com a mediana histórica geral de R$ 245 mil (que reflete unidades menores de 56 m²).

---

### 1. Zona Sul — Jardim Botânico

#### Alto Padrão

**Perfil Imobiliário:**

Vetor de maior valorização de Ribeirão Preto. Apartamento de 85 m² avaliado a **R$ 631,3 mil (R$ 7.428/m²)**.

O índice do bairro é **1,07x**, o que reflete a qualidade do condomínio com lazer clube frente à mediana de R$ 6.934/m².

**🗣️ Argumento de Fechamento:**

> "No Jardim Botânico, unidades de 3 quartos com 2 vagas sustentam valor acima de R$ 7,4 mil/m² devido à alta demanda e parques. Pela regra RN-05, nosso piso de segurança é R$ 568,2 mil para pagamento à vista, com teto de vitrine em R$ 694,5 mil."

---

### 2. Zona Leste — Jardim Paulista

#### Classe Média

**Perfil Imobiliário:**

Região de classe média consolidada (Av. Treze de Maio). Apartamento de 68 m² previsto a **R$ 312,8 mil (R$ 4.600/m²)**, exatamente alinhado ao m² mediano do bairro (índice **1,00x**), assegurando excelente liquidez.

**🗣️ Argumento de Fechamento:**

> "O valor deste apartamento está perfeitamente alinhado à média do Jardim Paulista. Não há risco de sobrepreço. O corredor operacional (RN-05) baliza propostas de R$ 281,5 mil (-10%) a R$ 344,0 mil (+10%), garantindo liquidez imediata."

---

### 3. Zona Oeste — Sumarezinho

#### Universitário / USP

**Perfil Imobiliário:**

Bairro dinâmico vizinho ao campus da USP. Apartamento de 65 m² estimado a **R$ 275 mil (R$ 4.230/m²)**.

O m² previsto está **3% abaixo da mediana local (índice 0,97x)**, configurando excelente oportunidade de rendimento.

**🗣️ Argumento de Fechamento:**

> "Excelente ativo para investidores: o custo de R$ 4.230/m² é competitivo. O valor previsto de R$ 275 mil está no centro exato da faixa segura da unidade (R$ 247,5 mil a R$ 302,5 mil), permitindo absorver propostas à vista a partir de R$ 247,5 mil sem comprometer a margem patrimonial."

---

### 4. Zona Norte — Campos Elíseos

#### 1º Imóvel / Popular

**Perfil Imobiliário:**

Bairro tradicional com forte comércio (Av. Saudade). Apartamento de 58 m² estimado a **R$ 238 mil (R$ 4.103/m²)**.

Ticket de entrada acessível com custo de manutenção condominial reduzido e farto transporte público.

**🗣️ Argumento de Fechamento:**

> "Ideal para quem quer sair do aluguel ou adquirir o primeiro apartamento. Com previsão de R$ 238 mil, a faixa segura de negociação (RN-05) situa-se entre R$ 214,2 mil e R$ 261,8 mil, viabilizando financiamentos facilitados."

---

### 5. Centro — Centro Histórico

#### Plantas Amplas / Retrofit

**Perfil Imobiliário:**

Edifícios consolidados com metragens amplas (mediana de 103 m²). Apartamento de 95 m² previsto a **R$ 365 mil (R$ 3.842/m²)**.

Apresenta o **menor valor por m² da cidade** entre bairros centrais, com enorme potencial de reforma (*retrofit*).

**🗣️ Argumento de Fechamento:**

> "Comprar 95 m² a R$ 3.842/m² no Centro é o melhor negócio em espaço por real da cidade. Avaliado em R$ 365 mil, seu corredor seguro (RN-05) vai de R$ 328,5 mil a R$ 401,5 mil, dando margem confortável para reformas."

# 5 - Interpretação dos resultados da API preditiva
## 📘 Manual de Vendas: Como o Corretor e o Gestor Devem Interpretar os 32 Campos da API

Quando o sistema de CRM ou aplicativo do corretor consome o endpoint `/invocations`, a resposta oferece 3 pilares de decisão para a mesa de negociação:

### 1. Preço de Entrada

#### `valor_previsto` & `valor_m2_previsto`

É a âncora técnica de valor justo. Use para balizar o anúncio inicial. O preço do m² permite comparar apartamentos de metragens diferentes no mesmo condomínio.

### 2. Posicionamento

#### `indice_imovel_bairro` & `diferenca_perc_*`

Mostra se a unidade é de padrão superior (índice > 1.15) ou oportunidade de liquidez rápida (índice < 0.85). Justifica o preço para o comprador com base em dados de mercado.

### 3. Balizador de Contraproposta

#### `*_faixa_segura_piso` & `*_desconto_10`

Define o limite máximo de concessão de desconto antes da venda se tornar prejudicial ao proprietário ou descolada da realidade local do bairro.

### Casos Práticos de Negociação

#### Cenário 1: O Proprietário Quer Pedir Muito Acima do Mercado

Se o proprietário quer anunciar por R$ 660 mil, mas a API retorna `valor_previsto: 600.000` e `diferenca_perc_bairro: +20%`: o corretor tem o argumento perfeito:

> "O valor pedido está 10% acima da avaliação algorítmica do modelo e 20% acima da média do bairro. Sem um ajuste ou diferencial exclusivo de acabamento, o imóvel corre alto risco de ficar meses encalhado."

#### Cenário 2: O Comprador Faz uma Proposta Muito Baixa

Se o comprador oferece R$ 450 mil em um imóvel avaliado em R$ 600 mil, o corretor consulta `bairro_faixa_segura_piso` (R$ 540 mil):

> "A sua proposta de R$ 450 mil está 17% abaixo do piso seguro de negociação da inteligência de dados para o Jardim Botânico. O valor mínimo viável para fechamento é R$ 540 mil."

#### Cenário 3: Avaliação de Carteira de Imóveis (Lotes Múltiplos)

Ao enviar um lote de 20 apartamentos de uma construtora, a API preenche `valor_previsto_zona` e `valor_previsto_bairro` com as médias consolidadas do lote, permitindo que a diretoria visualize instantaneamente o ticket médio por empreendimento e região.

---

### 🔍 Exemplo Prático Real: Apartamento no Sumarezinho (Zona Oeste)

#### Caso de Análise Real

Veja a interpretação dos dados retornados para uma unidade de **85 m² no Sumarezinho**:

| Indicador | Valor | Interpretação |
|---|---|---|
| **Preço Previsto** | **R$ 291.609,93** | R$ 3.430,71 / m² |
| **Posição no Bairro** | **+3,49%** | Alinhado com a média local |
| **Índice do m²** | **0,78x** | 22% abaixo da mediana do m² |

**Diagnóstico de Negócio:**

Embora o preço total (R$ 291,6 mil) esteja quase 20% acima da média da Zona Oeste (que é puxada para baixo por unidades minúsculas de 40–50 m²), o **custo por metro quadrado deste imóvel (R$ 3.430/m²) é extremamente vantajoso** (22% abaixo da mediana de R$ 4.415/m² do Sumarezinho).

**🗣️ Argumento de Venda:**

> "Apartamento com metragem ampla (85m²) oferecendo custo por m² de oportunidade, pronto para compradores que priorizam espaço sem pagar o valor inflacionado das zonas centrais."

---

### 💎 Estudo de Caso de Alto Padrão: Jardim Botânico (R$ 801.852,60)

#### Interpretação Completa dos 32 Campos

Abaixo está a **tradução comercial elemento por elemento** do retorno da API para um apartamento de 85 m² no **Jardim Botânico (Zona Sul)**, orientando corretores e diretores comerciais sobre como usar cada métrica para fechar vendas e defender margens:

#### 1. Identificação e Precificação Base do Imóvel

| Campo no JSON | Valor Retornado | Conceito Imobiliário | Interpretação Prática para a Equipe de Negócio & Roteiro de Conversa |
|---|---|---|---|
| `valor_previsto` | **R$ 801.852,60** | Preço Justo Algorítmico | **Âncora de Avaliação:** O valor central estimado pelo comitê de Machine Learning. Use como o preço sugerido de anúncio ou proposta inicial firme. |
| `valor_m2_previsto` | **R$ 9.433,56 / m²** | Valor Unitário de Área | **Métrica de Equalização:** Permite comparar esta unidade de 85 m² com apartamentos de outros tamanhos no mesmo bairro ou condomínio. |
| `Zona` | `"Zona Sul"` | Macrorregião | Região de maior prestígio e valorização imobiliária em Ribeirão Preto, com maior liquidez para o segmento de médio-alto padrão. |
| `Bairro` | `"Jardim Botânico"` | Micro-localização | Bairro nobre consolidado com ampla infraestrutura, parques e gastronomia, justificando ticket médio superior à média municipal. |

---

#### 2. Contexto Macrolocal: Cidade de Ribeirão Preto (Nível Global)

| Campo no JSON | Valor Retornado | Conceito Imobiliário | Interpretação Prática para a Equipe de Negócio & Roteiro de Conversa |
|---|---|---|---|
| `global_mediana_mercado` | R$ 290.000,00 | Mediana Municipal | Metade dos apartamentos vendidos em toda a cidade custa até R$ 290 mil. Demonstra o padrão popular/médio do município como um todo. |
| `global_media_mercado` | R$ 422.828,81 | Média Municipal | Média aritmética de Ribeirão Preto (puxada para cima pelos imóveis de alto padrão). |
| `global_mediana_m2_mercado` | R$ 4.444,44 / m² | Mediana do m² Urbano | Preço mediano do metro quadrado na cidade. Balizador macro de custo construtivo e terreno em Ribeirão Preto. |
| `diferenca_perc_global` | **+176,50%** | Desvio sobre a Média Urbana | **Enquadramento de Alto Padrão:** Este imóvel custa quase 3 vezes a média da cidade. Mostra ao cliente que se trata de um produto exclusivo do topo da pirâmide. |
| `indice_imovel_global` | **2,12x** | Múltiplo do m² Municipal | O m² desta unidade (R$ 9.433/m²) é **2,12 vezes mais valorizado** que a mediana municipal (R$ 4.444/m²), atestando a nobreza da localização. |
| `global_faixa_segura_piso` | R$ 721.667,34 | Piso Global Seguro (−10%) | Valor mínimo de proteção patrimonial considerando o benchmark de 10% de margem operacional. |
| `global_faixa_segura_teto` | R$ 882.037,86 | Teto Global Seguro (+10%) | Valor máximo aceitável de anúncio caso o apartamento conte com diferenciais exclusivos (andar alto, armários planejados de luxo). |

---

#### 3. Contexto Regional: Zona Sul

| Campo no JSON | Valor Retornado | Conceito Imobiliário | Interpretação Prática para a Equipe de Negócio & Roteiro de Conversa |
|---|---|---|---|
| `zona_mediana_mercado` | R$ 480.000,00 | Mediana da Zona Sul | O ponto médio dos imóveis da Zona Sul é R$ 480 mil, mostrando que a zona opera em patamar 65% acima da cidade. |
| `zona_media_mercado` | R$ 548.232,96 | Média da Zona Sul | Valor médio de negociação dos apartamentos situados na Zona Sul. |
| `zona_mediana_m2_mercado` | R$ 6.470,59 / m² | Mediana do m² na Zona | Patamar de referência do metro quadrado da Zona Sul como um todo. |
| `zona_valor_previsto` | R$ 801.852,60 | Média do Lote na Zona | Média das previsões para os imóveis da Zona Sul enviados na chamada (idêntica ao valor unitário em requisição individual). |
| `valor_m2_previsto_zona` | R$ 9.433,56 / m² | Média do m² no Lote | Preço médio do metro quadrado previsto para as unidades da Zona Sul no lote avaliado. |
| `diferenca_perc_zona` | **+67,05%** | Desvio sobre a Zona Sul | O apartamento está **67% acima da média da Zona Sul**, justificando que não é um padrão comum da região, mas sim um produto premium. |
| `indice_imovel_zona` | **1,46x** | Múltiplo do m² da Zona | O metro quadrado está **46% acima da mediana da Zona Sul**, evidenciando o diferencial competitivo do Jardim Botânico. |
| `zona_faixa_segura_piso` | R$ 721.667,34 | Piso Regional (−10%) | Piso de proteção de margem para evitar subprecificação em negociações na Zona Sul. |
| `zona_faixa_segura_teto` | R$ 882.037,86 | Teto Regional (+10%) | Teto recomendado para margem de negociação e gordura de contraproposta na Zona Sul. |

---

#### 4. Contexto Microlocal: Jardim Botânico (Nível Bairro)

| Campo no JSON | Valor Retornado | Conceito Imobiliário | Interpretação Prática para a Equipe de Negócio & Roteiro de Conversa |
|---|---|---|---|
| `bairro_mediana_mercado` | R$ 440.000,00 | Mediana do Bairro | Metade dos apartamentos do Jardim Botânico custa até R$ 440 mil (bairro com mix de unidades menores de 1-2 quartos e grandes plantas). |
| `bairro_media_mercado` | R$ 506.000,00 | Média do Bairro | Ticket médio dos apartamentos negociados no Jardim Botânico. |
| `bairro_mediana_m2_mercado` | R$ 7.428,00 / m² | Mediana do m² no Bairro | **Referência Chave:** O m² de mercado típico no Jardim Botânico gira em torno de R$ 7,4 mil/m². |
| `bairro_valor_previsto` | R$ 801.852,60 | Média do Lote no Bairro | Média das previsões para os apartamentos do Jardim Botânico incluídos na mesma requisição. |
| `valor_m2_previsto_bairro` | R$ 9.433,56 / m² | Média do m² no Bairro | Preço médio previsto por m² para as unidades do Jardim Botânico no lote. |
| `diferenca_perc_bairro` | **+82,24%** | Desvio sobre o Bairro | O imóvel custa 82% a mais que a média do Jardim Botânico porque combina **planta de 85 m² + 2 vagas + 2 banheiros**, superando as plantas compactas locais. |
| `indice_imovel_bairro` | **1,27x** | Múltiplo do m² do Bairro | **Defesa do Preço:** O m² previsto (R$ 9.433/m²) é **27% mais valorizado** que a mediana do bairro (R$ 7.428/m²). Justifica-se por padrão construtivo superior, vista ou lazer completo. |

---

#### 5. Estratégia de Negociação, Margens de Desconto e Fechamento

| Campo no JSON | Valor Retornado | Conceito Imobiliário | Interpretação Prática para a Equipe de Negócio & Roteiro de Conversa |
|---|---|---|---|
| `bairro_desconto_5` | R$ 761.759,97 | Desconto Cortesia (5%) | **Negociação Rápida:** Concessão inicial de fechamento para pagamento à vista ou cliente com crédito pré-aprovado. Preserva 95% da margem. |
| `bairro_desconto_10` | R$ 721.667,34 | Desconto Comercial (10%) | **Limite de Flexibilidade do Corretor:** Oferta para destravar negociação travada. Corresponde exatamente ao piso de segurança. |
| `bairro_desconto_15` | R$ 681.574,71 | Desconto Agressivo (15%) | **Zona de Alerta de Deságio:** Aceitável apenas em condições extraordinárias (proprietário com urgência extrema de liquidez). |
| `bairro_faixa_segura_piso` | **R$ 721.667,34** | Piso Seguro de Venda | **Linha Vermelha:** O corretor *nunca deve aceitar propostas abaixo deste valor* sem autorização expressa da diretoria comercial. |
| `bairro_faixa_segura_teto` | **R$ 882.037,86** | Teto Seguro de Venda | **Gordura de Anúncio:** Preço máximo inicial para colocar no portal imobiliário garantindo margem para concessão de desconto sem corroer o preço justo. |

---

### 🎙️ Pitch Comercial Pronto para o Corretor Fechar a R$ 800 mil

> "O valor de R$ 801.850,00 foi validado pelo comitê de inteligência de mercado com base em 5.712 transações reais de Ribeirão Preto. No Jardim Botânico, a mediana do metro quadrado é de R$ 7.428,00; este apartamento foi precificado a R$ 9.433,00/m² (índice 1,27x) devido aos diferenciais de 85 m², 2 vagas cobertas e condomínio de alto padrão. Para fechamento à vista nesta semana, temos margem aprovada até o piso seguro de R$ 761.700,00 (desconto de 5%), garantindo uma compra sólida dentro do corredor de segurança patrimonial."

# 6 – Demonstração do projeto:*



[![Assistir ao vídeo de demonstração do projeto](https://img.shields.io/badge/🎬%20Assistir%20ao%20vídeo-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtu.be/gMFt4OhIO2w)


[Link do reposítório](https://github.com/rodrigorocha1234/machine-learning-previsao_preco_ap_v3)