# 🧠 MLP Ablation Study - Adult Income Classification

Projeto experimental de Machine Learning utilizando uma rede neural MLP (Multi-Layer Perceptron) desenvolvida e executada sobre o framework `bert_cpu`.

O objetivo deste trabalho é analisar rigorosamente como diferentes escolhas de hiperparâmetros (funções de ativação, tamanho da camada oculta e taxa de aprendizado) influenciam o desempenho preditivo, o custo computacional (mensurado em GFLOPs) e a eficiência geral da rede aplicada ao problema de classificação de renda do _Adult Income Dataset_.

---

# 📌 Objetivo do Experimento

O estudo de ablação avalia o impacto isolado de três fatores principais na arquitetura:

## 1. Funções de Ativação

Comparação de propriedades matemáticas e dinâmicas de convergência entre:

- **ReLU** ($f(x) = \max(0, x)$)
- **GELU** ($f(x) = x \cdot \Phi(x)$)
- **Tanh** ($f(x) = \tanh(x)$)

**Parâmetros de Controle:** Camada oculta fixa em `64` neurônios e Taxa de Aprendizado fixa em `0.01`.

---

## 2. Número de Neurônios Ocultos (Capacidade Arquitetural)

Avaliação do limite de escalonamento e capacidade de representação interna através dos tamanhos:

- **32** neurônios
- **64** neurônios
- **128** neurônios
- **256** neurônios

**Parâmetros de Controle:** Ativação fixa em `ReLU` e Taxa de Aprendizado fixa em `0.01`.

---

## 3. Taxa de Aprendizado (Learning Rate)

Avaliação da eficiência do otimizador e velocidade de convergência com os valores:

- **0.001** (Convergência conservadora)
- **0.005**
- **0.01** (Padrão)
- **0.05** (Passos agressivos/risco de divergência)

**Parâmetros de Controle:** Ativação fixa em `ReLU` e Camada oculta fixa em `64` neurônios.

---

# 🏗 Arquitetura da Rede

A rede neural segue uma arquitetura feedforward com uma camada oculta,
composta por duas camadas lineares e uma função de ativação intermediária,
mapeando as características demográficas processadas do Adult Dataset para uma classificação binária ($\le 50K$ ou $>50K$).
Input Features (108)
|
v
Linear(108, Hidden)
|
v
Activation Function (ReLU / GELU / Tanh)
|
v
Linear(Hidden, 2)
|
v
Classification Output (Cross Entropy)

### Configurações de Treinamento

- **Otimizador:** Adam
- **Épocas:** 100 por modelo
- **Função de Perda:** Cross Entropy Loss
- **Base de Dados:** Adult Income Dataset (Extraído do Censo de 1994, com ~108 dimensões após codificação de variáveis categóricas).

---

# 📊 Métricas Avaliadas

Para cada variação de hiperparâmetro, o pipeline extrai três dimensões de dados:

### Desempenho Preditivo

- **Train Accuracy:** Ajuste ao conjunto de treino.
- **Validation Accuracy:** Capacidade de generalização intermediária (ajuste de hiperparâmetros).
- **Test Accuracy:** Métrica final de validação em dados não vistos.

### Custo Computacional

- **FLOPs por época:** Quantidade de operações de ponto flutuante em uma passada.
- **FLOPs total:** Acumulado computacional ao fim do treinamento.
- **GFLOPs:** Unidade em bilhões de operações para análise de eficiência de hardware.

### Dinâmica de Treinamento

- **Training Loss / Validation Loss:** Comportamento das curvas ao longo das épocas.
- **Tempo por época:** Custo temporal puro em CPU.

---

# 📁 Estrutura do Projeto

BERT-CPU/
│
├── exercises/
│ └── task_binary_classification.py # Script principal de execução dos experimentos
│── app.py # Código do Dashboard Interativo
│
├── resultados.csv # Métricas consolidadas por experimento
├── historico.csv # Histórico detalhado época por época
│
├── requirements.txt # Dependências do ambiente Python
└── README.md # Documentação acadêmica do projeto

---

# ⚙️ Instalação e Configuração

Siga os passos abaixo para preparar o ambiente de execução local.

### 1. Clonar o Repositório

- git clone https://github.com/GomesJoas/BERT-CPU.git
- cd BERT-CPU

### 2. Criar e Ativar o Ambiente Virtual

**No Windows:**

- python -m venv .venv
- .venv\Scripts\activate

**No Linux / macOS:**

- python -m venv .venv
- source .venv/bin/activate

### 3. Instalar as Dependências

Com o ambiente virtual ativo, instale os pacotes necessários:

- pip install -r requirements.txt

# ▶️ Executando e Reproduzindo os Resultados

**Passo 1:** Execução do Pipeline de Treinamento
Para rodar a rotina automatizada que treina todas as variações descritas na ablação, execute:

- python -m exercises.task_binary_classification
- O script realizará o download/carregamento do Adult Dataset, aplicará o pré-processamento, rodará as 100 épocas de cada modelo configurado e salvará os logs em resultados.csv e historico.csv.

**Passo 2:** Inicialização do Dashboard Streamlit

- Para visualizar as análises de eficiência (Acurácia por GFLOP), curvas de Loss e tabelas comparativas de forma interativa, inicialize o servidor do Streamlit:

- streamlit run streamlit/app.py
- O terminal exibirá um endereço local (ex: http://localhost:8501) para abrir no navegador.

# 🔬 Customização dos Experimentos

Os testes que a fila de execução rodará são totalmente controlados via código. Se você desejar testar novas configurações de arquitetura ou taxas de aprendizado, abra o arquivo:

exercises/task_binary_classification.py

E localize a lista Python encarregada de definir os dicionários de parâmetros:

# Exemplo de dicionário interno na lista de experimentos

experiments = [
{"activation": "relu", "hidden": 64, "lr": 0.01},
{"activation": "gelu", "hidden": 64, "lr": 0.01},

# Adicione novos cenários mantendo este formato...

]
📄 Detalhamento dos Arquivos de Saída
resultados.csv
Armazena a fotografia final de cada modelo após o término do treinamento. Estrutura das colunas:

**activation:** Nome da função utilizada.

**hidden:** Quantidade de neurônios ocultos.

**lr:** Taxa de aprendizado aplicada.

**train_acc** / **val_acc** / **test_acc:** Acurácias finais alcançadas.

**gflops:** Custo computacional total acumulado da arquitetura.

**historico.csv**
Dados temporais e de convergência registrados passo a passo, permitindo a plotagem de gráficos de linha:

**epoch:** Índice da época atual (1 a 100).

**train_loss** / **val_loss:** Progresso da função de perda.

**epoch_flops** / **total_flops:** O comportamento do consumo de computação conforme o treino avança.

**epoch_time:** Tempo em segundos que a CPU levou para processar a época.

# 🔁 Guia Rápido de Replicação

Se você precisa apenas limpar o ambiente e garantir que tudo roda do início com um único bloco de comandos:

# Instala os pacotes necessários

pip install -r requirements.txt

# Executa o treino massivo de ablação

python -m exercises.task_binary_classification

# Abre a interface de análise visual

streamlit run streamlit/app.py

# 🧪 Ambiente e Tecnologias

**Linguagem Principal:** Python 3.x

**Manipulação de Dados:** NumPy, Pandas, Scikit-Learn

**Visualização de Dados:** Plotly, Streamlit

**Framework de Execução:** bert_cpu
