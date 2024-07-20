# Análise e Projeção do Preço de Tokens

![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-100%25-blue)
![Last Commit](https://img.shields.io/badge/Last%20Commit-July%202024-yellow)
![Contribuições](https://img.shields.io/badge/Contribuições-Bem%20Vindas-brightgreen)

## Descrição do Projeto

### Visão Geral

Este projeto realiza a análise e projeção de preços de tokens de criptomoedas utilizando uma combinação de modelos estatísticos, Inteligência Artificial Generativa da Google e um Sistema de Multiagentes Crew AI. Ele visa fornecer previsões precisas e análises detalhadas para ajudar traders e investidores a tomarem decisões informadas.

### Tecnologias Utilizadas

- **Python**: Linguagem de programação principal utilizada para desenvolver o projeto.
- **Crew AI**: Framework para criação e gestão de agentes inteligentes.
- **LangChain**: Biblioteca para integração com modelos de linguagem.
- **Google Generative AI**: Serviço de IA generativa para criar análises textuais.
- **TensorFlow**: Biblioteca de aprendizado de máquina utilizada para construir e treinar modelos LSTM.
- **scikit-learn**: Biblioteca para aprendizado de máquina utilizada para pré-processamento de dados e validação cruzada.
- **TA-Lib**: Biblioteca para análise técnica de séries temporais financeiras.

### Modelos Estatísticos e Técnicas Aplicadas

1. **Indicadores Técnicos**:
   - **RSI (Relative Strength Index)**: Indicador de momentum que mede a velocidade e a mudança dos movimentos de preço.
   - **EMA (Exponential Moving Average)**: Média móvel exponencial que dá mais peso aos preços recentes.
   - **ADX (Average Directional Index)**: Indicador que mede a força da tendência.
   - **Bollinger Bands**: Faixas que indicam a volatilidade do mercado.
   - **MACD (Moving Average Convergence Divergence)**: Indicador de tendência que mostra a relação entre duas médias móveis de preços.
   - **Ichimoku Cloud**: Indicador que define níveis de suporte e resistência, identifica a direção da tendência, mede o momentum e fornece sinais de trading.
   - **MFI (Money Flow Index)**: Indicador que mede a entrada e saída de dinheiro de um ativo.

2. **Modelos de Aprendizado de Máquina**:
   - **LSTM (Long Short-Term Memory)**: Tipo de rede neural recorrente usada para prever séries temporais devido à sua capacidade de aprender dependências de longo prazo.
   - **CNN-LSTM (Convolutional Neural Network - Long Short-Term Memory)**: Combinação de CNN para extração de características e LSTM para previsão de séries temporais.

### Resultados Apresentados

O projeto fornece os seguintes resultados para cada criptomoeda analisada:

1. **Preço Atual**: O preço mais recente da criptomoeda.
2. **Indicadores Técnicos**: Valores calculados para RSI, EMA, ADX, Bollinger Bands, MACD, Ichimoku Cloud, MFI e VWAP.
3. **Previsão de Preço**:
   - **Projeção Usando Indicadores**: Previsão baseada nos valores dos indicadores técnicos.
   - **Projeção Usando LSTM**: Previsão baseada em um modelo CNN-LSTM treinado com os dados históricos.
   - **Previsão para os Próximos 5 Dias**: Previsões diárias para os próximos 5 dias usando o modelo LSTM.
4. **Erro Médio Quadrático (MSE)**: Métrica de desempenho do modelo LSTM.
5. **Análise Textual**: Análise detalhada e contextualizada gerada pela Inteligência Artificial Generativa da Google, considerando os dados históricos e notícias recentes.

Esses resultados são enviados diretamente para um chat no Telegram, proporcionando acesso rápido e conveniente às análises e previsões.

## Estrutura do Projeto

```sh
├── Analise_Crewai.py
├── CoinGecko_Analise_Token.py
├── CoinGecko_Analise_Token_Com_AI.py
├── Envio_Resultado_Telegram.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Dependências

Para executar este projeto, você precisará das seguintes bibliotecas e dependências:

- `requests`: Biblioteca para fazer requisições HTTP.
- `pandas`: Biblioteca para manipulação e análise de dados.
- `numpy`: Biblioteca para computação numérica.
- `ta`: Biblioteca para análise técnica de séries temporais financeiras.
- `scikit-learn`: Biblioteca para aprendizado de máquina, usada para pré-processamento de dados e validação cruzada.
- `tensorflow`: Biblioteca de aprendizado de máquina usada para construir e treinar modelos LSTM.
- `crewai`: Biblioteca para criação e gestão de agentes inteligentes.
- `crewai_tools`: Ferramentas adicionais para integração com Crew AI.
- `langchain_groq`: Biblioteca para integração com modelos de linguagem Groq.
- `langchain_google_genai`: Biblioteca para integração com a IA Generativa da Google.
- `langchain_community.tools`: Ferramentas da comunidade para integração com LangChain.
- `langchain_core.tools`: Ferramentas principais para integração com LangChain.
- `langchain_openai`: Biblioteca para integração com modelos OpenAI.

### Instalação

1. Clone o repositório:
   ```sh
   git clone https://github.com/BrunoAraujo84/Modelo-Analise-Projecao-Token.git
   cd Modelo-Analise-Projecao-Token
   ```

2. Crie um ambiente virtual e ative-o:
   ```sh
   python -m venv venv
   source venv/bin/activate  # No Windows use \`venv\Scripts\activate\`
   ```

3. Instale as dependências listadas no arquivo `requirements.txt`:
   ```sh
   pip install -r requirements.txt
   ```

4. Configure as seguintes variáveis de ambiente com suas respectivas chaves de API:
   ```
   - GEMINI_API_KEY
   - LLAMA3_API_KEY
   - SERPER_API_KEY
   - OPENAI_API_KEY
   - BOT_TELEGRAM_API_KEY
   - BOT_TELEGRAM_CHAT_ID
   ```
