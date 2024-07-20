# Importando Bibliotecas
import os
import time
import ast
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, BaseTool
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from CoinGecko_Analise_Token import main
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

# Buscando chaves de API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
LLAMA3_API_KEY = os.getenv('LLAMA3_API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Verificando chaves de API
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set")
if not LLAMA3_API_KEY:
    raise ValueError("LLAMA3_API_KEY is not set")
if not SERPER_API_KEY:
    raise ValueError("SERPER_API_KEY is not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")

# Configurando as variáveis de ambiente
os.environ['SERPER_API_KEY'] = SERPER_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Configurando os LLMs a serem utilizados pelos Multiagentes
llama3 = ChatGroq(
    api_key=LLAMA3_API_KEY,
    model="llama3-70b-8192",
    temperature=1.8
)

gemini = ChatGoogleGenerativeAI(
    google_api_key=GEMINI_API_KEY,
    model="gemini-pro",
    temperature=0.8
)

openai_gpt = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo",
    temperature=1.8
)

# Configurando parâmetros iniciais
default_params = {
    'rsi_period': 14,
    'ema_period': 50,
    'adx_period': 14,
    'bb_period': 20,
    'stoch_period': 14,
    'mfi_period': 14,
    'macd_fast_period': 12,
    'macd_slow_period': 26,
    'macd_signal_period': 9,
    'ichimoku_tenkan_period': 9,
    'ichimoku_kijun_period': 26,
    'ichimoku_senkou_span_b_period': 52,
    'neurons': 150,
    'layers': 2,
    'dropout_rate': 0.2,
    'l2_reg': 0.01,
    'epochs': 150,
    'batch_size': 64
}

# Ferramenta de pesquisa
search_tool = SerperDevTool()
search_tool_duck = DuckDuckGoSearchRun()

# Definindo a classe personalizada para pesquisa DuckDuckGo
class DuckDuckGoTool(BaseTool):
    name: str = "DuckDuckGoSearch"
    description: str = "A tool to perform web searches using DuckDuckGo"

    def _run(self, query: str) -> str:
        search_results = search_tool_duck.run(query)
        return search_results

# Instanciando a ferramenta personalizada
duck_tool = DuckDuckGoTool()
#duck_tool = Tool(
#    name="DuckDuckGoSearch",
#    description="A tool to perform web searches using DuckDuckGo",
#    func=search_tool_duck.run
#)

# Descrição do modelo a ser otimizado
model_description = """
The cryptocurrency price prediction model uses LSTM to forecast future prices based on various technical indicators. The indicators include RSI, EMA, ADX, Bollinger Bands, Stochastic, MFI, MACD, and Ichimoku. The LSTM is configured with hyperparameters such as the number of neurons, layers, epochs, and batch size. The goal is to optimize both the periods of the technical indicators and the LSTM hyperparameters to achieve the best possible accuracy in predictions.

Main functions of the model:
- calculate_indicators: Calculates the values of the technical indicators using parameters such as 'rsi_period', 'ema_period', etc.
- build_and_train_lstm: Builds and trains an LSTM model with hyperparameters such as 'neurons', 'layers', 'dropout_rate', 'l2_reg', 'epochs', and 'batch_size'.
- predict_with_lstm: Uses the trained LSTM model to predict future prices.
"""

# Agente Pesquisador de Indicadores
indicador_researcher = Agent(
    role='Indicator Researcher',
    goal='Optimize the periods of technical indicators to improve model performance.',
    verbose=True,
    memory=True,
    llm=llama3,
    backstory=(
        "You are a technical analyst specializing in cryptocurrencies."
        "Your goal is to adjust the periods of the technical indicators to find the best configuration."
    ),
    tools=[search_tool, duck_tool],
    allow_delegation=True,
    max_iter=8,
    max_rpm=15,
    cache=True
)

# Agente Pesquisador LSTM
lstm_researcher = Agent(
    role='LSTM Researcher',
    goal='Optimize the LSTM hyperparameters to improve price prediction.',
    verbose=True,
    memory=True,
    llm=gemini,
    backstory=(
        "You are a machine learning expert focused on sequential models."
        "Your goal is to adjust the LSTM hyperparameters to achieve the best possible accuracy."
    ),
    tools=[search_tool, duck_tool],
    allow_delegation=True,
    max_iter=8,
    max_rpm=10,
    cache=True
)

# Agente de Compilação de Resultado
compilation_agent = Agent(
    role='Results Compiler',
    goal='Compile the best parameters for the technical indicators and LSTM hyperparameters into a single dictionary.',
    verbose=True,
    memory=True,
    llm=gemini,
    backstory=(
        "You are a results compiler, specializing in gathering information from multiple sources and consolidating it into a useful format."
    ),
    tools=[],
    allow_delegation=False,
    max_iter=5,
    max_rpm=5,
    cache=True
)

# Tarefa de Otimização de Indicadores
indicadores_task = Task(
    description=(
        "You must adjust the periods of the following technical indicators: RSI, EMA, ADX, Bollinger Bands, Stochastic, MFI, MACD, Ichimoku."
        "First, understand the price prediction model that uses these indicators. Evaluate the model's performance with different periods for each indicator and determine the best values."
        "The response should be in the format: {'rsi_period': VALUE, 'ema_period': VALUE, 'adx_period': VALUE, 'bb_period': VALUE, 'stoch_period': VALUE, 'mfi_period': VALUE, 'macd_fast_period': VALUE, 'macd_slow_period': VALUE, 'macd_signal_period': VALUE, 'ichimoku_tenkan_period': VALUE, 'ichimoku_kijun_period': VALUE, 'ichimoku_senkou_span_b_period': VALUE}"
        f"\n\nModel description:\n{model_description}"
    ),
    expected_output='A list of the best periods for each indicator based on the model performance.',
    agent=indicador_researcher,
    async_execution=False
)

# Tarefa de otimização LSTM
lstm_task = Task(
    description=(
        "You must adjust the LSTM hyperparameters, such as the number of neurons, layers, epochs, and batch size."
        "First, understand the price prediction model that uses LSTM. Evaluate the model's performance with different hyperparameter configurations and determine the best values."
        "The response should be in the format: {'neurons': VALUE, 'layers': VALUE, 'dropout_rate': VALUE, 'l2_reg': VALUE, 'epochs': VALUE, 'batch_size': VALUE}"
        f"\n\nModel description:\n{model_description}"
    ),
    expected_output='A list of the best hyperparameter values for the LSTM based on the model performance.',
    agent=lstm_researcher,
    async_execution=False
)

# Tarefa de compilação de resultados
compilation_task = Task(
    description=(
        "Compile the results provided by the 'Indicator Researcher' and 'LSTM Researcher' agents."
        "The compiled results should be a dictionary with the following parameters:"
        "{'rsi_period': VALUE, 'ema_period': VALUE, 'adx_period': VALUE, 'bb_period': VALUE, 'stoch_period': VALUE, 'mfi_period': VALUE, 'macd_fast_period': VALUE, 'macd_slow_period': VALUE, 'macd_signal_period': VALUE, 'ichimoku_tenkan_period': VALUE, 'ichimoku_kijun_period': VALUE, 'ichimoku_senkou_span_b_period': VALUE, 'neurons': VALUE, 'layers': VALUE, 'dropout_rate': VALUE, 'l2_reg': VALUE, 'epochs': VALUE, 'batch_size': VALUE}"
    ),
    expected_output='A compiled dictionary with the best parameters for the technical indicators and LSTM hyperparameters.',
    agent=compilation_agent,
    context=[indicadores_task, lstm_task],
    async_execution=False
)

#Função para executar a análise com nova tentativa em caso de limite de taxa das LLMs
def execute_crew_with_retry(crew, max_retries=5, retry_delay=60):
    retries = 0
    while retries < max_retries:
        try:
            results = crew.kickoff(inputs={})
            return results
        except Exception as e:
            print(f"Erro de limite de taxa ou outro erro: {e}. Tentando novamente em {retry_delay} segundos...")
            time.sleep(retry_delay)
            retries += 1
    raise Exception("Número máximo de tentativas excedido devido aos limites de taxa.")

# Configurando a Crew
crew = Crew(
    agents=[indicador_researcher, lstm_researcher, compilation_agent],
    tasks=[indicadores_task, lstm_task, compilation_task],
    process=Process.sequential
)

#Executando a crew e coletando resultados
results = execute_crew_with_retry(crew)

# print(f"RESULT BEFORE extract_suggested_params FUNCTION:\n{results}")

# Função para extrair os parâmetros sugeridos dos resultados
def extract_suggested_params(results):
    params = default_params.copy()
    try:        
        # Converter a string JSON para um dicionário
        results_dict = ast.literal_eval(results)
        
        # Certifique-se de que os resultados sejam um dicionário
        if isinstance(results_dict, dict):
            for key in results_dict.keys():
                # Verifique se a chave está presente nos parâmetros
                if key in default_params and isinstance(results_dict[key], (int, float)):
                    params[key] = results_dict[key]
        else:
            print("Resultados não são um dicionário.")

    except Exception as e:
        print(f"Error processing optimized results: {e}")
    return params

# Processando os resultados corretamente
optimized_params = extract_suggested_params(results)

# optimized_params = results
print(f"RESULT AFTER extract_suggested_params FUNCTION:\n{optimized_params}")

# Chamando a função principal do script original com os parâmetros otimizados
main(optimized_params)
