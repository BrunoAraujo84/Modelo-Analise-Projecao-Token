# IMPORTANTO A BIBLIOTECA DE GENERATIVA AI DA GOOGLE
import os
import google.generativeai as GenAI

# RESGATANDO A CHAVE PARA A API DA LLM GOOGLE
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# INFORMANDO A CHAVE DA API, PARA CRIAR SEGUE LINK ABAIXO:
# https://aistudio.google.com/app/apikey
GenAI.configure(api_key=GEMINI_API_KEY)

# FUNÇÃO PARA GERAR O TEXTO RESPOSTA UTILIZANDO A ENGINE DA GEMINI DA GOOGLE
def gerar_texto(prompt):
    # USANDO O MÉTODO DE TRATAMENTO DE ERRO
    try:
        # INFORMANDO QUAL ENGINE (MODELO) SERÁ UTILIZADO
        modelo_nome = GenAI.GenerativeModel('gemini-1.5-pro-latest')

        # GERANDO TEXTO COM A ENGINE ESCOLHIDA
        resposta = modelo_nome.generate_content(prompt)
        
        # RETORNANDO O RESULTADO DA RESPOSTA DO MODELO
        return resposta.text
        
    # SE IDENTIFICAR ALGUM ERRO É EXECUTADO O PROCESSO DE EXCEÇÃO
    except Exception as e:
        # TRATAMENTO DAS EXCEÇÕES DE ACORDO COM A VARIÁVEL "e"
        print("Ocorreu um erro:", str(e))
        return None


def processar_dados(informacoes):
    # DEFININDO O PROMPT QUE SERÁ GERADO O TEXTO PELA ENGINE DA GEMINI AI
    # prompt = f"Escreva um resumo da análise dos dados em texto corrido com uma conclusão, segue a informação:\n{informacoes}"
    prompt = f"Você é um especialista em análise de dados de criptomoedas. Tenho dados históricos de preços, volumes de negociação, indicadores técnicos (RSI, MACD, etc.) referente a informações do token e se possível adicionar na análise notícias recentes relacionado ao token. Use esses dados para realizar uma análise completa e fornecer uma conclusão sobre a tendência nos próximos dias. Indique se espera uma tendência de alta, baixa ou lateral, com base nos dados fornecidos. Sua resposta deve apresentar uma linha contínua com apenas um único paragrafo para ter uma melhor leitura. Considere os seguintes pontos:\n1. Analise as séries temporais dos preços e volumes para identificar padrões.\n2. Avalie os indicadores técnicos para obter insights sobre a força das tendências atuais.\n3. Considere o impacto de notícias recentes no mercado.\n4. Faça uma previsão com base nos dados fornecidos.\nSegue a informação:\n{informacoes}. habilite a web para que o especialista possa pesquisar."
    # CHAMANDO A FUNÇÃO PARA GERAR O TEXTO COMO RETORNO DA RESPOSTA
    resposta = gerar_texto(prompt)

    # CASO APRESENTE UM RETORNO NA VARIÁVEL É APRESENTANDO A RESPOSTA IMPRESSO NO TERMINAL
    if resposta:
        return resposta

