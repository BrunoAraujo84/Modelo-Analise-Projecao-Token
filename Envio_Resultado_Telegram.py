# Importando a biblioteca
import requests
import os

# CHAVE DO BOT TELEGRAM
TELEGRAM_TOKEN = os.getenv('BOT_TELEGRAM_API_KEY')
CHAT_ID = os.getenv('BOT_TELEGRAM_CHAT_ID')

# URL da API do Telegram
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={CHAT_ID}&text="
        
# Função para enviar mensagem para o numero de WhatsApp
def enviar_mensagem_telegram(mensagem):
    resultado = ""
    try:
        # Fazer a solicitação POST para a API do Telegram
        resposta = requests.get(TELEGRAM_API_URL+mensagem)
        # Verificar se a mensagem foi enviada com sucesso
        if resposta.status_code == 200:
            return "Mensagem enviada com sucesso!"
        else:
            return f"Erro ao enviar mensagem para o Telegram: {resposta.text}"
    except Exception as e:
       resultado = f"Erro ao enviar mensagem para o Telegram: {e}"
    
    return resultado
