# Importando as bibliotecas para utilização do modelo
import requests
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, ADXIndicator, MACD, IchimokuIndicator
from ta.volatility import BollingerBands
from ta.volume import MFIIndicator, VolumeWeightedAveragePrice
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, RepeatVector
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import CoinGecko_Analise_Token_Com_AI as GenAI
import Envio_Resultado_Telegram as Telegram
from time import sleep
import os
import datetime
import logging
import warnings
warnings.filterwarnings("ignore")

# Desativar avisos do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Selecionando as criptomoedas para calcular os indicadores
# portfolio = ['bitcoin','ethereum','polygon-ecosystem-token','solana','pendle','render-token','chainlink','near','injective-protocol','axelar','polkadot']
portfolio = ['bitcoin','ethereum','polygon-ecosystem-token','solana','pendle','render-token','chainlink','near']

# Gostaria de ter a análise pela GenIA com o sentimento dos traders?
sentimento_traders = True

# período para selecionar e treinar o modelo
numero_periodicidade = 180

# Informações para utilizar e resgatar dados de operações de futuros
coin_symbol_map = {
    'bitcoin': 'BTCUSDT',
    'ethereum': 'ETHUSDT',
    'polygon-ecosystem-token': 'MATICUSDT',
    'solana': 'SOLUSDT',
    'pendle': 'PENDLEUSDT',
    'render-token': 'RNDRUSDT',
    'chainlink': 'LINKUSDT',
    'near': 'NEARUSDT',
    'injective-protocol': 'INJUSDT',
    'axelar': 'AXLUSDT',
    'polkadot': 'DOTUSDT'
}

# Informação da data de processamento do modelo
dataProcessamento = datetime.date.today()

# Configurar logging
logging.basicConfig(level=logging.INFO)


# Função para obter dados históricos de preços
def get_historical_data(coin_id, days):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {'vs_currency': 'usd', 'days': days}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        
        prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])

        # Convertendo timestamps para datetime e unindo ambos os DataFrames
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms')
        volumes_df['timestamp'] = pd.to_datetime(volumes_df['timestamp'], unit='ms')
        prices_df.set_index('timestamp', inplace=True)
        volumes_df.set_index('timestamp', inplace=True)
        
        data_df = prices_df.join(volumes_df)
        return data_df
    else:
        print(f"Erro ao buscar dados para {coin_id}: {response.status_code}")
        return pd.DataFrame()


# Função para obter dados de futuros de criptomoedas da Binance
def get_binance_futures_data(symbol):
    url = f'https://fapi.binance.com/futures/data/globalLongShortAccountRatio'
    params = {'symbol': symbol, 'period': '1d', 'limit': 500}  # Ajuste o limite e o período conforme necessário
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['longShortRatio'] = df['longShortRatio'].astype(float)
            return df
        else:
            print(f"Dados de futuros para {symbol} não encontrados.")
            return pd.DataFrame()
    else:
        print(f"Erro ao buscar dados de futuros na Binance: {response.status_code}")
        return pd.DataFrame()


# Função para calcular os indicadores
def calculate_indicators(data, params):
    logging.info("Calculando indicadores...")
    rsi_period = params['rsi_period']
    ema_period = params['ema_period']
    adx_period = params['adx_period']
    bb_period = params['bb_period']
    stoch_period = params['stoch_period']
    mfi_period = params['mfi_period']
    macd_fast_period = params['macd_fast_period']
    macd_slow_period = params['macd_slow_period']
    macd_signal_period = params['macd_signal_period']
    ichimoku_tenkan_period = params['ichimoku_tenkan_period']
    ichimoku_kijun_period = params['ichimoku_kijun_period']
    ichimoku_senkou_span_b_period = params['ichimoku_senkou_span_b_period']
    
    if data.empty:
        print("Dados insuficientes para calcular os indicadores.")
        return data

    rsi = RSIIndicator(close=data['price'], window=rsi_period)
    ema = EMAIndicator(close=data['price'], window=ema_period)
    bb = BollingerBands(close=data['price'], window=bb_period)
    adx = ADXIndicator(high=data['price'], low=data['price'], close=data['price'], window=adx_period)
    macd = MACD(close=data['price'], window_slow=macd_slow_period, window_fast=macd_fast_period, window_sign=macd_signal_period)
    stoch = StochasticOscillator(high=data['price'], low=data['price'], close=data['price'], window=stoch_period)
    mfi = MFIIndicator(high=data['price'], low=data['price'], close=data['price'], volume=data['volume'], window=mfi_period)
    ichimoku = IchimokuIndicator(high=data['price'], low=data['price'], window1=ichimoku_tenkan_period, window2=ichimoku_kijun_period, window3=ichimoku_senkou_span_b_period)
    vwap = VolumeWeightedAveragePrice(high=data['price'], low=data['price'], close=data['price'], volume=data['volume'])

    data['rsi'] = rsi.rsi()
    data['ema'] = ema.ema_indicator()
    data['bb_upper'] = bb.bollinger_hband()
    data['bb_lower'] = bb.bollinger_lband()
    data['adx'] = adx.adx()
    data['macd_line'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['stochastic'] = stoch.stoch()
    data['mfi'] = mfi.money_flow_index()
    data['ichimoku_a'] = ichimoku.ichimoku_a()
    data['ichimoku_b'] = ichimoku.ichimoku_b()
    data['vwap'] = vwap.volume_weighted_average_price()

    data['resistance'] = data['price'][-150:].max()
    data['support'] = data['price'][-150:].min()
    
    logging.info("Indicadores calculados.")
    return data.dropna()


# Função para inferir a projeção de preços usando os indicadores
def infer_price_projection(data):
    if data['adx'].iloc[-1] > 20:
        if data['macd_line'].iloc[-1] > data['macd_signal'].iloc[-1]:
            if data['rsi'].iloc[-1] < 70 and data['stochastic'].iloc[-1] < 80:
                return data['price'].iloc[-1] * 1.05
        elif data['macd_line'].iloc[-1] < data['macd_signal'].iloc[-1]:
            if data['rsi'].iloc[-1] > 30 and data['stochastic'].iloc[-1] > 20:
                return data['price'].iloc[-1] * 0.95
    return data['price'].iloc[-1]


# Função para preparar dados para LSTM
def prepare_data_lstm(data, n_steps):
    logging.info("Preparando dados para LSTM...")
    
    features = ['price', 'volume', 'rsi', 'ema', 'bb_upper', 'bb_lower', 'adx', 'macd_line', 'macd_signal', 'stochastic', 'mfi', 'ichimoku_a', 'ichimoku_b', 'vwap', 'resistance', 'support']
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[features])
    X, y = [], []
    for i in range(len(data_scaled) - n_steps):
        X.append(data_scaled[i:i+n_steps])
        y.append(data_scaled[i+n_steps, 0])
    X = np.array(X)
    y = np.array(y)
    
    logging.info("Dados preparados.")
    return X, y, scaler


# Função para construir e treinar LSTM
def build_and_train_cnn_lstm(X_train, y_train, n_steps, params):
    logging.info("Construindo e treinando LSTM...")
    neurons = params['neurons']
    layers = params['layers']
    dropout_rate = params['dropout_rate']
    epochs = params['epochs']
    batch_size = params['batch_size']
    l2_reg = params['l2_reg']
    
    model = Sequential()
    model.add(Input(shape=(n_steps, X_train.shape[2])))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(n_steps))
    model.add(LSTM(neurons, activation='relu', return_sequences=True, kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    
    for _ in range(layers - 1):
        model.add(LSTM(neurons, activation='relu', return_sequences=True, kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout_rate))
    
    model.add(LSTM(neurons, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    logging.info("Iniciando treinamento...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)
    
    logging.info("Treinamento concluído.")
    return model


# Função para prever usando CNN-LSTM
def predict_with_cnn_lstm(data, params):
    logging.info("Prevendo com CNN-LSTM...")
    
    n_steps = 20
    X, y, scaler = prepare_data_lstm(data, n_steps)
    
    kfold = KFold(n_splits=5, shuffle=False)  # 5-fold cross-validation
    fold_no = 1
    mse_scores = []

    for train_index, test_index in kfold.split(X):
        logging.info(f"Treinando fold {fold_no}...")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = build_and_train_cnn_lstm(X_train, y_train, n_steps, params)
        predictions = model.predict(X_test)
        predictions_rescaled = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], X_train.shape[2] - 1))), axis=1))[:, 0]
        
        mse = mean_squared_error(y_test, predictions_rescaled)
        mse_scores.append(mse)
        logging.info(f"Fold {fold_no} - MSE: {mse}")
        fold_no += 1
    
    avg_mse = np.mean(mse_scores)
    logging.info(f"Desempenho médio do modelo CNN-LSTM:\nErro médio quadrático: {avg_mse}\n")
    
    # Treinando no conjunto completo para projeções finais
    model = build_and_train_cnn_lstm(X, y, n_steps, params)
    predictions = model.predict(X)
    predictions_rescaled = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], X.shape[2] - 1))), axis=1))[:, 0]
    predictions_5_days = predict_n_days_lstm(model, data, n_steps, 5, scaler)
    
    return predictions_rescaled, predictions_5_days, avg_mse


# Função para prever os próximos 5 dias usando o modelo LSTM
def predict_n_days_lstm(model, data, n_steps, n_days, scaler):
    logging.info("Prevendo próximos 5 dias com LSTM...")
    
    input_seq = data[-n_steps:]
    predictions = []
    for _ in range(n_days):
        input_seq_scaled = scaler.transform(input_seq)
        input_seq_reshaped = input_seq_scaled.reshape((1, n_steps, input_seq.shape[1]))
        next_price_scaled = model.predict(input_seq_reshaped)
        next_price = scaler.inverse_transform(np.concatenate((next_price_scaled, np.zeros((1, input_seq.shape[1] - 1))), axis=1))[0, 0]
        predictions.append(next_price)
        next_data_point = pd.DataFrame({col: [next_price] if col == 'price' else [input_seq[col].iloc[-1]] for col in input_seq.columns})
        input_seq = pd.concat([input_seq.iloc[1:], next_data_point], ignore_index=True)
    
    logging.info("Previsões dos próximos 5 dias concluídas.")
    return predictions


# Função para obter informações da criptomoeda
def get_crypto_info(coin_id, symbol, params):
    logging.info(f"Obtendo dados históricos para {coin_id}...")
    
    current_data = get_historical_data(coin_id, 1)
    historical_data = get_historical_data(coin_id, numero_periodicidade)
    futures_data = get_binance_futures_data(symbol)
  
    if historical_data.empty:
        return {}

    logging.info(f"Calculando indicadores para {coin_id}...")
    data_with_indicators = calculate_indicators(historical_data, params)
    projected_price = infer_price_projection(data_with_indicators)
    
    logging.info(f"Prevendo preço para {coin_id} com LSTM...")
    projected_price_ml, predictions_5_days, avg_mse = predict_with_cnn_lstm(data_with_indicators, params)
  
    info = {
        'current_price': current_data['price'].iloc[-1] if not current_data.empty else None,
        'rsi': data_with_indicators['rsi'].iloc[-1],
        'ema': data_with_indicators['ema'].iloc[-1],
        'bb_upper': data_with_indicators['bb_upper'].iloc[-1],
        'bb_lower': data_with_indicators['bb_lower'].iloc[-1],
        'adx': data_with_indicators['adx'].iloc[-1],
        'macd_line': data_with_indicators['macd_line'].iloc[-1],
        'macd_signal': data_with_indicators['macd_signal'].iloc[-1],
        'stochastic': data_with_indicators['stochastic'].iloc[-1],
        'mfi': data_with_indicators['mfi'].iloc[-1],
        'ichimoku_a': data_with_indicators['ichimoku_a'].iloc[-1],
        'ichimoku_b': data_with_indicators['ichimoku_b'].iloc[-1],
        'vwap': data_with_indicators['vwap'].iloc[-1],
        'resistance': data_with_indicators['resistance'].iloc[-1],
        'support': data_with_indicators['support'].iloc[-1],
        'projected_price': projected_price,
        'projected_price_ml': projected_price_ml[-1],
        'predictions_5_days': predictions_5_days[4],
        'avg_mse': avg_mse,
        'long_short_ratio': futures_data['longShortRatio'].iloc[-1] if not futures_data.empty else 'Dados não disponíveis'
    }
    return info


# Apresentando o resultado no terminal após o processamento do modelo
# Função principal para executar a análise
def main(params):
    for coin in portfolio:
        symbol = coin_symbol_map[coin]
        info = get_crypto_info(coin, symbol, params)
        informacoes = ""
    
        if sentimento_traders == True:
            if info:
                print(f"Informações para {coin}:")
                print(f"Preço atual: {info['current_price']}")
                print(f"RSI: {info['rsi']}")
                print(f"EMA: {info['ema']}")
                print(f"Bollinger Bands Superior: {info['bb_upper']}")
                print(f"Bollinger Bands Inferior: {info['bb_lower']}")
                print(f"ADX: {info['adx']}")
                print(f"Linha MACD: {info['macd_line']}")
                print(f"Sinal MACD: {info['macd_signal']}")
                print(f"Estocástico: {info['stochastic']}")
                print(f"MFI: {info['mfi']}")
                print(f"Ichimoku A: {info['ichimoku_a']}")
                print(f"Ichimoku B: {info['ichimoku_b']}")
                print(f"VWAP: {info['vwap']}")
                print(f"Resistência: {info['resistance']}")
                print(f"Suporte: {info['support']}")
                print(f"Relação Long/Short: {info['long_short_ratio']}")
                print(f"Preço projetado (Usando Indicadores): {info['projected_price']}")
                print(f"Preço projetado (Aprendizado de Máquina): {info['projected_price_ml']}")
                print(f"Previsão dos próximos 5 dias (LSTM): {info['predictions_5_days']}")
                print(f"Erro médio quadrático médio (Cross-validation): {info['avg_mse']}")
                print()
            else:
                print(f"Não foi possível obter informações para {coin}")
    
            # Variável que será utilizado para a Inteligencia Artificial
            informacoes = f"Informações para {coin}:\nPreço atual: {info['current_price']}\nRSI: {info['rsi']}\nEMA: {info['ema']}\nBollinger Bands Superior: {info['bb_upper']}\nBollinger Bands Inferior: {info['bb_lower']}\nADX: {info['adx']}\nLinha MACD: {info['macd_line']}\nSinal MACD: {info['macd_signal']}\nEstocástico: {info['stochastic']}\nMFI: {info['mfi']}\nIchimoku A: {info['ichimoku_a']}\nIchimoku B: {info['ichimoku_b']}\nVWAP: {info['vwap']}\nRelação Long/Short: {info['long_short_ratio']}\nData da Análise: {dataProcessamento}"

        else:
            if info:
                print(f"Informações para {coin}:")
                print(f"Preço atual: {info['current_price']}")
                print(f"RSI: {info['rsi']}")
                print(f"EMA: {info['ema']}")
                print(f"Bollinger Bands Superior: {info['bb_upper']}")
                print(f"Bollinger Bands Inferior: {info['bb_lower']}")
                print(f"ADX: {info['adx']}")
                print(f"Linha MACD: {info['macd_line']}")
                print(f"Sinal MACD: {info['macd_signal']}")
                print(f"Estocástico: {info['stochastic']}")
                print(f"MFI: {info['mfi']}")
                print(f"Ichimoku A: {info['ichimoku_a']}")
                print(f"Ichimoku B: {info['ichimoku_b']}")
                print(f"VWAP: {info['vwap']}")
                print(f"Resistência: {info['resistance']}")
                print(f"Suporte: {info['support']}")
                print(f"Preço projetado (Usando Indicadores): {info['projected_price']}")
                print(f"Preço projetado (Aprendizado de Máquina): {info['projected_price_ml']}")
                print(f"Previsão dos próximos 5 dias (LSTM): {info['predictions_5_days']}")
                print(f"Erro médio quadrático médio (Cross-validation): {info['avg_mse']}")
                print()
            else:
                print(f"Não foi possível obter informações para {coin}")
    
            # Variável que será utilizado para a Inteligencia Artificial
            informacoes = f"Informações para {coin}:\nPreço atual: {info['current_price']}\nRSI: {info['rsi']}\nEMA: {info['ema']}\nBollinger Bands Superior: {info['bb_upper']}\nBollinger Bands Inferior: {info['bb_lower']}\nADX: {info['adx']}\nLinha MACD: {info['macd_line']}\nSinal MACD: {info['macd_signal']}\nEstocástico: {info['stochastic']}\nMFI: {info['mfi']}\nIchimoku A: {info['ichimoku_a']}\nIchimoku B: {info['ichimoku_b']}\nVWAP: {info['vwap']}\nData da Análise: {dataProcessamento}"



        # Processar os dados para retornar a análise da IA Generativa Gemini
        texto_gerado = GenAI.processar_dados(informacoes)
    
        print("Análise geral:")
        print(texto_gerado)

        mensagem = f"{coin}\n\nPreço atual: {info['current_price']}\nPrevisão para os próximos 5 dias (LSTM): {info['predictions_5_days']}\n\n{texto_gerado}\n\n"
        enviado = Telegram.enviar_mensagem_telegram(mensagem)
        
        # Esperando...
        print(enviado)
        print()
        sleep(30)


if __name__ == "__main__":
    default_params = {
        'rsi_period': 14,
        'ema_period': 21,
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
        'epochs': 1,
        'batch_size': 64
    }
    main(default_params)
