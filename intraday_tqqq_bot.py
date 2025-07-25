
import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# Constants
TICKERS = ["TQQQ", "SQQQ"]
INTERVAL = "15m"
PERIOD = "2d"

# Market hours in UTC for 9:30 AM to 2:00 AM SGT
SGT = pytz.timezone("Asia/Singapore")
now = datetime.now(SGT)
hour = now.hour
minute = now.minute
allowed_hours = list(range(9, 24)) + list(range(0, 2))  # 9AM - 2AM SGT

def fetch_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=PERIOD, interval=INTERVAL)
    return hist

def calculate_indicators(df):
    df['EMA9'] = df['Close'].ewm(span=9).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    df['AvgVolume'] = df['Volume'].rolling(window=20).mean()
    df['VolumeSpike'] = df['Volume'] > 1.5 * df['AvgVolume']
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

    return df

def generate_signal(df, ticker):
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    if ticker == "TQQQ":
        if (
            latest["RSI"] < 35 and
            latest["RSI"] > previous["RSI"] and
            latest["MACD_Hist"] > previous["MACD_Hist"] and
            latest["Close"] > latest["EMA9"] > latest["VWAP"] and
            latest["VolumeSpike"]
        ):
            return "🟢 TQQQ BUY SIGNAL @ ${:.2f}".format(latest["Close"])
    elif ticker == "SQQQ":
        if (
            latest["RSI"] > 65 and
            latest["RSI"] < previous["RSI"] and
            latest["MACD_Hist"] < previous["MACD_Hist"] and
            latest["Close"] < latest["EMA9"] < latest["VWAP"] and
            latest["VolumeSpike"]
        ):
            return "🔴 SQQQ BUY SIGNAL @ ${:.2f}".format(latest["Close"])
    return None

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, data=payload)
        return response.status_code == 200
    except Exception as e:
        print(f"Error sending message: {e}")
        return False

def run_intraday_signal_bot():
    if hour not in allowed_hours:
        print("Outside trading signal window.")
        return

    messages = []
    for ticker in TICKERS:
        df = fetch_data(ticker)
        df = calculate_indicators(df)
        signal = generate_signal(df, ticker)
        if signal:
            messages.append(signal)

    if messages:
        full_message = f"📈 *Intraday Trading Signals* - {now.strftime('%Y-%m-%d %H:%M')}\n\n" + "\n".join(messages)

    else:
        full_message = f"📈 *Intraday Trading Signals* - {now.strftime('%Y-%m-%d %H:%M')}\n\nNo actionable signals."


    send_telegram_message(full_message)

run_intraday_signal_bot()
