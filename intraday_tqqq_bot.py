
import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import pytz

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

INTERVAL = "15m"
PERIOD = "2d"
SGT = pytz.timezone("Asia/Singapore")
now = datetime.now(SGT)
hour = now.hour
allowed_hours = list(range(9, 24)) + list(range(0, 2))  # 9AM - 2AM SGT

def fetch_data(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period=PERIOD, interval=INTERVAL)

def calculate_indicators(df):
    df['EMA9'] = df['Close'].ewm(span=9).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    macd_fast = df['Close'].ewm(span=12, adjust=False).mean()
    macd_slow = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = macd_fast - macd_slow
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['AvgVolume'] = df['Volume'].rolling(window=20).mean()
    df['VolumeSpike'] = df['Volume'] > 1.5 * df['AvgVolume']
    df['Support'] = df['Low'].rolling(window=20).min()
    df['Resistance'] = df['High'].rolling(window=20).max()
    return df

def fetch_spy_vix():
    spy = yf.Ticker("SPY").history(period="2d", interval="15m")
    vix = yf.Ticker("^VIX").history(period="2d", interval="15m")
    spy_trend = "Up" if spy["Close"].iloc[-1] > spy["Close"].iloc[-10] else "Down"
    vix_level = vix["Close"].iloc[-1]
    return spy_trend, vix_level

def analyze(df, ticker, spy_trend, vix_level):
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    votes = []
    notes = []

    # RSI
    rsi = latest["RSI"]
    rsi_signal = "Buy" if ticker == "TQQQ" and rsi < 35 and rsi > previous["RSI"] else                  "Sell" if ticker == "SQQQ" and rsi > 65 and rsi < previous["RSI"] else "Wait"
    votes.append(rsi_signal)
    notes.append(f"• RSI: {rsi:.1f} → {rsi_signal}")

    # MACD Histogram
    macd_now = latest["MACD_Hist"]
    macd_prev = previous["MACD_Hist"]
    macd_signal = "Buy" if ticker == "TQQQ" and macd_now > macd_prev else                   "Sell" if ticker == "SQQQ" and macd_now < macd_prev else "Wait"
    votes.append(macd_signal)
    notes.append(f"• MACD Hist: {macd_now:+.4f} → {macd_signal}")

    # EMA/VWAP Trend
    price, ema, vwap = latest["Close"], latest["EMA9"], latest["VWAP"]
    trend_signal = "Buy" if ticker == "TQQQ" and price > ema > vwap else                    "Sell" if ticker == "SQQQ" and price < ema < vwap else "Wait"
    votes.append(trend_signal)
    notes.append(f"• Trend: ${price:.2f} vs EMA9 ${ema:.2f}, VWAP ${vwap:.2f} → {trend_signal}")

    # Volume
    volume_signal = "Buy" if ticker == "TQQQ" and latest["VolumeSpike"] else                     "Sell" if ticker == "SQQQ" and latest["VolumeSpike"] else "Wait"
    votes.append(volume_signal)
    notes.append(f"• Volume Spike: {'Yes' if latest['VolumeSpike'] else 'No'} → {volume_signal}")

    # Support/Resistance Proximity
    support, resistance = latest["Support"], latest["Resistance"]
    if ticker == "TQQQ" and price <= support * 1.01:
        votes.append("Buy")
        notes.append(f"• Near Support (${support:.2f}) → Buy Zone")
    elif ticker == "SQQQ" and price >= resistance * 0.99:
        votes.append("Sell")
        notes.append(f"• Near Resistance (${resistance:.2f}) → Sell Zone")
    else:
        votes.append("Wait")
        notes.append(f"• Support/Resistance → Wait")

    # SPY/VIX context
    context_signal = "Buy" if ticker == "TQQQ" and spy_trend == "Up" and vix_level < 17 else                      "Sell" if ticker == "SQQQ" and spy_trend == "Down" and vix_level > 20 else "Wait"
    votes.append(context_signal)
    notes.append(f"• SPY Trend: {spy_trend}, VIX: {vix_level:.1f} → {context_signal}")

    score = votes.count("Buy") * 20 if ticker == "TQQQ" else votes.count("Sell") * 20
    final = f"✅ *BUY {ticker}* @ ${price:.2f}" if score >= 80 else f"⏳ *WAIT for {ticker}*"

    return notes, final, score

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, data=payload)
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

def format_output(ticker, notes, summary, score):
    section = f"🎯 Analyzing: *{ticker}*\n\n"
    section += "\n".join(notes) + "\n"
    section += f"\n📊 *Confidence Score:* {score}/100\n{summary}\n"
    return section

def run_bot():
    if hour not in allowed_hours:
        return

    df_tqqq = calculate_indicators(fetch_data("TQQQ"))
    df_sqqq = calculate_indicators(fetch_data("SQQQ"))
    spy_trend, vix_level = fetch_spy_vix()

    notes_tqqq, summary_tqqq, score_tqqq = analyze(df_tqqq, "TQQQ", spy_trend, vix_level)
    notes_sqqq, summary_sqqq, score_sqqq = analyze(df_sqqq, "SQQQ", spy_trend, vix_level)

    now_str = now.strftime('%Y-%m-%d %H:%M')
    message = f"📈 *Intraday TQQQ/SQQQ Signal Report* - {now_str}\n\n"
    message += format_output("TQQQ", notes_tqqq, summary_tqqq, score_tqqq)
    message += format_output("SQQQ", notes_sqqq, summary_sqqq, score_sqqq)

    send_telegram_message(message)

run_bot()
