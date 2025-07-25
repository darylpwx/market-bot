
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
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd - signal
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['AvgVolume'] = df['Volume'].rolling(window=20).mean()
    df['VolumeSpike'] = df['Volume'] > 1.5 * df['AvgVolume']
    return df

def analyze_signals(df, ticker):
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    results = []

    # RSI Logic
    rsi = latest["RSI"]
    prev_rsi = previous["RSI"]
    if ticker == "TQQQ":
        if rsi < 35 and rsi > prev_rsi:
            results.append(("RSI", f"{rsi:.1f}", "✅ Buy Bias"))
            rsi_signal = "Buy"
        else:
            results.append(("RSI", f"{rsi:.1f}", "❌ Wait"))
            rsi_signal = "Wait"
    else:
        if rsi > 65 and rsi < prev_rsi:
            results.append(("RSI", f"{rsi:.1f}", "✅ Sell Bias"))
            rsi_signal = "Sell"
        else:
            results.append(("RSI", f"{rsi:.1f}", "❌ Wait"))
            rsi_signal = "Wait"

    # MACD Logic
    macd_hist = latest["MACD_Hist"]
    prev_macd_hist = previous["MACD_Hist"]
    if ticker == "TQQQ":
        if macd_hist > prev_macd_hist:
            results.append(("MACD Histogram", f"{macd_hist:+.4f}", "✅ Buy Bias"))
            macd_signal = "Buy"
        else:
            results.append(("MACD Histogram", f"{macd_hist:+.4f}", "❌ Wait"))
            macd_signal = "Wait"
    else:
        if macd_hist < prev_macd_hist:
            results.append(("MACD Histogram", f"{macd_hist:+.4f}", "✅ Sell Bias"))
            macd_signal = "Sell"
        else:
            results.append(("MACD Histogram", f"{macd_hist:+.4f}", "❌ Wait"))
            macd_signal = "Wait"

    # Trend Logic
    price = latest["Close"]
    ema = latest["EMA9"]
    vwap = latest["VWAP"]
    if ticker == "TQQQ":
        if price > ema > vwap:
            results.append(("Price vs EMA9 & VWAP", f"${price:.2f} > ${ema:.2f} > ${vwap:.2f}", "✅ Bullish Trend"))
            trend_signal = "Buy"
        else:
            results.append(("Price vs EMA9 & VWAP", f"${price:.2f}, ${ema:.2f}, ${vwap:.2f}", "❌ Wait"))
            trend_signal = "Wait"
    else:
        if price < ema < vwap:
            results.append(("Price vs EMA9 & VWAP", f"${price:.2f} < ${ema:.2f} < ${vwap:.2f}", "✅ Bearish Trend"))
            trend_signal = "Sell"
        else:
            results.append(("Price vs EMA9 & VWAP", f"${price:.2f}, ${ema:.2f}, ${vwap:.2f}", "❌ Wait"))
            trend_signal = "Wait"

    # Volume Spike Logic
    if latest["VolumeSpike"]:
        results.append(("Volume Spike", "Yes", "✅ Confirmed Interest"))
        volume_signal = "Buy" if ticker == "TQQQ" else "Sell"
    else:
        results.append(("Volume Spike", "No", "❌ Weak"))
        volume_signal = "Wait"

    votes = [rsi_signal, macd_signal, trend_signal, volume_signal]
    if votes.count("Buy") >= 3 and ticker == "TQQQ":
        overall = f"✅ *Overall Signal: BUY {ticker} @ ${price:.2f}*"
    elif votes.count("Sell") >= 3 and ticker == "SQQQ":
        overall = f"🔴 *Overall Signal: BUY {ticker} @ ${price:.2f}*"
    else:
        overall = f"⏳ *Overall Signal: WAIT for {ticker}*"

    return results, overall

def format_message(results_tqqq, summary_tqqq, results_sqqq, summary_sqqq):
    time_str = now.strftime('%Y-%m-%d %H:%M')
    msg = f"📈 *Intraday TQQQ/SQQQ Signals* - {time_str}

"

    msg += "🎯 Analyzing: *TQQQ*

🔍 *Indicator Breakdown:*
"
    for name, value, comment in results_tqqq:
        msg += f"• {name}: {value} → {comment}
"
    msg += f"
{summary_tqqq}

"

    msg += "🎯 Analyzing: *SQQQ*

🔍 *Indicator Breakdown:*
"
    for name, value, comment in results_sqqq:
        msg += f"• {name}: {value} → {comment}
"
    msg += f"
{summary_sqqq}"

    return msg

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

    df_tqqq = calculate_indicators(fetch_data("TQQQ"))
    df_sqqq = calculate_indicators(fetch_data("SQQQ"))

    results_tqqq, summary_tqqq = analyze_signals(df_tqqq, "TQQQ")
    results_sqqq, summary_sqqq = analyze_signals(df_sqqq, "SQQQ")

    message = format_message(results_tqqq, summary_tqqq, results_sqqq, summary_sqqq)
    send_telegram_message(message)

run_intraday_signal_bot()
