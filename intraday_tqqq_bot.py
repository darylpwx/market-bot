import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import pytz
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

# -----------------------------
# Enhanced Technical Indicators (from yy.py)
# -----------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(series, period=20, std_dev=2):
    rolling_mean = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band, rolling_mean, lower_band

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def calculate_williams_r(high, low, close, period=14):
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r

def calculate_momentum(series, period=10):
    return series.pct_change(periods=period) * 100

def calculate_indicators(df):
    # Flatten MultiIndex columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # Core indicators
    df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    
    # MACD with different settings for faster signals
    df['MACD'] = df['Close'].ewm(span=8, adjust=False).mean() - df['Close'].ewm(span=21, adjust=False).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=5, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_signal']
    
    # RSI with multiple timeframes
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['RSI_Fast'] = compute_rsi(df['Close'], 7)
    
    # Volume indicators
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['AvgVolume'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['AvgVolume']
    df['VolumeSpike'] = df['Volume'] > 1.5 * df['AvgVolume']
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'], 20, 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Stochastic
    df['Stoch_K'], df['Stoch_D'] = calculate_stochastic(df['High'], df['Low'], df['Close'], 14, 3)
    
    # Williams %R
    df['Williams_R'] = calculate_williams_r(df['High'], df['Low'], df['Close'], 14)
    
    # Momentum and volatility
    df['Momentum'] = calculate_momentum(df['Close'], 5)
    df['Price_Change'] = df['Close'].pct_change() * 100
    df['Volatility'] = df['Close'].rolling(window=10).std()
    
    # Trend indicators
    df['EMA_Slope'] = df['EMA9'].diff() / df['EMA9'].shift()
    df['Price_vs_EMA5'] = (df['Close'] - df['EMA5']) / df['EMA5'] * 100
    df['Price_vs_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP'] * 100
    
    # Support/Resistance
    df['Support'] = df['Low'].rolling(window=20).min()
    df['Resistance'] = df['High'].rolling(window=20).max()
    df['Recent_High'] = df['High'].rolling(window=5).max()
    df['Recent_Low'] = df['Low'].rolling(window=5).min()
    
    return df

def fetch_spy_vix():
    spy = yf.Ticker("SPY").history(period="2d", interval="15m")
    vix = yf.Ticker("^VIX").history(period="2d", interval="15m")
    spy_trend = "Up" if spy["Close"].iloc[-1] > spy["Close"].iloc[-10] else "Down"
    vix_level = vix["Close"].iloc[-1]
    return spy_trend, vix_level

# -----------------------------
# Enhanced Signal Analysis (based on yy.py logic)
# -----------------------------
def evaluate_buy_signals(row):
    """Enhanced buy signal evaluation"""
    buy_score = 0
    buy_reasons = []
    
    # Extract values
    rsi = row['RSI']
    rsi_fast = row['RSI_Fast']
    macd = row['MACD']
    macd_signal = row['MACD_signal']
    macd_hist = row['MACD_Hist']
    price = row['Close']
    vwap = row['VWAP']
    ema5 = row['EMA5']
    ema9 = row['EMA9']
    ema21 = row['EMA21']
    bb_position = row['BB_Position']
    stoch_k = row['Stoch_K']
    williams_r = row['Williams_R']
    volume_ratio = row['Volume_Ratio']
    momentum = row['Momentum']
    price_vs_ema5 = row['Price_vs_EMA5']
    price_vs_vwap = row['Price_vs_VWAP']
    ema_slope = row['EMA_Slope']
    
    # 1. Strong momentum conditions
    if momentum > 2:  # Strong positive momentum
        buy_score += 25
        buy_reasons.append(f"Strong Momentum (+{momentum:.1f}%)")
    
    # 2. RSI conditions - looking for oversold bounce or momentum
    if 25 < rsi < 45 and rsi_fast > rsi:  # Oversold with recovery
        buy_score += 20
        buy_reasons.append("RSI Oversold Recovery")
    elif 45 < rsi < 60 and rsi_fast > 55:  # Building momentum
        buy_score += 15
        buy_reasons.append("RSI Building Momentum")
    
    # 3. MACD strong bullish
    if macd > macd_signal and macd_hist > 0:
        buy_score += 20
        buy_reasons.append("MACD Bullish")
    
    # 4. Price positioning
    if price > vwap and price_vs_vwap > 0.3:  # Above VWAP with strength
        buy_score += 15
        buy_reasons.append(f"Strong above VWAP (+{price_vs_vwap:.1f}%)")
    
    if price > ema5 and price > ema9 and ema9 > ema21:  # Trend alignment
        buy_score += 20
        buy_reasons.append("Bullish Trend Alignment")
    
    # 5. Volume confirmation
    if volume_ratio > 1.5:  # High volume
        buy_score += 15
        buy_reasons.append(f"High Volume ({volume_ratio:.1f}x)")
    
    # 6. Stochastic conditions
    if stoch_k < 30:  # Oversold
        buy_score += 10
        buy_reasons.append("Stochastic Oversold")
    elif 40 < stoch_k < 70:  # Good momentum zone
        buy_score += 10
        buy_reasons.append("Stochastic Momentum Zone")
    
    # 7. Williams %R
    if -50 < williams_r < -20:  # Good momentum area
        buy_score += 10
        buy_reasons.append("Williams %R Momentum")
    
    # 8. Bollinger Bands
    if 0.2 < bb_position < 0.7:  # Good position in BB
        buy_score += 10
        buy_reasons.append("Good BB Position")
    elif bb_position < 0.1:  # Oversold on BB
        buy_score += 15
        buy_reasons.append("BB Oversold")
    
    # 9. EMA slope (trend strength)
    if ema_slope > 0:
        buy_score += 10
        buy_reasons.append("EMA Uptrend")
    
    return buy_score, buy_reasons

def evaluate_sell_signals(row):
    """Enhanced sell signal evaluation for SQQQ"""
    sell_score = 0
    sell_reasons = []
    
    # Extract values
    rsi = row['RSI']
    rsi_fast = row['RSI_Fast']
    macd = row['MACD']
    macd_signal = row['MACD_signal']
    macd_hist = row['MACD_Hist']
    price = row['Close']
    vwap = row['VWAP']
    ema5 = row['EMA5']
    ema9 = row['EMA9']
    ema21 = row['EMA21']
    bb_position = row['BB_Position']
    stoch_k = row['Stoch_K']
    williams_r = row['Williams_R']
    volume_ratio = row['Volume_Ratio']
    momentum = row['Momentum']
    price_vs_vwap = row['Price_vs_VWAP']
    
    # 1. Strong negative momentum
    if momentum < -2:  # Strong negative momentum
        sell_score += 25
        sell_reasons.append(f"Strong Negative Momentum ({momentum:.1f}%)")
    
    # 2. RSI conditions for SQQQ (opposite of TQQQ)
    if rsi > 75 or (rsi > 65 and rsi_fast < rsi):  # Overbought or losing momentum
        sell_score += 20
        sell_reasons.append("RSI Overbought/Weakening")
    
    # 3. MACD bearish
    if macd < macd_signal and macd_hist < 0:
        sell_score += 20
        sell_reasons.append("MACD Bearish")
    
    # 4. Price positioning (bearish for SQQQ means QQQ is weak)
    if price < vwap and price_vs_vwap < -0.3:  # Below VWAP with weakness
        sell_score += 15
        sell_reasons.append(f"Weak below VWAP ({price_vs_vwap:.1f}%)")
    
    if price < ema5 and price < ema9 and ema9 < ema21:  # Bearish trend alignment
        sell_score += 20
        sell_reasons.append("Bearish Trend Alignment")
    
    # 5. Volume confirmation
    if volume_ratio > 1.5:  # High volume on decline
        sell_score += 15
        sell_reasons.append(f"High Volume on Decline ({volume_ratio:.1f}x)")
    
    # 6. Stochastic overbought
    if stoch_k > 70:  # Overbought
        sell_score += 15
        sell_reasons.append("Stochastic Overbought")
    
    # 7. Williams %R overbought
    if williams_r > -20:  # Overbought
        sell_score += 10
        sell_reasons.append("Williams %R Overbought")
    
    # 8. Bollinger Bands
    if bb_position > 0.8:  # Near upper band
        sell_score += 15
        sell_reasons.append("BB Upper Band")
    
    return sell_score, sell_reasons

def analyze_enhanced(df, ticker, spy_trend, vix_level):
    """Enhanced analysis using yy.py methodology"""
    latest = df.iloc[-1]
    
    if ticker == "TQQQ":
        score, reasons = evaluate_buy_signals(latest)
        signal_type = "BUY"
    else:  # SQQQ
        score, reasons = evaluate_sell_signals(latest)
        signal_type = "SELL"
    
    # Market context adjustment
    context_bonus = 0
    context_notes = []
    
    if ticker == "TQQQ":
        if spy_trend == "Up" and vix_level < 17:
            context_bonus += 15
            context_notes.append("Favorable Market (SPY↑, VIX Low)")
        elif spy_trend == "Down" or vix_level > 25:
            context_bonus -= 10
            context_notes.append("Unfavorable Market (Risk Off)")
    else:  # SQQQ
        if spy_trend == "Down" and vix_level > 20:
            context_bonus += 15
            context_notes.append("Favorable Market (SPY↓, VIX High)")
        elif spy_trend == "Up" and vix_level < 15:
            context_bonus -= 10
            context_notes.append("Unfavorable Market (Risk On)")
    
    final_score = score + context_bonus
    
    # Generate summary
    price = latest['Close']
    rsi = latest['RSI']
    macd_hist = latest['MACD_Hist']
    momentum = latest['Momentum']
    
    if final_score >= 70:
        confidence = "HIGH"
        action = f"✅ *{signal_type} {ticker}*"
        emoji = "🚀" if ticker == "TQQQ" else "📉"
    elif final_score >= 50:
        confidence = "MEDIUM"
        action = f"⚠️ *CONSIDER {signal_type} {ticker}*"
        emoji = "⚡"
    else:
        confidence = "LOW"
        action = f"⏳ *WAIT for {ticker}*"
        emoji = "⏸️"
    
    summary = f"{emoji} {action} @ ${price:.2f}"
    
    # Format detailed notes
    notes = []
    notes.append(f"• Price: ${price:.2f} | RSI: {rsi:.1f} | Momentum: {momentum:+.1f}%")
    notes.append(f"• MACD Hist: {macd_hist:+.4f}")
    
    # Add top 3 signal reasons
    for i, reason in enumerate(reasons[:3]):
        notes.append(f"• {reason}")
    
    # Add context
    if context_notes:
        notes.extend([f"• {note}" for note in context_notes])
    
    notes.append(f"• SPY: {spy_trend} | VIX: {vix_level:.1f}")
    
    return notes, summary, final_score, confidence

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, data=payload)
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

def format_output(ticker, notes, summary, score, confidence):
    section = f"🎯 *{ticker} Analysis*\n\n"
    section += "\n".join(notes) + "\n"
    section += f"\n📊 *Score:* {score}/100 (*{confidence}* Confidence)\n"
    section += f"{summary}\n"
    return section

def run_enhanced_bot():
    if hour not in allowed_hours:
        return

    try:
        # Fetch and analyze data
        df_tqqq = calculate_indicators(fetch_data("TQQQ"))
        df_sqqq = calculate_indicators(fetch_data("SQQQ"))
        spy_trend, vix_level = fetch_spy_vix()

        # Enhanced analysis
        notes_tqqq, summary_tqqq, score_tqqq, conf_tqqq = analyze_enhanced(df_tqqq, "TQQQ", spy_trend, vix_level)
        notes_sqqq, summary_sqqq, score_sqqq, conf_sqqq = analyze_enhanced(df_sqqq, "SQQQ", spy_trend, vix_level)

        # Format message
        now_str = now.strftime('%Y-%m-%d %H:%M SGT')
        message = f"📈 *Enhanced TQQQ/SQQQ Signal* - {now_str}\n\n"
        message += format_output("TQQQ", notes_tqqq, summary_tqqq, score_tqqq, conf_tqqq)
        message += "➖" * 30 + "\n\n"
        message += format_output("SQQQ", notes_sqqq, summary_sqqq, score_sqqq, conf_sqqq)
        
        # Add market overview
        message += f"\n🌍 *Market Context*\n"
        message += f"• SPY Trend: {spy_trend}\n"
        message += f"• VIX Level: {vix_level:.1f}\n"
        
        # Risk warning
        if vix_level > 25:
            message += f"\n⚠️ *High Volatility Warning* (VIX > 25)"

        send_telegram_message(message)
        print(f"✅ Enhanced signal sent at {now_str}")
        
    except Exception as e:
        error_msg = f"❌ Bot Error: {str(e)}"
        send_telegram_message(error_msg)
        print(error_msg)

# Run the enhanced bot
if __name__ == "__main__":
    run_enhanced_bot()
