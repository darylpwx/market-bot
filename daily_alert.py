import os
from dotenv import load_dotenv
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")

# Init OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def get_us_market_data():
    """Get comprehensive US market data focusing on SPY, QQQ, and TQQQ"""
    tickers = {
        'SPY': 'S&P 500',
        'QQQ': 'Nasdaq-100',
        'TQQQ': 'Nasdaq 3x Bull',
        '^VIX': 'Volatility Index',
        'UUP': 'Dollar Index',
        'TLT': '20Y Treasury',
        '^TNX': '10Y Treasury Yield',
        'XLF': 'Financials',
        'XLK': 'Technology',
        'XLC': 'Communications',
        'XLY': 'Consumer Discretionary',
        'XLI': 'Industrials'
    }

    market_data = {}

    for ticker, name in tickers.items():
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="30d", interval="1d")

            if len(hist) < 2:
                continue

            current = hist['Close'].iloc[-1]
            previous = hist['Close'].iloc[-2]
            change = current - previous
            pct_change = (change / previous) * 100

            # Technical indicators for TQQQ trading
            sma_10 = hist['Close'].rolling(window=10).mean().iloc[-1]
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(window=min(50, len(hist))).mean().iloc[-1]

            # Enhanced RSI calculation
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # MACD for trend confirmation
            exp1 = hist['Close'].ewm(span=12).mean()
            exp2 = hist['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            macd_histogram = macd - signal

            # Volume analysis
            avg_volume = hist['Volume'].rolling(window=20).mean().iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            # Support and resistance levels (20-day high/low)
            resistance = hist['High'].rolling(window=20).max().iloc[-1]
            support = hist['Low'].rolling(window=20).min().iloc[-1]

            market_data[ticker] = {
                'name': name,
                'price': current,
                'change': change,
                'pct_change': pct_change,
                'sma_10': sma_10,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': rsi.iloc[-1] if not rsi.empty else 50,
                'macd': macd.iloc[-1] if not macd.empty else 0,
                'macd_signal': signal.iloc[-1] if not signal.empty else 0,
                'macd_histogram': macd_histogram.iloc[-1] if not macd_histogram.empty else 0,
                'volume_ratio': volume_ratio,
                'resistance': resistance,
                'support': support,
                'above_sma10': current > sma_10,
                'above_sma20': current > sma_20,
                'above_sma50': current > sma_50
            }

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            continue

    return market_data

def get_us_economic_data():
    """Get US-specific economic indicators"""
    economic_data = {}

    try:
        # 10-Year Treasury yield
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="5d")
        if not hist.empty:
            economic_data['10y_yield'] = hist['Close'].iloc[-1]
            economic_data['10y_change'] = hist['Close'].iloc[-1] - hist['Close'].iloc[-2]

        # Dollar strength (DXY)
        dxy = yf.Ticker("DX-Y.NYB")
        hist = dxy.history(period="5d")
        if not hist.empty:
            economic_data['dxy'] = hist['Close'].iloc[-1]
            economic_data['dxy_change'] = hist['Close'].iloc[-1] - hist['Close'].iloc[-2]

        # Add placeholder for real economic data (normally from FRED API)
        # These would be fetched from actual sources in production
        economic_data['core_inflation'] = 3.2  # Placeholder - would fetch from FRED
        economic_data['unemployment_rate'] = 3.7  # Placeholder - would fetch from BLS
        economic_data['initial_claims'] = 230000  # Placeholder - weekly jobless claims
        economic_data['retail_sales'] = 0.4  # Placeholder - monthly change %
        economic_data['industrial_production'] = 0.2  # Placeholder - monthly change %

    except Exception as e:
        print(f"Error fetching economic data: {e}")

    return economic_data

def get_us_sector_analysis(market_data):
    """Analyze US sector performance"""
    us_sectors = {
        'XLK': 'Technology',
        'XLF': 'Financials', 
        'XLC': 'Communications',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLV': 'Healthcare',
        'XLI': 'Industrials',
        'XLE': 'Energy',
        'XLU': 'Utilities',
        'XLRE': 'Real Estate'
    }

    sector_performance = {}
    for ticker, name in us_sectors.items():
        if ticker in market_data:
            sector_performance[name] = {
                'pct_change': market_data[ticker]['pct_change'],
                'rsi': market_data[ticker]['rsi'],
                'above_sma20': market_data[ticker]['above_sma20']
            }

    # Sort by performance
    sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1]['pct_change'], reverse=True)
    return sorted_sectors

def analyze_tqqq_signals(market_data):
    """Advanced TQQQ trading signal analysis"""
    if 'TQQQ' not in market_data or 'QQQ' not in market_data or 'SPY' not in market_data:
        return {'signal': 'WAIT', 'confidence': 0, 'reason': 'Insufficient data'}

    tqqq = market_data['TQQQ']
    qqq = market_data['QQQ']
    spy = market_data['SPY']
    vix = market_data.get('^VIX', {})

    signals = []
    confidence_score = 0

    # Trend Analysis (30% weight)
    if tqqq['above_sma10'] and tqqq['above_sma20'] and qqq['above_sma20']:
        signals.append('Bullish trend intact')
        confidence_score += 30
        trend_bias = 'BULLISH'
    elif not tqqq['above_sma10'] and not tqqq['above_sma20'] and not qqq['above_sma20']:
        signals.append('Bearish trend confirmed')
        confidence_score += 30
        trend_bias = 'BEARISH'
    else:
        signals.append('Mixed trend signals')
        confidence_score += 10
        trend_bias = 'NEUTRAL'

    # RSI Analysis (25% weight)
    tqqq_rsi = tqqq['rsi']
    qqq_rsi = qqq['rsi']
    
    if tqqq_rsi < 30 and qqq_rsi < 40:
        signals.append('Oversold - potential bounce')
        confidence_score += 25
        rsi_signal = 'BUY'
    elif tqqq_rsi > 70 and qqq_rsi > 60:
        signals.append('Overbought - potential pullback')
        confidence_score += 25
        rsi_signal = 'SELL'
    elif 40 <= tqqq_rsi <= 60:
        signals.append('Neutral RSI zone')
        confidence_score += 15
        rsi_signal = 'NEUTRAL'
    else:
        signals.append('RSI transitioning')
        confidence_score += 10
        rsi_signal = 'WAIT'

    # MACD Analysis (20% weight)
    if tqqq['macd'] > tqqq['macd_signal'] and tqqq['macd_histogram'] > 0:
        signals.append('MACD bullish crossover')
        confidence_score += 20
        macd_signal = 'BUY'
    elif tqqq['macd'] < tqqq['macd_signal'] and tqqq['macd_histogram'] < 0:
        signals.append('MACD bearish crossover')
        confidence_score += 20
        macd_signal = 'SELL'
    else:
        signals.append('MACD mixed signals')
        confidence_score += 5
        macd_signal = 'NEUTRAL'

    # VIX Analysis (15% weight)
    vix_level = vix.get('price', 20)
    if vix_level > 25:
        signals.append('High volatility - reduce size')
        confidence_score += 10
        vix_signal = 'CAUTION'
    elif vix_level < 15:
        signals.append('Low volatility - normal risk')
        confidence_score += 15
        vix_signal = 'NORMAL'
    else:
        signals.append('Moderate volatility')
        confidence_score += 12
        vix_signal = 'NORMAL'

    # Volume Analysis (10% weight)
    if tqqq['volume_ratio'] > 1.5:
        signals.append('High volume confirmation')
        confidence_score += 10
    elif tqqq['volume_ratio'] < 0.8:
        signals.append('Low volume - weak signal')
        confidence_score -= 5
    else:
        signals.append('Normal volume')
        confidence_score += 5

    # Generate final signal
    if confidence_score >= 75:
        if trend_bias == 'BULLISH' and rsi_signal in ['BUY', 'NEUTRAL'] and macd_signal != 'SELL':
            final_signal = 'BUY'
        elif trend_bias == 'BEARISH' and rsi_signal in ['SELL', 'NEUTRAL'] and macd_signal != 'BUY':
            final_signal = 'SELL'
        else:
            final_signal = 'WAIT'
    elif confidence_score >= 50:
        if rsi_signal == 'BUY' and macd_signal == 'BUY':
            final_signal = 'WEAK BUY'
        elif rsi_signal == 'SELL' and macd_signal == 'SELL':
            final_signal = 'WEAK SELL'
        else:
            final_signal = 'WAIT'
    else:
        final_signal = 'WAIT'

    # Calculate entry/exit levels
    current_price = tqqq['price']
    support_level = tqqq['support']
    resistance_level = tqqq['resistance']
    
    if final_signal in ['BUY', 'WEAK BUY']:
        entry_zone = f"{current_price * 0.98:.2f} - {current_price * 1.02:.2f}"
        stop_loss = f"{support_level * 0.97:.2f}"
        target = f"{resistance_level * 0.98:.2f}"
    elif final_signal in ['SELL', 'WEAK SELL']:
        entry_zone = f"{current_price * 0.98:.2f} - {current_price * 1.02:.2f}"
        stop_loss = f"{resistance_level * 1.03:.2f}"
        target = f"{support_level * 1.02:.2f}"
    else:
        entry_zone = "Wait for better setup"
        stop_loss = "N/A"
        target = "N/A"

    return {
        'signal': final_signal,
        'confidence': min(confidence_score, 100),
        'signals': signals,
        'current_price': current_price,
        'entry_zone': entry_zone,
        'stop_loss': stop_loss,
        'target': target,
        'vix_level': vix_level,
        'trend_bias': trend_bias
    }

def get_us_market_news():
    """Get US market-focused news"""
    us_market_queries = {
        'fed': 'Federal Reserve OR Jerome Powell OR interest rates OR FOMC OR monetary policy',
        'earnings': 'earnings OR quarterly results OR S&P 500 OR Nasdaq OR US stocks',  
        'economy': 'US economy OR inflation OR unemployment OR GDP OR retail sales',
        'tech': 'US tech stocks OR Apple OR Microsoft OR Google OR Amazon OR Meta OR Tesla',
        'market': 'US stock market OR S&P 500 OR Nasdaq OR Dow Jones OR Wall Street'
    }

    all_news = {}

    for category, query in us_market_queries.items():
        url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&pageSize=5&from={(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')}&apiKey={NEWS_API_KEY}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                articles = response.json().get("articles", [])
                # Filter for US market relevance
                filtered_articles = []
                for a in articles:
                    title_lower = a['title'].lower()
                    if any(keyword in title_lower for keyword in ['us', 'america', 'nasdaq', 's&p', 'dow', 'fed', 'wall street', 'stock']):
                        filtered_articles.append({
                            'title': a['title'],
                            'source': a['source']['name'],
                            'url': a['url'],
                            'publishedAt': a['publishedAt'],
                            'description': a.get('description', '')[:200] + '...' if a.get('description') else ''
                        })
                all_news[category] = filtered_articles[:3]  # Limit to top 3 per category
            else:
                print(f"News API error for {category}: {response.status_code}")
                all_news[category] = []
        except Exception as e:
            print(f"Error fetching {category} news: {e}")
            all_news[category] = []

    return all_news

def get_top_us_stories(news_data, market_data, limit=4):
    """Get most relevant US market stories"""
    all_stories = []
    
    # Prioritize based on market conditions
    spy_change = market_data.get('SPY', {}).get('pct_change', 0)
    vix_level = market_data.get('^VIX', {}).get('price', 20)

    if abs(spy_change) > 1 or vix_level > 25:
        priority_categories = ['market', 'fed', 'economy', 'earnings']
    else:
        priority_categories = ['earnings', 'tech', 'fed', 'market']

    for category in priority_categories:
        for story in news_data.get(category, []):
            if story not in all_stories:
                story['category'] = category
                all_stories.append(story)

    # Sort by recency
    all_stories.sort(key=lambda x: x['publishedAt'], reverse=True)
    return all_stories[:limit]

def generate_us_market_analysis(market_data, economic_data, sector_analysis, news_data, tqqq_signals, top_stories):
    """Generate comprehensive US market analysis using AI"""
    
    def safe_format(value, decimals=2):
        if value is None or value == 'N/A':
            return 'N/A'
        try:
            return f"{float(value):.{decimals}f}"
        except:
            return str(value)

    def safe_format_change(value, decimals=2):
        if value is None or value == 'N/A':
            return 'N/A'
        try:
            return f"{float(value):+.{decimals}f}"
        except:
            return str(value)

    # Prepare market data
    spy_info = market_data.get('SPY', {})
    qqq_info = market_data.get('QQQ', {})
    vix_info = market_data.get('^VIX', {})

    # Build top stories string
    news_str = ""
    for i, story in enumerate(top_stories, 1):
        news_str += f"{i}. {story['title']} ({story['source']})\n"

    system_msg = """You are a professional US market analyst focused on providing actionable insights for US equity markets. 

Generate a comprehensive daily market brief with these sections:

1. **US MARKET OVERVIEW**: Focus on SPY performance, key macro/micro factors affecting US markets, economic data impact
2. **US SECTOR HIGHLIGHTS**: Top 3 performing US sectors with brief analysis
3. **MARKET IMPACT ANALYSIS**: How current economic factors are affecting US markets specifically  
4. **INVESTOR IMPLICATIONS**: What this means for US equity investors with clear actionable insights
5. **CALL TO ACTION**: Specific recommendations for positioning in current environment

Keep analysis focused on US markets, be concise but insightful, and provide actionable intelligence."""

    # Build comprehensive data summary
    user_msg = f"""🇺🇸 US MARKET DATA - {datetime.now().strftime("%B %d, %Y")}

MARKET PERFORMANCE:
• SPY: ${safe_format(spy_info.get('price', 'N/A'))} ({safe_format_change(spy_info.get('pct_change', 0))}%)
• QQQ: ${safe_format(qqq_info.get('price', 'N/A'))} ({safe_format_change(qqq_info.get('pct_change', 0))}%)
• VIX: {safe_format(vix_info.get('price', 'N/A'))} ({safe_format_change(vix_info.get('pct_change', 0))}%)

ECONOMIC DATA:
• 10Y Yield: {safe_format(economic_data.get('10y_yield', 'N/A'))}% ({safe_format_change(economic_data.get('10y_change', 0))}%)
• Core Inflation: {safe_format(economic_data.get('core_inflation', 'N/A'))}%
• Unemployment: {safe_format(economic_data.get('unemployment_rate', 'N/A'))}%
• Initial Claims: {economic_data.get('initial_claims', 'N/A'):,}
• Dollar Index: {safe_format(economic_data.get('dxy', 'N/A'))}

TOP SECTOR PERFORMERS:
"""

    # Add top 3 sectors
    for i, (sector, data) in enumerate(sector_analysis[:3]):
        user_msg += f"• {sector}: {safe_format_change(data['pct_change'])}%\n"

    user_msg += f"""
KEY HEADLINES:
{news_str}

Provide analysis focusing on US market implications, economic factor impacts, and actionable investor guidance."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI error: {e}")
        return generate_fallback_analysis(market_data, economic_data, sector_analysis, top_stories)

def generate_fallback_analysis(market_data, economic_data, sector_analysis, top_stories):
    """Fallback analysis without OpenAI"""
    spy_info = market_data.get('SPY', {})
    qqq_info = market_data.get('QQQ', {})
    vix_info = market_data.get('^VIX', {})

    def safe_format(value, decimals=2):
        try:
            return f"{float(value):.{decimals}f}"
        except:
            return str(value)

    def safe_format_change(value, decimals=2):
        try:
            return f"{float(value):+.{decimals}f}"
        except:
            return str(value)

    analysis = f"""🇺🇸 **US MARKET OVERVIEW**
The S&P 500 closed at ${safe_format(spy_info.get('price', 'N/A'))}, {safe_format_change(spy_info.get('pct_change', 0))}% for the session. The Nasdaq-100 moved {safe_format_change(qqq_info.get('pct_change', 0))}% with current volatility at {safe_format(vix_info.get('price', 'N/A'))} VIX level.

Key economic factors: 10-year yields at {safe_format(economic_data.get('10y_yield', 'N/A'))}%, core inflation at {safe_format(economic_data.get('core_inflation', 'N/A'))}%, unemployment at {safe_format(economic_data.get('unemployment_rate', 'N/A'))}%.

📊 **US SECTOR HIGHLIGHTS**"""

    for i, (sector, data) in enumerate(sector_analysis[:3], 1):
        analysis += f"\n{i}. {sector}: {safe_format_change(data['pct_change'])}% - {'Above' if data['above_sma20'] else 'Below'} 20-day average"

    analysis += f"""

💼 **MARKET IMPACT ANALYSIS**
Current economic conditions suggest {'elevated caution' if vix_info.get('price', 20) > 20 else 'stable conditions'} for US equities. The {'rising' if economic_data.get('10y_change', 0) > 0 else 'falling'} 10-year yield environment is {'supportive' if economic_data.get('10y_change', 0) < 0 else 'challenging'} for growth stocks.

🎯 **INVESTOR IMPLICATIONS** 
Investors should {'maintain defensive positioning' if vix_info.get('price', 20) > 25 else 'consider tactical opportunities'} given current market conditions. Focus on {'quality names with strong fundamentals' if spy_info.get('pct_change', 0) < 0 else 'momentum plays in leading sectors'}.

⚡ **CALL TO ACTION**
{'Reduce risk exposure and increase cash positions' if vix_info.get('price', 20) > 25 else 'Monitor for entry opportunities in quality names on any weakness'}.
"""

    return analysis

def create_us_market_charts():
    """Create charts for SPY, QQQ, and TQQQ"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('US Market Performance - 7 Days', fontsize=16, fontweight='bold')

        tickers = ['SPY', 'QQQ', 'TQQQ', '^VIX']
        axes = [ax1, ax2, ax3, ax4]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for idx, (ticker, ax, color) in enumerate(zip(tickers, axes, colors)):
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="7d", interval="1h")

                if len(hist) < 2:
                    continue

                # Plot price
                ax.plot(hist.index, hist['Close'], color=color, linewidth=2)
                ax.fill_between(hist.index, hist['Close'], alpha=0.3, color=color)

                # Calculate metrics
                current_price = hist['Close'].iloc[-1]
                start_price = hist['Close'].iloc[0]
                change = current_price - start_price
                pct_change = (change / start_price) * 100

                # Title with current info
                title_color = 'green' if pct_change >= 0 else 'red'
                ax.set_title(f'{ticker} - ${current_price:.2f} ({pct_change:+.2f}%)', 
                           fontsize=12, fontweight='bold', color=title_color)

                # Format axes
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

                ax.grid(True, alpha=0.3)
                ax.set_ylabel('Price ($)', fontsize=10)

            except Exception as e:
                print(f"Error creating chart for {ticker}: {e}")
                continue

        plt.tight_layout()

        # Save to BytesIO
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        img_buffer.seek(0)
        plt.close()

        return img_buffer

    except Exception as e:
        print(f"Error creating charts: {e}")
        return None

def send_chart_to_telegram(chart_buffer):
    """Send chart to Telegram"""
    if not chart_buffer:
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

    try:
        files = {'photo': ('us_market_chart.png', chart_buffer, 'image/png')}
        data = {
            'chat_id': CHAT_ID,
            'caption': f'📈 US Market Charts - {datetime.now().strftime("%B %d, %Y")}'
        }

        response = requests.post(url, files=files, data=data, timeout=30)
        return response.status_code == 200

    except Exception as e:
        print(f"Error sending chart: {e}")
        return False

def send_to_telegram(message):
    """Send message to Telegram with formatting"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    # Handle long messages
    if len(message) > 4000:
        chunks = []
        lines = message.split('\n')
        current_chunk = ""

        for line in lines:
            if len(current_chunk + line + '\n') < 4000:
                current_chunk += line + '\n'
            else:
                chunks.append(current_chunk)
                current_chunk = line + '\n'
        
        if current_chunk:
            chunks.append(current_chunk)

        for chunk in chunks:
            payload = {
                "chat_id": CHAT_ID,
                "text": chunk,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            }
            try:
                requests.post(url, data=payload, timeout=30)
            except Exception as e:
                print(f"Error sending chunk: {e}")
    else:
        payload = {
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "Markdown", 
            "disable_web_page_preview": True
        }
        try:
            requests.post(url, data=payload, timeout=30)
        except Exception as e:
            print(f"Error sending message: {e}")

def format_tqqq_trading_signal(tqqq_signals, market_data):
    """Format TQQQ trading signals for Telegram"""
    signal_data = tqqq_signals
    tqqq_price = market_data.get('TQQQ', {}).get('price', 0)
    
    # Signal emoji
    signal_emoji = {
        'BUY': '🟢',
        'WEAK BUY': '🟡', 
        'SELL': '🔴',
        'WEAK SELL': '🟠',
        'WAIT': '⚪'
    }

    confidence_bars = "█" * (signal_data['confidence'] // 10)
    
    message = f"""🎯 **TQQQ TRADING SIGNAL** - {datetime.now().strftime('%B %d, %Y')}
{signal_emoji.get(signal_data['signal'], '⚪')} **Signal: {signal_data['signal']}**

📊 **Confidence:** {signal_data['confidence']}/100
{confidence_bars}

💰 **Current Price:** ${signal_data['current_price']:.2f}
🎯 **Entry Zone:** {signal_data['entry_zone']}
🛡️ **Stop Loss:** {signal_data['stop_loss']}
🎯 **Target:** {signal_data['target']}

📈 **Market Context:**
• Trend Bias: {signal_data['trend_bias']}
• VIX Level: {signal_data['vix_level']:.1f}

🔍 **Signal Analysis:**"""

    for signal in signal_data['signals']:
        message += f"\n• {signal}"

    # Add action recommendation
    if signal_data['signal'] in ['BUY', 'WEAK BUY']:
        message += f"\n\n✅ **ACTION:** Consider opening long position in entry zone with stop at {signal_data['stop_loss']}"
    elif signal_data['signal'] in ['SELL', 'WEAK SELL']:
        message += f"\n\n🔴 **ACTION:** Consider short position or exit longs with stop at {signal_data['stop_loss']}"
    else:
        message += f"\n\n⏳ **ACTION:** Wait for clearer signals. Monitor for breakout above resistance or breakdown below support"

    message += f"\n\n⚠️ **Risk Management:** Use proper position sizing (1-2% risk per trade)"
    
    return message

def send_news_articles(top_stories):
    """Send formatted US market news to Telegram"""
    if not top_stories:
        return

    news_message = f"📰 **US MARKET DRIVERS** - {datetime.now().strftime('%B %d, %Y')}\n\n"

    for i, story in enumerate(top_stories, 1):
        news_message += f"**{i}. {story['title']}**\n"
        news_message += f"🔗 Source: {story['source']}\n"
        if story['description']:
            news_message += f"📝 {story['description']}\n"
        news_message += f"🔗 [Read More]({story['url']})\n\n"

    news_message += f"🤖 US Market Focus | {datetime.now().strftime('%H:%M:%S')} EST"
    send_to_telegram(news_message)

def main():
    """Main execution function for US Market & TQQQ Alert"""
    try:
        print("🇺🇸 Starting US Market & TQQQ Analysis...")
        
        # Get market data
        print("📊 Fetching US market data...")
        market_data = get_us_market_data()
        
        if not market_data:
            raise Exception("Failed to retrieve market data")

        # Get economic data
        print("📈 Gathering US economic indicators...")  
        economic_data = get_us_economic_data()

        # Analyze sectors
        print("🏭 Analyzing US sector performance...")
        sector_analysis = get_us_sector_analysis(market_data)

        # Get news
        print("📰 Fetching US market news...")
        news_data = get_us_market_news()
        top_stories = get_top_us_stories(news_data, market_data, limit=4)

        # TQQQ signal analysis
        print("🎯 Analyzing TQQQ trading signals...")
        tqqq_signals = analyze_tqqq_signals(market_data)

        # Generate analysis
        print("✍️ Generating market analysis...")
        market_analysis = generate_us_market_analysis(
            market_data, economic_data, sector_analysis, 
            news_data, tqqq_signals, top_stories
        )

        # Create charts
        print("📊 Creating market charts...")
        chart_buffer = create_us_market_charts()

        # Send to Telegram
        print("📨 Sending alerts...")
        
        # Send chart first
        if chart_buffer:
            send_chart_to_telegram(chart_buffer)

        # Send main analysis
        send_to_telegram(market_analysis)

        # Send TQQQ trading signal
        tqqq_signal_message = format_tqqq_trading_signal(tqqq_signals, market_data)
        send_to_telegram(tqqq_signal_message)

        # Send news
        if top_stories:
            send_news_articles(top_stories)

        print("✅ US Market & TQQQ Alert completed successfully!")

    except Exception as e:
        error_msg = f"⚠️ ERROR: US Market Alert failed - {str(e)}"
        print(error_msg)
        send_to_telegram(error_msg)

if __name__ == "__main__":
    main()
