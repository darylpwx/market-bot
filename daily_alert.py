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
FMP_API_KEY = os.getenv("FMP_API_KEY")  # Financial Modeling Prep for additional data

# Init OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def create_market_charts():
    """Create charts for SPY and QQQ over the last 7 days"""
    try:
        # Set up the figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('7-Day Market Performance', fontsize=16, fontweight='bold')
        
        # Chart styling
        plt.style.use('seaborn-v0_8')
        
        # Fetch 7-day data for SPY and QQQ
        tickers = ['SPY', 'QQQ']
        colors = ['#1f77b4', '#ff7f0e']  # Blue for SPY, Orange for QQQ
        
        for i, ticker in enumerate(tickers):
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="7d", interval="1h")
                
                if len(hist) < 2:
                    continue
                
                ax = ax1 if i == 0 else ax2
                
                # Plot price line
                ax.plot(hist.index, hist['Close'], color=colors[i], linewidth=2, label=f'{ticker} Close')
                ax.fill_between(hist.index, hist['Close'], alpha=0.3, color=colors[i])
                
                # Calculate and display key metrics
                current_price = hist['Close'].iloc[-1]
                start_price = hist['Close'].iloc[0]
                change = current_price - start_price
                pct_change = (change / start_price) * 100
                
                # Format the title with current info
                title_color = 'green' if pct_change >= 0 else 'red'
                ax.set_title(f'{ticker} - ${current_price:.2f} ({pct_change:+.2f}%)', 
                           fontsize=14, fontweight='bold', color=title_color)
                
                # Format x-axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # Add grid and styling
                ax.grid(True, alpha=0.3)
                ax.set_ylabel('Price ($)', fontsize=10)
                
                # Add volume subplot (smaller)
                ax_vol = ax.twinx()
                ax_vol.bar(hist.index, hist['Volume'], alpha=0.3, color=colors[i], width=0.02)
                ax_vol.set_ylabel('Volume', fontsize=8, alpha=0.7)
                ax_vol.tick_params(labelsize=8)
                
            except Exception as e:
                print(f"Error creating chart for {ticker}: {e}")
                continue
        
        # Adjust layout
        plt.tight_layout()
        
        # Save to BytesIO
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        img_buffer.seek(0)
        
        plt.close()  # Close the figure to free memory
        
        return img_buffer
        
    except Exception as e:
        print(f"Error creating charts: {e}")
        return None

def get_comprehensive_market_data():
    """Get comprehensive market data including multiple indices, sectors, and technical indicators"""
    tickers = {
        'SPY': 'S&P 500',
        'QQQ': 'Nasdaq-100',
        'IWM': 'Russell 2000',
        '^VIX': 'Volatility Index',
        'UUP': 'Dollar Index',
        'TLT': '20Y Treasury',
        'GLD': 'Gold',
        'XLF': 'Financials',
        'XLK': 'Technology',
        'XLE': 'Energy',
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
            
            # Technical indicators
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(window=min(50, len(hist))).mean().iloc[-1]
            
            # RSI calculation
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Volume analysis
            avg_volume = hist['Volume'].rolling(window=20).mean().iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            market_data[ticker] = {
                'name': name,
                'price': current,
                'change': change,
                'pct_change': pct_change,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': rsi.iloc[-1] if not rsi.empty else 50,
                'volume_ratio': volume_ratio,
                'resistance': hist['High'].rolling(window=20).max().iloc[-1],
                'support': hist['Low'].rolling(window=20).min().iloc[-1]
            }
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            continue
    
    return market_data

def get_economic_indicators():
    """Get key economic indicators and Fed data"""
    indicators = {}
    
    try:
        # Treasury yields
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="5d")
        if not hist.empty:
            indicators['10y_yield'] = hist['Close'].iloc[-1]
            indicators['10y_change'] = hist['Close'].iloc[-1] - hist['Close'].iloc[-2]
        
        # Dollar strength
        dxy = yf.Ticker("DX-Y.NYB")
        hist = dxy.history(period="5d")
        if not hist.empty:
            indicators['dxy'] = hist['Close'].iloc[-1]
            indicators['dxy_change'] = hist['Close'].iloc[-1] - hist['Close'].iloc[-2]
            
    except Exception as e:
        print(f"Error fetching economic indicators: {e}")
    
    return indicators

def get_sector_rotation_analysis(market_data):
    """Analyze sector rotation patterns"""
    sectors = {
        'XLF': 'Financials',
        'XLK': 'Technology', 
        'XLE': 'Energy',
        'XLI': 'Industrials',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLV': 'Healthcare',
        'XLU': 'Utilities',
        'XLRE': 'Real Estate'
    }
    
    sector_performance = {}
    for ticker, name in sectors.items():
        if ticker in market_data:
            sector_performance[name] = market_data[ticker]['pct_change']
    
    # Sort by performance
    sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_sectors

def get_market_sentiment_indicators(market_data):
    """Calculate market sentiment indicators"""
    sentiment = {}
    
    # Fear & Greed components
    vix_key = '^VIX' if '^VIX' in market_data else 'VIX'
    if vix_key in market_data:
        vix = market_data[vix_key]['price']
        if vix < 20:
            sentiment['vix_signal'] = 'Complacent'
        elif vix > 30:
            sentiment['vix_signal'] = 'Fearful'
        else:
            sentiment['vix_signal'] = 'Neutral'
    
    # Put/Call ratio proxy using VIX vs SPY
    if 'SPY' in market_data and vix_key in market_data:
        spy_rsi = market_data['SPY']['rsi']
        vix_level = market_data[vix_key]['price']
        
        if spy_rsi > 70 and vix_level < 20:
            sentiment['market_regime'] = 'Euphoric - Caution Warranted'
        elif spy_rsi < 30 and vix_level > 30:
            sentiment['market_regime'] = 'Oversold - Opportunity Zone'
        else:
            sentiment['market_regime'] = 'Normal Trading Range'
    
    return sentiment

def get_enhanced_news():
    """Get market news with better filtering and categorization"""
    categories = {
        'fed': 'Federal Reserve OR interest rates OR monetary policy OR Jerome Powell',
        'earnings': 'earnings OR quarterly results OR guidance OR revenue',
        'geopolitical': 'geopolitical OR China OR Russia OR trade war OR inflation',
        'tech': 'AI OR semiconductor OR tech stocks OR NVDA OR AAPL OR Microsoft',
        'macro': 'GDP OR unemployment OR economic data OR recession OR inflation',
        'breaking': 'stock market OR S&P 500 OR Nasdaq OR market news'
    }
    
    all_news = {}
    
    for category, query in categories.items():
        url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&pageSize=5&from={(datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')}&apiKey={NEWS_API_KEY}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                articles = response.json().get("articles", [])
                all_news[category] = [
                    {
                        'title': a['title'],
                        'source': a['source']['name'],
                        'url': a['url'],
                        'publishedAt': a['publishedAt'],
                        'description': a.get('description', '')[:200] + '...' if a.get('description') else ''
                    }
                    for a in articles if a['title'] and 'stock' in a['title'].lower() or 'market' in a['title'].lower()
                ]
            else:
                print(f"News API error for {category}: {response.status_code}")
                all_news[category] = []
        except Exception as e:
            print(f"Error fetching {category} news: {e}")
            all_news[category] = []
    
    return all_news

def get_top_market_stories(news_data, market_data, limit=3):
    """Get the most relevant market stories based on current market conditions"""
    all_stories = []
    
    # Weight news categories based on market conditions
    spy_change = market_data.get('SPY', {}).get('pct_change', 0)
    vix_level = market_data.get('^VIX', {}).get('price', 20)
    
    # If market is volatile, prioritize breaking news and fed news
    if abs(spy_change) > 1 or vix_level > 25:
        priority_categories = ['breaking', 'fed', 'macro', 'earnings']
    else:
        priority_categories = ['earnings', 'tech', 'fed', 'breaking']
    
    # Collect stories from priority categories
    for category in priority_categories:
        for story in news_data.get(category, []):
            if story not in all_stories:
                story['category'] = category
                all_stories.append(story)
    
    # Sort by recency and relevance
    all_stories.sort(key=lambda x: x['publishedAt'], reverse=True)
    
    return all_stories[:limit]

def calculate_risk_metrics(market_data):
    """Calculate portfolio risk metrics"""
    risk_metrics = {}
    
    vix_key = '^VIX' if '^VIX' in market_data else 'VIX'
    if 'SPY' in market_data and vix_key in market_data:
        # Market stress indicator
        vix = market_data[vix_key]['price']
        spy_rsi = market_data['SPY']['rsi']
        
        if vix > 25 or spy_rsi < 35:
            risk_metrics['risk_level'] = 'Elevated'
            risk_metrics['position_sizing'] = 'Reduce position sizes, increase cash'
        elif vix < 15 and spy_rsi > 65:
            risk_metrics['risk_level'] = 'Complacent'
            risk_metrics['position_sizing'] = 'Consider hedging, avoid FOMO'
        else:
            risk_metrics['risk_level'] = 'Normal'
            risk_metrics['position_sizing'] = 'Standard allocation appropriate'
    
    return risk_metrics

def generate_professional_summary(market_data, economic_indicators, news_data, sentiment, risk_metrics, sector_rotation, top_stories):
    """Generate sophisticated market analysis with news integration"""
    
    # Prepare data for GPT with safe formatting
    spy_info = market_data.get('SPY', {})
    vix_key = '^VIX' if '^VIX' in market_data else 'VIX'
    vix_info = market_data.get(vix_key, {})
    
    # Safe number formatting
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
    
    system_msg = "You are a market analyst. Summarize what happened, the impact on the market, and what it means for investors in a concise tone."

    # Build top stories string
    news_str = ""
    for i, story in enumerate(top_stories, 1):
        news_str += f"{i}. {story['title']} ({story['source']})\n"
        if story['description']:
            news_str += f"   {story['description']}\n"
    
    # Build message with safe formatting
    spy_price = safe_format(spy_info.get('price', 'N/A'))
    spy_change = safe_format_change(spy_info.get('pct_change', 0))
    spy_rsi = safe_format(spy_info.get('rsi', 50), 1)
    
    vix_price = safe_format(vix_info.get('price', 'N/A'))
    vix_change = safe_format_change(vix_info.get('pct_change', 0))
    
    ten_y_yield = safe_format(economic_indicators.get('10y_yield', 'N/A'))
    ten_y_change = safe_format_change(economic_indicators.get('10y_change', 0))
    
    support_level = safe_format(spy_info.get('support', 'N/A'))
    resistance_level = safe_format(spy_info.get('resistance', 'N/A'))
    volume_ratio = safe_format(spy_info.get('volume_ratio', 1), 1)
    
    # Build sector rotation string
    sector_str = ""
    for sector, perf in sector_rotation[:5]:
        sector_str += f"• {sector}: {safe_format_change(perf)}%\n"

    user_msg = f"""🧠 MARKET SNAPSHOT - {datetime.now().strftime("%b %d, %Y")}

SPY: {spy_price} ({spy_change}%) | RSI: {spy_rsi}
VIX: {vix_price} | Regime: {sentiment.get("market_regime", "Normal")}
10Y Yield: {ten_y_yield}%
Key Levels: Support={support_level}, Resistance={resistance_level}

Top Headlines:
1. {top_stories[0]["title"]} ({top_stories[0]["source"]})
2. {top_stories[1]["title"]} ({top_stories[1]["source"]})
3. {top_stories[2]["title"]} ({top_stories[2]["source"]})

Summarize what happened, the impact on markets, and what it means to investors."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        # Fallback to simple summary if OpenAI fails
        return generate_simple_summary(market_data, economic_indicators, news_data, sentiment, risk_metrics, sector_rotation, top_stories)

def send_chart_to_telegram(chart_buffer):
    """Send chart image to Telegram"""
    if not chart_buffer:
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    
    try:
        files = {
            'photo': ('market_chart.png', chart_buffer, 'image/png')
        }
        data = {
            'chat_id': CHAT_ID,
            'caption': f'📈 7-Day Market Charts - {datetime.now().strftime("%B %d, %Y")}'
        }
        
        response = requests.post(url, files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            print("✅ Chart sent successfully to Telegram")
            return True
        else:
            print(f"❌ Failed to send chart: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error sending chart: {e}")
        return False

def send_to_telegram(message):
    """Send message to Telegram with better formatting and error handling"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # Telegram message limit is 4096 characters
    if len(message) > 4000:
        # Split message into chunks
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
        
        # Send each chunk
        for i, chunk in enumerate(chunks):
            payload = {
                "chat_id": CHAT_ID,
                "text": chunk,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            }
            
            try:
                r = requests.post(url, data=payload, timeout=30)
                if r.status_code == 200:
                    print(f"✅ Message chunk {i+1}/{len(chunks)} sent successfully")
                else:
                    print(f"❌ Telegram error for chunk {i+1}: {r.status_code} - {r.text}")
                    # Try without markdown if it fails
                    payload["parse_mode"] = "HTML"
                    r2 = requests.post(url, data=payload, timeout=30)
                    if r2.status_code == 200:
                        print(f"✅ Chunk {i+1} sent with HTML formatting")
                    else:
                        print(f"❌ Failed completely: {r2.text}")
            except requests.exceptions.RequestException as e:
                print(f"❌ Network error sending chunk {i+1}: {e}")
    else:
        # Send single message
        payload = {
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        
        try:
            r = requests.post(url, data=payload, timeout=30)
            if r.status_code == 200:
                print("✅ Professional market brief sent to Telegram")
            else:
                print(f"❌ Telegram error: {r.status_code} - {r.text}")
                # Try without markdown if it fails
                payload["parse_mode"] = "HTML"
                r2 = requests.post(url, data=payload, timeout=30)
                if r2.status_code == 200:
                    print("✅ Message sent with HTML formatting")
                else:
                    # Last resort - send as plain text
                    payload.pop("parse_mode", None)
                    r3 = requests.post(url, data=payload, timeout=30)
                    if r3.status_code == 200:
                        print("✅ Message sent as plain text")
                    else:
                        print(f"❌ Failed completely: {r3.text}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")

def send_news_articles(top_stories):
    """Send formatted news articles to Telegram"""
    if not top_stories:
        return
    
    news_message = f"📰 **KEY MARKET DRIVERS** - {datetime.now().strftime('%B %d, %Y')}\n\n"
    
    for i, story in enumerate(top_stories, 1):
        news_message += f"**{i}. {story['title']}**\n"
        news_message += f"🔗 Source: {story['source']}\n"
        if story['description']:
            news_message += f"📝 {story['description']}\n"
        news_message += f"🔗 [Read More]({story['url']})\n\n"
    
    news_message += f"🤖 Curated at {datetime.now().strftime('%H:%M:%S')} EST"
    
    send_to_telegram(news_message)

def test_telegram_connection():
    """Test if Telegram bot is working"""
    test_message = "🔧 Testing connection... Enhanced market bot is online!"
    send_to_telegram(test_message)

def generate_simple_summary(market_data, economic_indicators, news_data, sentiment, risk_metrics, sector_rotation, top_stories=None):
    """Generate a simple market summary without OpenAI (fallback)"""
    
    spy_info = market_data.get('SPY', {})
    vix_key = '^VIX' if '^VIX' in market_data else 'VIX'
    vix_info = market_data.get(vix_key, {})
    
    # Safe formatting
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
    
    # Build summary
    summary = f"""📊 MARKET BRIEF - {datetime.now().strftime('%B %d, %Y')}

🎯 MARKET SNAPSHOT:
• SPY: ${safe_format(spy_info.get('price', 'N/A'))} ({safe_format_change(spy_info.get('pct_change', 0))}%)
• VIX: {safe_format(vix_info.get('price', 'N/A'))} ({safe_format_change(vix_info.get('pct_change', 0))}%)
• RSI: {safe_format(spy_info.get('rsi', 50), 1)}
• Market Regime: {sentiment.get('market_regime', 'Normal')}

📈 TECHNICAL LEVELS:
• Support: ${safe_format(spy_info.get('support', 'N/A'))}
• Resistance: ${safe_format(spy_info.get('resistance', 'N/A'))}
• Volume: {safe_format(spy_info.get('volume_ratio', 1), 1)}x avg

⚠️ RISK ASSESSMENT:
• Risk Level: {risk_metrics.get('risk_level', 'Normal')}
• Position Sizing: {risk_metrics.get('position_sizing', 'Standard allocation appropriate')}

🔄 TOP SECTORS TODAY:"""
    
    # Add sector performance
    for sector, perf in sector_rotation[:5]:
        summary += f"\n• {sector}: {safe_format_change(perf)}%"
    
    # Add top stories if available
    if top_stories:
        summary += "\n\n📰 KEY MARKET DRIVERS:"
        for i, story in enumerate(top_stories[:3], 1):
            summary += f"\n{i}. {story['title']} ({story['source']})"
    
    summary += f"\n\n🤖 Auto-generated at {datetime.now().strftime('%H:%M:%S')} EST"
    
    return summary

def main():
    """Main execution function"""
    try:
        # Test connection first
        print("🔧 Testing Telegram connection...")
        test_telegram_connection()
        
        print("🔄 Gathering comprehensive market data...")
        market_data = get_comprehensive_market_data()
        
        if not market_data:
            raise Exception("No market data retrieved")
        
        print("📊 Fetching economic indicators...")
        economic_indicators = get_economic_indicators()
        
        print("📰 Analyzing market news...")
        news_data = get_enhanced_news()
        
        print("🎯 Calculating sentiment indicators...")
        sentiment = get_market_sentiment_indicators(market_data)
        
        print("⚠️ Assessing risk metrics...")
        risk_metrics = calculate_risk_metrics(market_data)
        
        print("🔄 Analyzing sector rotation...")
        sector_rotation = get_sector_rotation_analysis(market_data)
        
        print("📈 Creating market charts...")
        chart_buffer = create_market_charts()
        
        print("📰 Selecting top market stories...")
        top_stories = get_top_market_stories(news_data, market_data, limit=3)
        
        print("✍️ Generating professional analysis...")
        try:
            # Try OpenAI first
            summary = generate_professional_summary(
                market_data, 
                economic_indicators, 
                news_data, 
                sentiment, 
                risk_metrics, 
                sector_rotation,
                top_stories
            )
        except Exception as openai_error:
            print(f"⚠️ OpenAI failed: {openai_error}")
            print("📝 Using simple summary instead...")
            summary = generate_simple_summary(
                market_data, 
                economic_indicators, 
                news_data, 
                sentiment, 
                risk_metrics, 
                sector_rotation,
                top_stories
            )
        
        if not summary or len(summary) < 100:
            raise Exception("Generated summary is too short or empty")
        
        print("📨 Sending to clients...")
        
        # Send chart first
        if chart_buffer:
            send_chart_to_telegram(chart_buffer)
        
        # Send main analysis
        send_to_telegram(summary)
        
        # Send news articles
        if top_stories:
            send_news_articles(top_stories)
        
        print("✅ Enhanced market brief completed successfully!")
        
    except Exception as e:
        error_msg = f"⚠️ SYSTEM ERROR: Enhanced market brief generation failed - {str(e)}"
        print(error_msg)
        send_to_telegram(error_msg)

if __name__ == "__main__":
    main()