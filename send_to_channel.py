import asyncio
from telegram import Bot

TELEGRAM_TOKEN = '8019479710:AAGkXJrUF9k_q9x7GSMtUg_A8PPlSKxzs_E'
CHAT_ID = -1002604709037  # ✅ Use numeric chat ID from getUpdates

async def main():
    bot = Bot(token=TELEGRAM_TOKEN)
    msg = await bot.send_message(chat_id=CHAT_ID, text="📢 MarketUpLatesBot is now LIVE!")
    print("Message sent. Telegram response:", msg)

asyncio.run(main())
