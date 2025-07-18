from newsapi import NewsApiClient

api_key = "433d2175377d428d9f6f75c056ad5ee4"
newsapi = NewsApiClient(api_key=api_key)

try:
    res = newsapi.get_top_headlines(q="stock", language="en", page_size=1)
    print("✅ SUCCESS")
    for article in res["articles"]:
        print(f"- {article['title']}")
except Exception as e:
    print("❌ ERROR:", e)
