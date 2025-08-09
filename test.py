import yfinance as yf
import logging

logging.basicConfig(level=logging.DEBUG)

try:
    stock = yf.Ticker("AAPL")
    data = stock.history(period="1y")
    print(data)
except Exception as e:
    logging.error(f"Error fetching data: {e}")