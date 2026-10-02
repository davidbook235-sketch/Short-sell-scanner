import pandas as pd
import yfinance as yf
import streamlit as st

@st.cache_data(ttl=3600)
def get_nifty500_symbols():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    df = pd.read_csv(url)
    # NSE symbol → Yahoo Finance symbol
    symbols = [f"{s}.NS" for s in df['Symbol'].tolist()]
    return symbols

def fetch_ohlcv(symbol, period="6mo", interval="1d"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df
    except Exception:
        return None
