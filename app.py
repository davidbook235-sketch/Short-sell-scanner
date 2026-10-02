import streamlit as st
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from data_loader import get_nifty500_symbols, fetch_ohlcv
from scanner import scan_short_candidates
from backtest import backtest_short, compute_metrics

st.set_page_config(page_title="N500 Short Scanner", page_icon="📉", layout="wide")

st.markdown("""
<style>
    .main { padding: 0.5rem; }
    .stDataFrame { font-size: 12px; }
    .stButton button { width: 100%; }
    @media (max-width: 768px) {
        h1 { font-size: 1.3rem !important; }
        h3 { font-size: 1rem !important; }
    }
</style>
""", unsafe_allow_html=True)

st.title("📉 Nifty 500 Short Sell Scanner")

tab1, tab2 = st.tabs(["🔍 Live Scanner", "🧪 Backtest"])


# ============================================================
# TAB 1: LIVE SCANNER (पहले जैसा)
# ============================================================
with tab1:
    with st.sidebar:
        st.header("⚙️ Scan Settings")
        max_stocks = st.slider("Stocks to Scan", 50, 500, 200, 50, key="scan_max")
        min_score = st.slider("Min Bearish Score", 1, 7, 4, key="scan_score")
        interval = st.selectbox("Timeframe", ["1d", "1h"], index=0, key="scan_tf")
        scan_btn = st.button("🚀 Start Scan", type="primary", key="scan_btn")

    if scan_btn:
        symbols = get_nifty500_symbols()[:max_stocks]
        progress = st.progress(0, text="Fetching data...")
        results = []

        def process(sym):
            df = fetch_ohlcv(sym, period="6mo", interval=interval)
            return scan_short_candidates(df, sym)

        with ThreadPoolExecutor(max_workers=10) as executor:
            for i, res in enumerate(executor.map(process, symbols)):
                if res and res["Score"] >= min_score:
                    results.append(res)
                progress.progress((i + 1) / len(symbols),
                                  text=f"Scanned {i+1}/{len(symbols)}")
        progress.empty()

        if results:
            df_res = pd.DataFrame(results).sort_values("Score", ascending=False)
            c1, c2, c3 = st.columns(3)
            c1.metric("Hits", len(df_res))
            c2.metric("Strong", len(df_res[df_res["Score"] >= 5]))
            c3.metric("Avg RSI", round(df_res["RSI"].mean(), 1))

            for _, row in df_res.iterrows():
                with st.expander(f"{row['Signal']} — {row['Symbol']} @ ₹{row['Price']}"):
                    c1, c2 = st.columns(2)
                    c1.metric("RSI", row['RSI'])
                    c2.metric("ADX", row['ADX'])
                    c1.metric("Vol Ratio", f"{row['Vol_Ratio']}x")
                    c2.metric("ATR", row['ATR'])

            csv = df_res.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ CSV", csv, "short_scan.csv", "text/csv")
        else:
            st.warning("कोई stock pass नहीं हुआ। Min Score कम करें।")


# ============================================================
# TAB 2: BACKTEST
# ============================================================
with tab2:
    st.subheader("🧪 Short Strategy Backtest")

    col1, col2 = st.columns(2)
    with col1:
        bt_symbol = st.text_input("Symbol (NSE)", value="RELIANCE",
                                  help="जैसे RELIANCE, TCS, INFY")
        bt_period = st.selectbox("History", ["1y", "2y", "5y"], index=1)
        bt_interval = st.selectbox("Candle", ["1d", "1h"], index=0)
    with col2:
        bt_capital = st.number_input("Initial Capital ₹", 10000, 10000000, 100000, 10000)
        bt_risk = st.slider("Risk per Trade %", 0.5, 5.0, 2.0, 0.5) / 100
        bt_sl_mult = st.slider("Stop-Loss (ATR ×)", 0.5, 3.0, 1.5, 0.5)
        bt_target_mult = st.slider("Target (ATR ×)", 1.0, 6.0, 3.0, 0.5)

    col3, col4 = st.columns(2)
    with col3:
        bt_min_score = st.slider("Min Bearish Score", 2, 7, 4)
    with col4:
        bt_max_hold = st.slider("Max Hold (candles)", 3, 30, 10)

    run_bt = st.button("▶️ Run Backtest", type="primary", key="bt_run")

    if run_bt:
        yf_symbol = f"{bt_symbol.upper()}.NS"
        with st.spinner(f"Fetching {yf_symbol} data..."):
            df = fetch_ohlcv(yf_symbol, period=bt_period, interval=bt_interval)

        if df is None or len(df) < 60:
            st.error("डेटा नहीं मिला या पर्याप्त नहीं है। Symbol check करें।")
        else:
            equity_df, trades_df = backtest_short(
                df,
                initial_capital=bt_capital,
                risk_per_trade=bt_risk,
                atr_sl_mult=bt_sl_mult,
                atr_target_mult=bt_target_mult,
                min_score=bt_min_score,
                max_hold_days=bt_max_hold
            )

            if trades_df.empty:
                st.warning("इस symbol पर कोई trade signal नहीं मिला। Score कम करें।")
            else:
                metrics = compute_metrics(equity_df, trades_df, bt_capital)

                # ---- Metrics Grid ----
                st.markdown("### 📊 Performance Metrics")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Return", f"{metrics['Total Return %']}%")
                m2.metric("Win Rate", f"{metrics['Win Rate %']}%")
                m3.metric("Profit Factor", metrics['Profit Factor'])
                m4.metric("Max Drawdown", f"{metrics['Max Drawdown %']}%")

                m5, m6, m7, m8 = st.columns(4)
                m5.metric("Total Trades", metrics['Total Trades'])
                m6.metric("Avg Win ₹", metrics['Avg Win ₹'])
                m7.metric("Avg Loss ₹", metrics['Avg Loss ₹'])
                m8.metric("Sharpe", metrics['Sharpe Ratio'])

                # ---- Equity Curve ----
                st.markdown("### 📈 Equity Curve")
                st.line_chart(equity_df['Equity'])

                # ---- Drawdown ----
                st.markdown("### 📉 Drawdown %")
                equity = equity_df['Equity']
                dd = (equity - equity.cummax()) / equity.cummax() * 100
                st.area_chart(dd)

                # ---- Trade Log ----
                st.markdown("### 📋 Trade Log")
                st.dataframe(trades_df, use_container_width=True, height=300)

                # ---- CSV Download ----
                c1, c2 = st.columns(2)
                c1.download_button("⬇️ Trades CSV",
                                   trades_df.to_csv(index=False).encode('utf-8'),
                                   f"{bt_symbol}_trades.csv")
                c2.download_button("⬇️ Equity CSV",
                                   equity_df.to_csv().encode('utf-8'),
                                   f"{bt_symbol}_equity.csv")

st.caption("⚠️ Educational only. Backtest past performance ≠ future results.")
