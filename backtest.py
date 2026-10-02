import pandas as pd
import numpy as np
from indicators import (
    calc_ema, calc_rsi, calc_adx, calc_macd, calc_atr, calc_bollinger
)


def generate_signals(df):
    """
    हर candle पर short entry/exit signal generate करता है।
    Entry: bearish setup (score >= 4)
    Exit: stop-loss hit, target hit, या reverse signal
    """
    close = df['Close']
    high = df['High']
    low = df['Low']

    # Indicators (full series)
    ema20 = calc_ema(close, 20)
    ema50 = calc_ema(close, 50)
    rsi = calc_rsi(close, 14)
    adx = calc_adx(df, 14)
    macd, signal, hist = calc_macd(close)
    atr = calc_atr(df, 14)
    upper, middle, lower = calc_bollinger(close)

    vol_avg = df['Volume'].rolling(20).mean()
    vol_ratio = df['Volume'] / vol_avg

    # Bearish score हर candle पर (0-7)
    score = (
        (close < ema20).astype(int) +
        (ema20 < ema50).astype(int) +
        (rsi < 45).astype(int) +
        (adx > 20).astype(int) +
        (hist < 0).astype(int) +
        (close < lower).astype(int) +
        (vol_ratio > 1.5).astype(int)
    )

    df = df.copy()
    df['EMA20'] = ema20
    df['EMA50'] = ema50
    df['RSI'] = rsi
    df['ADX'] = adx
    df['MACD_Hist'] = hist
    df['ATR'] = atr
    df['Score'] = score
    df['BB_Lower'] = lower
    df['Vol_Ratio'] = vol_ratio

    return df


def backtest_short(df, initial_capital=100000,
                   risk_per_trade=0.02,
                   atr_sl_mult=1.5,
                   atr_target_mult=3.0,
                   min_score=4,
                   max_hold_days=10):
    """
    Short-only backtest.

    Parameters:
    - initial_capital: शुरुआती पूंजी
    - risk_per_trade: हर trade में capital का कितना % risk (0.02 = 2%)
    - atr_sl_mult: stop-loss = entry + (ATR × multiplier)
    - atr_target_mult: target = entry - (ATR × multiplier)
    - min_score: entry के लिए minimum bearish score
    - max_hold_days: max days to hold (time-based exit)
    """
    df = generate_signals(df)
    capital = initial_capital
    equity_curve = []
    trades = []

    in_position = False
    entry_price = 0
    entry_date = None
    stop_loss = 0
    target = 0
    qty = 0
    entry_idx = 0

    for i in range(50, len(df)):
        row = df.iloc[i]
        date = df.index[i]

        # ---- EXIT LOGIC ----
        if in_position:
            exit_price = None
            exit_reason = None

            # Stop-loss hit (intraday check on High)
            if row['High'] >= stop_loss:
                exit_price = stop_loss
                exit_reason = "SL"
            # Target hit (intraday check on Low)
            elif row['Low'] <= target:
                exit_price = target
                exit_reason = "Target"
            # Time exit
            elif (i - entry_idx) >= max_hold_days:
                exit_price = row['Close']
                exit_reason = "Time"
            # Reverse signal: price above EMA20 और RSI > 55
            elif row['Close'] > row['EMA20'] and row['RSI'] > 55:
                exit_price = row['Close']
                exit_reason = "Reverse"

            if exit_price is not None:
                # Short profit = entry - exit
                pnl = (entry_price - exit_price) * qty
                capital += pnl
                trades.append({
                    "Entry_Date": entry_date,
                    "Exit_Date": date,
                    "Entry": round(entry_price, 2),
                    "Exit": round(exit_price, 2),
                    "Qty": qty,
                    "PnL": round(pnl, 2),
                    "Return_%": round((entry_price - exit_price) / entry_price * 100, 2),
                    "Reason": exit_reason,
                    "Hold_Days": i - entry_idx
                })
                in_position = False

        # ---- ENTRY LOGIC ----
        if not in_position and row['Score'] >= min_score:
            entry_price = row['Close']
            entry_date = date
            entry_idx = i
            atr_val = row['ATR']

            if pd.isna(atr_val) or atr_val <= 0:
                continue

            stop_loss = entry_price + (atr_val * atr_sl_mult)
            target = entry_price - (atr_val * atr_target_mult)

            # Position sizing: risk_per_trade % of capital
            risk_amount = capital * risk_per_trade
            risk_per_share = stop_loss - entry_price
            qty = max(int(risk_amount / risk_per_share), 1)

            in_position = True

        equity_curve.append({
            "Date": date,
            "Equity": capital
        })

    equity_df = pd.DataFrame(equity_curve).set_index("Date")
    trades_df = pd.DataFrame(trades)

    return equity_df, trades_df


def compute_metrics(equity_df, trades_df, initial_capital):
    """Performance metrics calculate करता है।"""
    if trades_df.empty:
        return {}

    final_capital = equity_df['Equity'].iloc[-1]
    total_return = (final_capital - initial_capital) / initial_capital * 100

    wins = trades_df[trades_df['PnL'] > 0]
    losses = trades_df[trades_df['PnL'] <= 0]

    win_rate = len(wins) / len(trades_df) * 100
    avg_win = wins['PnL'].mean() if not wins.empty else 0
    avg_loss = losses['PnL'].mean() if not losses.empty else 0

    gross_profit = wins['PnL'].sum()
    gross_loss = abs(losses['PnL'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown
    equity = equity_df['Equity']
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max * 100
    max_dd = drawdown.min()

    # Sharpe (daily returns, annualized)
    daily_ret = equity.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    return {
        "Final Capital": round(final_capital, 2),
        "Total Return %": round(total_return, 2),
        "Total Trades": len(trades_df),
        "Win Rate %": round(win_rate, 2),
        "Avg Win ₹": round(avg_win, 2),
        "Avg Loss ₹": round(avg_loss, 2),
        "Profit Factor": round(profit_factor, 2),
        "Max Drawdown %": round(max_dd, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Avg Hold Days": round(trades_df['Hold_Days'].mean(), 1),
              }
