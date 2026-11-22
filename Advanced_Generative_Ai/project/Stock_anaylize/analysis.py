"""
Stock analysis engine.

Pipeline:
    1. fetch_prices()      -> download OHLCV data from Yahoo Finance (yfinance)
    2. add_indicators()    -> SMA / RSI / MACD / Bollinger Bands (ta library)
    3. risk_metrics()      -> volatility, drawdown, Sharpe, VaR, beta
    4. classify_risk()     -> Low / Medium / High risk label
    5. window_table()      -> same metrics across several investment windows
    6. build_chart()       -> interactive Plotly figure
    7. get_verdict()       -> Gemini (via LangChain) returns INVEST / DO NOT INVEST

Use it from the Streamlit app (app.py) or from the command line:
    python analysis.py AAPL --window "6 Months" --risk Medium
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

load_dotenv()

TRADING_DAYS_PER_YEAR = 252

# Investment windows offered in the app, measured in trading days.
WINDOWS: dict[str, int] = {
    "1 Month": 21,
    "3 Months": 63,
    "6 Months": 126,
    "1 Year": 252,
    "5 Years": 1260,
}

# Extra history downloaded before the window so long indicators (SMA 200)
# are already valid on the first day of the window.
INDICATOR_LOOKBACK = 250

RISK_LEVELS = ["Low", "Medium", "High"]

# Annualised volatility and max-drawdown thresholds used to label risk.
VOL_THRESHOLDS = (0.20, 0.35)        # < 20% Low, 20-35% Medium, > 35% High
DRAWDOWN_THRESHOLDS = (-0.15, -0.30)  # > -15% Low, -15..-30% Medium, < -30% High

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
DEFAULT_BENCHMARK = "SPY"


# --------------------------------------------------------------------------- #
# 1. Data
# --------------------------------------------------------------------------- #
def fetch_prices(ticker: str, trading_days: int) -> pd.DataFrame:
    """Download enough daily history for the window plus indicator lookback."""
    calendar_days = int((trading_days + INDICATOR_LOOKBACK) * 365 / TRADING_DAYS_PER_YEAR) + 10
    start = (datetime.now() - timedelta(days=calendar_days)).strftime("%Y-%m-%d")

    df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
    if df.empty:
        raise ValueError(
            f"No price data found for '{ticker}'. Check the ticker symbol "
            "(e.g. AAPL, MSFT, RELIANCE.NS for NSE stocks)."
        )
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


# --------------------------------------------------------------------------- #
# 2. Technical indicators
# --------------------------------------------------------------------------- #
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]

    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()
    df["RSI"] = RSIIndicator(close, window=14).rsi()

    macd = MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()

    bb = BollingerBands(close, window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()

    df["Return"] = close.pct_change()
    return df


# --------------------------------------------------------------------------- #
# 3-4. Risk metrics and classification
# --------------------------------------------------------------------------- #
def risk_metrics(close: pd.Series, benchmark_close: pd.Series | None = None) -> dict:
    """Core risk/return numbers for one price series (one window)."""
    returns = close.pct_change().dropna()
    if len(returns) < 2:
        raise ValueError("Not enough price data in this window to measure risk.")

    std = returns.std()
    drawdown = close / close.cummax() - 1

    metrics = {
        "start_price": float(close.iloc[0]),
        "end_price": float(close.iloc[-1]),
        "total_return": float(close.iloc[-1] / close.iloc[0] - 1),
        "annual_volatility": float(std * math.sqrt(TRADING_DAYS_PER_YEAR)),
        "max_drawdown": float(drawdown.min()),
        # Risk-free rate assumed 0 to keep the number easy to interpret.
        "sharpe_ratio": float(returns.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR)) if std > 0 else float("nan"),
        # Historical 1-day Value at Risk at 95%: "on a bad day (1 in 20) expect to lose at least this much".
        "var_95_daily": float(-np.percentile(returns, 5)),
        "positive_days_pct": float((returns > 0).mean()),
        "beta": float("nan"),
    }

    if benchmark_close is not None and not benchmark_close.empty:
        bench_returns = benchmark_close.pct_change().dropna()
        aligned = pd.concat([returns, bench_returns], axis=1, join="inner").dropna()
        if len(aligned) > 5 and aligned.iloc[:, 1].var() > 0:
            metrics["beta"] = float(aligned.cov().iloc[0, 1] / aligned.iloc[:, 1].var())

    return metrics


def _bucket(value: float, low_cut: float, high_cut: float, higher_is_riskier: bool) -> int:
    if higher_is_riskier:
        return 0 if value < low_cut else 1 if value < high_cut else 2
    return 0 if value > low_cut else 1 if value > high_cut else 2


def classify_risk(metrics: dict) -> str:
    """Label risk by the worse of volatility and drawdown (conservative)."""
    vol_score = _bucket(metrics["annual_volatility"], *VOL_THRESHOLDS, higher_is_riskier=True)
    dd_score = _bucket(metrics["max_drawdown"], *DRAWDOWN_THRESHOLDS, higher_is_riskier=False)
    return RISK_LEVELS[max(vol_score, dd_score)]


def risk_fit(stock_risk: str, user_risk: str) -> str:
    if RISK_LEVELS.index(stock_risk) <= RISK_LEVELS.index(user_risk):
        return "Within your risk tolerance"
    return "Above your risk tolerance"


# --------------------------------------------------------------------------- #
# 5. Multi-window view
# --------------------------------------------------------------------------- #
def window_table(close: pd.Series, benchmark_close: pd.Series | None, max_days: int) -> pd.DataFrame:
    """Return / volatility / drawdown / risk for every window up to max_days."""
    rows = []
    for name, days in WINDOWS.items():
        if days > max_days or days >= len(close):
            continue
        window_close = close.iloc[-(days + 1):]
        bench = benchmark_close.loc[benchmark_close.index >= window_close.index[0]] if benchmark_close is not None else None
        m = risk_metrics(window_close, bench)
        rows.append({
            "Window": name,
            "Return": m["total_return"],
            "Annual volatility": m["annual_volatility"],
            "Max drawdown": m["max_drawdown"],
            "Sharpe": m["sharpe_ratio"],
            "Risk level": classify_risk(m),
        })
    return pd.DataFrame(rows).set_index("Window")


def trend_signals(df: pd.DataFrame) -> dict:
    """Snapshot of where the latest price sits relative to its indicators."""
    last = df.iloc[-1]
    year = df["Close"].iloc[-TRADING_DAYS_PER_YEAR:]

    def above(col: str):
        return None if pd.isna(last[col]) else bool(last["Close"] > last[col])

    rsi = float(last["RSI"])
    return {
        "price_above_sma50": above("SMA50"),
        "price_above_sma200": above("SMA200"),
        "golden_cross_active": None if pd.isna(last["SMA200"]) else bool(last["SMA50"] > last["SMA200"]),
        "rsi_14": rsi,
        "rsi_state": "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral",
        "macd_momentum": "bullish" if last["MACD_hist"] > 0 else "bearish",
        "pct_below_52w_high": float(last["Close"] / year.max() - 1),
        "pct_above_52w_low": float(last["Close"] / year.min() - 1),
    }


# --------------------------------------------------------------------------- #
# 6. Chart
# --------------------------------------------------------------------------- #
def build_chart(df: pd.DataFrame, ticker: str, window: str, risk_level: str) -> go.Figure:
    """Price + indicators, volume, RSI and drawdown on one shared time axis."""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.5, 0.14, 0.16, 0.2],
        subplot_titles=("Price", "Volume", "RSI (14)", "Drawdown from peak"),
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price",
    ), row=1, col=1)
    for col, color in (("SMA20", "#f59e0b"), ("SMA50", "#3b82f6"), ("SMA200", "#8b5cf6")):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1.4, color=color)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="Bollinger upper",
                             line=dict(width=0.8, dash="dot", color="gray")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="Bollinger lower",
                             line=dict(width=0.8, dash="dot", color="gray"),
                             fill="tonexty", fillcolor="rgba(128,128,128,0.08)"), row=1, col=1)

    up = df["Close"] >= df["Open"]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", showlegend=False,
                         marker_color=np.where(up, "#16a34a", "#dc2626")), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", showlegend=False,
                             line=dict(color="#0ea5e9")), row=3, col=1)
    for level in (70, 30):
        fig.add_hline(y=level, line_dash="dash", line_color="gray", line_width=1, row=3, col=1)

    drawdown = df["Close"] / df["Close"].cummax() - 1
    fig.add_trace(go.Scatter(x=df.index, y=drawdown, name="Drawdown", showlegend=False,
                             fill="tozeroy", line=dict(color="#dc2626")), row=4, col=1)

    fig.update_yaxes(tickformat=".0%", row=4, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])  # hide weekends
    fig.update_layout(
        title=f"{ticker} — {window} — Risk level: {risk_level}",
        height=850, xaxis_rangeslider_visible=False, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=40, r=20, t=90, b=30),
    )
    return fig


# --------------------------------------------------------------------------- #
# 7. Gemini verdict
# --------------------------------------------------------------------------- #
class InvestmentVerdict(BaseModel):
    decision: Literal["INVEST", "DO NOT INVEST"] = Field(description="Final call for this investor.")
    confidence: Literal["Low", "Medium", "High"]
    summary: str = Field(description="2-3 sentence plain-English explanation of the decision.")
    reasons: list[str] = Field(description="3-5 short bullet points supporting the decision, citing the numbers.")
    key_risks: list[str] = Field(description="2-4 short bullet points on what could go wrong.")


SYSTEM_PROMPT = """You are a cautious equity analyst helping a retail investor.
You receive pre-computed risk metrics and technical signals for one stock.
Decide INVEST or DO NOT INVEST for THIS investor, considering:
- whether the stock's risk level fits the investor's risk tolerance (a mismatch should weigh heavily),
- risk-adjusted return (Sharpe), drawdowns and volatility across windows,
- trend and momentum signals (SMA 50/200, RSI, MACD).
Base every reason on the numbers provided; do not invent news, fundamentals or price targets.
Percentages in the data are decimals (0.12 = 12%)."""


def _clean(value):
    """Make metrics JSON-friendly and compact for the prompt."""
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, 4)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    return value


def get_verdict(payload: dict, model: str = DEFAULT_MODEL) -> InvestmentVerdict:
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to the .env file.")

    # Bounded retries/timeout: if Gemini is overloaded (503) or out of quota (429)
    # the app reports it within ~1 minute instead of retrying with back-off for many minutes.
    llm = ChatGoogleGenerativeAI(model=model, api_key=api_key, temperature=0.2, max_retries=1, timeout=30)
    structured_llm = llm.with_structured_output(InvestmentVerdict)
    return structured_llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", "Analyse this stock and give your verdict:\n" + json.dumps(_clean(payload), indent=2)),
    ])


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class AnalysisResult:
    ticker: str
    window: str
    user_risk: str
    data: pd.DataFrame                  # selected window, with indicators
    metrics: dict
    risk_level: str
    risk_fit: str
    windows: pd.DataFrame
    signals: dict
    figure: go.Figure
    verdict: InvestmentVerdict | None = None
    verdict_error: str | None = None
    notes: list[str] = field(default_factory=list)


def run_analysis(
    ticker: str,
    window: str = "6 Months",
    user_risk: str = "Medium",
    use_llm: bool = True,
    benchmark: str = DEFAULT_BENCHMARK,
    model: str = DEFAULT_MODEL,
) -> AnalysisResult:
    ticker = ticker.strip().upper()
    if window not in WINDOWS:
        raise ValueError(f"Unknown window '{window}'. Choose one of: {', '.join(WINDOWS)}")
    if user_risk not in RISK_LEVELS:
        raise ValueError(f"Unknown risk level '{user_risk}'. Choose one of: {', '.join(RISK_LEVELS)}")

    days = WINDOWS[window]
    # Always fetch at least 1 year so the multi-window table and 52-week stats are available.
    fetch_days = max(days, TRADING_DAYS_PER_YEAR)
    full = add_indicators(fetch_prices(ticker, fetch_days))
    notes: list[str] = []

    bench_close = None
    if benchmark and benchmark.upper() != ticker:
        try:
            bench_close = fetch_prices(benchmark, fetch_days)["Close"]
        except Exception as exc:  # benchmark is optional; keep going without beta
            notes.append(f"Benchmark {benchmark} unavailable, beta skipped ({exc}).")

    window_df = full.iloc[-(days + 1):]
    if len(window_df) < days + 1:
        notes.append(f"Only {len(window_df)} trading days of history available for this window.")

    window_bench = bench_close.loc[bench_close.index >= window_df.index[0]] if bench_close is not None else None
    metrics = risk_metrics(window_df["Close"], window_bench)
    level = classify_risk(metrics)
    fit = risk_fit(level, user_risk)
    windows = window_table(full["Close"], bench_close, max_days=fetch_days)
    signals = trend_signals(full)
    figure = build_chart(window_df, ticker, window, level)

    result = AnalysisResult(
        ticker=ticker, window=window, user_risk=user_risk, data=window_df,
        metrics=metrics, risk_level=level, risk_fit=fit, windows=windows,
        signals=signals, figure=figure, notes=notes,
    )

    if use_llm:
        try:
            result.verdict = get_verdict(llm_payload(result, benchmark), model=model)
        except Exception as exc:  # keep chart + metrics even if the LLM call fails
            result.verdict_error = f"{type(exc).__name__}: {exc}"

    return result


def llm_payload(result: AnalysisResult, benchmark: str = DEFAULT_BENCHMARK) -> dict:
    """Everything Gemini needs to judge the stock, as plain JSON-able data."""
    return _clean({
        "ticker": result.ticker,
        "benchmark_for_beta": benchmark,
        "investor_profile": {"risk_tolerance": result.user_risk, "investment_window": result.window},
        "selected_window_metrics": result.metrics,
        "computed_risk_level": result.risk_level,
        "risk_fit": result.risk_fit,
        "metrics_by_window": [_clean(row) for row in result.windows.reset_index().to_dict(orient="records")],
        "technical_signals": result.signals,
    })


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.2%}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse a stock's risk and get a Gemini investment verdict.")
    parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL or RELIANCE.NS")
    parser.add_argument("--window", default="6 Months", choices=list(WINDOWS))
    parser.add_argument("--risk", default="Medium", choices=RISK_LEVELS, help="Your risk tolerance")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Benchmark for beta (e.g. SPY, ^NSEI)")
    parser.add_argument("--no-llm", action="store_true", help="Skip the Gemini verdict")
    parser.add_argument("--html", help="Where to save the chart (default: <TICKER>_analysis.html)")
    args = parser.parse_args()

    r = run_analysis(args.ticker, args.window, args.risk, use_llm=not args.no_llm, benchmark=args.benchmark)
    m = r.metrics

    print(f"\n{r.ticker} | window: {r.window} | your risk tolerance: {r.user_risk}")
    print(f"  Price            {m['start_price']:.2f} -> {m['end_price']:.2f}  ({_pct(m['total_return'])})")
    print(f"  Annual volatility {m['annual_volatility']:.2%}")
    print(f"  Max drawdown     {m['max_drawdown']:.2%}")
    print(f"  Sharpe ratio     {m['sharpe_ratio']:.2f}")
    print(f"  1-day VaR (95%)  {m['var_95_daily']:.2%}")
    beta = "n/a" if math.isnan(m["beta"]) else f"{m['beta']:.2f}"
    print(f"  Beta vs {args.benchmark:<8} {beta}")
    print(f"  Risk level       {r.risk_level}  ->  {r.risk_fit}")
    print("\nAcross windows:")
    print(r.windows.to_string(float_format=lambda v: f"{v:.2f}"))
    for note in r.notes:
        print(f"\nNote: {note}")

    if r.verdict:
        v = r.verdict
        print(f"\nGemini verdict: {v.decision}  (confidence: {v.confidence})")
        print(f"  {v.summary}")
        print("  Reasons:")
        for reason in v.reasons:
            print(f"   - {reason}")
        print("  Key risks:")
        for risk in v.key_risks:
            print(f"   - {risk}")
    elif r.verdict_error:
        print(f"\nGemini verdict unavailable: {r.verdict_error}")

    out = args.html or f"{r.ticker.replace('^', '')}_analysis.html"
    r.figure.write_html(out)
    print(f"\nChart saved to {out}")
    print("\nThis is an educational tool, not financial advice.")


if __name__ == "__main__":
    main()
