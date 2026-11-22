# Stock Analysis App (Gemini + LangChain)

A Streamlit app that analyses a stock over the investment window you choose. It measures how risky the stock has been, checks whether that matches your own risk tolerance, draws an interactive Plotly chart, and then asks Google Gemini (through LangChain) for a final **INVEST / DO NOT INVEST** verdict.

> Educational project. Not financial advice.

## How it works

```
 User input (app.py)                 analysis.py
 ───────────────────                 ───────────────────────────────────────────────
 Ticker  ──────────┐
 Window  ──────────┼──► fetch_prices()     Yahoo Finance daily OHLCV (yfinance)
 Risk tolerance ───┘          │
                              ▼
                       add_indicators()    SMA 20/50/200, RSI 14, MACD, Bollinger Bands (ta)
                              │
                              ▼
                       risk_metrics()      return, volatility, max drawdown, Sharpe, VaR, beta vs SPY
                       classify_risk()     Low / Medium / High
                       risk_fit()          compares stock risk with YOUR risk tolerance
                       window_table()      the same metrics over 1M / 3M / 6M / 1Y (/ 5Y)
                       trend_signals()     price vs SMAs, RSI state, MACD momentum, 52-week range
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
         build_chart()               get_verdict()
         Plotly: candles + SMAs,     Gemini via LangChain, structured output:
         volume, RSI, drawdown       decision, confidence, summary, reasons, key risks
```

The numbers are calculated in Python and Gemini only interprets them. The model never has to fetch prices or do arithmetic, so the verdict is grounded in measured data. The prompt also forbids it from inventing news or price targets.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: inputs, metric cards, chart, window table, verdict |
| `analysis.py` | Analysis engine (data, indicators, risk, chart, LLM). Also runs as a CLI |
| `test.py` | Quick check that the Gemini connection works |
| `requirements.txt` | Python dependencies |
| `.env` | Your `GEMINI_API_KEY` (git-ignored, never commit it) |

## Setup (macOS / Linux)

```bash
cd Advanced_Generative_Ai/Stock_anaylize
python3 -m venv .venv
source .venv/bin/activate          # prompt should now show (.venv)
pip install -r requirements.txt
```

On Windows PowerShell, use `.venv\Scripts\Activate.ps1` instead of `source ...`.

Create a `.env` file in this folder:

```
GEMINI_API_KEY=your-key-from-aistudio.google.com
# optional: override the model
# GEMINI_MODEL=gemini-3.8-flash
```

## Run

**Web app**

```bash
streamlit run app.py
```

Enter a ticker, pick the window and your risk tolerance, then click **Analyze Stock**.

**Command line** (handy for testing; saves the chart as HTML)

```bash
python analysis.py AAPL --window "6 Months" --risk Medium
python analysis.py RELIANCE.NS --window "1 Year" --risk Low --benchmark ^NSEI
python analysis.py TSLA --no-llm          # metrics + chart only, no Gemini call
```

## Risk methodology

All metrics use daily adjusted closing prices over the selected window.

| Metric | Meaning |
|---|---|
| Total return | Price change from the first to the last day of the window |
| Annual volatility | Std. dev. of daily returns × √252. How much the price swings |
| Max drawdown | Largest fall from a previous peak inside the window |
| Sharpe ratio | Mean daily return ÷ std. dev. × √252 (risk-free rate taken as 0). Return per unit of risk |
| 1-day VaR (95%) | On 1 day in 20, expect to lose at least this much |
| Beta | Sensitivity to the benchmark (default `SPY`; use `^NSEI` for Indian stocks) |

**Risk level** is the *worse* of these two scores:

| Level | Annual volatility | Max drawdown |
|---|---|---|
| Low | < 20% | better than −15% |
| Medium | 20–35% | −15% to −30% |
| High | > 35% | worse than −30% |

If the stock's risk level is higher than your chosen tolerance, the app flags it as **Above your risk tolerance**, and Gemini is told to weigh that mismatch heavily. The thresholds are constants at the top of `analysis.py`, so you can tune them there.

**Investment windows.** The same metrics are computed for every standard window (1 Month → 1 Year, plus 5 Years when selected). This shows whether a stock that looks calm over 1 month was much riskier over a year, or the other way round.

## The Gemini verdict

`get_verdict()` uses `ChatGoogleGenerativeAI.with_structured_output()` with a Pydantic model, so the reply is always parsed into these fields:

```python
decision:   "INVEST" | "DO NOT INVEST"
confidence: "Low" | "Medium" | "High"
summary:    str
reasons:    list[str]
key_risks:  list[str]
```

If the Gemini call fails (missing key, quota, network, wrong model name), the app still shows the metrics and chart, and displays the error where the verdict would be.

## Limitations

- It uses price history only: no fundamentals, earnings or news. Past volatility doesn't guarantee future behaviour.
- Yahoo Finance data can be delayed or occasionally missing for some tickers.
- The `SMA200` line is only available when there are 200+ trading days before the end of the window. The app downloads extra history for this automatically.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'dotenv'` | You're running a different Python. Activate `.venv`, or in VS Code run **Python: Select Interpreter** → `Stock_anaylize/.venv/bin/python` |
| `No price data found for 'XYZ'` | Check the ticker. Non-US stocks need a suffix (`.NS`, `.L`, `.TO`, …) |
| `Gemini verdict unavailable: ... GEMINI_API_KEY is not set` | Add the key to `.env` in this folder |
| Gemini model not found | Set `GEMINI_MODEL` in `.env` to a model available on your key |
| `503 UNAVAILABLE ... high demand` | Google's servers are overloaded for that model. Wait and retry, or set `GEMINI_MODEL` to another model in `.env`. Metrics and chart still work |
