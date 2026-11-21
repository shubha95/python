import json
import math

import streamlit

from analysis import get_verdict, llm_payload, run_analysis


@streamlit.cache_data(ttl=900, show_spinner=False)
def cached_analysis(ticker: str, window: str, risk: str):
    """Data + metrics + chart only (fast). Cached 15 min so re-clicking doesn't re-download."""
    return run_analysis(ticker, window, risk, use_llm=False)


@streamlit.cache_data(ttl=900, show_spinner=False)
def cached_verdict(payload_json: str):
    """Gemini call. Only successful answers are cached; errors are retried on the next click."""
    return get_verdict(json.loads(payload_json))


def fmt_pct(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.2%}"


streamlit.set_page_config(
    page_title="Stock Analysis App",
    page_icon="📈",
    layout="wide",
    )
streamlit.title("Stock Analysis App")
streamlit.write("This app allows you to analyze stock data and visualize trends over time. You can input a stock ticker symbol, select a date range, and view various charts and metrics related to the stock's performance.")

#user input
stock = streamlit.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT, GOOGL):", "AAPL")

# selectbox for time duration
time_duration = streamlit.selectbox(
    "Select Time Duration for Analysis:",
    ("1 Month", "3 Months", "6 Months", "1 Year", "5 Years"),
    index=0,
)

#risk selection
risk_level = streamlit.selectbox(
    "Select Risk Level for Analysis:",
    ("Low", "Medium", "High"),
    index=1,
)
#button to trigger analysis
if streamlit.button("Analyze Stock"):
    with streamlit.spinner(f"Fetching {stock.upper()} data and computing risk for {time_duration}..."):
        try:
            result = cached_analysis(stock, time_duration, risk_level)
        except Exception as exc:
            streamlit.error(f"Analysis failed: {exc}")
            streamlit.stop()

    m = result.metrics

    # --- Key metrics ---
    streamlit.subheader(f"{result.ticker} — {result.window}")
    c1, c2, c3, c4, c5, c6 = streamlit.columns(6)
    c1.metric("Return", fmt_pct(m["total_return"]))
    c2.metric("Annual volatility", fmt_pct(m["annual_volatility"]))
    c3.metric("Max drawdown", fmt_pct(m["max_drawdown"]))
    c4.metric("Sharpe ratio", "n/a" if math.isnan(m["sharpe_ratio"]) else f"{m['sharpe_ratio']:.2f}")
    c5.metric("Beta vs SPY", "n/a" if math.isnan(m["beta"]) else f"{m['beta']:.2f}")
    c6.metric("Stock risk level", result.risk_level)

    if result.risk_fit.startswith("Within"):
        streamlit.success(f"Risk level **{result.risk_level}** — {result.risk_fit} (**{result.user_risk}**).")
    else:
        streamlit.warning(f"Risk level **{result.risk_level}** — {result.risk_fit} (**{result.user_risk}**).")

    # --- Chart ---
    streamlit.plotly_chart(result.figure, use_container_width=True)

    # --- Investment windows ---
    streamlit.subheader("Risk across investment windows")
    streamlit.dataframe(
        result.windows.style.format({
            "Return": "{:+.2%}", "Annual volatility": "{:.2%}",
            "Max drawdown": "{:.2%}", "Sharpe": "{:.2f}",
        }),
        use_container_width=True,
    )

    # --- Gemini verdict ---
    # Runs after everything above is already on screen, so a slow or failing
    # Gemini call never hides the chart and metrics.
    streamlit.subheader("Gemini verdict")
    try:
        with streamlit.spinner("Asking Gemini for a verdict (up to ~1 minute)..."):
            v = cached_verdict(json.dumps(llm_payload(result)))
        box = streamlit.success if v.decision == "INVEST" else streamlit.error
        box(f"**{v.decision}** (confidence: {v.confidence})\n\n{v.summary}")
        left, right = streamlit.columns(2)
        left.markdown("**Why**\n" + "\n".join(f"- {r}" for r in v.reasons))
        right.markdown("**Key risks**\n" + "\n".join(f"- {r}" for r in v.key_risks))
    except Exception as exc:
        message = str(exc)
        if "429" in message or "RESOURCE_EXHAUSTED" in message:
            hint = "Your Gemini quota is used up. Wait a minute (or until the daily reset) and click Analyze again."
        elif "503" in message or "UNAVAILABLE" in message:
            hint = "Gemini is overloaded right now. Try again shortly, or set GEMINI_MODEL in .env to another model."
        else:
            hint = "See the error below."
        streamlit.warning(f"Gemini verdict unavailable. {hint}")
        with streamlit.expander("Error details"):
            streamlit.code(f"{type(exc).__name__}: {message}")

    for note in result.notes:
        streamlit.info(note)
    streamlit.caption("Educational tool only — not financial advice.")
    