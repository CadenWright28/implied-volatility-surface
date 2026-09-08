"""Simple implied-volatility surface project.

The script downloads an option chain, solves Black-Scholes implied volatility
from market prices, and plots the result across moneyness and time.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.optimize import brentq
from scipy.stats import norm


FALLBACK_RATE = 0.04
MONEYNESS_PRESETS = {
    "focused": (0.80, 1.40),
    "standard": (0.60, 1.80),
    "wide": (0.40, 2.50),
}


@dataclass
class Inputs:
    ticker: str = "CLSK"
    max_dte: int = 120
    side: str = "calls"
    moneyness: str = "focused"


# ----------------------------- Black-Scholes -----------------------------


def parse_side(value: str) -> str:
    value = value.strip().lower()
    if value in {"call", "calls", "c"}:
        return "calls"
    if value in {"put", "puts", "p"}:
        return "puts"
    raise ValueError("side must be calls or puts")


def bs_price(
    spot: float,
    strike: float,
    time: float,
    rate: float,
    vol: float,
    side: str = "calls",
) -> float:
    side = parse_side(side)
    if time <= 0:
        return max(spot - strike, 0.0) if side == "calls" else max(strike - spot, 0.0)
    if vol <= 0:
        forward_pv = spot - strike * np.exp(-rate * time)
        return max(forward_pv, 0.0) if side == "calls" else max(-forward_pv, 0.0)

    root_t = np.sqrt(time)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * time) / (vol * root_t)
    d2 = d1 - vol * root_t

    if side == "calls":
        return float(spot * norm.cdf(d1) - strike * np.exp(-rate * time) * norm.cdf(d2))
    return float(strike * np.exp(-rate * time) * norm.cdf(-d2) - spot * norm.cdf(-d1))


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    time: float,
    rate: float,
    side: str,
) -> float:
    side = parse_side(side)
    if price <= 0 or spot <= 0 or strike <= 0 or time <= 0:
        return np.nan

    intrinsic = max(spot - strike * np.exp(-rate * time), 0.0)
    if side == "puts":
        intrinsic = max(strike * np.exp(-rate * time) - spot, 0.0)
    if price < intrinsic - 1e-8:
        return np.nan

    def error(vol: float) -> float:
        return bs_price(spot, strike, time, rate, vol, side) - price

    try:
        return float(brentq(error, 1e-6, 8.0, maxiter=150))
    except ValueError:
        return np.nan


def option_greeks(
    spot: float,
    strike: float,
    time: float,
    rate: float,
    vol: float,
    side: str,
) -> dict[str, float]:
    side = parse_side(side)
    if time <= 0 or vol <= 0:
        return {"delta": np.nan, "gamma": np.nan, "vega": np.nan, "theta_day": np.nan}

    root_t = np.sqrt(time)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * time) / (vol * root_t)
    d2 = d1 - vol * root_t
    pdf = norm.pdf(d1)

    if side == "calls":
        delta = norm.cdf(d1)
        theta = -(spot * pdf * vol) / (2 * root_t) - rate * strike * np.exp(-rate * time) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = -(spot * pdf * vol) / (2 * root_t) + rate * strike * np.exp(-rate * time) * norm.cdf(-d2)

    gamma = pdf / (spot * vol * root_t)
    vega = spot * pdf * root_t / 100.0
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta_day": float(theta / 365.0),
    }


# ----------------------------- market data -----------------------------


def _load_yfinance():
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance with: pip install yfinance") from exc
    return yf


def load_short_rate() -> tuple[float, str]:
    """Use the 13-week Treasury yield when available, otherwise 4%."""
    yf = _load_yfinance()
    try:
        history = yf.Ticker("^IRX").history(period="5d", auto_adjust=False)
        value = float(history["Close"].dropna().iloc[-1]) / 100.0
        return value, "^IRX"
    except Exception:
        return FALLBACK_RATE, "4% fallback"


def load_option_chain(inputs: Inputs) -> tuple[pd.DataFrame, float, float, str]:
    yf = _load_yfinance()
    ticker = yf.Ticker(inputs.ticker.upper())

    history = ticker.history(period="5d", auto_adjust=False)
    if history.empty:
        raise RuntimeError(f"No price history returned for {inputs.ticker}.")
    spot = float(history["Close"].dropna().iloc[-1])

    rate, rate_source = load_short_rate()
    today = pd.Timestamp.now().normalize()
    rows = []

    for expiration_text in ticker.options:
        expiration = pd.Timestamp(expiration_text)
        dte = int((expiration.normalize() - today).days)
        if dte < 2 or dte > inputs.max_dte:
            continue

        chain = ticker.option_chain(expiration_text)
        frame = chain.calls if inputs.side == "calls" else chain.puts
        if frame.empty:
            continue

        frame = frame.copy()
        frame["expiration"] = expiration
        frame["dte"] = dte
        rows.append(frame)

    if not rows:
        raise RuntimeError("No option contracts matched the requested date range.")

    data = pd.concat(rows, ignore_index=True)
    data["mid"] = np.where(
        (data["bid"] > 0) & (data["ask"] >= data["bid"]),
        (data["bid"] + data["ask"]) / 2,
        data["lastPrice"],
    )
    data["moneyness"] = data["strike"] / spot
    data["time"] = data["dte"] / 365.0

    low, high = MONEYNESS_PRESETS[inputs.moneyness]
    data = data[
        (data["moneyness"] >= low)
        & (data["moneyness"] <= high)
        & (data["mid"] > 0)
        & (data["strike"] > 0)
    ].copy()

    data["iv"] = [
        implied_volatility(price, spot, strike, time, rate, inputs.side)
        for price, strike, time in zip(data["mid"], data["strike"], data["time"])
    ]
    data = data[np.isfinite(data["iv"]) & (data["iv"] > 0) & (data["iv"] < 8)].copy()

    if len(data) < 8:
        raise RuntimeError("Not enough clean contracts to build a surface.")

    return data, spot, rate, rate_source


# ----------------------------- charts and table -----------------------------


def build_surface(data: pd.DataFrame, ticker: str) -> go.Figure:
    x = data["moneyness"].to_numpy(float)
    y = data["dte"].to_numpy(float)
    z = 100 * data["iv"].to_numpy(float)

    fig = go.Figure()

    # If there are enough strikes and expirations, interpolate between the
    # observed contracts to make the plane easier to read.
    if len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
        x_grid = np.linspace(x.min(), x.max(), 70)
        y_grid = np.linspace(y.min(), y.max(), 55)
        X, Y = np.meshgrid(x_grid, y_grid)
        try:
            Z = griddata((x, y), z, (X, Y), method="linear")
            if np.isnan(Z).any():
                nearest = griddata((x, y), z, (X, Y), method="nearest")
                Z = np.where(np.isnan(Z), nearest, Z)
            fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.9, showscale=True))
        except Exception:
            pass

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker={"size": 2},
            name="contracts",
        )
    )
    fig.update_layout(
        title=f"{ticker.upper()} implied volatility plane",
        scene={
            "xaxis_title": "Strike / Spot",
            "yaxis_title": "Days to expiration",
            "zaxis_title": "IV (%)",
        },
    )
    return fig


def build_expiration_slice(data: pd.DataFrame, ticker: str) -> go.Figure:
    counts = data.groupby("expiration").size().sort_values(ascending=False)
    expiration = counts.index[0]
    slice_data = data[data["expiration"] == expiration].sort_values("moneyness")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=slice_data["moneyness"],
            y=100 * slice_data["iv"],
            mode="lines+markers",
            name=str(pd.Timestamp(expiration).date()),
        )
    )
    fig.add_vline(x=1.0, line_dash="dash")
    fig.update_layout(
        title=f"{ticker.upper()} IV slice - {pd.Timestamp(expiration).date()}",
        xaxis_title="Strike / Spot",
        yaxis_title="IV (%)",
    )
    return fig


def comparison_table(
    data: pd.DataFrame,
    spot: float,
    rate: float,
    side: str,
    n: int = 8,
) -> pd.DataFrame:
    work = data.copy()
    work["distance"] = (work["moneyness"] - 1).abs()
    work = work.sort_values(["dte", "distance"]).head(n)

    greek_rows = [
        option_greeks(spot, row.strike, row.time, rate, row.iv, side)
        for row in work.itertuples()
    ]
    greeks = pd.DataFrame(greek_rows, index=work.index)
    work = pd.concat([work, greeks], axis=1)

    columns = [
        "expiration",
        "strike",
        "mid",
        "iv",
        "volume",
        "openInterest",
        "delta",
        "gamma",
        "vega",
        "theta_day",
    ]
    return work[columns].reset_index(drop=True)


# ----------------------------- run -----------------------------


def read_inputs() -> Inputs:
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
        max_dte = int(sys.argv[2]) if len(sys.argv) > 2 else 120
        side = parse_side(sys.argv[3]) if len(sys.argv) > 3 else "calls"
        preset = sys.argv[4].lower() if len(sys.argv) > 4 else "focused"
    else:
        ticker = input("Ticker [CLSK]: ").strip() or "CLSK"
        max_dte = int(input("Max days to expiration [120]: ").strip() or "120")
        side = parse_side(input("Calls or puts [calls]: ").strip() or "calls")
        preset = input("Moneyness view (focused/standard/wide) [focused]: ").strip().lower() or "focused"

    if preset not in MONEYNESS_PRESETS:
        raise ValueError(f"Unknown moneyness preset: {preset}")
    return Inputs(ticker.upper(), max_dte, side, preset)


def main() -> None:
    inputs = read_inputs()
    data, spot, rate, rate_source = load_option_chain(inputs)

    print(f"\n{inputs.ticker} spot: ${spot:.2f}")
    print(f"Risk-free rate: {rate:.2%} ({rate_source})")
    print(f"Contracts used: {len(data)}")
    print("\nNear-ATM contracts:")
    print(comparison_table(data, spot, rate, inputs.side).to_string(index=False))

    surface = build_surface(data, inputs.ticker)
    iv_slice = build_expiration_slice(data, inputs.ticker)

    surface_file = f"{inputs.ticker.lower()}_iv_plane.html"
    slice_file = f"{inputs.ticker.lower()}_iv_slice.html"
    surface.write_html(surface_file)
    iv_slice.write_html(slice_file)

    print(f"\nSaved {surface_file}")
    print(f"Saved {slice_file}")

    surface.show()
    iv_slice.show()


if __name__ == "__main__":
    main()
