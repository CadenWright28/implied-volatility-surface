from __future__ import annotations

import sys
import math
from io import StringIO
from pathlib import Path

import requests

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

try:
    from scipy.interpolate import Rbf, PchipInterpolator
    from scipy.optimize import minimize
except Exception as exc:
    raise ImportError(
        "This script needs scipy for the smoothed IV surface, slice curve, and optimizer.\n"
        "Install with: pip install scipy"
    ) from exc


# ============================================================
# DEFAULTS / FRONT-AND-CENTER CONTROLS
# ============================================================
APP_NAME = "IV3D"
APP_VERSION = "1.0"
APP_TAG = "iv3d_v1_0"
DEFAULT_TICKER = "CLSK"
DEFAULT_MAX_DTE = 120
DEFAULT_SLICE_TARGET = "auto"   # "auto", YYYY-MM-DD, or numeric DTE like 7 / 14 / 30
MIN_DTE = 4
OPTION_SIDE = "calls"
MAX_REASONABLE_IV = 8.0                # hard cap: 800%
GRID_SIZE_X = 75
GRID_SIZE_DTE = 60
SMOOTHING = 0.65                       # stronger smoothing than v6
RBF_FUNCTION = "multiquadric"
SAVE_HTML = True
OPEN_BROWSER = True
FORCE_PAUSE_ON_EXIT = True
EXPORT_DIAGNOSTIC_CSVS = False
PLOT_THEME = "plotly_dark"
PLOT_WIDTH = None
PLOT_HEIGHT = None
SLICE_WIDTH = None
SLICE_HEIGHT = None
COLOR_SCALE = "Viridis"
SURFACE_OPACITY = 1.0
OUTPUT_DIR = Path.home() / "AppData" / "Local" / "IVPlaneOutputs"
AUTO_OPEN_PLANE = True
AUTO_OPEN_SLICE = True

# ============================================================
# VIEW CONTROLS
# ============================================================
X_AXIS_MODE = "moneyness"            # "strike" or "moneyness"
SURFACE_USE_MONEYNESS_DISPLAY_BAND = True
SURFACE_MIN_MONEYNESS_DISPLAY = 0.80
SURFACE_MAX_MONEYNESS_DISPLAY = 1.60
SURFACE_MONEYNESS_PRESET = "focused"
SURFACE_MONEYNESS_PRESETS = {
    "focused": (0.80, 1.60),
    "conservative": (0.80, 1.20),
    "standard": (0.50, 2.00),
    "aggressive": (0.30, 3.00),
}
FOCUS_ON_TRUSTED_VIEW = True
SHOW_EXCLUDED_POINTS = True
SHOW_EXCLUDED_POINTS_IN_FOCUS = False
ZOOM_X_PAD_PCT = 0.06
ZOOM_Y_PAD_PCT = 0.08
ZOOM_Z_PAD_PCT = 0.14
Z_UPPER_PERCENTILE = 0.985
CAMERA_EYE = dict(x=1.55, y=1.32, z=0.92)

# 3D plane display controls
PLANE_SQUARE_GRIDS = True
PLANE_GRID_COLOR = "rgba(150, 180, 230, 0.38)"
PLANE_BACKGROUND_COLOR = "rgba(18, 22, 32, 1.0)"

# 2D slice display controls
SLICE_Y_USE_TRUSTED_RANGE = True
SLICE_Y_UPPER_PERCENTILE = 0.985
SLICE_Y_LOWER_PAD_PCT = 0.06
SLICE_Y_UPPER_PAD_PCT = 0.10
SLICE_SHOW_GRID = True
SLICE_GRID_COLOR = "rgba(90, 120, 170, 0.34)"
SLICE_ZERO_LINE_COLOR = "rgba(180, 180, 180, 0.40)"

# ============================================================
# MINIMAL SANITY CLEANUP + MIXTURE-OF-BLACK-SCHOLES CONTROLS
# ============================================================
MIN_PRICE_EPS = 1e-8
MAX_ABS_SPREAD_TO_PRICE_RATIO = 4.0
MIN_STRIKE_MULTIPLE = 0.20
MAX_STRIKE_MULTIPLE = 4.00

# ============================================================
# MIXTURE-OF-BLACK-SCHOLES ("GMM-STYLE") FIT CONTROLS
# ============================================================
MIXTURE_COMPONENTS = 3
FIT_PRICE_SOURCE = "auto"                 # auto / mid / last
FIT_USE_BID_ASK_WEIGHTS = True
FIT_MULTI_STARTS = 12
FIT_OPT_METHODS = ("L-BFGS-B", "Powell")
FIT_MAX_ITER = 1200
FIT_MIN_POINTS_PER_EXPIRY = 6
FIT_COMPONENT_FALLBACKS = (3, 2, 1)
FIT_OBJECTIVE_FAIL_PENALTY = 1e12
FIT_OBJECTIVE_REJECT_RATIO = 0.999999
FIT_COMPONENT_FORWARD_MULT_MIN = 0.50
FIT_COMPONENT_FORWARD_MULT_MAX = 1.75
FIT_COMPONENT_VOL_MIN = 0.05
FIT_COMPONENT_VOL_MAX = 4.00
FIT_PSEUDO_HUBER_DELTA = 1.10
FIT_L2_REG = 2e-4
FIT_PRICE_SCALE_FLOOR = 0.03
FIT_PRICE_SCALE_PCT = 0.05
FIT_SHORT_DTE_THRESHOLD = 25.0
FIT_MIN_EXTRINSIC_FOR_INFORMATIVE = 0.03
FIT_MIN_EXTRINSIC_PCT = 0.03
FIT_MAX_SPREAD_PCT_FOR_INFORMATIVE = 1.25
FIT_MIN_INFORMATIVE_WEIGHT = 0.15
FIT_INCLUDE_OUTLIERS_IN_SURFACE = False
FIT_OUTLIER_ABS_THRESHOLD = 0.15
FIT_OUTLIER_REL_THRESHOLD = 0.35
FIT_SURFACE_MIN_POINTS = 6
FIT_DENSE_SLICE_POINTS = 220

FITTED_MODELS: dict[str, dict[str, object]] = {}

# ============================================================
# 2D EXPIRATION SLICE TOOL
# ============================================================
MAKE_SLICE_FIGURE = True
SLICE_TARGET = DEFAULT_SLICE_TARGET
SLICE_SHOW_EXCLUDED_POINTS = True
SLICE_USE_TRUSTED_POINTS_ONLY_FOR_CURVE = True
SLICE_MIN_TRUSTED_POINTS_FOR_CURVE = 4
SLICE_MARK_SPOT_LINE = True




# ============================================================
# INTERACTIVE STRIKE COMPARISON (AFTER SLICE IS BUILT)
# ============================================================
COMPARE_AFTER_SLICE = True
COMPARE_SAVE_CSV = False
COMPARE_NEAREST_TOLERANCE = 0.26   # max absolute strike difference when auto-matching nearest

# ============================================================
# REFERENCE PRICING / NO-MOVE DECAY ASSUMPTIONS FOR COMPARISON TABLE
# ============================================================
REFERENCE_RISK_FREE_RATE = 0.04    # fallback flat rate if live Treasury fetch fails
USE_LIVE_TREASURY_RATES = True
TREASURY_REQUEST_TIMEOUT_SEC = 12
TREASURY_BILL_URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_bill_rates"
TREASURY_CURVE_URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve"
TREASURY_YFINANCE_FALLBACK_SYMBOL = "^IRX"
PROJECT_NO_MOVE_DAYS = (1, 2, 3)   # no-move / no-IV-change scenario
COMPARISON_PRICE_SOURCE = "mid"    # "mid", "last", or "vendor_iv"
PLANE_IV_SOURCE = "mid"            # "vendor", "mid", "bid", or "ask"
SLICE_DISPLAY_IV_SOURCE = "plane"  # "plane", "vendor", "mid", "bid", or "ask"
SLICE_CURVE_IV_SOURCE = "display"    # "display", "plane", "vendor", "mid", "bid", or "ask"
IV_CONFIDENCE_TIGHT_PCT = 0.05
IV_CONFIDENCE_MODERATE_PCT = 0.12
IV_SOLVER_LOW = 1e-6
IV_SOLVER_HIGH = 10.0
IV_SOLVER_TOL = 1e-8
IV_SOLVER_MAX_ITER = 200
LIVE_TREASURY_INFO: dict[str, object] = {
    "enabled": USE_LIVE_TREASURY_RATES,
    "used_live": False,
    "source": "fallback_flat",
    "as_of": "N/A",
    "curve_points": {},
    "message": "Using fallback flat rate.",
}



def _normalize_treasury_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).replace(" ", " ").strip() for col in out.columns]
    return out


def _find_date_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if "date" in str(col).strip().lower():
            return col
    return None


def _coerce_latest_row(df: pd.DataFrame) -> tuple[pd.Series, str]:
    df = _normalize_treasury_columns(df)
    date_col = _find_date_column(df)
    if date_col is None:
        raise ValueError("Treasury table did not contain a date column.")

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col])
    if work.empty:
        raise ValueError("Treasury table did not contain any usable dated rows.")

    latest_idx = work[date_col].idxmax()
    latest_row = work.loc[latest_idx]
    as_of = pd.Timestamp(latest_row[date_col]).strftime("%Y-%m-%d")
    return latest_row, as_of


def _extract_numeric_percent_from_row(row: pd.Series, candidate_names: list[str]) -> float | None:
    lowered = {str(k).strip().lower(): k for k in row.index}
    for candidate in candidate_names:
        for key_lower, orig_key in lowered.items():
            if candidate in key_lower:
                val = row[orig_key]
                try:
                    num = float(str(val).replace('%', '').strip())
                except Exception:
                    continue
                if np.isfinite(num):
                    return num / 100.0
    return None


def _extract_latest_matching_treasury_row(tables: list[pd.DataFrame], label_map: dict[float, list[str]]) -> tuple[pd.Series, str]:
    candidates: list[tuple[pd.Timestamp, int, pd.Series]] = []

    for table in tables:
        try:
            work = _normalize_treasury_columns(table)
            date_col = _find_date_column(work)
            if date_col is None:
                continue
            work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
            work = work.dropna(subset=[date_col])
            if work.empty:
                continue

            for _, row in work.iterrows():
                match_count = 0
                for candidate_names in label_map.values():
                    if _extract_numeric_percent_from_row(row, candidate_names) is not None:
                        match_count += 1
                if match_count > 0:
                    candidates.append((pd.Timestamp(row[date_col]), match_count, row))
        except Exception:
            continue

    if not candidates:
        raise ValueError("No usable Treasury table rows with matching maturities were found.")

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_ts, _, best_row = candidates[0]
    return best_row, best_ts.strftime("%Y-%m-%d")


def _fetch_treasury_curve_points_official() -> tuple[dict[float, float], str, str]:
    curve_points: dict[float, float] = {}
    messages: list[str] = []

    for url, labels in [
        (TREASURY_BILL_URL, {28/365.0: ["4 weeks", "4-week", "4 wk"], 13/52.0: ["13 weeks", "13-week", "13 wk"], 26/52.0: ["26 weeks", "26-week", "26 wk"], 52/52.0: ["52 weeks", "52-week", "52 wk"]}),
        (TREASURY_CURVE_URL, {1/12.0: ["1 mo"], 2/12.0: ["2 mo"], 3/12.0: ["3 mo"], 4/12.0: ["4 mo"], 6/12.0: ["6 mo"], 1.0: ["1 yr", "1 year"]}),
    ]:
        response = requests.get(url, timeout=TREASURY_REQUEST_TIMEOUT_SEC, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        if not tables:
            raise ValueError(f"No Treasury tables found at {url}")
        latest_row, as_of = _extract_latest_matching_treasury_row(tables, labels)
        matched_any = False
        for maturity_years, candidates in labels.items():
            val = _extract_numeric_percent_from_row(latest_row, candidates)
            if val is not None:
                curve_points[maturity_years] = val
                matched_any = True
        if matched_any:
            messages.append(as_of)

    if not curve_points:
        raise ValueError("Official Treasury pages were reachable, but no usable front-end rates were parsed.")

    as_of_final = max(messages) if messages else "N/A"
    return dict(sorted(curve_points.items())), as_of_final, "official_treasury_html"


def _fetch_treasury_curve_points_yfinance() -> tuple[dict[float, float], str, str]:
    ticker = yf.Ticker(TREASURY_YFINANCE_FALLBACK_SYMBOL)
    hist = ticker.history(period="5d", auto_adjust=False)
    if hist.empty or hist["Close"].dropna().empty:
        raise ValueError("No fallback Treasury market history returned.")
    latest_close = float(hist["Close"].dropna().iloc[-1]) / 100.0
    as_of = pd.Timestamp(hist.index[-1]).strftime("%Y-%m-%d")
    return {1/12.0: latest_close, 0.25: latest_close, 0.5: latest_close, 1.0: latest_close}, as_of, "yfinance_^IRX_fallback"


def load_live_treasury_curve() -> dict[str, object]:
    info = {
        "enabled": USE_LIVE_TREASURY_RATES,
        "used_live": False,
        "source": "fallback_flat",
        "as_of": "N/A",
        "curve_points": {},
        "message": f"Using fallback flat rate {REFERENCE_RISK_FREE_RATE:.2%}.",
    }

    if not USE_LIVE_TREASURY_RATES:
        return info

    for loader in (_fetch_treasury_curve_points_official, _fetch_treasury_curve_points_yfinance):
        try:
            curve_points, as_of, source = loader()
            info.update({
                "used_live": True,
                "source": source,
                "as_of": as_of,
                "curve_points": curve_points,
                "message": f"Loaded live Treasury data from {source} as of {as_of}.",
            })
            return info
        except Exception as exc:
            info["message"] = f"Live Treasury fetch failed ({exc}). Using fallback flat rate {REFERENCE_RISK_FREE_RATE:.2%}."

    return info


def _interpolate_rate_from_curve(dte_days: float, curve_points: dict[float, float]) -> float:
    if not curve_points:
        return REFERENCE_RISK_FREE_RATE

    t = max(float(dte_days) / 365.0, 1e-8)
    xs = np.array(sorted(curve_points.keys()), dtype=float)
    ys = np.array([curve_points[x] for x in xs], dtype=float)

    if t <= xs[0]:
        return float(ys[0])
    if t >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(t, xs, ys))


def assign_risk_free_rates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    curve_points = LIVE_TREASURY_INFO.get("curve_points", {}) if isinstance(LIVE_TREASURY_INFO, dict) else {}
    out["risk_free_rate"] = out["dte"].map(lambda d: _interpolate_rate_from_curve(d, curve_points))
    out["risk_free_source"] = str(LIVE_TREASURY_INFO.get("source", "fallback_flat"))
    out["risk_free_as_of"] = str(LIVE_TREASURY_INFO.get("as_of", "N/A"))
    return out


def _safe_int_str(val: float | int | None) -> str:
    if val is None or pd.isna(val):
        return "N/A"
    try:
        return str(int(val))
    except Exception:
        return "N/A"



def _safe_price_str(val: float | int | None, decimals: int = 2) -> str:
    if val is None or pd.isna(val):
        return "N/A"
    return f"{float(val):,.{decimals}f}"



def _safe_pct_str(val: float | int | None, decimals: int = 1) -> str:
    if val is None or pd.isna(val):
        return "N/A"
    return f"{float(val):.{decimals}%}"


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)



def _bs_option_snapshot(
    spot: float,
    strike: float,
    t_years: float,
    sigma: float,
    option_side: str,
    rate: float = REFERENCE_RISK_FREE_RATE,
) -> dict[str, float]:
    t_years = max(float(t_years), 1e-8)
    sigma = max(float(sigma), 1e-8)

    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * t_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    nd1 = _norm_cdf(d1)
    nd2 = _norm_cdf(d2)
    pdf_d1 = _norm_pdf(d1)
    discount = math.exp(-rate * t_years)

    if option_side == "puts":
        price = strike * discount * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
        delta = nd1 - 1.0
        theta_annual = -(spot * pdf_d1 * sigma) / (2.0 * sqrt_t) + rate * strike * discount * _norm_cdf(-d2)
        intrinsic = max(strike - spot, 0.0)
        lower_bound = max(strike * discount - spot, 0.0)
        upper_bound = strike * discount
    else:
        price = spot * nd1 - strike * discount * nd2
        delta = nd1
        theta_annual = -(spot * pdf_d1 * sigma) / (2.0 * sqrt_t) - rate * strike * discount * nd2
        intrinsic = max(spot - strike, 0.0)
        lower_bound = max(spot - strike * discount, 0.0)
        upper_bound = spot

    theta_day = theta_annual / 365.0
    extrinsic = max(price - intrinsic, 0.0)

    return {
        "price": price,
        "delta": delta,
        "theta_day": theta_day,
        "intrinsic": intrinsic,
        "extrinsic": extrinsic,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }


def _solve_implied_vol_from_price(
    target_price: float,
    spot: float,
    strike: float,
    t_years: float,
    option_side: str,
    rate: float = REFERENCE_RISK_FREE_RATE,
    sigma_low: float = IV_SOLVER_LOW,
    sigma_high: float = IV_SOLVER_HIGH,
    tol: float = IV_SOLVER_TOL,
    max_iter: int = IV_SOLVER_MAX_ITER,
) -> float:
    target_price = float(target_price)
    t_years = max(float(t_years), 1e-8)

    low_snap = _bs_option_snapshot(spot, strike, t_years, sigma_low, option_side, rate)
    high_snap = _bs_option_snapshot(spot, strike, t_years, sigma_high, option_side, rate)

    lower_bound = low_snap["lower_bound"]
    upper_bound = high_snap["upper_bound"]

    if not np.isfinite(target_price):
        raise ValueError("Target option price is not finite.")
    if target_price < lower_bound - 1e-10:
        raise ValueError("Target price is below the no-arbitrage lower bound.")
    if target_price > upper_bound + 1e-10:
        raise ValueError("Target price is above the no-arbitrage upper bound.")

    low = float(sigma_low)
    high = float(sigma_high)
    low_price = float(low_snap["price"])
    high_price = float(high_snap["price"])

    # Expand the upper vol cap if needed.
    expand_count = 0
    while high_price < target_price and expand_count < 8:
        high *= 2.0
        high_snap = _bs_option_snapshot(spot, strike, t_years, high, option_side, rate)
        high_price = float(high_snap["price"])
        upper_bound = high_snap["upper_bound"]
        if target_price > upper_bound + 1e-10:
            raise ValueError("Target price is above the expanded no-arbitrage upper bound.")
        expand_count += 1

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        mid_price = float(_bs_option_snapshot(spot, strike, t_years, mid, option_side, rate)["price"])
        if abs(mid_price - target_price) <= tol:
            return mid
        if mid_price < target_price:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high)


def _resolve_comparison_price(row: pd.Series) -> tuple[float, str]:
    mid_val = row.get("mid", np.nan)
    last_val = row.get("lastPrice", np.nan)

    if COMPARISON_PRICE_SOURCE == "last" and pd.notna(last_val) and float(last_val) > 0:
        return float(last_val), "last"
    if COMPARISON_PRICE_SOURCE == "mid" and pd.notna(mid_val) and float(mid_val) > 0:
        return float(mid_val), "mid"
    if pd.notna(mid_val) and float(mid_val) > 0:
        return float(mid_val), "mid_fallback"
    if pd.notna(last_val) and float(last_val) > 0:
        return float(last_val), "last_fallback"
    raise ValueError("No valid market price available for comparison IV inversion.")


def _solve_quote_iv(
    row: pd.Series,
    quote_field: str,
    option_side: str,
    rate: float | None = None,
) -> float:
    quote_val = row.get(quote_field, np.nan)
    if pd.isna(quote_val) or float(quote_val) <= 0:
        raise ValueError(f"No valid {quote_field} price available for IV solve.")

    spot = float(row["spot"])
    strike = float(row["strike"])
    t_years = max(float(row["dte"]) / 365.0, 1e-8)

    effective_rate = float(row.get("risk_free_rate", REFERENCE_RISK_FREE_RATE)) if rate is None else float(rate)

    return _solve_implied_vol_from_price(
        float(quote_val),
        spot,
        strike,
        t_years,
        option_side,
        effective_rate,
    )


def _iv_confidence_label(spread_pct_mid: float | int | None) -> str:
    if spread_pct_mid is None or pd.isna(spread_pct_mid):
        return "Unknown"
    spread_pct_mid = float(spread_pct_mid)
    if spread_pct_mid <= IV_CONFIDENCE_TIGHT_PCT:
        return "Tight"
    if spread_pct_mid <= IV_CONFIDENCE_MODERATE_PCT:
        return "Moderate"
    return "Wide"


def _safe_iv_width(iv_low: float | int | None, iv_high: float | int | None) -> float:
    if iv_low is None or iv_high is None or pd.isna(iv_low) or pd.isna(iv_high):
        return np.nan
    return float(iv_high) - float(iv_low)


def enrich_with_solved_iv_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    bid_ivs: list[float] = []
    mid_ivs: list[float] = []
    ask_ivs: list[float] = []
    plane_ivs: list[float] = []
    plane_src: list[str] = []
    iv_widths: list[float] = []
    iv_conf: list[str] = []

    for _, row in out.iterrows():
        vendor_iv = row.get("impliedVolatility", np.nan)

        try:
            bid_iv = _solve_quote_iv(row, "bid", OPTION_SIDE)
        except Exception:
            bid_iv = np.nan

        try:
            mid_iv = _solve_quote_iv(row, "mid", OPTION_SIDE)
        except Exception:
            mid_iv = np.nan

        try:
            ask_iv = _solve_quote_iv(row, "ask", OPTION_SIDE)
        except Exception:
            ask_iv = np.nan

        source_map = {
            "vendor": vendor_iv,
            "bid": bid_iv,
            "mid": mid_iv,
            "ask": ask_iv,
        }
        selected_iv = source_map.get(PLANE_IV_SOURCE, np.nan)
        selected_source = PLANE_IV_SOURCE

        if pd.isna(selected_iv):
            for fallback_source in ("mid", "vendor", "ask", "bid"):
                fallback_iv = source_map.get(fallback_source, np.nan)
                if pd.notna(fallback_iv):
                    selected_iv = fallback_iv
                    selected_source = f"{fallback_source}_fallback"
                    break

        bid_ivs.append(bid_iv)
        mid_ivs.append(mid_iv)
        ask_ivs.append(ask_iv)
        plane_ivs.append(selected_iv)
        plane_src.append(selected_source)
        iv_widths.append(_safe_iv_width(bid_iv, ask_iv))
        iv_conf.append(_iv_confidence_label(row.get("spread_pct_mid", np.nan)))

    out["bid_solved_iv"] = bid_ivs
    out["mid_solved_iv"] = mid_ivs
    out["ask_solved_iv"] = ask_ivs
    out["plane_iv"] = plane_ivs
    out["plane_iv_source"] = plane_src
    out["solved_iv_width"] = iv_widths
    out["iv_confidence"] = iv_conf
    return out


def _compute_compare_metrics(row: pd.Series) -> dict[str, float]:
    base_nan_metrics = {
        "delta": np.nan,
        "theta_day": np.nan,
        "model_price": np.nan,
        "intrinsic": np.nan,
        "extrinsic": np.nan,
        "market_minus_model": np.nan,
        "pct_vs_model": np.nan,
        "vendor_iv": np.nan,
        "compare_iv": np.nan,
        "bid_iv": np.nan,
        "mid_iv": np.nan,
        "ask_iv": np.nan,
        "iv_width": np.nan,
        "vendor_model_price": np.nan,
        "compare_minus_vendor_model": np.nan,
        "price_source_used": "N/A",
        "market_price_used": np.nan,
        "iv_confidence": "Unknown",
    }
    for d in PROJECT_NO_MOVE_DAYS:
        base_nan_metrics[f"proj_{d}d_price"] = np.nan
        base_nan_metrics[f"proj_{d}d_decay"] = np.nan

    try:
        spot = float(row["spot"])
        strike = float(row["strike"])
        vendor_sigma = float(row["impliedVolatility"])
        bid_sigma = row.get("bid_solved_iv", np.nan)
        mid_sigma = row.get("mid_solved_iv", np.nan)
        ask_sigma = row.get("ask_solved_iv", np.nan)
        t_years = max(float(row["dte"]) / 365.0, 1e-8)
        option_side = str(OPTION_SIDE)
        market_price, price_source_used = _resolve_comparison_price(row)
        iv_confidence = row.get("iv_confidence", "Unknown")
        rate = float(row.get("risk_free_rate", REFERENCE_RISK_FREE_RATE))
    except Exception:
        return base_nan_metrics

    try:
        compare_sigma = _solve_implied_vol_from_price(
            market_price,
            spot,
            strike,
            t_years,
            option_side,
            rate,
        )
        compare_iv_source = "solved_from_market_price"
    except Exception:
        compare_sigma = vendor_sigma
        compare_iv_source = "vendor_iv_fallback"

    base = _bs_option_snapshot(spot, strike, t_years, compare_sigma, option_side, rate)
    vendor_base = _bs_option_snapshot(spot, strike, t_years, vendor_sigma, option_side, rate)

    market_minus_model = float(market_price) - base["price"] if pd.notna(market_price) else np.nan
    pct_vs_model = (float(market_price) / base["price"] - 1.0) if pd.notna(market_price) and abs(base["price"]) > 1e-12 else np.nan

    metrics = {
        "delta": base["delta"],
        "theta_day": base["theta_day"],
        "model_price": base["price"],
        "intrinsic": base["intrinsic"],
        "extrinsic": base["extrinsic"],
        "market_minus_model": market_minus_model,
        "pct_vs_model": pct_vs_model,
        "vendor_iv": vendor_sigma,
        "compare_iv": compare_sigma,
        "bid_iv": bid_sigma,
        "mid_iv": mid_sigma,
        "ask_iv": ask_sigma,
        "iv_width": _safe_iv_width(bid_sigma, ask_sigma),
        "vendor_model_price": vendor_base["price"],
        "compare_minus_vendor_model": base["price"] - vendor_base["price"],
        "price_source_used": f"{price_source_used} ({compare_iv_source})",
        "market_price_used": market_price,
        "iv_confidence": iv_confidence,
        "risk_free_rate": rate,
    }

    for d in PROJECT_NO_MOVE_DAYS:
        future_t = max(t_years - d / 365.0, 1e-8)
        future = _bs_option_snapshot(spot, strike, future_t, compare_sigma, option_side, rate)
        metrics[f"proj_{d}d_price"] = future["price"]
        metrics[f"proj_{d}d_decay"] = future["price"] - base["price"]

    return metrics



def build_slice_compare_frame(
    all_df: pd.DataFrame,
    plane_df: pd.DataFrame,
    faint_df: pd.DataFrame,
    slice_expiration: str,
) -> tuple[pd.DataFrame, tuple[np.ndarray | None, np.ndarray | None]]:
    slice_all = all_df[all_df["expiration"] == slice_expiration].copy().sort_values("strike")
    slice_plane = plane_df[plane_df["expiration"] == slice_expiration].copy().sort_values("strike")
    slice_faint = faint_df[faint_df["expiration"] == slice_expiration].copy().sort_values("strike")

    slice_plane = slice_plane.copy()
    slice_faint = slice_faint.copy()

    slice_plane["point_status"] = np.where(slice_plane.get("gmm_outlier_flag", False), "Outlier", "Fitted")
    slice_faint["point_status"] = "Outlier"

    compare_df = pd.concat([slice_plane, slice_faint], ignore_index=True)
    compare_df = compare_df.drop_duplicates(subset=["contractSymbol"], keep="first") if "contractSymbol" in compare_df.columns else compare_df
    compare_df = compare_df.sort_values("strike").reset_index(drop=True)

    x_curve, y_curve = build_slice_curve(slice_expiration)
    return compare_df, (x_curve, y_curve)



def print_available_slice_strikes(compare_df: pd.DataFrame) -> None:
    if compare_df.empty:
        print("\nNo slice contracts available to compare.")
        return

    strikes = compare_df["strike"].dropna().astype(float).sort_values().unique().tolist()
    formatted = ", ".join(f"{x:.2f}" for x in strikes)
    print("\nAvailable strikes in selected slice:")
    print(formatted)



def build_comparison_rows(compare_df: pd.DataFrame, requested_strikes: list[float], x_curve, y_curve) -> pd.DataFrame:
    rows = []
    if compare_df.empty:
        return pd.DataFrame()

    strikes_array = compare_df["strike"].to_numpy(dtype=float)

    for requested in requested_strikes:
        idx = int(np.argmin(np.abs(strikes_array - requested)))
        row = compare_df.iloc[idx]
        matched = float(row["strike"])
        diff = abs(matched - requested)
        if diff > COMPARE_NEAREST_TOLERANCE:
            print(f"Requested strike {requested:.2f} is too far from any available strike in this slice. Closest found: {matched:.2f}")
            continue

        metrics = _compute_compare_metrics(row)

        row_dict = {
            "Requested": requested,
            "Matched": matched,
            "Status": row.get("point_status", "Fitted"),
            "Moneyness": float(row["moneyness"]),
            "Vendor IV": metrics["vendor_iv"],
            "Compare IV": metrics["compare_iv"],
            "Bid IV": metrics["bid_iv"],
            "Mid IV": metrics["mid_iv"],
            "Ask IV": metrics["ask_iv"],
            "IV Width": metrics["iv_width"],
            "IV Confidence": metrics["iv_confidence"],
            "Risk-Free Rate": metrics["risk_free_rate"],
            "Delta": metrics["delta"],
            "Theta / Day": metrics["theta_day"],
            "Reference Px (BS)": metrics["model_price"],
            "Vendor Px (BS @ Vendor IV)": metrics["vendor_model_price"],
            "Ref Px - Vendor Px": metrics["compare_minus_vendor_model"],
            "Market Px Used": metrics["market_price_used"],
            "Px Used - Ref Px": metrics["market_minus_model"],
            "% vs Ref Px": metrics["pct_vs_model"],
            "Intrinsic": metrics["intrinsic"],
            "Extrinsic": metrics["extrinsic"],
            "Price Source": metrics["price_source_used"],
            "GMM Fitted Px": row.get("gmm_fitted_price", np.nan),
            "GMM Abs Error": row.get("gmm_abs_error", np.nan),
            "GMM Rel Error": row.get("gmm_rel_error", np.nan),
            "GMM Fitted IV": row.get("gmm_fitted_iv", np.nan),
            "Last": row.get("lastPrice", np.nan),
            "Bid": row.get("bid", np.nan),
            "Ask": row.get("ask", np.nan),
            "Mid": row.get("mid", np.nan),
            "Spread % Mid": row.get("spread_pct_mid", np.nan),
            "Volume": row.get("volume", np.nan),
            "Open Interest": row.get("openInterest", np.nan),
            "Contract": row.get("contractSymbol", "N/A"),
        }
        for d in PROJECT_NO_MOVE_DAYS:
            row_dict[f"{d}D No-Move Px"] = metrics[f"proj_{d}d_price"]
            row_dict[f"{d}D No-Move Decay"] = metrics[f"proj_{d}d_decay"]

        rows.append(row_dict)

    return pd.DataFrame(rows)



def print_comparison_table(table_df: pd.DataFrame) -> None:
    if table_df.empty:
        print("\nNo valid comparison rows to show.")
        return

    display_df = table_df.copy()

    price_cols = [
        "Reference Px (BS)",
        "Vendor Px (BS @ Vendor IV)",
        "Ref Px - Vendor Px",
        "Market Px Used",
        "Px Used - Ref Px",
        "GMM Fitted Px",
        "GMM Abs Error",
        "Mid",
        "Intrinsic",
        "Extrinsic",
        "Last",
        "Bid",
        "Ask",
    ] + [f"{d}D No-Move Px" for d in PROJECT_NO_MOVE_DAYS] + [f"{d}D No-Move Decay" for d in PROJECT_NO_MOVE_DAYS]
    pct_cols = ["Vendor IV", "Compare IV", "Bid IV", "Mid IV", "Ask IV", "IV Width", "% vs Ref Px", "Spread % Mid", "Risk-Free Rate", "GMM Rel Error", "GMM Fitted IV"]
    theta_cols = ["Theta / Day"]
    decimal_cols = ["Delta"]

    for col in ["Requested", "Matched"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.2f}")
    display_df["Moneyness"] = display_df["Moneyness"].map(lambda x: f"{x:.3f}")

    for col in pct_cols:
        if col in display_df.columns:
            dec = 2 if col in {"% vs Ref Px", "GMM Rel Error"} else 1
            display_df[col] = display_df[col].map(lambda x: _safe_pct_str(x, dec))

    for col in decimal_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: _safe_price_str(x, 3))

    for col in theta_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: _safe_price_str(x, 4))

    for col in price_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: _safe_price_str(x, 3))

    for col in ["Volume", "Open Interest"]:
        display_df[col] = display_df[col].map(_safe_int_str)

    strike_labels = [f"Strike {x}" for x in display_df["Matched"].tolist()]

    def _section(title: str, metrics: list[str]) -> None:
        section = display_df[metrics].T
        section.columns = strike_labels
        print("-" * 112)
        print(title)
        print(section.to_string())

    print("\n" + "=" * 112)
    print("SLICE STRIKE COMPARISON")
    print("Notes:")
    print(f"  - Reference Px (BS) now uses Compare IV, which is solved from the chosen market price source when possible.")
    print(f"  - Current comparison price source setting: {COMPARISON_PRICE_SOURCE}.")
    print(f"  - Risk-free rate source: {LIVE_TREASURY_INFO.get('source', 'fallback_flat')} | as of {LIVE_TREASURY_INFO.get('as_of', 'N/A')}.")
    print("  - GMM fields compare each quote to the arbitrage-consistent fitted slice.")

    _section(
        "1) SURFACE / SHAPE CONTEXT",
        ["Status", "Moneyness", "Vendor IV", "Bid IV", "Mid IV", "Ask IV", "Compare IV", "GMM Fitted IV", "IV Width", "IV Confidence", "Risk-Free Rate"],
    )
    _section(
        "2) GREEKS / SENSITIVITY",
        ["Delta", "Theta / Day"],
    )
    _section(
        "3) MODEL / PRICE CHECK",
        ["Reference Px (BS)", "Vendor Px (BS @ Vendor IV)", "Ref Px - Vendor Px", "Market Px Used", "Px Used - Ref Px", "GMM Fitted Px", "GMM Abs Error", "GMM Rel Error", "% vs Ref Px", "Intrinsic", "Extrinsic"],
    )
    _section(
        "4) LIVE MARKET QUOTES",
        ["Price Source", "Last", "Bid", "Ask", "Mid", "Spread % Mid"],
    )
    _section(
        "5) NO-MOVE TIME DECAY SCENARIO",
        [f"{d}D No-Move Px" for d in PROJECT_NO_MOVE_DAYS] + [f"{d}D No-Move Decay" for d in PROJECT_NO_MOVE_DAYS],
    )
    _section(
        "6) LIQUIDITY / CONTRACT INFO",
        ["Volume", "Open Interest", "Contract"],
    )

    print("=" * 112)


def maybe_save_comparison_csv(table_df: pd.DataFrame, ticker_symbol: str, slice_expiration: str) -> None:
    if not COMPARE_SAVE_CSV or table_df.empty:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{ticker_symbol.lower()}_slice_compare_{slice_expiration}.csv"
    table_df.to_csv(path, index=False)
    print(f"Saved comparison CSV to: {path.resolve()}")



def interactive_slice_compare_loop(
    all_df: pd.DataFrame,
    plane_df: pd.DataFrame,
    faint_df: pd.DataFrame,
    ticker_symbol: str,
    slice_expiration: str,
) -> None:
    if not COMPARE_AFTER_SLICE:
        return

    compare_df, (x_curve, y_curve) = build_slice_compare_frame(all_df, plane_df, faint_df, slice_expiration)
    if compare_df.empty:
        print("\nNo contracts available for interactive comparison in this slice.")
        return

    print("\nInteractive strike comparison is ready.")
    print("Type comma-separated strikes like: 9, 9.5")
    print("Type 'list' to show available strikes, or press Enter to exit.\nYou can keep comparing as many times as you want.\n")

    while True:
        user_in = safe_input("Enter strikes to compare for this slice: ").strip()
        if not user_in:
            print("Exiting strike comparison.")
            break
        if user_in.lower() in {"list", "ls", "show"}:
            print_available_slice_strikes(compare_df)
            continue

        try:
            requested = [float(x.strip()) for x in user_in.split(",") if x.strip()]
        except ValueError:
            print("Could not parse that input. Example: 9, 9.5")
            continue

        if not requested:
            print("No valid strikes entered.")
            continue

        table_df = build_comparison_rows(compare_df, requested, x_curve, y_curve)
        print_comparison_table(table_df)
        maybe_save_comparison_csv(table_df, ticker_symbol, slice_expiration)
        print()


# ============================================================
# INPUT
# ============================================================

def normalize_option_side(value: str) -> str:
    value = (value or "").strip().lower()
    if value in {"calls", "call", "c"}:
        return "calls"
    if value in {"puts", "put", "p"}:
        return "puts"
    raise ValueError("Option side must be calls/call/c or puts/put/p.")


def resolve_surface_moneyness_preset(value: str) -> tuple[str, tuple[float, float] | None]:
    raw = (value or "").strip().lower()
    if not raw:
        return SURFACE_MONEYNESS_PRESET, SURFACE_MONEYNESS_PRESETS.get(SURFACE_MONEYNESS_PRESET)

    aliases = {
        "f": "focused",
        "focus": "focused",
        "focused": "focused",
        "c": "conservative",
        "cons": "conservative",
        "conservative": "conservative",
        "s": "standard",
        "std": "standard",
        "standard": "standard",
        "a": "aggressive",
        "agg": "aggressive",
        "aggressive": "aggressive",
        "off": "off",
        "none": "off",
        "disable": "off",
        "disabled": "off",
        "custom": "custom",
    }
    key = aliases.get(raw, raw)
    if key == "off":
        return "off", None
    if key == "custom":
        return "custom", None
    if key in SURFACE_MONEYNESS_PRESETS:
        return key, SURFACE_MONEYNESS_PRESETS[key]
    raise ValueError(
        "Preset must be focused, conservative, standard, aggressive, custom, or off."
    )


def get_inputs_from_user() -> tuple[str, int, str, str, str]:
    global SURFACE_USE_MONEYNESS_DISPLAY_BAND, SURFACE_MIN_MONEYNESS_DISPLAY, SURFACE_MAX_MONEYNESS_DISPLAY, SURFACE_MONEYNESS_PRESET

    ticker = DEFAULT_TICKER
    max_dte = DEFAULT_MAX_DTE
    slice_target = DEFAULT_SLICE_TARGET
    option_side = OPTION_SIDE
    surface_preset = SURFACE_MONEYNESS_PRESET

    if len(sys.argv) > 1 and sys.argv[1].strip():
        ticker = sys.argv[1].strip().upper()

    if len(sys.argv) > 2 and sys.argv[2].strip():
        try:
            max_dte = int(sys.argv[2].strip())
        except ValueError:
            pass

    if len(sys.argv) > 3 and sys.argv[3].strip():
        slice_target = sys.argv[3].strip()

    if len(sys.argv) > 4 and sys.argv[4].strip():
        try:
            option_side = normalize_option_side(sys.argv[4].strip())
        except ValueError:
            pass

    if len(sys.argv) > 5 and sys.argv[5].strip():
        try:
            surface_preset, preset_band = resolve_surface_moneyness_preset(sys.argv[5].strip())
            if preset_band is None:
                if surface_preset == "off":
                    SURFACE_USE_MONEYNESS_DISPLAY_BAND = False
                else:
                    surface_preset = SURFACE_MONEYNESS_PRESET
            else:
                SURFACE_USE_MONEYNESS_DISPLAY_BAND = True
                SURFACE_MIN_MONEYNESS_DISPLAY, SURFACE_MAX_MONEYNESS_DISPLAY = preset_band
        except ValueError:
            pass

    if len(sys.argv) <= 1:
        user_ticker = safe_input(f"Enter ticker [{ticker}]: ").strip().upper()
        if user_ticker:
            ticker = user_ticker

        user_max_dte = safe_input(f"Enter max DTE [{max_dte}]: ").strip()
        if user_max_dte:
            try:
                max_dte = int(user_max_dte)
            except ValueError:
                print(f"Invalid DTE input. Using default {max_dte}.")

        user_slice = safe_input(
            f"Enter slice expiration / target DTE / auto [{slice_target}]: "
        ).strip()
        if user_slice:
            slice_target = user_slice

        user_side = safe_input(f"Enter option side calls/puts [{option_side}]: ").strip()
        if user_side:
            try:
                option_side = normalize_option_side(user_side)
            except ValueError:
                print(f"Invalid option side. Using default {option_side}.")

        preset_prompt = (
            "Enter surface moneyness preset "
            f"[focused=0.80-1.60, conservative=0.80-1.20, standard=0.50-2.00, aggressive=0.30-3.00, custom, off] [{surface_preset}]: "
        )
        user_preset = safe_input(preset_prompt).strip().lower()
        if user_preset:
            try:
                resolved_preset, preset_band = resolve_surface_moneyness_preset(user_preset)
                surface_preset = resolved_preset
                if preset_band is None:
                    if resolved_preset == "off":
                        SURFACE_USE_MONEYNESS_DISPLAY_BAND = False
                    else:
                        custom_min = safe_input("Enter custom min moneyness [0.80]: ").strip()
                        custom_max = safe_input("Enter custom max moneyness [1.60]: ").strip()
                        try:
                            custom_min_val = float(custom_min) if custom_min else 0.80
                            custom_max_val = float(custom_max) if custom_max else 1.60
                            if custom_min_val <= 0 or custom_max_val <= custom_min_val:
                                raise ValueError
                            SURFACE_USE_MONEYNESS_DISPLAY_BAND = True
                            SURFACE_MIN_MONEYNESS_DISPLAY = custom_min_val
                            SURFACE_MAX_MONEYNESS_DISPLAY = custom_max_val
                            surface_preset = f"custom({custom_min_val:.2f}-{custom_max_val:.2f})"
                        except ValueError:
                            print("Invalid custom moneyness band. Keeping previous band settings.")
                else:
                    SURFACE_USE_MONEYNESS_DISPLAY_BAND = True
                    SURFACE_MIN_MONEYNESS_DISPLAY, SURFACE_MAX_MONEYNESS_DISPLAY = preset_band
            except ValueError:
                print("Invalid surface moneyness preset. Keeping previous band settings.")

    if max_dte < 1:
        max_dte = DEFAULT_MAX_DTE

    if SURFACE_USE_MONEYNESS_DISPLAY_BAND and SURFACE_MAX_MONEYNESS_DISPLAY <= SURFACE_MIN_MONEYNESS_DISPLAY:
        SURFACE_MIN_MONEYNESS_DISPLAY, SURFACE_MAX_MONEYNESS_DISPLAY = SURFACE_MONEYNESS_PRESETS[SURFACE_MONEYNESS_PRESET]
        surface_preset = SURFACE_MONEYNESS_PRESET

    SURFACE_MONEYNESS_PRESET = surface_preset
    return ticker, max_dte, slice_target, option_side, surface_preset


# ============================================================
# DATA
# ============================================================
def get_spot_price(ticker_obj: yf.Ticker) -> float:
    fast_info = getattr(ticker_obj, "fast_info", {}) or {}
    spot = (
        fast_info.get("lastPrice")
        or fast_info.get("regularMarketPrice")
        or fast_info.get("previousClose")
    )

    if spot is not None and pd.notna(spot):
        return float(spot)

    hist = ticker_obj.history(period="5d", auto_adjust=False)
    if hist.empty or hist["Close"].dropna().empty:
        raise ValueError("Could not determine spot price.")
    return float(hist["Close"].dropna().iloc[-1])



def compute_dte(expiration_str: str) -> float:
    exp_ts = pd.Timestamp(expiration_str) + pd.Timedelta(hours=16)
    now_ts = pd.Timestamp.now()
    return (exp_ts - now_ts).total_seconds() / 86400.0



def fetch_option_points(ticker_symbol: str, max_dte: int) -> tuple[pd.DataFrame, float]:
    ticker = yf.Ticker(ticker_symbol)
    expirations = ticker.options

    if not expirations:
        raise ValueError(f"No option expirations found for {ticker_symbol}.")

    spot = get_spot_price(ticker)
    frames = []

    for exp_str in expirations:
        dte = compute_dte(exp_str)

        if dte < MIN_DTE or dte > max_dte:
            continue

        try:
            chain = ticker.option_chain(exp_str)
            side_df = getattr(chain, OPTION_SIDE).copy()

            if side_df.empty:
                continue

            side_df["expiration"] = exp_str
            side_df["dte"] = dte
            frames.append(side_df)

        except Exception as exc:
            print(f"Skipping {exp_str}: {exc}")

    if not frames:
        raise ValueError(f"No {OPTION_SIDE} data found for {ticker_symbol} in the chosen DTE range.")

    raw = pd.concat(frames, ignore_index=True)

    keep_cols = [
        "contractSymbol",
        "strike",
        "lastPrice",
        "bid",
        "ask",
        "volume",
        "openInterest",
        "impliedVolatility",
        "inTheMoney",
        "expiration",
        "dte",
    ]
    keep_cols = [c for c in keep_cols if c in raw.columns]
    raw = raw[keep_cols].copy()

    raw["spot"] = spot
    raw["moneyness"] = raw["strike"] / spot

    valid_bid = raw["bid"].notna() & (raw["bid"] > 0)
    valid_ask = raw["ask"].notna() & (raw["ask"] > 0)
    two_sided = valid_bid & valid_ask & (raw["ask"] >= raw["bid"])

    raw["mid"] = np.nan
    raw.loc[two_sided, "mid"] = (raw.loc[two_sided, "bid"] + raw.loc[two_sided, "ask"]) / 2.0
    raw["spread"] = np.nan
    raw.loc[two_sided, "spread"] = raw.loc[two_sided, "ask"] - raw.loc[two_sided, "bid"]
    raw["spread_pct_mid"] = np.where(raw["mid"] > 0, raw["spread"] / raw["mid"], np.nan)

    return raw, spot


# ============================================================
# CLEANING / TRUST CLASSIFICATION
# ============================================================
def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))



def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    z = x - np.max(x)
    e = np.exp(z)
    denom = np.sum(e)
    if denom <= 0:
        return np.ones_like(x) / len(x)
    return e / denom



def _price_within_no_arb_bounds(price: float, row: pd.Series, option_side: str) -> bool:
    if not np.isfinite(price) or price <= 0:
        return False
    try:
        snap = _bs_option_snapshot(
            float(row["spot"]),
            float(row["strike"]),
            max(float(row["dte"]) / 365.0, 1e-8),
            0.20,
            option_side,
            float(row.get("risk_free_rate", REFERENCE_RISK_FREE_RATE)),
        )
        lower = max(0.0, snap["lower_bound"] - 1e-8)
        upper = snap["upper_bound"] + 1e-8
        return lower <= float(price) <= upper
    except Exception:
        return False



def lightly_clean_option_points(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = out.dropna(subset=["strike", "dte"])
    for col in ["strike", "dte"]:
        out = out[np.isfinite(out[col])]

    out = out[out["strike"] > 0]
    out = out[out["dte"] > 0]

    spot = float(out["spot"].iloc[0]) if not out.empty else np.nan
    if np.isfinite(spot) and spot > 0:
        out = out[out["strike"] >= spot * MIN_STRIKE_MULTIPLE]
        out = out[out["strike"] <= spot * MAX_STRIKE_MULTIPLE]

    bid = out.get("bid", pd.Series(np.nan, index=out.index)).fillna(0)
    ask = out.get("ask", pd.Series(np.nan, index=out.index)).fillna(0)
    last = out.get("lastPrice", pd.Series(np.nan, index=out.index)).fillna(0)
    vol = out.get("volume", pd.Series(np.nan, index=out.index)).fillna(0)
    oi = out.get("openInterest", pd.Series(np.nan, index=out.index)).fillna(0)

    dead = (bid <= 0) & (ask <= 0) & (last <= 0) & (vol <= 0) & (oi <= 0)
    out = out[~dead]

    crossed = (out["ask"].fillna(0) > 0) & (out["bid"].fillna(0) > 0) & (out["ask"] < out["bid"])
    out = out[~crossed]

    insane_spread = out["spread_pct_mid"].fillna(0) > MAX_ABS_SPREAD_TO_PRICE_RATIO
    out = out[~insane_spread]

    if "contractSymbol" in out.columns:
        out = out.drop_duplicates(subset=["contractSymbol"], keep="last")

    return out.sort_values(["expiration", "strike"]).reset_index(drop=True)



def choose_market_price_for_fit(row: pd.Series) -> tuple[float, str, float]:
    bid = row.get("bid", np.nan)
    ask = row.get("ask", np.nan)
    mid = row.get("mid", np.nan)
    last = row.get("lastPrice", np.nan)
    spread_pct = row.get("spread_pct_mid", np.nan)
    oi = float(row.get("openInterest", 0) or 0)
    vol = float(row.get("volume", 0) or 0)
    rate = float(row.get("risk_free_rate", REFERENCE_RISK_FREE_RATE))

    spot = float(row["spot"])
    strike = float(row["strike"])
    t_years = max(float(row["dte"]) / 365.0, 1e-8)
    discount = math.exp(-rate * t_years)

    if OPTION_SIDE == "puts":
        intrinsic_floor = max(strike * discount - spot, 0.0)
    else:
        intrinsic_floor = max(spot - strike * discount, 0.0)

    def liquidity_multiplier() -> float:
        return float(np.clip(0.85 + 0.08 * np.log1p(max(oi + vol, 0.0)), 0.85, 1.20))

    def quote_weight(price: float, source: str) -> float:
        base = {"mid": 1.0, "blend": 0.82, "last": 0.42, "bid": 0.22, "ask": 0.18}.get(source, 0.15)
        if pd.notna(spread_pct):
            sp = float(spread_pct)
            if sp <= 0.05:
                spread_mult = 1.00
            elif sp <= 0.12:
                spread_mult = 0.92
            elif sp <= 0.25:
                spread_mult = 0.78
            elif sp <= 0.50:
                spread_mult = 0.58
            elif sp <= 1.00:
                spread_mult = 0.36
            else:
                spread_mult = 0.18
        else:
            spread_mult = 0.45 if source in {"bid", "ask", "last"} else 0.25

        extrinsic = max(float(price) - intrinsic_floor, 0.0)
        extrinsic_floor = max(FIT_MIN_EXTRINSIC_FOR_INFORMATIVE, FIT_MIN_EXTRINSIC_PCT * max(float(price), FIT_PRICE_SCALE_FLOOR))
        extrinsic_mult = 1.0 if extrinsic >= extrinsic_floor else float(np.clip(extrinsic / max(extrinsic_floor, 1e-8), 0.05, 0.35))
        return float(np.clip(base * spread_mult * extrinsic_mult * liquidity_multiplier(), 0.02, 1.50))

    if pd.notna(bid) and pd.notna(ask) and float(bid) > 0 and float(ask) > float(bid):
        if pd.notna(spread_pct) and float(spread_pct) <= 0.60:
            candidate = float(mid)
            source = "mid"
        else:
            candidate = 0.65 * float(bid) + 0.35 * float(ask)
            source = "blend"
        if _price_within_no_arb_bounds(float(candidate), row, OPTION_SIDE):
            return float(candidate), source, quote_weight(float(candidate), source)

    if FIT_PRICE_SOURCE in {"last", "auto"} and pd.notna(last) and float(last) > 0 and _price_within_no_arb_bounds(float(last), row, OPTION_SIDE):
        candidate = float(last)
        if pd.notna(mid) and float(mid) > 0:
            rel_gap = abs(candidate - float(mid)) / max(float(mid), FIT_PRICE_SCALE_FLOOR)
            if rel_gap <= 0.10:
                return candidate, "last", quote_weight(candidate, "last") * 1.20
            if rel_gap <= 0.25:
                return candidate, "last", quote_weight(candidate, "last")
        else:
            return candidate, "last", quote_weight(candidate, "last")

    if pd.notna(bid) and float(bid) > 0 and _price_within_no_arb_bounds(float(bid), row, OPTION_SIDE):
        return float(bid), "bid", quote_weight(float(bid), "bid")

    if pd.notna(ask) and float(ask) > 0 and _price_within_no_arb_bounds(float(ask), row, OPTION_SIDE):
        return float(ask), "ask", quote_weight(float(ask), "ask")

    raise ValueError("No sane market price available for GMM fit.")



def build_expiry_fit_frame(exp_df: pd.DataFrame) -> pd.DataFrame:
    out = exp_df.copy()
    prices = []
    sources = []
    weights = []
    fit_input = []
    half_spreads = []
    extrinsics = []
    price_scales = []
    informative_flags = []

    for _, row in out.iterrows():
        try:
            px, src, wt = choose_market_price_for_fit(row)
            rate = float(row.get("risk_free_rate", REFERENCE_RISK_FREE_RATE))
            spot = float(row["spot"])
            strike = float(row["strike"])
            t_years = max(float(row["dte"]) / 365.0, 1e-8)
            discount = math.exp(-rate * t_years)
            if OPTION_SIDE == "puts":
                intrinsic_floor = max(strike * discount - spot, 0.0)
            else:
                intrinsic_floor = max(spot - strike * discount, 0.0)
            extrinsic = max(float(px) - intrinsic_floor, 0.0)

            bid = row.get("bid", np.nan)
            ask = row.get("ask", np.nan)
            if pd.notna(bid) and pd.notna(ask) and float(ask) > float(bid) > 0:
                half_spread = 0.5 * float(ask - bid)
            else:
                half_spread = max(0.25 * float(px), FIT_PRICE_SCALE_FLOOR)

            price_scale = max(FIT_PRICE_SCALE_FLOOR, half_spread, FIT_PRICE_SCALE_PCT * float(px))
            extrinsic_floor = max(FIT_MIN_EXTRINSIC_FOR_INFORMATIVE, FIT_MIN_EXTRINSIC_PCT * max(float(px), FIT_PRICE_SCALE_FLOOR))
            spread_pct = row.get("spread_pct_mid", np.nan)
            informative = (
                wt >= FIT_MIN_INFORMATIVE_WEIGHT
                and extrinsic >= extrinsic_floor
                and (pd.isna(spread_pct) or float(spread_pct) <= FIT_MAX_SPREAD_PCT_FOR_INFORMATIVE)
            )

            prices.append(px)
            sources.append(src)
            weights.append(wt)
            fit_input.append(True)
            half_spreads.append(half_spread)
            extrinsics.append(extrinsic)
            price_scales.append(price_scale)
            informative_flags.append(bool(informative))
        except Exception:
            prices.append(np.nan)
            sources.append("none")
            weights.append(0.0)
            fit_input.append(False)
            half_spreads.append(np.nan)
            extrinsics.append(np.nan)
            price_scales.append(np.nan)
            informative_flags.append(False)

    out["market_price_used"] = prices
    out["surface_price_source"] = sources
    out["fit_weight"] = weights
    out["gmm_fit_input"] = fit_input
    out["half_spread_used"] = half_spreads
    out["extrinsic_value"] = extrinsics
    out["fit_price_scale"] = price_scales
    out["gmm_informative_input"] = informative_flags
    return out



def _bs_call_from_forward(forward: np.ndarray | float, strike: np.ndarray | float, sigma: np.ndarray | float, t_years: float) -> np.ndarray:
    forward = np.asarray(forward, dtype=float)
    strike = np.asarray(strike, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    t = max(float(t_years), 1e-12)
    sig_sqrt_t = np.maximum(sigma * math.sqrt(t), 1e-12)
    d1 = (np.log(np.maximum(forward, 1e-12) / np.maximum(strike, 1e-12)) + 0.5 * sigma * sigma * t) / sig_sqrt_t
    d2 = d1 - sig_sqrt_t
    return forward * (0.5 * (1.0 + np.vectorize(math.erf)(d1 / math.sqrt(2.0)))) - strike * (0.5 * (1.0 + np.vectorize(math.erf)(d2 / math.sqrt(2.0))))



def _decode_mixture_params(raw_params: np.ndarray, forward0: float, n_components: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(n_components or (len(raw_params) // 3))
    raw_w = raw_params[:n]
    raw_m = raw_params[n:2*n]
    raw_v = raw_params[2*n:3*n]

    weights = _softmax(raw_w)
    mult = FIT_COMPONENT_FORWARD_MULT_MIN + (FIT_COMPONENT_FORWARD_MULT_MAX - FIT_COMPONENT_FORWARD_MULT_MIN) * _sigmoid(raw_m)
    vols = FIT_COMPONENT_VOL_MIN + (FIT_COMPONENT_VOL_MAX - FIT_COMPONENT_VOL_MIN) * _sigmoid(raw_v)

    denom = float(np.sum(weights * mult))
    denom = max(denom, 1e-12)
    forwards = forward0 * mult / denom

    order = np.argsort(forwards)
    return weights[order], forwards[order], vols[order]



def mixture_option_price(
    strikes: np.ndarray,
    t_years: float,
    rate: float,
    option_side: str,
    raw_params: np.ndarray,
    spot: float,
    n_components: int | None = None,
) -> np.ndarray:
    strikes = np.asarray(strikes, dtype=float)
    forward0 = float(spot) * math.exp(float(rate) * max(float(t_years), 1e-12))
    weights, forwards, vols = _decode_mixture_params(np.asarray(raw_params, dtype=float), forward0, n_components=n_components)
    undiscounted = np.zeros_like(strikes, dtype=float)
    for w, fwd_i, vol_i in zip(weights, forwards, vols):
        undiscounted += float(w) * _bs_call_from_forward(fwd_i, strikes, vol_i, t_years)
    calls = math.exp(-float(rate) * max(float(t_years), 1e-12)) * undiscounted
    if option_side == "puts":
        return calls - float(spot) + strikes * math.exp(-float(rate) * max(float(t_years), 1e-12))
    return calls



def _pseudo_huber(x: np.ndarray, delta: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    d = max(float(delta), 1e-8)
    return d * d * (np.sqrt(1.0 + (x / d) ** 2) - 1.0)



def _initial_param_guesses(fit_df: pd.DataFrame, n_components: int | None = None) -> list[np.ndarray]:
    n = int(n_components or MIXTURE_COMPONENTS)
    guesses: list[np.ndarray] = []

    work = fit_df.copy().sort_values("strike")
    spot = float(work["spot"].iloc[0])
    rate = float(work["risk_free_rate"].median())
    t_years = max(float(work["dte"].median()) / 365.0, 1e-8)

    if "moneyness" in work.columns and not work.empty:
        atm_pos = int(np.argmin(np.abs(work["moneyness"].to_numpy(dtype=float) - 1.0)))
        atm_row = work.iloc[atm_pos]
    else:
        atm_row = work.iloc[len(work) // 2]

    iv_candidates = [
        atm_row.get("mid_solved_iv", np.nan),
        atm_row.get("bid_solved_iv", np.nan),
        atm_row.get("ask_solved_iv", np.nan),
        atm_row.get("impliedVolatility", np.nan),
        work["mid_solved_iv"].median() if "mid_solved_iv" in work.columns else np.nan,
        work["impliedVolatility"].median() if "impliedVolatility" in work.columns else np.nan,
    ]
    atm_iv = next((float(x) for x in iv_candidates if pd.notna(x) and np.isfinite(x) and float(x) > 0), 0.85)
    atm_iv = float(np.clip(atm_iv, 0.12, 2.50))

    short_expiry = t_years * 365.0 <= FIT_SHORT_DTE_THRESHOLD

    if short_expiry:
        if n == 1:
            seed_sets = [([1.00], [atm_iv]), ([1.00], [max(0.18, 0.8 * atm_iv)]), ([1.00], [1.25 * atm_iv])]
        elif n == 2:
            seed_sets = [
                ([0.94, 1.06], [0.85 * atm_iv, 1.20 * atm_iv]),
                ([0.90, 1.10], [0.75 * atm_iv, 1.40 * atm_iv]),
                ([0.97, 1.03], [0.95 * atm_iv, 1.10 * atm_iv]),
            ]
        else:
            seed_sets = [
                ([0.92, 1.00, 1.08], [0.80 * atm_iv, 1.00 * atm_iv, 1.35 * atm_iv]),
                ([0.88, 1.00, 1.12], [0.70 * atm_iv, 1.00 * atm_iv, 1.50 * atm_iv]),
                ([0.95, 1.00, 1.05], [0.90 * atm_iv, 1.05 * atm_iv, 1.20 * atm_iv]),
                ([0.90, 1.00, 1.10], [0.75 * atm_iv, 0.95 * atm_iv, 1.30 * atm_iv]),
            ]
    else:
        if n == 1:
            seed_sets = [([1.00], [atm_iv]), ([1.00], [0.80 * atm_iv]), ([1.00], [1.20 * atm_iv])]
        elif n == 2:
            seed_sets = [
                ([0.88, 1.12], [0.75 * atm_iv, 1.25 * atm_iv]),
                ([0.82, 1.18], [0.60 * atm_iv, 1.40 * atm_iv]),
                ([0.94, 1.06], [0.90 * atm_iv, 1.15 * atm_iv]),
            ]
        else:
            seed_sets = [
                ([0.86, 1.00, 1.14], [0.70 * atm_iv, 1.00 * atm_iv, 1.45 * atm_iv]),
                ([0.80, 1.00, 1.20], [0.60 * atm_iv, 0.95 * atm_iv, 1.60 * atm_iv]),
                ([0.92, 1.00, 1.08], [0.85 * atm_iv, 1.05 * atm_iv, 1.25 * atm_iv]),
                ([0.88, 1.00, 1.12], [0.75 * atm_iv, 0.95 * atm_iv, 1.35 * atm_iv]),
            ]

    def inv_sigmoid_bounded(values: list[float], lo: float, hi: float) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        scaled = np.clip((arr - lo) / max(hi - lo, 1e-12), 1e-6, 1 - 1e-6)
        return np.log(scaled / (1.0 - scaled))

    weight_templates = {
        1: [[1.0]],
        2: [[0.50, 0.50], [0.65, 0.35], [0.35, 0.65]],
        3: [[0.20, 0.60, 0.20], [0.10, 0.80, 0.10], [0.33, 0.34, 0.33], [0.25, 0.50, 0.25]],
    }

    while len(seed_sets) < FIT_MULTI_STARTS:
        mult_lo = 0.90 if short_expiry else 0.80
        mult_hi = 1.10 if short_expiry else 1.20
        mults = list(np.linspace(mult_lo, mult_hi, n))
        vols = list(np.linspace(max(0.15, 0.75 * atm_iv), min(FIT_COMPONENT_VOL_MAX * 0.9, 1.40 * atm_iv), n))
        seed_sets.append((mults, vols))

    for idx in range(FIT_MULTI_STARTS):
        mult_seed, vol_seed = seed_sets[idx % len(seed_sets)]
        raw_m = inv_sigmoid_bounded(list(np.clip(mult_seed[:n], FIT_COMPONENT_FORWARD_MULT_MIN + 1e-4, FIT_COMPONENT_FORWARD_MULT_MAX - 1e-4)), FIT_COMPONENT_FORWARD_MULT_MIN, FIT_COMPONENT_FORWARD_MULT_MAX)
        raw_v = inv_sigmoid_bounded(list(np.clip(vol_seed[:n], FIT_COMPONENT_VOL_MIN + 1e-4, FIT_COMPONENT_VOL_MAX - 1e-4)), FIT_COMPONENT_VOL_MIN, FIT_COMPONENT_VOL_MAX)

        templates = weight_templates.get(n, [list(np.ones(n) / n)])
        w_template = np.asarray(templates[idx % len(templates)], dtype=float)
        w_template = np.maximum(w_template, 1e-6)
        w_template = w_template / np.sum(w_template)
        raw_w = np.log(w_template)

        rng = np.random.default_rng(1000 + idx + 37 * n + int(round(t_years * 365.0)))
        raw_w = raw_w + rng.normal(0.0, 0.10 if short_expiry else 0.16, n)
        raw_m = raw_m + rng.normal(0.0, 0.08 if short_expiry else 0.12, n)
        raw_v = raw_v + rng.normal(0.0, 0.08 if short_expiry else 0.12, n)

        guesses.append(np.concatenate([raw_w, raw_m, raw_v]))
    return guesses



def fit_mixture_bs_slice(exp_df: pd.DataFrame, n_components: int | None = None) -> dict[str, object]:
    n = int(n_components or MIXTURE_COMPONENTS)
    fit_df = build_expiry_fit_frame(exp_df)
    usable_all = fit_df[fit_df["gmm_fit_input"]].copy().sort_values("strike")
    if usable_all.empty:
        raise ValueError(f"No usable prices to fit expiry {exp_df['expiration'].iloc[0]} with {n} components.")

    t_years = max(float(usable_all["dte"].median()) / 365.0, 1e-8)
    short_expiry = t_years * 365.0 <= FIT_SHORT_DTE_THRESHOLD
    informative = usable_all[usable_all["gmm_informative_input"]].copy().sort_values("strike")

    min_needed = max(4, min(FIT_MIN_POINTS_PER_EXPIRY, 2 * n))
    usable = informative if len(informative) >= min_needed else usable_all
    if len(usable) < min_needed:
        raise ValueError(f"Not enough usable prices to fit expiry {exp_df['expiration'].iloc[0]} with {n} components.")

    strikes = usable["strike"].to_numpy(dtype=float)
    market_prices = usable["market_price_used"].to_numpy(dtype=float)
    fit_weights = usable["fit_weight"].to_numpy(dtype=float)
    price_scales = usable["fit_price_scale"].to_numpy(dtype=float)
    extrinsic = usable["extrinsic_value"].to_numpy(dtype=float)
    spot = float(usable["spot"].iloc[0])
    rate = float(usable["risk_free_rate"].median())
    option_side = str(OPTION_SIDE)

    fit_weights = np.maximum(fit_weights, 1e-6)
    price_scales = np.maximum(price_scales, FIT_PRICE_SCALE_FLOOR)
    extrinsic_floor = np.maximum(FIT_MIN_EXTRINSIC_FOR_INFORMATIVE, FIT_MIN_EXTRINSIC_PCT * np.maximum(market_prices, FIT_PRICE_SCALE_FLOOR))
    extrinsic_weight = np.where(extrinsic >= extrinsic_floor, 1.0, np.clip(extrinsic / np.maximum(extrinsic_floor, 1e-8), 0.08, 0.40))
    fit_weights = fit_weights * extrinsic_weight
    forward0 = spot * math.exp(rate * t_years)

    def objective(raw_params: np.ndarray) -> float:
        try:
            model_prices = mixture_option_price(strikes, t_years, rate, option_side, raw_params, spot, n_components=n)
            if not np.all(np.isfinite(model_prices)):
                return float(FIT_OBJECTIVE_FAIL_PENALTY)
            if np.any(model_prices < -1e-8):
                return float(FIT_OBJECTIVE_FAIL_PENALTY)

            resid = (model_prices - market_prices) / price_scales
            loss = np.sum(fit_weights * _pseudo_huber(resid, FIT_PSEUDO_HUBER_DELTA))

            _, forwards, vols = _decode_mixture_params(raw_params, forward0, n_components=n)
            width_penalty = np.sum(np.diff(forwards / max(forward0, 1e-8)) ** 2) if len(forwards) > 1 else 0.0
            reg = FIT_L2_REG * (
                np.sum((vols - np.median(vols)) ** 2)
                + 0.35 * np.sum((forwards / max(forward0, 1e-8) - 1.0) ** 2)
                + 0.15 * width_penalty
            )

            total = float(loss + reg)
            return total if np.isfinite(total) else float(FIT_OBJECTIVE_FAIL_PENALTY)
        except Exception:
            return float(FIT_OBJECTIVE_FAIL_PENALTY)

    param_count = 3 * n
    bounds = [(-6.0, 6.0)] * param_count

    best_params = None
    best_val = np.inf
    for guess in _initial_param_guesses(usable, n_components=n):
        init_val = objective(guess)
        if np.isfinite(init_val) and init_val < best_val:
            best_val = init_val
            best_params = np.asarray(guess, dtype=float)

        for method in FIT_OPT_METHODS:
            try:
                kwargs = dict(
                    fun=objective,
                    x0=np.asarray(guess, dtype=float),
                    method=method,
                    options={"maxiter": FIT_MAX_ITER},
                )
                if method in {"L-BFGS-B", "Powell", "TNC", "SLSQP"}:
                    kwargs["bounds"] = bounds
                res = minimize(**kwargs)
                candidate_x = np.asarray(res.x, dtype=float) if hasattr(res, "x") else np.asarray(guess, dtype=float)
                candidate_val = objective(candidate_x)
                if np.isfinite(candidate_val) and candidate_val < best_val:
                    best_val = candidate_val
                    best_params = candidate_x
            except Exception:
                continue

    if best_params is None or not np.isfinite(best_val):
        raise ValueError(f"Mixture fit failed on all starts for {n} components.")
    if float(best_val) >= float(FIT_OBJECTIVE_FAIL_PENALTY) * float(FIT_OBJECTIVE_REJECT_RATIO):
        raise ValueError(f"Mixture fit converged only to penalty solutions for {n} components.")

    weights, forwards, vols = _decode_mixture_params(best_params, forward0, n_components=n)

    return {
        "fit_df": fit_df,
        "usable_df": usable,
        "params": best_params,
        "weights": weights,
        "forwards": forwards,
        "vols": vols,
        "spot": spot,
        "rate": rate,
        "t_years": t_years,
        "option_side": option_side,
        "objective": best_val,
        "n_components": n,
        "used_informative_subset": bool(short_expiry and len(informative) >= min_needed),
    }



def solve_fitted_iv_from_fitted_prices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    fitted_iv = []
    for _, row in out.iterrows():
        price = row.get("gmm_fitted_price", np.nan)
        try:
            iv = _solve_implied_vol_from_price(
                float(price),
                float(row["spot"]),
                float(row["strike"]),
                max(float(row["dte"]) / 365.0, 1e-8),
                OPTION_SIDE,
                float(row.get("risk_free_rate", REFERENCE_RISK_FREE_RATE)),
            )
        except Exception:
            iv = np.nan
        fitted_iv.append(iv)
    out["gmm_fitted_iv"] = fitted_iv
    out["plane_iv"] = out["gmm_fitted_iv"]
    out["plane_iv_source"] = "gmm_fitted"
    return out



def _classify_gmm_outliers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gmm_price_error"] = out["market_price_used"] - out["gmm_fitted_price"]
    out["gmm_abs_error"] = out["gmm_price_error"].abs()
    out["gmm_rel_error"] = out["gmm_abs_error"] / np.maximum(out["market_price_used"].fillna(0), 0.10)
    out["gmm_outlier_flag"] = (
        out["gmm_abs_error"].fillna(0) >= FIT_OUTLIER_ABS_THRESHOLD
    ) & (
        out["gmm_rel_error"].fillna(0) >= FIT_OUTLIER_REL_THRESHOLD
    )
    return out



def _evaluate_slice_curve_iv(expiration: str, strikes: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    model = FITTED_MODELS.get(str(expiration))
    if model is None or len(strikes) == 0:
        return None, None
    prices = mixture_option_price(
        np.asarray(strikes, dtype=float),
        float(model["t_years"]),
        float(model["rate"]),
        str(model["option_side"]),
        np.asarray(model["params"], dtype=float),
        float(model["spot"]),
        n_components=int(model.get("n_components", MIXTURE_COMPONENTS)),
    )
    ivs = []
    for k, px in zip(strikes, prices):
        try:
            iv = _solve_implied_vol_from_price(
                float(px),
                float(model["spot"]),
                float(k),
                max(float(model["t_years"]), 1e-8),
                str(model["option_side"]),
                float(model["rate"]),
            )
        except Exception:
            iv = np.nan
        ivs.append(iv)
    ivs = np.asarray(ivs, dtype=float)
    mask = np.isfinite(ivs)
    if not mask.any():
        return None, None
    return np.asarray(strikes, dtype=float)[mask], ivs[mask]



def fit_all_expirations_with_gmm(all_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    fitted_frames: list[pd.DataFrame] = []
    model_cache: dict[str, dict[str, object]] = {}
    skipped: list[str] = []

    for expiration, exp_df in all_df.groupby("expiration", sort=True):
        result = None
        last_exc = None
        component_attempts = [int(x) for x in FIT_COMPONENT_FALLBACKS if int(x) >= 1 and int(x) <= MIXTURE_COMPONENTS]
        if MIXTURE_COMPONENTS not in component_attempts:
            component_attempts.insert(0, int(MIXTURE_COMPONENTS))
        component_attempts = list(dict.fromkeys(component_attempts))

        for n_try in component_attempts:
            try:
                result = fit_mixture_bs_slice(exp_df.copy(), n_components=n_try)
                break
            except Exception as exc:
                last_exc = exc

        if result is None:
            print(f"Skipping GMM fit for {expiration}: {last_exc}")
            skipped.append(str(expiration))
            continue

        fit_df = result["fit_df"].copy()
        fit_df["gmm_fitted_price"] = mixture_option_price(
            fit_df["strike"].to_numpy(dtype=float),
            float(result["t_years"]),
            float(result["rate"]),
            str(result["option_side"]),
            np.asarray(result["params"], dtype=float),
            float(result["spot"]),
            n_components=int(result.get("n_components", MIXTURE_COMPONENTS)),
        )
        fit_df = solve_fitted_iv_from_fitted_prices(fit_df)
        fit_df = _classify_gmm_outliers(fit_df)
        fit_df["point_status"] = np.where(fit_df["gmm_outlier_flag"], "Outlier", "Fitted")
        fit_df["plane_reason"] = np.where(fit_df["gmm_outlier_flag"], "gmm_residual_outlier", "gmm_fitted")
        fit_df["fit_objective"] = float(result["objective"])
        fit_df["gmm_components"] = int(result.get("n_components", MIXTURE_COMPONENTS))
        fit_df["mix_weight_1"] = float(result["weights"][0]) if len(result["weights"]) > 0 else np.nan
        fit_df["mix_weight_2"] = float(result["weights"][1]) if len(result["weights"]) > 1 else np.nan
        fit_df["mix_weight_3"] = float(result["weights"][2]) if len(result["weights"]) > 2 else np.nan
        fitted_frames.append(fit_df)
        model_cache[str(expiration)] = result

    if not fitted_frames:
        raise ValueError("No expirations could be fitted with the mixture-of-Black-Scholes engine.")

    fit_all = pd.concat(fitted_frames, ignore_index=True)
    fit_all = fit_all[np.isfinite(fit_all["gmm_fitted_iv"])].copy().reset_index(drop=True)

    outliers = fit_all[fit_all["gmm_outlier_flag"]].copy().reset_index(drop=True)
    surface_df = fit_all.copy() if FIT_INCLUDE_OUTLIERS_IN_SURFACE else fit_all[~fit_all["gmm_outlier_flag"]].copy().reset_index(drop=True)

    if len(surface_df) < FIT_SURFACE_MIN_POINTS:
        raise ValueError("Too few fitted IV points survived to build the final surface.")

    return surface_df, outliers, model_cache


# ============================================================
# SURFACE + VIEW HELPERS
# ============================================================
def _get_x_series(df: pd.DataFrame) -> pd.Series:
    return df["moneyness"] if X_AXIS_MODE == "moneyness" else df["strike"]


def _get_x_value(row: pd.Series) -> float:
    return float(row["moneyness"]) if X_AXIS_MODE == "moneyness" else float(row["strike"])



def _get_x_label() -> str:
    return "Moneyness (Strike / Spot)" if X_AXIS_MODE == "moneyness" else "Strike Price"


def apply_moneyness_band(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df.copy()
    if X_AXIS_MODE != "moneyness" or not SURFACE_USE_MONEYNESS_DISPLAY_BAND:
        return df.copy()
    mask = df["moneyness"].between(SURFACE_MIN_MONEYNESS_DISPLAY, SURFACE_MAX_MONEYNESS_DISPLAY, inclusive="both")
    return df.loc[mask].copy().reset_index(drop=True)



def apply_fit_universe_moneyness_band(df: pd.DataFrame) -> pd.DataFrame:
    return apply_moneyness_band(df)



def apply_surface_display_band(df: pd.DataFrame) -> pd.DataFrame:
    return apply_moneyness_band(df)

def build_smooth_iv_surface(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(df) < FIT_SURFACE_MIN_POINTS:
        raise ValueError("Not enough fitted option points to build a surface.")

    x = _get_x_series(df).to_numpy(dtype=float)
    y = df["dte"].to_numpy(dtype=float)
    z = df["gmm_fitted_iv"].to_numpy(dtype=float)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    X_grid, Y_grid = np.meshgrid(
        np.linspace(x_min, x_max, GRID_SIZE_X),
        np.linspace(y_min, y_max, GRID_SIZE_DTE),
    )

    x_scale = max(x_max - x_min, 1e-9)
    y_scale = max(y_max - y_min, 1e-9)

    x_norm = (x - x_min) / x_scale
    y_norm = (y - y_min) / y_scale
    X_norm = (X_grid - x_min) / x_scale
    Y_norm = (Y_grid - y_min) / y_scale

    rbf = Rbf(x_norm, y_norm, z, function=RBF_FUNCTION, smooth=SMOOTHING)
    Z_grid = rbf(X_norm, Y_norm)
    Z_grid = np.maximum(Z_grid, 0)

    return X_grid, Y_grid, Z_grid



def _padded_range(values: np.ndarray, pad_pct: float) -> list[float]:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = max(vmax - vmin, 1e-9)
    pad = span * pad_pct
    return [vmin - pad, vmax + pad]



def compute_focus_ranges(plane_df: pd.DataFrame) -> tuple[list[float], list[float], list[float]]:
    x_vals = _get_x_series(plane_df).to_numpy(dtype=float)
    y_vals = plane_df["dte"].to_numpy(dtype=float)
    z_vals = plane_df["gmm_fitted_iv"].to_numpy(dtype=float)

    x_range = _padded_range(x_vals, ZOOM_X_PAD_PCT)
    y_range = _padded_range(y_vals, ZOOM_Y_PAD_PCT)

    z_low = max(0.0, float(np.min(z_vals)) * (1.0 - ZOOM_Z_PAD_PCT))
    z_high_base = float(np.quantile(z_vals, Z_UPPER_PERCENTILE))
    z_high = z_high_base * (1.0 + ZOOM_Z_PAD_PCT)
    if z_high <= z_low:
        z_high = float(np.max(z_vals)) * 1.05

    return x_range, y_range, [z_low, z_high]



def make_hover_text(df: pd.DataFrame, include_reason: bool = False) -> list[str]:
    texts = []
    for _, row in df.iterrows():
        bid = row.get("bid", np.nan)
        ask = row.get("ask", np.nan)
        last_price = row.get("lastPrice", np.nan)
        volume = int(row["volume"]) if pd.notna(row.get("volume", np.nan)) else "N/A"
        oi = int(row["openInterest"]) if pd.notna(row.get("openInterest", np.nan)) else "N/A"
        itm = row.get("inTheMoney", "N/A")
        status_html = f"Point Status: {row.get('point_status', row.get('plane_reason', 'Fitted'))}<br>"
        if include_reason:
            status_html += f"Residual Outlier: {bool(row.get('gmm_outlier_flag', False))}<br>"

        mid_val = row.get("mid", np.nan)
        spread_pct = row.get("spread_pct_mid", np.nan)
        fit_price = row.get("gmm_fitted_price", np.nan)
        fit_iv = row.get("gmm_fitted_iv", np.nan)
        abs_err = row.get("gmm_abs_error", np.nan)
        rel_err = row.get("gmm_rel_error", np.nan)
        market_px = row.get("market_price_used", np.nan)

        def fmt_money(x):
            return "N/A" if pd.isna(x) else f"{float(x):.4f}"
        def fmt_pct(x):
            return "N/A" if pd.isna(x) else f"{float(x):.2%}"

        texts.append(
            f"Expiration: {row['expiration']}<br>"
            f"Strike: ${row['strike']:.2f}<br>"
            f"Moneyness: {row['moneyness']:.3f}<br>"
            f"DTE: {row['dte']:.1f}<br>"
            f"Market Px Used: {fmt_money(market_px)} ({row.get('surface_price_source', 'none')})<br>"
            f"GMM Fitted Px: {fmt_money(fit_price)}<br>"
            f"GMM Abs Error: {fmt_money(abs_err)}<br>"
            f"GMM Rel Error: {fmt_pct(rel_err)}<br>"
            f"GMM Fitted IV: {fmt_pct(fit_iv)}<br>"
            f"Vendor IV: {fmt_pct(row.get('impliedVolatility', np.nan))}<br>"
            f"Bid IV: {fmt_pct(row.get('bid_solved_iv', np.nan))}<br>"
            f"Mid IV: {fmt_pct(row.get('mid_solved_iv', np.nan))}<br>"
            f"Ask IV: {fmt_pct(row.get('ask_solved_iv', np.nan))}<br>"
            f"Spot: ${row['spot']:.2f}<br>"
            f"Mid: {fmt_money(mid_val)}<br>"
            f"Spread % Mid: {fmt_pct(spread_pct)}<br>"
            f"Last: {fmt_money(last_price)}<br>"
            f"Bid: {fmt_money(bid)}<br>"
            f"Ask: {fmt_money(ask)}<br>"
            f"Volume: {volume}<br>"
            f"Open Interest: {oi}<br>"
            f"ITM: {itm}<br>"
            f"{status_html}"
        )
    return texts


# ============================================================
# 2D SLICE HELPERS
# ============================================================
def resolve_slice_expiration(all_df: pd.DataFrame, slice_target: str) -> str:
    expirations = sorted(all_df["expiration"].astype(str).unique().tolist())
    if not expirations:
        raise ValueError("No expirations available for slice figure.")

    target = str(slice_target).strip()
    if not target or target.lower() == "auto":
        counts = all_df.groupby("expiration").size().sort_values(ascending=False)
        return str(counts.index[0])

    if target in expirations:
        return target

    try:
        target_dte = float(target)
        dte_map = all_df.groupby("expiration")["dte"].median()
        best_exp = (dte_map - target_dte).abs().idxmin()
        return str(best_exp)
    except ValueError:
        pass

    try:
        ts = pd.Timestamp(target).strftime("%Y-%m-%d")
        if ts in expirations:
            return ts
    except Exception:
        pass

    counts = all_df.groupby("expiration").size().sort_values(ascending=False)
    return str(counts.index[0])



def _resolve_iv_field_name(source_name: str) -> str:
    mapping = {
        "vendor": "impliedVolatility",
        "bid": "bid_solved_iv",
        "mid": "mid_solved_iv",
        "ask": "ask_solved_iv",
        "plane": "gmm_fitted_iv",
        "gmm": "gmm_fitted_iv",
    }
    if source_name == "display":
        return _resolve_iv_field_name(SLICE_DISPLAY_IV_SOURCE)
    return mapping.get(source_name, "gmm_fitted_iv")



def build_slice_curve(slice_expiration: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    model = FITTED_MODELS.get(str(slice_expiration))
    if model is None:
        return None, None
    strikes = np.linspace(
        float(model["usable_df"]["strike"].min()),
        float(model["usable_df"]["strike"].max()),
        FIT_DENSE_SLICE_POINTS,
    )
    return _evaluate_slice_curve_iv(str(slice_expiration), strikes)



def _compute_slice_y_range(slice_plane: pd.DataFrame, slice_all: pd.DataFrame, y_curve: np.ndarray | None) -> list[float]:
    iv_field = _resolve_iv_field_name(SLICE_CURVE_IV_SOURCE)
    if not slice_plane.empty:
        y_source = slice_plane[iv_field].to_numpy(dtype=float)
    else:
        y_source = slice_all[iv_field].to_numpy(dtype=float)

    y_source = y_source[np.isfinite(y_source)]

    if y_curve is not None:
        curve_vals = np.asarray(y_curve, dtype=float)
        curve_vals = curve_vals[np.isfinite(curve_vals)]
        if curve_vals.size > 0:
            y_source = np.concatenate([y_source, curve_vals]) if y_source.size > 0 else curve_vals

    if y_source.size == 0:
        return [0.0, 1.0]

    y_low_base = float(np.min(y_source))
    y_high_base = float(np.quantile(y_source, SLICE_Y_UPPER_PERCENTILE))

    y_low = max(0.0, y_low_base * (1.0 - SLICE_Y_LOWER_PAD_PCT))
    y_high = y_high_base * (1.0 + SLICE_Y_UPPER_PAD_PCT)

    if y_high <= y_low:
        y_high = max(y_low + 0.05, float(np.max(y_source)) * 1.05)

    return [y_low, y_high]



def make_slice_figure(all_df: pd.DataFrame, plane_df: pd.DataFrame, faint_df: pd.DataFrame, spot: float, ticker_symbol: str, slice_expiration: str) -> go.Figure:
    slice_all = all_df[all_df["expiration"] == slice_expiration].copy().sort_values("strike")
    slice_plane = plane_df[plane_df["expiration"] == slice_expiration].copy().sort_values("strike")
    slice_faint = faint_df[faint_df["expiration"] == slice_expiration].copy().sort_values("strike")

    if slice_all.empty:
        raise ValueError(f"No contracts found for slice expiration {slice_expiration}.")

    x_curve_strike, y_curve = build_slice_curve(slice_expiration)
    if x_curve_strike is not None and X_AXIS_MODE == "moneyness":
        x_curve = x_curve_strike / float(spot)
    else:
        x_curve = x_curve_strike

    slice_display_iv_field = _resolve_iv_field_name(SLICE_DISPLAY_IV_SOURCE)

    fig = go.Figure()

    if SLICE_SHOW_EXCLUDED_POINTS and not slice_faint.empty:
        fig.add_trace(
            go.Scatter(
                x=_get_x_series(slice_faint),
                y=slice_faint[slice_display_iv_field],
                mode="markers",
                name="Residual Outliers",
                text=make_hover_text(slice_faint, include_reason=True),
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    size=8,
                    color=slice_faint[slice_display_iv_field],
                    colorscale=COLOR_SCALE,
                    opacity=0.20,
                    line=dict(width=0),
                    showscale=False,
                ),
            )
        )

    if not slice_plane.empty:
        fig.add_trace(
            go.Scatter(
                x=_get_x_series(slice_plane),
                y=slice_plane[slice_display_iv_field],
                mode="markers",
                name="Fitted Quotes",
                text=make_hover_text(slice_plane, include_reason=False),
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    size=9,
                    color=slice_plane[slice_display_iv_field],
                    colorscale=COLOR_SCALE,
                    opacity=0.98,
                    line=dict(width=0),
                    showscale=False,
                ),
            )
        )

    if x_curve is not None and y_curve is not None:
        fig.add_trace(
            go.Scatter(
                x=x_curve,
                y=y_curve,
                mode="lines",
                name="Model-Fitted Slice",
                line=dict(width=4),
                hovertemplate=f"{_get_x_label()}: %{{x:.3f}}<br>GMM Fitted IV: %{{y:.2%}}<extra></extra>",
            )
        )

    if SLICE_MARK_SPOT_LINE and X_AXIS_MODE == "moneyness":
        fig.add_vline(x=1.0, line_width=2, line_dash="dash", annotation_text="ATM", annotation_position="top")

    median_dte = float(slice_all["dte"].median())
    y_range = _compute_slice_y_range(slice_plane, slice_all, y_curve)
    x_vals_source = _get_x_series(slice_plane if not slice_plane.empty else slice_all).to_numpy(dtype=float)
    x_range = _padded_range(x_vals_source, 0.08)

    fig.update_layout(
        title=(
            f"{ticker_symbol} {APP_NAME} {OPTION_SIDE.capitalize()} Expiration Slice"
            f"<br><sup>Expiration = {slice_expiration} | Median DTE = {median_dte:.1f} | Model points = {len(slice_plane):,} | Residual outliers = {len(slice_faint):,}</sup>"
        ),
        template=PLOT_THEME,
        margin=dict(l=60, r=30, b=60, t=85),
        xaxis=dict(
            title=_get_x_label(),
            range=x_range,
            showgrid=SLICE_SHOW_GRID,
            gridcolor=SLICE_GRID_COLOR,
            zeroline=True,
            zerolinecolor=SLICE_ZERO_LINE_COLOR,
        ),
        yaxis=dict(
            title="Implied Volatility",
            tickformat=".0%",
            range=y_range,
            showgrid=SLICE_SHOW_GRID,
            gridcolor=SLICE_GRID_COLOR,
            zeroline=True,
            zerolinecolor=SLICE_ZERO_LINE_COLOR,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )

    return fig


# ============================================================
# PLOT
# ============================================================
def make_figure(plane_df: pd.DataFrame, faint_df: pd.DataFrame, spot: float, ticker_symbol: str, max_dte: int) -> go.Figure:
    X, Y, Z = build_smooth_iv_surface(plane_df)
    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale=COLOR_SCALE,
            opacity=SURFACE_OPACITY,
            colorbar=dict(title="Implied Vol"),
            hovertemplate=(
                f"{_get_x_label()}: %{{x:.3f}}<br>"
                "DTE: %{y:.1f}<br>"
                "Model-Fitted IV: %{z:.2%}<extra></extra>"
            ),
            name="IV3D Surface",
            showscale=True,
        )
    )

    if SHOW_EXCLUDED_POINTS and not faint_df.empty and (not FOCUS_ON_TRUSTED_VIEW or SHOW_EXCLUDED_POINTS_IN_FOCUS):
        fig.add_trace(
            go.Scatter3d(
                x=_get_x_series(faint_df),
                y=faint_df["dte"],
                z=faint_df["gmm_fitted_iv"],
                mode="markers",
                name="Residual Outliers",
                text=make_hover_text(faint_df, include_reason=True),
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    size=3,
                    color=faint_df["gmm_fitted_iv"],
                    colorscale=COLOR_SCALE,
                    opacity=0.15,
                    showscale=False,
                    line=dict(width=0),
                ),
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=_get_x_series(plane_df),
            y=plane_df["dte"],
            z=plane_df["gmm_fitted_iv"],
            mode="markers",
            name="Fitted Contracts",
            text=make_hover_text(plane_df, include_reason=False),
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                size=4,
                color=plane_df["gmm_fitted_iv"],
                colorscale=COLOR_SCALE,
                opacity=0.98,
                showscale=False,
                line=dict(width=0),
            ),
        )
    )

    scene = dict(
        xaxis=dict(
            title=_get_x_label(),
            showbackground=True,
            backgroundcolor=PLANE_BACKGROUND_COLOR,
            gridcolor=PLANE_GRID_COLOR,
            showgrid=True,
            zerolinecolor=PLANE_GRID_COLOR,
        ),
        yaxis=dict(
            title="Days to Expiration",
            showbackground=True,
            backgroundcolor=PLANE_BACKGROUND_COLOR,
            gridcolor=PLANE_GRID_COLOR,
            showgrid=True,
            zerolinecolor=PLANE_GRID_COLOR,
        ),
        zaxis=dict(
            title="Implied Volatility",
            tickformat=".0%",
            showbackground=True,
            backgroundcolor=PLANE_BACKGROUND_COLOR,
            gridcolor=PLANE_GRID_COLOR,
            showgrid=True,
            zerolinecolor=PLANE_GRID_COLOR,
        ),
        camera=dict(eye=CAMERA_EYE),
    )

    if PLANE_SQUARE_GRIDS:
        scene["aspectmode"] = "cube"

    if FOCUS_ON_TRUSTED_VIEW:
        x_range, y_range, z_range = compute_focus_ranges(plane_df)
        scene["xaxis"]["range"] = x_range
        scene["yaxis"]["range"] = y_range
        scene["zaxis"]["range"] = z_range

    fig.update_layout(
        title=(
            f"{ticker_symbol} {APP_NAME} Implied Volatility Plane"
            f"<br><sup>Spot = ${spot:.2f} | Max DTE = {max_dte} | X Axis = {X_AXIS_MODE}"
            f"{(' | Surface preset = ' + str(SURFACE_MONEYNESS_PRESET) + ' | Surface moneyness band = ' + str(SURFACE_MIN_MONEYNESS_DISPLAY) + ' to ' + str(SURFACE_MAX_MONEYNESS_DISPLAY)) if X_AXIS_MODE == 'moneyness' and SURFACE_USE_MONEYNESS_DISPLAY_BAND else ''}"
            f" | Surface points = {len(plane_df):,} | Residual outliers = {len(faint_df):,} | Smoothing = {SMOOTHING:.2f}</sup>"
        ),
        template=PLOT_THEME,
        margin=dict(l=0, r=0, b=0, t=80),
        scene=scene,
    )

    return fig


# ============================================================
# SUMMARY
# ============================================================
def print_summary(raw_df: pd.DataFrame, all_df: pd.DataFrame, plane_df: pd.DataFrame, faint_df: pd.DataFrame, spot: float, ticker_symbol: str, max_dte: int, slice_expiration: str) -> None:
    print("=" * 100)
    print(f"APPLICATION: {APP_NAME}")
    print(f"VERSION: {APP_VERSION}")
    print(f"TICKER: {ticker_symbol}")
    print(f"SPOT PRICE: ${spot:.2f}")
    print(f"MAX DTE CHOSEN: {max_dte}")
    print(f"MIN DTE INCLUDED: {MIN_DTE}")
    print(f"OPTION SIDE: {OPTION_SIDE}")
    print(f"RAW CONTRACTS PULLED: {len(raw_df):,}")
    print(f"REAL CONTRACTS AFTER MINIMAL CLEANING: {len(all_df):,}")
    print(f"GMM SURFACE POINTS: {len(plane_df):,}")
    print(f"GMM RESIDUAL OUTLIERS: {len(faint_df):,}")
    print(f"X AXIS MODE: {X_AXIS_MODE}")
    if X_AXIS_MODE == "moneyness" and SURFACE_USE_MONEYNESS_DISPLAY_BAND:
        print(f"SURFACE MONEYNESS PRESET: {SURFACE_MONEYNESS_PRESET}")
        print(f"SURFACE MONEYNESS DISPLAY BAND: {SURFACE_MIN_MONEYNESS_DISPLAY:.2f} to {SURFACE_MAX_MONEYNESS_DISPLAY:.2f}")
    print(f"FOCUSED VIEW: {FOCUS_ON_TRUSTED_VIEW}")
    print(f"SURFACE SMOOTHING: {SMOOTHING:.2f}")
    print(f"SLICE EXPIRATION SELECTED: {slice_expiration}")
    print(f"MIXTURE COMPONENTS: {MIXTURE_COMPONENTS}")
    print(f"FIT PRICE SOURCE: {FIT_PRICE_SOURCE}")
    print(f"FIT MULTI-STARTS: {FIT_MULTI_STARTS}")
    print(f"OUTLIER ABS THRESHOLD: {FIT_OUTLIER_ABS_THRESHOLD:.3f}")
    print(f"OUTLIER REL THRESHOLD: {FIT_OUTLIER_REL_THRESHOLD:.1%}")
    if FOCUS_ON_TRUSTED_VIEW and len(plane_df) > 0:
        x_range, y_range, z_range = compute_focus_ranges(plane_df)
        print(f"FOCUS X RANGE: {x_range[0]:.3f} to {x_range[1]:.3f}")
        print(f"FOCUS DTE RANGE: {y_range[0]:.2f} to {y_range[1]:.2f}")
        print(f"FOCUS Z RANGE: {z_range[0]:.2%} to {z_range[1]:.2%}")
    print(f"RISK-FREE SOURCE: {LIVE_TREASURY_INFO.get('source', 'fallback_flat')} | as of {LIVE_TREASURY_INFO.get('as_of', 'N/A')}")
    print(f"FALLBACK FLAT RATE: {REFERENCE_RISK_FREE_RATE:.2%}")
    print("PLANE IV SOURCE: Model-fitted prices -> implied vols")
    print("MODEL TYPE: Mixture-of-Black-Scholes")
    print(f"COMPARISON PRICE SOURCE: {COMPARISON_PRICE_SOURCE}")

    if FITTED_MODELS:
        print("\nFitted expirations summary:")
        for exp in sorted(FITTED_MODELS.keys()):
            model = FITTED_MODELS[exp]
            w = model.get("weights", [])
            w_txt = ", ".join(f"{float(x):.1%}" for x in w)
            comp_txt = int(model.get('n_components', MIXTURE_COMPONENTS))
            print(f"  - {exp}: comps={comp_txt} | objective={float(model.get('objective', np.nan)):.4f} | weights={w_txt}")

    if not faint_df.empty:
        print("\nResidual outlier breakdown by expiration:")
        for exp, count in faint_df.groupby("expiration").size().items():
            print(f"  - {exp}: {int(count):,}")

    print("=" * 100)


# ============================================================
# DOUBLE-CLICK / EXIT HELPERS
# ============================================================
def safe_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def pause_before_exit() -> None:
    if not FORCE_PAUSE_ON_EXIT:
        return
    try:
        input("\nPress Enter to close this window...")
    except EOFError:
        pass


def safe_write_html(fig, path: Path, auto_open: bool = False, label: str = "HTML") -> Path:
    """
    Save a full-page responsive HTML so the chart fills the browser window with less clutter.
    Falls back to a timestamped filename if the target file is locked.
    """
    import webbrowser

    html_fragment = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displaylogo": False},
    )

    full_page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{label}</title>
<style>
html, body {{
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: #000;
}}
.plot-wrap {{
    width: 100vw;
    height: 100vh;
}}
.plot-wrap > div {{
    width: 100% !important;
    height: 100% !important;
}}
</style>
</head>
<body>
    <div class="plot-wrap">
        {html_fragment}
    </div>
</body>
</html>
"""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(full_page, encoding="utf-8")
        saved_path = path
    except PermissionError:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(full_page, encoding="utf-8")
        print(f"\n{label} file was locked/open, so it was saved to a new filename instead:")
        print(f"  {fallback.resolve()}")
        saved_path = fallback

    if auto_open:
        webbrowser.open(saved_path.resolve().as_uri())

    return saved_path


def export_diagnostic_csvs(all_df: pd.DataFrame, plane_df: pd.DataFrame, faint_df: pd.DataFrame, ticker_symbol: str, max_dte: int) -> None:
    if not EXPORT_DIAGNOSTIC_CSVS:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_out = all_df.copy()
    all_out["point_status"] = "All Cleaned Rows"

    plane_out = plane_df.copy()
    plane_out["point_status"] = "Trusted"

    faint_out = faint_df.copy()
    faint_out["point_status"] = "Excluded"

    all_path = OUTPUT_DIR / f"{ticker_symbol.lower()}_{OPTION_SIDE}_input_rows_{APP_TAG}_{max_dte}dte.csv"
    plane_path = OUTPUT_DIR / f"{ticker_symbol.lower()}_{OPTION_SIDE}_surface_points_{APP_TAG}_{max_dte}dte.csv"
    faint_path = OUTPUT_DIR / f"{ticker_symbol.lower()}_{OPTION_SIDE}_residual_outliers_{APP_TAG}_{max_dte}dte.csv"

    all_out.to_csv(all_path, index=False)
    plane_out.to_csv(plane_path, index=False)
    faint_out.to_csv(faint_path, index=False)

    print(f"Saved cleaned rows CSV: {all_path.resolve()}")
    print(f"Saved surface points CSV: {plane_path.resolve()}")
    print(f"Saved residual outliers CSV: {faint_path.resolve()}")


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    global OPTION_SIDE, LIVE_TREASURY_INFO, FITTED_MODELS
    ticker_symbol, max_dte, slice_target, option_side, surface_preset = get_inputs_from_user()
    OPTION_SIDE = option_side

    LIVE_TREASURY_INFO = load_live_treasury_curve()
    print(LIVE_TREASURY_INFO.get("message", ""))

    raw_df, spot = fetch_option_points(ticker_symbol, max_dte)
    cleaned_df = lightly_clean_option_points(raw_df)

    if cleaned_df.empty:
        raise ValueError("No usable option rows survived the minimal sanity cleaning.")

    fit_universe_df = apply_fit_universe_moneyness_band(cleaned_df)
    if fit_universe_df.empty:
        raise ValueError(
            "Fit-universe moneyness band removed every cleaned row. "
            "Widen SURFACE_MIN_MONEYNESS_DISPLAY / SURFACE_MAX_MONEYNESS_DISPLAY."
        )

    if X_AXIS_MODE == "moneyness" and SURFACE_USE_MONEYNESS_DISPLAY_BAND:
        removed_rows = len(cleaned_df) - len(fit_universe_df)
        print(
            f"Applied fit-universe moneyness band before IV solving/fitting: "
            f"{SURFACE_MIN_MONEYNESS_DISPLAY:.2f} to {SURFACE_MAX_MONEYNESS_DISPLAY:.2f} | "
            f"kept {len(fit_universe_df):,} of {len(cleaned_df):,} cleaned rows "
            f"({removed_rows:,} removed)."
        )

    all_df = assign_risk_free_rates(fit_universe_df)
    all_df = enrich_with_solved_iv_columns(all_df)
    plane_df, faint_df, FITTED_MODELS = fit_all_expirations_with_gmm(all_df)
    slice_expiration = resolve_slice_expiration(plane_df if not plane_df.empty else all_df, slice_target)

    surface_plane_df = apply_surface_display_band(plane_df)
    surface_faint_df = apply_surface_display_band(faint_df)
    if len(surface_plane_df) < FIT_SURFACE_MIN_POINTS:
        raise ValueError(
            "Surface display band removed too many fitted points. "
            "Widen SURFACE_MIN_MONEYNESS_DISPLAY / SURFACE_MAX_MONEYNESS_DISPLAY."
        )

    print_summary(raw_df, all_df, surface_plane_df, surface_faint_df, spot, ticker_symbol, max_dte, slice_expiration)

    fig = make_figure(surface_plane_df, surface_faint_df, spot, ticker_symbol, max_dte)

    slice_fig = None
    if MAKE_SLICE_FIGURE:
        slice_fig = make_slice_figure(all_df, plane_df, faint_df, spot, ticker_symbol, slice_expiration)

    if SAVE_HTML:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        html_path = OUTPUT_DIR / f"{ticker_symbol.lower()}_{OPTION_SIDE}_iv3d_plane_{APP_TAG}_{max_dte}dte.html"
        saved_plane_path = safe_write_html(fig, html_path, auto_open=AUTO_OPEN_PLANE, label="Plane")
        print(f"\nSaved interactive IV plane to: {saved_plane_path.resolve()}")

        if slice_fig is not None:
            slice_html_path = OUTPUT_DIR / f"{ticker_symbol.lower()}_{OPTION_SIDE}_iv3d_slice_{APP_TAG}_{slice_expiration}.html"
            saved_slice_path = safe_write_html(slice_fig, slice_html_path, auto_open=AUTO_OPEN_SLICE, label="Slice")
            print(f"Saved interactive IV slice to: {saved_slice_path.resolve()}")

    if MAKE_SLICE_FIGURE:
        print("\n" + "=" * 100)
        print(f"{APP_NAME} STRIKE COMPARISON")
        print(f"Selected slice expiration: {slice_expiration}")
        print("After the graphs open, come back to this window to compare strikes.")
        print("Example input: 9, 9.5")
        print("=" * 100)
        interactive_slice_compare_loop(all_df, plane_df, faint_df, ticker_symbol, slice_expiration)

    export_diagnostic_csvs(all_df, plane_df, faint_df, ticker_symbol, max_dte)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        pause_before_exit()