# Implied Volatility Surface

I built this project to get more comfortable with Black-Scholes and to see how implied volatility changes across strikes and expirations instead of looking at one option at a time.

The script downloads a live option chain, uses market prices to solve for implied volatility, and plots the results as a 3D surface and a single-expiration slice.

## Why I built it

I understood the idea of implied volatility before I built this, but I wanted to see what it looked like across an actual option chain. The useful part for me was working backward from an option price to volatility and then seeing the smile/skew show up across strike and time.

## What it does

- downloads calls or puts with `yfinance`
- uses the latest stock price as spot
- uses the 13-week Treasury yield as a simple risk-free-rate input when available
- calculates mid prices from bid and ask quotes
- solves Black-Scholes implied volatility contract by contract
- plots IV against moneyness and days to expiration
- shows a 2D expiration slice
- prints a small near-ATM table with Greeks

## Run it

```bash
pip install -r requirements.txt
python iv_surface.py
```

You can also pass the inputs directly:

```bash
python iv_surface.py CLSK 120 calls focused
```

The last argument controls the moneyness window: `focused`, `standard`, or `wide`.

## Tests

```bash
python -m unittest test_iv_engine.py
```

The tests check Black-Scholes call-put parity, implied-volatility inversion, the option-side parser, and basic Greeks.

## Files

```text
iv_surface.py       data loading, Black-Scholes, IV solving, and charts
test_iv_engine.py   small deterministic test suite
requirements.txt
```

## Limits

This is a learning project, not a production volatility model. It uses a basic Black-Scholes setup with no dividend yield, depends on the quality of Yahoo Finance option quotes, and the plotted surface is an interpolation of observed contracts rather than a fully arbitrage-free calibrated volatility surface.
