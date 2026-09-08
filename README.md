# Implied Volatility Surface

I built this project because I wanted to understand implied volatility across an entire option chain instead of only solving IV for one contract at a time.

The project pulls live option data, cleans the quotes, fits each expiration, and turns the fitted prices back into implied volatilities. It then shows the result as an interactive 3D IV plane and a 2D expiration slice.

The full calibration model is still part of the project. I kept it because that was a large part of what I was trying to learn.

## Sample outputs

### IV plane

![3D implied volatility surface](images/iv_surface_3d.png)

### Expiration slice

![Implied volatility expiration slice](images/expiration_slice.png)

## What the model does

For each expiration, the program can fit a one-, two-, or three-component mixture of Black-Scholes prices. The fit allows the component weights, forward levels, and volatilities to move so the model can match an observed option chain more closely than a single flat-vol Black-Scholes model.

The calibration uses multiple starting points and constrained optimization. Quote quality and bid-ask spreads affect the weights, and the loss function is made less sensitive to extreme residuals so a few bad quotes do not completely control the fit.

After calibration, the fitted option prices are inverted through a Black-Scholes IV solver. Those fitted IV observations are then used to build the surface.

## What is included

- live option chains and spot prices through `yfinance`
- calls or puts over a configurable expiration range
- short-term Treasury rates, with a fallback if the live rate source fails
- bid, ask, mid, and vendor-IV handling
- Black-Scholes pricing and IV inversion
- one- to three-component mixture-of-Black-Scholes calibration by expiration
- multi-start optimization with `L-BFGS-B` and `Powell`
- bid-ask-aware weighting and robust residual loss
- residual-outlier filtering
- interactive 3D IV surface by moneyness or strike
- 2D expiration slice with ATM reference
- strike comparison tables with Greeks and liquidity information
- projected no-move time-decay calculations
- optional diagnostic CSV exports
- interactive HTML graph exports

## Run it

```bash
python -m pip install -r requirements.txt
python iv_surface.py
```

You can also pass inputs on the command line:

```bash
python iv_surface.py CLSK 120 auto calls focused
```

The program can ask for the ticker, maximum DTE, target expiration slice, calls or puts, and the moneyness view.

## Outputs

A normal run can produce:

- an interactive 3D implied-volatility plane
- an interactive expiration slice
- strike-level comparison tables in the terminal
- optional cleaned-contract, fitted-surface, and outlier CSVs

The HTML graphs can be reopened later and rotated or zoomed in the browser.

## Tests

```bash
python -m unittest test_iv_engine.py
```

The tests cover option-side parsing, Black-Scholes call-put parity, IV inversion, moneyness presets, and monotonic mixture call prices.

## Files

```text
iv_surface.py       small file used to start the program
iv_engine.py        pricing, calibration, data work, tables, and graphs
test_iv_engine.py   deterministic model tests
requirements.txt
images/             sample IV plane and expiration slice
```

## Limits

This is a learning project, not a production volatility surface. The fit depends heavily on the quality of the option quotes going into it, especially when contracts are illiquid or spreads are wide. Each expiration is fitted separately, so the final surface is not guaranteed to be fully arbitrage-free across both strike and maturity. The surface interpolation is mainly for analysis and visualization, not execution or production pricing.
