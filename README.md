# Stock Market Investment Timing Calculator

A comprehensive analysis tool for Bitcoin and S&P 500 investment strategies using Kelly Criterion optimization and Monte Carlo simulations.

## 📁 Project Structure

```
Stock-Market-Investment-Timing-Calculator/
├── Data Sets/                     # Raw and cleaned data files
│   ├── Bitcoin Data/             # Bitcoin historical price data
│   └── S&P 500 Data Sets/        # S&P 500 historical price data
│
├── Scripts/                      # All analysis and simulation scripts
│   ├── Bitcoin/                  # Bitcoin-specific analysis scripts
│   │   ├── bitcoin_lumpsum_growth_model_simulation.py
│   │   ├── bitcoin_monte_carlo_fixed.py
│   │   ├── bitcoin_monte_carlo_price_paths.py
│   │   └── bitcoin_lump_sum_summary.py
│   │
│   ├── SP500/                    # S&P 500 analysis scripts
│   │   ├── sp500_lumpsum_simulation.py
│   │   └── sp500_monte_carlo_simulation.py
│   │
│   ├── Portfolio/                # Portfolio optimization scripts
│   │   ├── kelly_criterion_portfolio.py
│   │   ├── no_leverage_kelly.py
│   │   └── three_asset_kelly.py
│   │
│   └── Data_Cleaning/            # Data preprocessing scripts
│       ├── Bitcoin Data cleaning/
│       └── Stock Market Data Cleaning/
│
├── Results/                      # Analysis results and outputs
│   ├── Bitcoin/                  # Bitcoin analysis results
│   │   ├── bitcoin_lumpsum_formula_summary.txt
│   │   ├── bitcoin_monte_carlo_lump_sum_results.csv
│   │   ├── bitcoin_volatility_analysis.pdf
│   │   └── *.png (charts and visualizations)
│   │
│   ├── SP500/                    # S&P 500 analysis results
│   │   ├── sp500_lumpsum_horizon_summary.txt
│   │   ├── sp500_monte_carlo_paths.csv
│   │   └── *.png (volatility charts)
│   │
│   └── Portfolio/                # Portfolio optimization results
│       ├── no_leverage_kelly_results.txt
│       └── three_asset_kelly_results.txt
│
├── Models/                       # Growth and volatility models
│   ├── Growth Models/            # Asset growth model coefficients
│   │   ├── bitcoin_growth_model_coefficients.txt
│   │   ├── bitcoin_growth_model_fit.py
│   │   ├── sp500_growth_model_coefficients.txt
│   │   └── sp500_growth_model_fit.py
│   │
│   └── Volatility Models/        # Volatility model coefficients
│       ├── bitcoin_volatility_model_coefficients.txt
│       ├── bitcoin_volatility_model_fit.py
│       └── bitcoin_volatility_inverse_model_coefficients.txt
│
├── Analysis/                     # Additional analysis scripts
│   └── analyze_bitcoin_volatility.py
│
├── Visualizations/               # Visualization scripts
│   └── bitcoin_rainbow_chart.py
│
└── README.md                     # This file
```

## 🚀 Key Features

### Growth Models
- **Bitcoin**: Logarithmic growth model with 94% R² fit
- **S&P 500**: Exponential growth model with 94% R² fit

### Investment Strategies
- **Lump Sum Analysis**: Pure growth model projections
- **Monte Carlo Simulations**: Risk-adjusted return projections
- **Kelly Criterion Optimization**: Optimal portfolio allocation

### Portfolio Optimization
- **No-Leverage Kelly**: Conservative allocation without borrowing
- **Three-Asset Kelly**: Bitcoin, S&P 500, and High-Yield Savings optimization
- **Risk-Adjusted Returns**: Sharpe ratio optimization

## 📊 Key Results

### Optimal Portfolio Allocation (No Leverage)
- **Bitcoin**: 40.3%
- **S&P 500**: 59.7%
- **Expected Return**: 13.23%
- **Volatility**: 28.53%

### Growth Projections (10-Year)
- **Bitcoin**: 21.31% CAGR (decreasing over time)
- **S&P 500**: 7.78% CAGR (steady)

## 🔧 Usage

### Running Simulations
```bash
# Bitcoin lump sum analysis
python Scripts/Bitcoin/bitcoin_lumpsum_growth_model_simulation.py

# Portfolio optimization
python Scripts/Portfolio/no_leverage_kelly.py

# Monte Carlo simulations
python Scripts/Bitcoin/bitcoin_monte_carlo_fixed.py
```

### Key Files
- **Growth Models**: `Models/Growth Models/`
- **Portfolio Results**: `Results/Portfolio/no_leverage_kelly_results.txt`
- **Simulation Results**: `Results/Bitcoin/` and `Results/SP500/`

## 📈 Investment Insights

1. **Optimal allocation balances Bitcoin's high returns with S&P 500's stability**
2. **Kelly Criterion suggests 40/60 Bitcoin/S&P 500 split for maximum risk-adjusted returns**
3. **No leverage strategy provides 13.23% expected returns with managed risk**
4. **Monte Carlo simulations show realistic return distributions with volatility**

## 🛠️ Dependencies
- Python 3.x
- NumPy
- Pandas
- SciPy
- Matplotlib

---
*Last updated: [Current Date]* 