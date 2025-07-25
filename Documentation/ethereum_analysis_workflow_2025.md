# Ethereum Analysis Workflow 2025

## Overview
This document outlines the comprehensive analysis workflow for Ethereum (ETH) investment strategies, including growth modeling, volatility analysis, and Monte Carlo simulations.

## Directory Structure
```
Etherium/
├── Data/
│   └── Ethereum Historical Data.csv
├── Models/
│   ├── Growth/
│   │   ├── Scripts/
│   │   │   └── ethereum_growth_model_fit.py
│   │   └── Formulas/
│   │       └── ethereum_growth_model_coefficients.txt
│   └── Volatility/
│       ├── Scripts/
│       │   ├── ethereum_volatility_model_fit.py
│       │   └── ethereum_365day_volatility_decay.py
│       └── Formulas/
│           ├── ethereum_volatility_model_coefficients.txt
│           └── ethereum_365day_volatility_decay_coefficients.txt
├── Results/
│   └── ethereum_gbm_paths_YYYYMMDD_HHMMSS.csv
├── simulation/
│   └── ethereum_gbm_simple_save.py
└── Visualizations/
    ├── Scripts/
    │   ├── ethereum_bitcoin_comparison.py
    │   ├── ethereum_bitcoin_comparison_log.py
    │   └── rainbow_chart_comparison.py
    └── Images/
        ├── ethereum_rainbow_chart_day365_autoflat.png
        ├── ethereum_bitcoin_comparison.png
        ├── ethereum_bitcoin_comparison_log.png
        ├── rainbow_chart_comparison.png
        ├── ethereum_volatility_decay_models.png
        └── ethereum_365day_volatility_decay.png
```

## Core Models

### 1. Growth Model (Rainbow Chart)
**Script**: `Etherium/Models/Growth/Scripts/ethereum_growth_model_fit.py`
**Output**: `Etherium/Models/Growth/Formulas/ethereum_growth_model_coefficients.txt`

**Model Details**:
- **Start Date**: Day 365 (2016-03-10)
- **Formula**: `log10(price) = slope * ln(day) + intercept`
- **Purpose**: Determines fair value based on Ethereum's age
- **Usage**: Starting point for GBM simulations

**Key Parameters**:
- Growth follows power law relationship
- Model starts from day 365 to avoid early volatility
- Used to calculate fair value for investment strategies

### 2. Volatility Models

#### A. Multi-Window Volatility Analysis
**Script**: `Etherium/Models/Volatility/Scripts/ethereum_volatility_model_fit.py`
**Output**: `Etherium/Models/Volatility/Formulas/ethereum_volatility_model_coefficients.txt`

**Purpose**: Analyze how volatility changes with different time windows (7, 14, 30, 60, 90, 180, 365 days)

**Results**:
- 7-day volatility: 81.88%
- 14-day volatility: 84.96%
- 30-day volatility: 87.48%
- 60-day volatility: 89.30%
- 90-day volatility: 90.15%
- 180-day volatility: 90.84%
- 365-day volatility: 92.21%

**Best Model**: Power Law (R² = 0.943)
- Formula: `log(volatility) = 0.029022 * log(window) + 4.363187`

#### B. 365-Day Volatility Decay Over Time
**Script**: `Etherium/Models/Volatility/Scripts/ethereum_365day_volatility_decay.py`
**Output**: `Etherium/Models/Volatility/Formulas/ethereum_365day_volatility_decay_coefficients.txt`

**Purpose**: Analyze how Ethereum's 365-day volatility changes as the asset matures

**Key Findings**:
- **Current Age**: 9.4 years (as of 2025)
- **Current 365-day Volatility**: 74.6%
- **Volatility Range**: 46.1% to 145.5%
- **Age Range**: 1.0 to 9.4 years

**Best Model**: Linear (R² = 0.735)
- Formula: `volatility = -8.492547 * age + 136.243807`

**For GBM**: Using Exponential Model (R² = 0.698)
- Formula: `log(volatility) = -0.097430 * age + 4.991914`
- Prevents negative volatility in future predictions

## Monte Carlo Simulation

### GBM Simulation
**Script**: `Etherium/simulation/ethereum_gbm_simple_save.py`
**Output**: `Etherium/Results/ethereum_gbm_paths_YYYYMMDD_HHMMSS.csv`

**Simulation Parameters**:
- **Starting Price**: Fair value from growth model
- **Time Horizon**: 10 years
- **Number of Paths**: 100 (quick test) or 1000 (full simulation)
- **Time Steps**: Daily (3653 steps)
- **Update Frequency**: Every day

**Dynamic Parameters**:
1. **Growth Rate**: Based on growth model fair value
   - Current fair value vs. future fair value
   - Formula: `mu = (future_fair_value / current_fair_value) ** (1/time) - 1`

2. **Volatility**: Based on exponential decay model
   - Formula: `sigma = exp(vol_slope * future_age + vol_intercept)`
   - Decays as Ethereum matures

**Key Features**:
- Immediate file saving (memory efficient)
- Dynamic parameter updates
- Progress tracking
- Comprehensive statistics

## Visualization Scripts

### 1. Price History Comparison
**Script**: `Etherium/Visualizations/Scripts/ethereum_bitcoin_comparison.py`
**Output**: `Etherium/Visualizations/Images/ethereum_bitcoin_comparison.png`

**Features**:
- Full Ethereum price history
- Bitcoin price history (2010-2020)
- Side-by-side comparison
- Linear scale

### 2. Log Scale Comparison
**Script**: `Etherium/Visualizations/Scripts/ethereum_bitcoin_comparison_log.py`
**Output**: `Etherium/Visualizations/Images/ethereum_bitcoin_comparison_log.png`

**Features**:
- Base-10 logarithmic scale
- Better visualization of exponential growth
- Same data as linear version

### 3. Rainbow Chart Comparison
**Script**: `Etherium/Visualizations/Scripts/rainbow_chart_comparison.py`
**Output**: `Etherium/Visualizations/Images/rainbow_chart_comparison.png`

**Features**:
- Side-by-side rainbow charts
- Bitcoin starts from day 365
- Ethereum starts from day 365
- Model information displayed

## Key Differences from Bitcoin

### 1. Age and Maturity
- **Ethereum**: ~9.4 years old (as of 2025)
- **Bitcoin**: ~15 years old
- **Impact**: Ethereum shows higher volatility and faster growth

### 2. Volatility Characteristics
- **Ethereum**: Higher current volatility (74.6% vs ~60% for Bitcoin)
- **Decay Rate**: Faster volatility decay as Ethereum matures
- **Model**: Exponential decay preferred over linear to prevent negative values

### 3. Growth Model
- **Start Point**: Both use day 365 to avoid early volatility
- **Growth Rate**: Ethereum shows different growth characteristics
- **Fair Value**: Different baseline for investment strategies

## Usage Workflow

### 1. Initial Setup
```bash
# Run growth model
python Etherium/Models/Growth/Scripts/ethereum_growth_model_fit.py

# Run volatility analysis
python Etherium/Models/Volatility/Scripts/ethereum_volatility_model_fit.py
python Etherium/Models/Volatility/Scripts/ethereum_365day_volatility_decay.py
```

### 2. Generate Visualizations
```bash
# Create price comparisons
python Etherium/Visualizations/Scripts/ethereum_bitcoin_comparison.py
python Etherium/Visualizations/Scripts/ethereum_bitcoin_comparison_log.py
python Etherium/Visualizations/Scripts/rainbow_chart_comparison.py
```

### 3. Run Monte Carlo Simulation
```bash
# Run GBM simulation
python Etherium/simulation/ethereum_gbm_simple_save.py
```

## Investment Strategy Applications

### 1. Fair Value Assessment
- Use growth model to determine if Ethereum is over/undervalued
- Compare current price to fair value
- Adjust investment timing based on valuation

### 2. Risk Assessment
- Use volatility models to understand risk characteristics
- Consider volatility decay in long-term planning
- Adjust position sizing based on current volatility

### 3. Portfolio Construction
- Compare Ethereum vs Bitcoin characteristics
- Use GBM simulations for scenario planning
- Consider correlation and diversification benefits

## Data Sources

### Historical Data
- **File**: `Etherium/Data/Ethereum Historical Data.csv`
- **Date Range**: 2016-03-10 to 2025-07-24
- **Data Points**: 3,424 days
- **Price Range**: $6.70 to $4,808.38

### Model Coefficients
- Growth model coefficients stored in `Etherium/Models/Growth/Formulas/`
- Volatility model coefficients stored in `Etherium/Models/Volatility/Formulas/`
- All models include R² values and statistical validation

## Notes and Considerations

### 1. Model Limitations
- Growth model assumes continued power law relationship
- Volatility decay may not continue indefinitely
- GBM assumes normal distribution of returns

### 2. Market Conditions
- Models based on historical data
- Future market conditions may differ
- Regular model updates recommended

### 3. Risk Management
- Use models as tools, not predictions
- Consider multiple scenarios
- Maintain appropriate position sizing

## Future Enhancements

### 1. Strategy Analysis
- Implement Ethereum-specific investment strategies
- Compare DCA vs lump sum approaches
- Add multi-tier reserve strategies

### 2. Portfolio Optimization
- Integrate with Bitcoin analysis
- Create multi-asset portfolio models
- Add correlation analysis

### 3. Real-time Updates
- Automated data fetching
- Model recalibration
- Strategy backtesting

---

*Last Updated: 2025-07-25*
*Ethereum Analysis Workflow v1.0* 