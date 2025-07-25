# Ethereum Analysis Workflow Summary 2025

## Quick Reference

### Core Scripts
- **Growth Model**: `Etherium/Models/Growth/Scripts/ethereum_growth_model_fit.py`
- **Volatility Analysis**: `Etherium/Models/Volatility/Scripts/ethereum_volatility_model_fit.py`
- **Volatility Decay**: `Etherium/Models/Volatility/Scripts/ethereum_365day_volatility_decay.py`
- **GBM Simulation**: `Etherium/simulation/ethereum_gbm_simple_save.py`

### Key Models

#### Growth Model (Rainbow Chart)
- **Start**: Day 365 (2016-03-10)
- **Formula**: `log10(price) = slope * ln(day) + intercept`
- **Purpose**: Fair value calculation

#### Volatility Models
1. **Multi-Window**: Power Law (R² = 0.943)
   - Formula: `log(volatility) = 0.029022 * log(window) + 4.363187`
   - 365-day volatility: 92.21%

2. **Decay Over Time**: Exponential (R² = 0.698)
   - Formula: `log(volatility) = -0.097430 * age + 4.991914`
   - Current volatility: 74.6% (age 9.4 years)

### Current Market Data
- **Age**: 9.4 years (as of 2025)
- **Current Price**: ~$4,800
- **Fair Value**: Calculated from growth model
- **Current Volatility**: 74.6%

### GBM Simulation Parameters
- **Starting Price**: Fair value from growth model
- **Time Horizon**: 10 years
- **Paths**: 100 (test) or 1000 (full)
- **Updates**: Daily parameter recalculation
- **Growth**: Dynamic based on fair value
- **Volatility**: Exponential decay with age

### Quick Commands
```bash
# Run complete analysis
python Etherium/Models/Growth/Scripts/ethereum_growth_model_fit.py
python Etherium/Models/Volatility/Scripts/ethereum_volatility_model_fit.py
python Etherium/Models/Volatility/Scripts/ethereum_365day_volatility_decay.py
python Etherium/simulation/ethereum_gbm_simple_save.py
```

### Key Differences from Bitcoin
- **Younger**: 9.4 vs 15 years
- **Higher Volatility**: 74.6% vs ~60%
- **Faster Decay**: Volatility decreases more rapidly
- **Different Growth**: Unique growth characteristics

### Output Files
- **Growth**: `Etherium/Models/Growth/Formulas/ethereum_growth_model_coefficients.txt`
- **Volatility**: `Etherium/Models/Volatility/Formulas/ethereum_volatility_model_coefficients.txt`
- **Decay**: `Etherium/Models/Volatility/Formulas/ethereum_365day_volatility_decay_coefficients.txt`
- **GBM Paths**: `Etherium/Results/ethereum_gbm_paths_YYYYMMDD_HHMMSS.csv`

---

*Last Updated: 2025-07-25* 