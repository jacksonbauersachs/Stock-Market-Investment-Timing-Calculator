# Stock Market Investment Timing Calculator

A comprehensive analysis tool for Bitcoin and S&P 500 investment timing strategies using mathematical models and Monte Carlo simulations.

## 🎯 **Current GBM Simulation (Latest)**

### **Bitcoin GBM - Fair Value Start**
- **Script**: `Scripts/Bitcoin/bitcoin_gbm_fair_value_start.py`
- **Starting Point**: Formula fair value (~$107k on 7/20/2025)
- **Components**: Dynamic growth + Dynamic volatility (NO mean reversion)
- **Validation**: 13.3% average difference from growth formula (GOOD)
- **Results**: 
  - 1 year: $141,848 (90% CI: $65,544 - $255,999)
  - 5 years: $392,084 (90% CI: $80,652 - $1,043,893)
  - 10 years: $1,302,917 (90% CI: $170,441 - $4,075,492)

### **Key Features:**
- ✅ **Dynamic Growth**: Uses growth formula to calculate future fair values
- ✅ **Dynamic Volatility**: Uses exponential decay volatility formula
- ✅ **No Mean Reversion**: Pure GBM with growth + volatility only
- ✅ **Validated**: Closely follows growth formula predictions
- ✅ **Realistic**: Shows proper volatility drag and uncertainty

### **Files:**
- **Paths**: `Results/Bitcoin/bitcoin_gbm_fair_value_start_20250720_172506.csv` (35MB)
- **Summary**: `Results/Bitcoin/bitcoin_gbm_fair_value_start_20250720_172506_summary.csv`
- **Validation**: `Results/Bitcoin/gbm_formula_validation_20250720_172611.png`
- **Comparison**: `Results/Bitcoin/gbm_formula_comparison_20250720_172612.csv`

## 📁 **Project Structure**

### **Scripts/**
- **Bitcoin/**: Investment strategies and GBM simulations
- **SP500/**: S&P 500 analysis and simulations
- **Portfolio/**: Multi-asset portfolio optimization
- **Data_Cleaning/**: Data preprocessing scripts

### **Models/**
- **Growth Models/**: Bitcoin and S&P 500 growth formulas
- **Volatility Models/**: Dynamic volatility decay models

### **Results/**
- **Bitcoin/**: Current GBM results and investment strategies
- **SP500/**: S&P 500 analysis results
- **Portfolio/**: Portfolio optimization results

### **Mean Reversion Theory/**
- **All theoretical mean reversion analysis** (separated from practical strategies)

## 🔬 **Key Models**

### **Bitcoin Growth Formula**
```
log10(price) = 1.827743 * ln(day) + -10.880943
```

### **Bitcoin Volatility Formula**
```
volatility = a * exp(-b * years) + c
```

### **GBM Simulation**
- **Starting Price**: Formula fair value ($107,641 on 7/20/2025)
- **Time Horizon**: 10 years
- **Paths**: 1,000 Monte Carlo paths
- **Updates**: Daily parameter updates
- **Components**: Dynamic growth + Dynamic volatility

## 📊 **Recent Results**

### **GBM Validation (7/20/2025)**
- **Formula vs GBM Mean**: 13.3% average difference
- **Formula vs GBM Median**: 5.4% average difference
- **Assessment**: GOOD - GBM properly follows growth formula
- **Volatility Effect**: Properly models uncertainty around trend

### **Investment Implications**
- **Short-term (1-2 years)**: Very close to formula predictions
- **Medium-term (3-5 years)**: Reasonable uncertainty range
- **Long-term (10 years)**: Higher uncertainty, but trend maintained

## 🚀 **Quick Start**

### **Run Current GBM Simulation:**
```bash
python Scripts/Bitcoin/bitcoin_gbm_fair_value_start.py
```

### **Validate Against Growth Formula:**
```bash
python Scripts/Bitcoin/validate_gbm_against_growth_formula.py
```

## 📝 **Notes**

- **Mean Reversion Analysis**: Moved to dedicated "Mean Reversion Theory" folder
- **Current Focus**: Dynamic growth + volatility GBM (no mean reversion)
- **Validation**: All simulations validated against historical data and formulas
- **Documentation**: Updated to reflect current working models only 