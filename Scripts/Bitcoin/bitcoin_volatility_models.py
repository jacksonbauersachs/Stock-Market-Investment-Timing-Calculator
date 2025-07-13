import numpy as np
import pandas as pd

def current_model(years):
    """
    Your current Bitcoin volatility model (inverse decay with unrealistic floor)
    Formula: volatility = (a / (1 + b * years) + c) / 100
    """
    a, b, c = 31.78, 22.19, 0.31
    return (a / (1 + b * years) + c) / 100

def better_model(years):
    """
    Proposed better Bitcoin volatility model (exponential decay to realistic floor)
    Formula: volatility = mature_vol + (initial_vol - mature_vol) * exp(-decay_rate * years)
    """
    initial_vol = 1.0  # 100% starting volatility
    mature_vol = 0.4   # 40% mature volatility floor
    decay_rate = 0.15  # moderate exponential decay
    return mature_vol + (initial_vol - mature_vol) * np.exp(-decay_rate * years)

def realistic_model(years):
    """
    Alternative realistic model based on empirical Bitcoin data
    Uses different decay rates for different phases
    """
    if years < 5:
        # Early high volatility phase
        return 0.8 + 0.2 * np.exp(-0.3 * years)
    elif years < 10:
        # Maturing phase
        return 0.6 + 0.2 * np.exp(-0.2 * (years - 5))
    else:
        # Mature phase
        return 0.45 + 0.15 * np.exp(-0.1 * (years - 10))

def calculate_model_predictions(years_list, model_func, model_name):
    """
    Calculate predictions for a given model over a list of years
    """
    predictions = []
    for years in years_list:
        vol = model_func(years)
        predictions.append(vol)
    return predictions

def compare_models(years_list=None):
    """
    Compare all volatility models side by side
    """
    if years_list is None:
        years_list = [1, 2, 3, 5, 7, 10, 15, 20]
    
    print("BITCOIN VOLATILITY MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Year':<4} | {'Current Model':<12} | {'Better Model':<12} | {'Realistic Model':<14} | {'Difference':<10}")
    print("-" * 60)
    
    for years in years_list:
        current = current_model(years)
        better = better_model(years)
        realistic = realistic_model(years)
        diff = better - current
        
        print(f"{years:<4} | {current:<11.1%} | {better:<11.1%} | {realistic:<13.1%} | {diff:+9.1%}")

def model_formulas():
    """
    Print the mathematical formulas for each model
    """
    print("BITCOIN VOLATILITY MODEL FORMULAS")
    print("=" * 40)
    print()
    
    print("1. CURRENT MODEL (Your Original):")
    print("   Formula: σ(t) = (a/(1 + b×t) + c)/100")
    print("   Parameters: a=31.78, b=22.19, c=0.31")
    print("   Problem: Approaches 0.31% (unrealistic)")
    print()
    
    print("2. BETTER MODEL (Exponential Decay):")
    print("   Formula: σ(t) = 0.4 + 0.6 × e^(-0.15t)")
    print("   Parameters: initial=100%, mature=40%, decay=0.15")
    print("   Advantage: Realistic 40% floor")
    print()
    
    print("3. REALISTIC MODEL (Multi-Phase):")
    print("   Formula: Different equations for different phases")
    print("   Phase 1 (0-5 years): σ(t) = 0.8 + 0.2 × e^(-0.3t)")
    print("   Phase 2 (5-10 years): σ(t) = 0.6 + 0.2 × e^(-0.2(t-5))")
    print("   Phase 3 (10+ years): σ(t) = 0.45 + 0.15 × e^(-0.1(t-10))")
    print("   Advantage: Captures different market maturity phases")

def validate_against_actual(actual_volatility=0.445, current_years=15):
    """
    Validate models against actual current Bitcoin volatility
    """
    print("MODEL VALIDATION AGAINST ACTUAL DATA")
    print("=" * 45)
    print(f"Actual Bitcoin volatility (current): {actual_volatility:.1%}")
    print(f"Current time period: Year {current_years}")
    print()
    
    current_pred = current_model(current_years)
    better_pred = better_model(current_years)
    realistic_pred = realistic_model(current_years)
    
    current_error = abs(actual_volatility - current_pred) / actual_volatility
    better_error = abs(actual_volatility - better_pred) / actual_volatility
    realistic_error = abs(actual_volatility - realistic_pred) / actual_volatility
    
    print(f"Current Model Prediction: {current_pred:.1%} (Error: {current_error:.1%})")
    print(f"Better Model Prediction: {better_pred:.1%} (Error: {better_error:.1%})")
    print(f"Realistic Model Prediction: {realistic_pred:.1%} (Error: {realistic_error:.1%})")
    print()
    
    best_model = min([
        ("Current", current_error),
        ("Better", better_error),
        ("Realistic", realistic_error)
    ], key=lambda x: x[1])
    
    print(f"Best performing model: {best_model[0]} (Error: {best_model[1]:.1%})")

def export_model_data(years_range=None, filename="bitcoin_volatility_models.csv"):
    """
    Export model predictions to CSV for further analysis
    """
    if years_range is None:
        years_range = np.linspace(0.5, 20, 40)
    
    data = []
    for years in years_range:
        data.append({
            'Years': years,
            'Current_Model': current_model(years),
            'Better_Model': better_model(years),
            'Realistic_Model': realistic_model(years)
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Model data exported to {filename}")
    return df

if __name__ == "__main__":
    # Run all analyses
    model_formulas()
    print("\n" + "=" * 60 + "\n")
    
    compare_models()
    print("\n" + "=" * 60 + "\n")
    
    validate_against_actual()
    print("\n" + "=" * 60 + "\n")
    
    # Export data
    export_model_data() 