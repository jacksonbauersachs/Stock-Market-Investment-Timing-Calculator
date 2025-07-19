import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

class BitcoinVolatilityDecayAnalyzer:
    def __init__(self, data_path):
        """Initialize with Bitcoin data for volatility decay analysis"""
        self.data = pd.read_csv(data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Handle price column format - use 'Price' column
        if 'Price' in self.data.columns:
            self.data['Price'] = pd.to_numeric(self.data['Price'], errors='coerce')
        else:
            print("Error: 'Price' column not found in data")
            return
        
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        
        # Calculate time metrics
        self.data['Days'] = (self.data['Date'] - self.data['Date'].min()).dt.days
        self.data['Years'] = self.data['Days'] / 365.25
        
        # Calculate various volatility metrics
        self._calculate_volatility_metrics()
        
    def _calculate_volatility_metrics(self):
        """Calculate multiple volatility metrics over time"""
        # Calculate returns
        self.data['Returns'] = self.data['Price'].pct_change()
        
        # Calculate rolling volatilities with different windows
        self.data['Volatility_7d'] = self.data['Returns'].rolling(7).std() * np.sqrt(365)
        self.data['Volatility_30d'] = self.data['Returns'].rolling(30).std() * np.sqrt(365)
        self.data['Volatility_90d'] = self.data['Returns'].rolling(90).std() * np.sqrt(365)
        self.data['Volatility_180d'] = self.data['Returns'].rolling(180).std() * np.sqrt(365)
        self.data['Volatility_365d'] = self.data['Returns'].rolling(365).std() * np.sqrt(365)
        
        # Calculate realized volatility (rolling standard deviation of returns)
        self.data['Realized_Volatility'] = self.data['Returns'].rolling(30).std() * np.sqrt(365)
        
        # Calculate volatility of volatility (how much volatility itself varies)
        self.data['Volatility_of_Volatility'] = self.data['Volatility_30d'].rolling(90).std()
        
        print("Volatility metrics calculated for analysis...")
        
    def analyze_volatility_decay(self, volatility_column='Volatility_30d'):
        """Analyze how volatility decays over time"""
        print(f"\n=== Analyzing {volatility_column} Decay ===")
        
        # Remove NaN values
        clean_data = self.data[['Years', volatility_column]].dropna()
        
        # Plot volatility over time
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Raw volatility over time
        plt.subplot(2, 3, 1)
        plt.scatter(clean_data['Years'], clean_data[volatility_column], alpha=0.6, s=10)
        plt.xlabel('Years Since Start')
        plt.ylabel('Annualized Volatility')
        plt.title(f'{volatility_column} Over Time')
        plt.grid(True, alpha=0.3)
        
        # Try different decay models
        models = {
            'Linear Decay': self._fit_linear_decay,
            'Exponential Decay': self._fit_exponential_decay,
            'Power Law Decay': self._fit_power_law_decay,
            'Logarithmic Decay': self._fit_logarithmic_decay,
            'Polynomial Decay': self._fit_polynomial_decay
        }
        
        best_model = None
        best_r_squared = -np.inf
        
        for model_name, fit_function in models.items():
            try:
                params, r_squared = fit_function(clean_data['Years'], clean_data[volatility_column])
                models[model_name] = (params, r_squared)
                
                if r_squared > best_r_squared:
                    best_r_squared = r_squared
                    best_model = model_name
                    
                print(f"{model_name}: R² = {r_squared:.4f}")
                
            except Exception as e:
                print(f"{model_name}: Failed to fit - {e}")
                models[model_name] = (None, -np.inf)
        
        # Plot the best model
        if best_model and models[best_model][0] is not None:
            params, r_squared = models[best_model]
            
            # Generate predictions
            years_range = np.linspace(clean_data['Years'].min(), clean_data['Years'].max(), 100)
            
            if best_model == 'Linear Decay':
                predictions = self._linear_decay_model(years_range, *params)
            elif best_model == 'Exponential Decay':
                predictions = self._exponential_decay_model(years_range, *params)
            elif best_model == 'Power Law Decay':
                predictions = self._power_law_decay_model(years_range, *params)
            elif best_model == 'Logarithmic Decay':
                predictions = self._logarithmic_decay_model(years_range, *params)
            elif best_model == 'Polynomial Decay':
                predictions = self._polynomial_decay_model(years_range, *params)
            
            # Plot best fit
            plt.subplot(2, 3, 2)
            plt.scatter(clean_data['Years'], clean_data[volatility_column], alpha=0.6, s=10, label='Actual')
            plt.plot(years_range, predictions, 'r-', linewidth=2, label=f'Best Fit ({best_model})')
            plt.xlabel('Years Since Start')
            plt.ylabel('Annualized Volatility')
            plt.title(f'Best Fit Model: {best_model}\nR² = {r_squared:.4f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Store the best model for future use
            self.best_decay_model = {
                'name': best_model,
                'params': params,
                'r_squared': r_squared,
                'function': self._get_decay_function(best_model)
            }
            
            print(f"\nBest Model: {best_model}")
            print(f"R² = {r_squared:.4f}")
            print(f"Parameters: {params}")
            
        # Plot 3: Volatility by year (averaged)
        plt.subplot(2, 3, 3)
        yearly_vol = clean_data.groupby(clean_data['Years'].astype(int))[volatility_column].mean()
        plt.bar(yearly_vol.index, yearly_vol.values, alpha=0.7)
        plt.xlabel('Year')
        plt.ylabel('Average Volatility')
        plt.title('Average Volatility by Year')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Volatility distribution
        plt.subplot(2, 3, 4)
        plt.hist(clean_data[volatility_column], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Volatility')
        plt.ylabel('Frequency')
        plt.title('Volatility Distribution')
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Volatility vs Years (log scale)
        plt.subplot(2, 3, 5)
        plt.scatter(clean_data['Years'], clean_data[volatility_column], alpha=0.6, s=10)
        plt.yscale('log')
        plt.xlabel('Years Since Start')
        plt.ylabel('Volatility (Log Scale)')
        plt.title('Volatility vs Time (Log Scale)')
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Rolling average volatility
        plt.subplot(2, 3, 6)
        rolling_avg = clean_data[volatility_column].rolling(90).mean()
        plt.plot(clean_data['Years'], clean_data[volatility_column], alpha=0.3, label='Daily')
        plt.plot(clean_data['Years'], rolling_avg, 'r-', linewidth=2, label='90-day Rolling Avg')
        plt.xlabel('Years Since Start')
        plt.ylabel('Volatility')
        plt.title('Volatility with Rolling Average')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return self.best_decay_model
    
    def _fit_linear_decay(self, x, y):
        """Fit linear decay model: volatility = a * years + b"""
        def linear_model(x, a, b):
            return a * x + b
        
        params, _ = curve_fit(linear_model, x, y)
        predictions = linear_model(x, *params)
        r_squared = 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        return params, r_squared
    
    def _fit_exponential_decay(self, x, y):
        """Fit exponential decay model: volatility = a * exp(-b * years) + c"""
        def exponential_model(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        # Initial guess
        p0 = [np.max(y), 0.1, np.min(y)]
        params, _ = curve_fit(exponential_model, x, y, p0=p0, maxfev=10000)
        predictions = exponential_model(x, *params)
        r_squared = 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        return params, r_squared
    
    def _fit_power_law_decay(self, x, y):
        """Fit power law decay model: volatility = a * years^(-b) + c"""
        def power_law_model(x, a, b, c):
            return a * (x ** (-b)) + c
        
        # Initial guess
        p0 = [np.max(y), 0.5, np.min(y)]
        params, _ = curve_fit(power_law_model, x, y, p0=p0, maxfev=10000)
        predictions = power_law_model(x, *params)
        r_squared = 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        return params, r_squared
    
    def _fit_logarithmic_decay(self, x, y):
        """Fit logarithmic decay model: volatility = a * ln(years + 1) + b"""
        def logarithmic_model(x, a, b):
            return a * np.log(x + 1) + b
        
        params, _ = curve_fit(logarithmic_model, x, y)
        predictions = logarithmic_model(x, *params)
        r_squared = 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        return params, r_squared
    
    def _fit_polynomial_decay(self, x, y):
        """Fit polynomial decay model: volatility = a * years^2 + b * years + c"""
        def polynomial_model(x, a, b, c):
            return a * (x ** 2) + b * x + c
        
        params, _ = curve_fit(polynomial_model, x, y)
        predictions = polynomial_model(x, *params)
        r_squared = 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        return params, r_squared
    
    def _linear_decay_model(self, x, a, b):
        return a * x + b
    
    def _exponential_decay_model(self, x, a, b, c):
        return a * np.exp(-b * x) + c
    
    def _power_law_decay_model(self, x, a, b, c):
        return a * (x ** (-b)) + c
    
    def _logarithmic_decay_model(self, x, a, b):
        return a * np.log(x + 1) + b
    
    def _polynomial_decay_model(self, x, a, b, c):
        return a * (x ** 2) + b * x + c
    
    def _get_decay_function(self, model_name):
        """Get the decay function for the specified model"""
        if model_name == 'Linear Decay':
            return self._linear_decay_model
        elif model_name == 'Exponential Decay':
            return self._exponential_decay_model
        elif model_name == 'Power Law Decay':
            return self._power_law_decay_model
        elif model_name == 'Logarithmic Decay':
            return self._logarithmic_decay_model
        elif model_name == 'Polynomial Decay':
            return self._polynomial_decay_model
        else:
            return None
    
    def predict_future_volatility(self, years_ahead=5):
        """Predict future volatility using the best-fit model"""
        if not hasattr(self, 'best_decay_model'):
            print("No decay model fitted yet. Run analyze_volatility_decay() first.")
            return None
        
        current_years = self.data['Years'].max()
        future_years = np.arange(current_years, current_years + years_ahead + 1, 0.1)
        
        # Predict volatility
        predictions = self.best_decay_model['function'](future_years, *self.best_decay_model['params'])
        
        # Ensure predictions are reasonable (not negative, not too high)
        predictions = np.maximum(predictions, 0.1)  # Minimum 10% volatility
        predictions = np.minimum(predictions, 2.0)   # Maximum 200% volatility
        
        # Plot predictions
        plt.figure(figsize=(12, 8))
        
        # Historical data
        clean_data = self.data[['Years', 'Volatility_30d']].dropna()
        plt.scatter(clean_data['Years'], clean_data['Volatility_30d'], alpha=0.6, s=10, label='Historical')
        
        # Model fit
        model_years = np.linspace(clean_data['Years'].min(), clean_data['Years'].max(), 100)
        model_predictions = self.best_decay_model['function'](model_years, *self.best_decay_model['params'])
        plt.plot(model_years, model_predictions, 'r-', linewidth=2, label=f'Model Fit ({self.best_decay_model["name"]})')
        
        # Future predictions
        plt.plot(future_years, predictions, 'g--', linewidth=2, label='Future Predictions')
        plt.axvline(x=current_years, color='black', linestyle=':', label='Current Time')
        
        plt.xlabel('Years Since Start')
        plt.ylabel('Predicted Annualized Volatility')
        plt.title(f'Bitcoin Volatility Predictions\nModel: {self.best_decay_model["name"]} (R² = {self.best_decay_model["r_squared"]:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Print predictions
        print(f"\n=== Future Volatility Predictions ===")
        print(f"Model: {self.best_decay_model['name']}")
        print(f"R² = {self.best_decay_model['r_squared']:.4f}")
        print(f"Parameters: {self.best_decay_model['params']}")
        print(f"\nPredicted Volatility:")
        
        for i, year in enumerate([1, 2, 3, 4, 5]):
            future_year = current_years + year
            vol_prediction = self.best_decay_model['function'](future_year, *self.best_decay_model['params'])
            vol_prediction = max(0.1, min(2.0, vol_prediction))
            print(f"  Year {year} from now: {vol_prediction:.1%}")
        
        return {
            'model': self.best_decay_model,
            'predictions': predictions,
            'years': future_years
        }
    
    def compare_volatility_metrics(self):
        """Compare different volatility metrics"""
        metrics = ['Volatility_7d', 'Volatility_30d', 'Volatility_90d', 'Volatility_180d', 'Volatility_365d']
        
        plt.figure(figsize=(15, 10))
        
        for i, metric in enumerate(metrics):
            plt.subplot(2, 3, i+1)
            
            clean_data = self.data[['Years', metric]].dropna()
            plt.scatter(clean_data['Years'], clean_data[metric], alpha=0.6, s=5)
            
            # Add trend line
            z = np.polyfit(clean_data['Years'], clean_data[metric], 1)
            p = np.poly1d(z)
            plt.plot(clean_data['Years'], p(clean_data['Years']), "r--", alpha=0.8)
            
            plt.xlabel('Years Since Start')
            plt.ylabel('Volatility')
            plt.title(f'{metric}\nTrend: {z[0]:.3f}x + {z[1]:.3f}')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n=== Volatility Metrics Comparison ===")
        for metric in metrics:
            clean_data = self.data[metric].dropna()
            print(f"{metric}:")
            print(f"  Mean: {clean_data.mean():.2%}")
            print(f"  Std: {clean_data.std():.2%}")
            print(f"  Min: {clean_data.min():.2%}")
            print(f"  Max: {clean_data.max():.2%}")
            print()


def main():
    """Main execution function"""
    # Initialize analyzer with the complete dataset
    data_path = "Data Sets/Bitcoin Data/Bitcoin_Final_Complete_Data_20250719.csv"
    analyzer = BitcoinVolatilityDecayAnalyzer(data_path)
    
    print("=== Bitcoin Volatility Decay Analysis ===")
    print(f"Data spans {analyzer.data['Years'].max():.1f} years")
    print(f"Total data points: {len(analyzer.data)}")
    
    # Compare different volatility metrics
    analyzer.compare_volatility_metrics()
    
    # Analyze volatility decay for different metrics
    metrics_to_analyze = ['Volatility_30d', 'Volatility_90d', 'Volatility_365d']
    
    best_models = {}
    for metric in metrics_to_analyze:
        print(f"\n{'='*50}")
        print(f"Analyzing {metric}")
        print(f"{'='*50}")
        
        best_model = analyzer.analyze_volatility_decay(metric)
        if best_model:
            best_models[metric] = best_model
    
    # Predict future volatility using the best model
    if best_models:
        best_metric = max(best_models.keys(), key=lambda x: best_models[x]['r_squared'])
        print(f"\n{'='*50}")
        print(f"Using best model: {best_metric}")
        print(f"{'='*50}")
        
        # Re-analyze with the best metric
        analyzer.analyze_volatility_decay(best_metric)
        
        # Predict future volatility
        predictions = analyzer.predict_future_volatility(years_ahead=5)
        
        if predictions:
            print(f"\n=== Summary ===")
            print(f"Best volatility metric: {best_metric}")
            print(f"Best decay model: {predictions['model']['name']}")
            print(f"Model R²: {predictions['model']['r_squared']:.4f}")
            print(f"Model parameters: {predictions['model']['params']}")
            print(f"\nThis model can be used in Monte Carlo simulations to predict future volatility!")


if __name__ == "__main__":
    main() 