import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

class CompleteBitcoinVolatilityAnalyzer:
    def __init__(self, data_path):
        print("Loading Bitcoin data...")
        self.data = pd.read_csv(data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Handle price column format (different column names in full dataset)
        if 'Price' in self.data.columns:
            price_col = 'Price'
        elif 'Close/Last' in self.data.columns:
            price_col = 'Close/Last'
        else:
            raise ValueError("No price column found")
        
        self.data['Price'] = pd.to_numeric(self.data[price_col], errors='coerce')
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        
        # Calculate time metrics
        self.data['Days'] = (self.data['Date'] - self.data['Date'].min()).dt.days
        self.data['Years'] = self.data['Days'] / 365.25
        
        print("Calculating volatility metrics...")
        self._calculate_volatility_metrics()
        
        print(f"Loaded Bitcoin data from {self.data['Date'].min().date()} to {self.data['Date'].max().date()}")
        print(f"Total time span: {self.data['Years'].max():.1f} years")
        print(f"Total data points: {len(self.data)}")
        
    def _calculate_volatility_metrics(self):
        print("Calculating returns...")
        self.data['Returns'] = self.data['Price'].pct_change()
        print("Calculating rolling volatilities...")
        self.data['Volatility_7d'] = self.data['Returns'].rolling(7).std() * np.sqrt(365)
        self.data['Volatility_30d'] = self.data['Returns'].rolling(30).std() * np.sqrt(365)
        self.data['Volatility_90d'] = self.data['Returns'].rolling(90).std() * np.sqrt(365)
        self.data['Volatility_180d'] = self.data['Returns'].rolling(180).std() * np.sqrt(365)
        self.data['Volatility_365d'] = self.data['Returns'].rolling(365).std() * np.sqrt(365)
        print("Removing extreme volatility outliers...")
        outliers_7d = (self.data['Volatility_7d'] > 10).sum()
        outliers_30d = (self.data['Volatility_30d'] > 10).sum()
        outliers_90d = (self.data['Volatility_90d'] > 5).sum()
        outliers_180d = (self.data['Volatility_180d'] > 4).sum()
        outliers_365d = (self.data['Volatility_365d'] > 3).sum()
        print(f"Outliers found - 7d: {outliers_7d}, 30d: {outliers_30d}, 90d: {outliers_90d}, 180d: {outliers_180d}, 365d: {outliers_365d}")
        self.data.loc[self.data['Volatility_7d'] > 10, 'Volatility_7d'] = np.nan
        self.data.loc[self.data['Volatility_30d'] > 10, 'Volatility_30d'] = np.nan
        self.data.loc[self.data['Volatility_90d'] > 5, 'Volatility_90d'] = np.nan
        self.data.loc[self.data['Volatility_180d'] > 4, 'Volatility_180d'] = np.nan
        self.data.loc[self.data['Volatility_365d'] > 3, 'Volatility_365d'] = np.nan
        print("Volatility metrics calculated for analysis...")
        
    def analyze_volatility_decay_comprehensive(self, volatility_column='Volatility_30d'):
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE VOLATILITY DECAY ANALYSIS")
        print(f"Analyzing: {volatility_column}")
        print(f"Using all years with extreme outliers removed")
        print(f"{'='*60}")
        print("Filtering data for analysis...")
        clean_data = self.data[['Years', volatility_column]].dropna()
        clean_data = clean_data[clean_data[volatility_column] > 0]
        print(f"Data points after outlier and zero removal: {len(clean_data)}")
        print(f"Years range: {clean_data['Years'].min():.1f} to {clean_data['Years'].max():.1f}")
        
        # Define multiple mathematical models to test
        models = {
            'Linear Decay': {
                'function': lambda x, a, b: a * x + b,
                'description': 'volatility = a*years + b',
                'initial_guess': [1, 1]
            },
            'Exponential Decay': {
                'function': lambda x, a, b, c: a * np.exp(-b * x) + c,
                'description': 'volatility = a*exp(-b*years) + c',
                'initial_guess': [np.max(clean_data[volatility_column]), 0.1, np.min(clean_data[volatility_column])]
            },
            'Power Law Decay': {
                'function': lambda x, a, b, c: a * (x ** (-b)) + c,
                'description': 'volatility = a*years^(-b) + c',
                'initial_guess': [np.max(clean_data[volatility_column]), 0.5, np.min(clean_data[volatility_column])]
            },
            'Logarithmic Decay': {
                'function': lambda x, a, b: a * np.log(x + 1) + b,
                'description': 'volatility = a*ln(years + 1) + b',
                'initial_guess': [1, 1]
            },
            'Inverse Decay': {
                'function': lambda x, a, b, c: a / (1 + b * x) + c,
                'description': 'volatility = a/(1 + b*years) + c',
                'initial_guess': [np.max(clean_data[volatility_column]), 0.1, np.min(clean_data[volatility_column])]
            },
            'Square Root Decay': {
                'function': lambda x, a, b, c: a / np.sqrt(1 + b * x) + c,
                'description': 'volatility = a/sqrt(1 + b*years) + c',
                'initial_guess': [np.max(clean_data[volatility_column]), 0.1, np.min(clean_data[volatility_column])]
            },
            'Polynomial Decay (2nd order)': {
                'function': lambda x, a, b, c: a * (x ** 2) + b * x + c,
                'description': 'volatility = a*years² + b*years + c',
                'initial_guess': [1, 1, 1]
            },
            'Polynomial Decay (3rd order)': {
                'function': lambda x, a, b, c, d: a * (x ** 3) + b * (x ** 2) + c * x + d,
                'description': 'volatility = a*years³ + b*years² + c*years + d',
                'initial_guess': [1, 1, 1, 1]
            }
        }
        
        # Test all models and store results
        model_results = {}
        best_model = None
        best_r_squared = -np.inf
        
        print(f"\nTesting {len(models)} mathematical models...")
        print(f"{'Model':<25} {'R²':<10} {'Parameters'}")
        print(f"{'-'*25} {'-'*10} {'-'*50}")
        
        for model_name, model_info in models.items():
            try:
                # Fit the model
                params, _ = curve_fit(model_info['function'], clean_data['Years'], clean_data[volatility_column], 
                                    p0=model_info['initial_guess'], maxfev=10000)
                
                # Calculate predictions
                predictions = model_info['function'](clean_data['Years'], *params)
                
                # Calculate R-squared
                ss_res = np.sum((clean_data[volatility_column] - predictions) ** 2)
                ss_tot = np.sum((clean_data[volatility_column] - np.mean(clean_data[volatility_column])) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Store results
                model_results[model_name] = {
                    'params': params,
                    'r_squared': r_squared,
                    'function': model_info['function'],
                    'description': model_info['description'],
                    'predictions': predictions
                }
                
                # Check if this is the best model
                if r_squared > best_r_squared:
                    best_r_squared = r_squared
                    best_model = model_name
                
                # Print results
                param_str = ', '.join([f'{p:.4f}' for p in params])
                print(f"{model_name:<25} {r_squared:<10.4f} {param_str}")
                
            except Exception as e:
                print(f"{model_name:<25} {'FAILED':<10} {str(e)[:40]}")
                model_results[model_name] = None
        
        # Store the best model
        if best_model:
            self.best_decay_model = {
                'name': best_model,
                'results': model_results[best_model]
            }
            
            print(f"\n{'='*60}")
            print(f"BEST MODEL: {best_model}")
            print(f"R² = {best_r_squared:.6f}")
            print(f"Formula: {model_results[best_model]['description']}")
            print(f"Parameters: {model_results[best_model]['params']}")
            print(f"{'='*60}")
            # Write summary to file
            with open('volatility_decay_summary.txt', 'w') as f:
                f.write(f"BEST MODEL: {best_model}\n")
                f.write(f"R² = {best_r_squared:.6f}\n")
                f.write(f"Formula: {model_results[best_model]['description']}\n")
                f.write(f"Parameters: {model_results[best_model]['params']}\n")
                f.write(f"\nAll Model Results:\n")
                for m, res in model_results.items():
                    if res:
                        f.write(f"{m}: R² = {res['r_squared']:.4f}, Params = {res['params']}\n")
        
        # Create comprehensive visualization
        self._plot_comprehensive_analysis(clean_data, model_results, best_model)
        
        return model_results
    
    def _plot_comprehensive_analysis(self, clean_data, model_results, best_model):
        print("Generating comprehensive plots...")
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'Bitcoin Volatility Decay Analysis: {clean_data["Years"].max():.1f} Years of Data', fontsize=16)
        
        # Plot 1: Raw data
        axes[0, 0].scatter(clean_data['Years'], clean_data.iloc[:, 1], alpha=0.6, s=10, color='blue')
        axes[0, 0].set_xlabel('Years Since Start')
        axes[0, 0].set_ylabel('Annualized Volatility')
        axes[0, 0].set_title('Raw Volatility Data')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Best model fit
        if best_model and model_results[best_model]:
            years_range = np.linspace(clean_data['Years'].min(), clean_data['Years'].max(), 1000)
            best_predictions = model_results[best_model]['function'](years_range, *model_results[best_model]['params'])
            
            axes[0, 1].scatter(clean_data['Years'], clean_data.iloc[:, 1], alpha=0.6, s=10, color='blue', label='Actual')
            axes[0, 1].plot(years_range, best_predictions, 'r-', linewidth=2, label=f'Best Fit: {best_model}')
            axes[0, 1].set_xlabel('Years Since Start')
            axes[0, 1].set_ylabel('Annualized Volatility')
            axes[0, 1].set_title(f'Best Model: {best_model}\nR² = {model_results[best_model]["r_squared"]:.4f}')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: All models comparison
        years_range = np.linspace(clean_data['Years'].min(), clean_data['Years'].max(), 1000)
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_results)))
        
        axes[0, 2].scatter(clean_data['Years'], clean_data.iloc[:, 1], alpha=0.6, s=10, color='black', label='Actual')
        
        for i, (model_name, results) in enumerate(model_results.items()):
            if results:
                predictions = results['function'](years_range, *results['params'])
                axes[0, 2].plot(years_range, predictions, color=colors[i], linewidth=1, alpha=0.7, label=f'{model_name} (R²={results["r_squared"]:.3f})')
        
        axes[0, 2].set_xlabel('Years Since Start')
        axes[0, 2].set_ylabel('Annualized Volatility')
        axes[0, 2].set_title('All Models Comparison')
        axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Volatility by year (averaged)
        yearly_vol = clean_data.groupby(clean_data['Years'].astype(int)).iloc[:, 1].mean()
        axes[1, 0].bar(yearly_vol.index, yearly_vol.values, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Average Volatility')
        axes[1, 0].set_title('Average Volatility by Year')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Volatility distribution
        axes[1, 1].hist(clean_data.iloc[:, 1], bins=50, alpha=0.7, edgecolor='black', color='orange')
        axes[1, 1].set_xlabel('Volatility')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Volatility Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Volatility vs Years (log scale)
        axes[1, 2].scatter(clean_data['Years'], clean_data.iloc[:, 1], alpha=0.6, s=10, color='purple')
        axes[1, 2].set_yscale('log')
        axes[1, 2].set_xlabel('Years Since Start')
        axes[1, 2].set_ylabel('Volatility (Log Scale)')
        axes[1, 2].set_title('Volatility vs Time (Log Scale)')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Plot 7: Rolling average volatility
        rolling_avg = clean_data.iloc[:, 1].rolling(365).mean()
        axes[2, 0].plot(clean_data['Years'], clean_data.iloc[:, 1], alpha=0.3, color='blue', label='Daily')
        axes[2, 0].plot(clean_data['Years'], rolling_avg, 'r-', linewidth=2, label='365-day Rolling Avg')
        axes[2, 0].set_xlabel('Years Since Start')
        axes[2, 0].set_ylabel('Volatility')
        axes[2, 0].set_title('Volatility with Rolling Average')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 8: Residuals of best model
        if best_model and model_results[best_model]:
            residuals = clean_data.iloc[:, 1] - model_results[best_model]['predictions']
            axes[2, 1].scatter(clean_data['Years'], residuals, alpha=0.6, s=10, color='red')
            axes[2, 1].axhline(y=0, color='black', linestyle='--')
            axes[2, 1].set_xlabel('Years Since Start')
            axes[2, 1].set_ylabel('Residuals')
            axes[2, 1].set_title(f'Residuals: {best_model}')
            axes[2, 1].grid(True, alpha=0.3)
        
        # Plot 9: Model comparison table
        axes[2, 2].axis('off')
        model_text = "Model Comparison:\n\n"
        for model_name, results in model_results.items():
            if results:
                model_text += f"{model_name}: R² = {results['r_squared']:.4f}\n"
        
        axes[2, 2].text(0.1, 0.9, model_text, transform=axes[2, 2].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
    
    def predict_future_volatility(self, years_ahead=10):
        """Predict future volatility using the best-fit model"""
        if not hasattr(self, 'best_decay_model'):
            print("No decay model fitted yet. Run analyze_volatility_decay_comprehensive() first.")
            return None
        
        current_years = self.data['Years'].max()
        future_years = np.arange(current_years, current_years + years_ahead + 1, 0.1)
        
        # Predict volatility using best model
        best_function = self.best_decay_model['results']['function']
        best_params = self.best_decay_model['results']['params']
        predictions = best_function(future_years, *best_params)
        
        # Ensure predictions are reasonable
        predictions = np.maximum(predictions, 0.05)  # Minimum 5% volatility
        predictions = np.minimum(predictions, 3.0)   # Maximum 300% volatility
        
        # Plot predictions
        plt.figure(figsize=(15, 10))
        
        # Historical data
        clean_data = self.data[['Years', 'Volatility_30d']].dropna()
        plt.scatter(clean_data['Years'], clean_data['Volatility_30d'], alpha=0.6, s=10, label='Historical', color='blue')
        
        # Model fit
        model_years = np.linspace(clean_data['Years'].min(), clean_data['Years'].max(), 1000)
        model_predictions = best_function(model_years, *best_params)
        plt.plot(model_years, model_predictions, 'r-', linewidth=2, label=f'Model Fit ({self.best_decay_model["name"]})')
        
        # Future predictions
        plt.plot(future_years, predictions, 'g--', linewidth=2, label='Future Predictions')
        plt.axvline(x=current_years, color='black', linestyle=':', label='Current Time')
        
        plt.xlabel('Years Since Start')
        plt.ylabel('Predicted Annualized Volatility')
        plt.title(f'Bitcoin Volatility Predictions\nModel: {self.best_decay_model["name"]} (R² = {self.best_decay_model["results"]["r_squared"]:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Print predictions
        print(f"\n{'='*60}")
        print(f"FUTURE VOLATILITY PREDICTIONS")
        print(f"{'='*60}")
        print(f"Model: {self.best_decay_model['name']}")
        print(f"Formula: {self.best_decay_model['results']['description']}")
        print(f"R² = {self.best_decay_model['results']['r_squared']:.6f}")
        print(f"Parameters: {best_params}")
        print(f"\nPredicted Volatility:")
        
        for i, year in enumerate([1, 2, 3, 5, 10]):
            future_year = current_years + year
            vol_prediction = best_function(future_year, *best_params)
            vol_prediction = max(0.05, min(3.0, vol_prediction))
            print(f"  Year {year} from now: {vol_prediction:.1%}")
        
        return {
            'model': self.best_decay_model,
            'predictions': predictions,
            'years': future_years
        }
    
    def compare_volatility_metrics(self):
        print("Comparing volatility metrics...")
        metrics = ['Volatility_7d', 'Volatility_30d', 'Volatility_90d', 'Volatility_180d', 'Volatility_365d']
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        summary_lines = []
        for i, metric in enumerate(metrics):
            print(f"Plotting {metric}...")
            row, col = divmod(i, 3)
            ax = axes[row, col]
            clean_data = self.data[['Years', metric]].dropna()
            clean_data = clean_data[clean_data[metric] > 0]
            ax.scatter(clean_data['Years'], clean_data[metric], alpha=0.6, s=5)
            z = np.polyfit(clean_data['Years'], clean_data[metric], 1)
            p = np.poly1d(z)
            ax.plot(clean_data['Years'], p(clean_data['Years']), "r--", alpha=0.8)
            ax.set_xlabel('Years Since Start')
            ax.set_ylabel('Volatility')
            ax.set_title(f'{metric}\nTrend: {z[0]:.3f}x + {z[1]:.3f}')
            ax.grid(True, alpha=0.3)
            summary_lines.append(f"{metric} Trend: {z[0]:.6f}x + {z[1]:.6f}")
        # Hide the last empty subplot if needed
        if len(metrics) < 6:
            axes[1, 2].axis('off')
        plt.tight_layout()
        print("Showing volatility plots...")
        plt.show()
        plt.close()
        print("Finished showing volatility plots.")
        print("Printing summary statistics...")
        print(f"\n{'='*60}")
        print(f"VOLATILITY METRICS COMPARISON")
        print(f"{'='*60}")
        output_path = 'Investment Strategy Analasis/Bitcoin Analysis/volatility_summary.txt'
        with open(output_path, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"VOLATILITY METRICS COMPARISON\n")
            f.write(f"{'='*60}\n")
            for metric in metrics:
                clean_data = self.data[metric].dropna()
                clean_data = clean_data[clean_data > 0]
                stats = [
                    f"\n{metric}:",
                    f"  Mean: {clean_data.mean():.2%}",
                    f"  Std: {clean_data.std():.2%}",
                    f"  Min: {clean_data.min():.2%}",
                    f"  Max: {clean_data.max():.2%}"
                ]
                for line in stats:
                    print(line)
                    f.write(line + '\n')
            f.write('\nTrend Lines:\n')
            for line in summary_lines:
                print(line)
                f.write(line + '\n')
        print(f"Summary saved to {output_path}")

    def fit_inverse_decay_all_windows(self):
        print("Fitting inverse decay model to all volatility windows...")
        from scipy.optimize import curve_fit
        metrics = ['Volatility_7d', 'Volatility_30d', 'Volatility_90d', 'Volatility_180d', 'Volatility_365d']
        results = []
        for metric in metrics:
            print(f"Fitting {metric}...")
            clean_data = self.data[['Years', metric]].dropna()
            clean_data = clean_data[clean_data[metric] > 0]
            x = clean_data['Years'].values
            y = clean_data[metric].values
            if len(x) == 0:
                print(f"No data for {metric} after cleaning.")
                continue
            def inverse_decay(x, a, b, c):
                return a / (1 + b * x) + c
            # Initial guess: a=max(y), b=1, c=min(y)
            p0 = [max(y), 1, min(y)]
            try:
                params, _ = curve_fit(inverse_decay, x, y, p0=p0, maxfev=10000)
                predictions = inverse_decay(x, *params)
                ss_res = ((y - predictions) ** 2).sum()
                ss_tot = ((y - y.mean()) ** 2).sum()
                r2 = 1 - ss_res / ss_tot
                results.append((metric, params, r2))
                print(f"{metric}: a={params[0]:.4f}, b={params[1]:.4f}, c={params[2]:.4f}, R^2={r2:.4f}")
            except Exception as e:
                print(f"{metric}: Fit failed ({e})")
        # Save results
        output_path = 'Investment Strategy Analasis/Bitcoin Analysis/inverse_decay_fits.txt'
        with open(output_path, 'w') as f:
            f.write('Inverse Decay Model Fits for Volatility Windows\n')
            f.write('Formula: volatility = a/(1 + b*years) + c\n\n')
            for metric, params, r2 in results:
                line = f"{metric}: a={params[0]:.6f}, b={params[1]:.6f}, c={params[2]:.6f}, R^2={r2:.4f}"
                print(line)
                f.write(line + '\n')
        print(f"All fits saved to {output_path}")


def main():
    """Main execution function"""
    # Initialize analyzer with full historical data
    data_path = "Data Sets/Bitcoin Data/Bitcoin Historical Data Full.csv"
    analyzer = CompleteBitcoinVolatilityAnalyzer(data_path)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE BITCOIN VOLATILITY DECAY ANALYSIS")
    print(f"{'='*60}")
    
    # Compare different volatility metrics
    analyzer.compare_volatility_metrics()
    analyzer.fit_inverse_decay_all_windows()
    
    # Analyze volatility decay for different metrics
    metrics_to_analyze = ['Volatility_30d', 'Volatility_90d', 'Volatility_365d']
    
    best_models = {}
    for metric in metrics_to_analyze:
        print(f"\n{'='*60}")
        print(f"Analyzing {metric}")
        print(f"{'='*60}")
        
        model_results = analyzer.analyze_volatility_decay_comprehensive(metric)
        
        # Find best model for this metric
        best_model = None
        best_r_squared = -np.inf
        for model_name, results in model_results.items():
            if results and results['r_squared'] > best_r_squared:
                best_r_squared = results['r_squared']
                best_model = model_name
        
        if best_model:
            best_models[metric] = {
                'name': best_model,
                'results': model_results[best_model]
            }
    
    # Use the best overall model for predictions
    if best_models:
        best_metric = max(best_models.keys(), key=lambda x: best_models[x]['results']['r_squared'])
        print(f"\n{'='*60}")
        print(f"USING BEST OVERALL MODEL: {best_metric}")
        print(f"{'='*60}")
        
        # Re-analyze with the best metric
        analyzer.analyze_volatility_decay_comprehensive(best_metric)
        
        # Predict future volatility
        predictions = analyzer.predict_future_volatility(years_ahead=10)
        
        if predictions:
            print(f"\n{'='*60}")
            print(f"SUMMARY")
            print(f"{'='*60}")
            print(f"Best volatility metric: {best_metric}")
            print(f"Best decay model: {predictions['model']['name']}")
            print(f"Model R²: {predictions['model']['results']['r_squared']:.6f}")
            print(f"Model formula: {predictions['model']['results']['description']}")
            print(f"Model parameters: {predictions['model']['results']['params']}")
            print(f"\nThis model can be used in Monte Carlo simulations to predict future volatility!")


if __name__ == "__main__":
    main() 