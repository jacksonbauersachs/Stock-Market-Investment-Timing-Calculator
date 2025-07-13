import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load Bitcoin data
df = pd.read_csv('Data Sets/Bitcoin Data/Bitcoin Historical Data Full.csv')
df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')
df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna(subset=['Price'])
df = df.sort_values('Date')

# Calculate daily returns
df['Returns'] = np.log(df['Price'] / df['Price'].shift(1))

# Calculate rolling volatilities
df['Vol_30d'] = df['Returns'].rolling(30).std() * np.sqrt(365)
df['Vol_90d'] = df['Returns'].rolling(90).std() * np.sqrt(365)
df['Vol_365d'] = df['Returns'].rolling(365).std() * np.sqrt(365)

# Calculate years since Bitcoin start
df['Years'] = (df['Date'] - df['Date'].min()).dt.days / 365.25

# Your current model
def current_model(years):
    a, b, c = 31.78, 22.19, 0.31
    return (a / (1 + b * years) + c) / 100

# Proposed better model
def better_model(years):
    initial_vol = 1.0  # 100%
    mature_vol = 0.4   # 40%
    decay_rate = 0.15
    return mature_vol + (initial_vol - mature_vol) * np.exp(-decay_rate * years)

# Create the visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# Plot 1: Volatility over time
ax1.plot(df['Date'], df['Vol_30d'], alpha=0.3, color='lightblue', label='30-day volatility')
ax1.plot(df['Date'], df['Vol_90d'], alpha=0.7, color='blue', label='90-day volatility', linewidth=2)
ax1.plot(df['Date'], df['Vol_365d'], alpha=0.9, color='darkblue', label='365-day volatility', linewidth=2)

ax1.set_ylabel('Annualized Volatility', fontsize=12)
ax1.set_title('Bitcoin Volatility Over Time', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 2.5)  # 0% to 250%

# Format y-axis as percentages
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

# Plot 2: Volatility vs Years with Model Comparison
# Filter data to avoid early noise
filtered_df = df[df['Years'] >= 1].copy()

# Plot actual volatility
ax2.scatter(filtered_df['Years'], filtered_df['Vol_90d'], alpha=0.3, color='blue', s=10, label='Actual 90-day volatility')

# Create smooth trend line
years_range = np.linspace(1, 15, 100)
current_model_line = [current_model(y) for y in years_range]
better_model_line = [better_model(y) for y in years_range]

ax2.plot(years_range, current_model_line, 'r--', linewidth=3, label='Your Current Model', alpha=0.8)
ax2.plot(years_range, better_model_line, 'g-', linewidth=3, label='Proposed Better Model', alpha=0.8)

# Add moving average trend
window_size = 365  # 1 year window
trend_data = filtered_df.groupby(filtered_df['Years'].round()).agg({
    'Vol_90d': 'mean',
    'Years': 'mean'
}).reset_index(drop=True)

ax2.plot(trend_data['Years'], trend_data['Vol_90d'], 'orange', linewidth=4, 
         label='Actual Annual Average', alpha=0.9)

ax2.set_xlabel('Years Since Bitcoin Start', fontsize=12)
ax2.set_ylabel('Annualized Volatility', fontsize=12)
ax2.set_title('Bitcoin Volatility Models vs Reality', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.5)  # 0% to 150%
ax2.set_xlim(1, 15)

# Format y-axis as percentages
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

# Add annotations
current_vol = df['Vol_90d'].iloc[-1]
current_years = df['Years'].iloc[-1]
current_model_pred = current_model(current_years)
better_model_pred = better_model(current_years)

ax2.annotate(f'Current Reality: {current_vol:.1%}', 
             xy=(current_years, current_vol), 
             xytext=(current_years-2, current_vol+0.2),
             arrowprops=dict(arrowstyle='->', color='blue'),
             fontsize=10, fontweight='bold', color='blue')

ax2.annotate(f'Your Model: {current_model_pred:.1%}', 
             xy=(current_years, current_model_pred), 
             xytext=(current_years-2, current_model_pred+0.15),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, fontweight='bold', color='red')

ax2.annotate(f'Better Model: {better_model_pred:.1%}', 
             xy=(current_years, better_model_pred), 
             xytext=(current_years-2, better_model_pred-0.15),
             arrowprops=dict(arrowstyle='->', color='green'),
             fontsize=10, fontweight='bold', color='green')

# Add summary statistics box
stats_text = f"""Key Statistics:
Current Volatility: {current_vol:.1%}
Mean Historical: {df['Vol_90d'].mean():.1%}
Median Historical: {df['Vol_90d'].median():.1%}
Min Historical: {df['Vol_90d'].min():.1%}
Max Historical: {df['Vol_90d'].max():.1%}"""

ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
         verticalalignment='top', fontsize=9, fontfamily='monospace')

plt.tight_layout()

# Save the plot
plt.savefig('bitcoin_volatility_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('bitcoin_volatility_analysis.pdf', bbox_inches='tight')

print("Bitcoin volatility analysis saved as:")
print("- bitcoin_volatility_analysis.png")
print("- bitcoin_volatility_analysis.pdf")

# Show the plot
plt.show()

# Print numerical comparison
print("\nNUMERICAL COMPARISON:")
print("="*50)
print(f"Current Bitcoin volatility: {current_vol:.1%}")
print(f"Your model prediction: {current_model_pred:.1%}")
print(f"Better model prediction: {better_model_pred:.1%}")
print(f"Your model error: {abs(current_vol - current_model_pred)/current_vol:.1%}")
print(f"Better model error: {abs(current_vol - better_model_pred)/current_vol:.1%}")

# Recent volatility trends
print("\nRECENT VOLATILITY TRENDS:")
print("="*30)
recent_years = df[df['Years'] >= 10].copy()
if len(recent_years) > 0:
    yearly_avg = recent_years.groupby(recent_years['Date'].dt.year)['Vol_90d'].mean()
    for year, vol in yearly_avg.items():
        print(f"{year}: {vol:.1%}") 