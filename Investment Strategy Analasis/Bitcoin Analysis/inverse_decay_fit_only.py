from Bitcoin_Complete_Volatility_Analysis import CompleteBitcoinVolatilityAnalyzer

def main():
    data_path = "Data Sets/Bitcoin Data/Bitcoin Historical Data Full.csv"
    analyzer = CompleteBitcoinVolatilityAnalyzer(data_path)
    analyzer.fit_inverse_decay_all_windows()

if __name__ == "__main__":
    main() 