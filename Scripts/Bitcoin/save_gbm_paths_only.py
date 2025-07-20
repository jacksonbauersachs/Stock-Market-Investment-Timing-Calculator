import pandas as pd
import numpy as np
from datetime import datetime
import os

def save_gbm_paths_only():
    """Just save the GBM price paths to test if saving works"""
    print("Testing GBM price paths saving...")
    
    # Check if we have the GBM paths from the recent run
    # Look for any recent GBM files
    results_dir = "Results/Bitcoin"
    gbm_files = [f for f in os.listdir(results_dir) if "gbm" in f.lower() and f.endswith('.csv')]
    
    if gbm_files:
        print(f"Found existing GBM files: {gbm_files}")
        return
    
    # If no files found, the simulation might not have completed saving
    print("No GBM CSV files found. The simulation might have:")
    print("1. Crashed during the save step")
    print("2. Been interrupted before saving")
    print("3. Had an error in the save function")
    
    print("\nLet's run a quick test to see if we can save files...")
    
    # Create a small test dataset
    test_paths = np.random.rand(10, 100) * 100000  # 10 paths, 100 time steps
    test_times = np.linspace(0, 10, 100)
    
    # Try to save
    try:
        test_filename = f'Results/Bitcoin/test_gbm_paths_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        test_df = pd.DataFrame(test_paths.T, index=test_times, columns=[f'Path_{i+1}' for i in range(10)])
        test_df.index.name = 'Years'
        test_df.to_csv(test_filename)
        print(f"✅ Test save successful: {test_filename}")
        
        # Clean up test file
        os.remove(test_filename)
        print("Test file cleaned up")
        
    except Exception as e:
        print(f"❌ Test save failed: {e}")
    
    print("\nRecommendation: Run the GBM simulation again and watch for any error messages")

if __name__ == "__main__":
    save_gbm_paths_only() 