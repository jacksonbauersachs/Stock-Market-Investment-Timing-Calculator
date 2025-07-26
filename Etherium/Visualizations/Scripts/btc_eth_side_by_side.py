import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

def create_side_by_side_comparison():
    """Create a side-by-side comparison of Bitcoin and Ethereum price charts"""
    
    # Load the existing images
    eth_image_path = 'Etherium/Visualizations/Images/ethereum_price_history_blue.png'
    btc_image_path = 'Bitcoin/Visualizations/Images/bitcoin_price_history.png'
    
    # Check if images exist
    if not os.path.exists(eth_image_path):
        print(f"Error: {eth_image_path} not found!")
        return
    if not os.path.exists(btc_image_path):
        print(f"Error: {btc_image_path} not found!")
        return
    
    # Load images
    eth_img = mpimg.imread(eth_image_path)
    btc_img = mpimg.imread(btc_image_path)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 18))
    
    # Set background to white
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    
    # Display images
    ax1.imshow(eth_img)
    ax1.axis('off')
    
    ax2.imshow(btc_img)
    ax2.axis('off')
    
    # Adjust layout
    plt.tight_layout(pad=3.0)
    
    # Save the combined image
    save_path = 'Etherium/Visualizations/Images/btc_eth_side_by_side.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Combined chart saved to: {save_path}")
    
    plt.show()

def main():
    """Main function"""
    print("Bitcoin vs Ethereum Side-by-Side Comparison")
    print("="*50)
    
    create_side_by_side_comparison()
    
    print("\n✅ Side-by-side comparison completed!")

if __name__ == "__main__":
    main() 