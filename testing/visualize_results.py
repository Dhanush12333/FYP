# testing/visualize_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_test_results():
    """Create comprehensive visualization of test results"""
    
    # Load metrics
    metrics_df = pd.read_csv("test_results/test_metrics.csv")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. SSIM Distribution
    axes[0,0].hist(metrics_df['ssim'], bins=20, edgecolor='black', alpha=0.7)
    axes[0,0].axvline(metrics_df['ssim'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {metrics_df["ssim"].mean():.3f}')
    axes[0,0].set_xlabel('SSIM')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('SSIM Distribution')
    axes[0,0].legend()
    
    # 2. PSNR Distribution
    axes[0,1].hist(metrics_df['psnr'], bins=20, edgecolor='black', alpha=0.7)
    axes[0,1].axvline(metrics_df['psnr'].mean(), color='red', linestyle='--',
                      label=f'Mean: {metrics_df["psnr"].mean():.1f} dB')
    axes[0,1].set_xlabel('PSNR (dB)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('PSNR Distribution')
    axes[0,1].legend()
    
    # 3. SSIM vs Missing Index
    axes[1,0].scatter(metrics_df['missing_idx'], metrics_df['ssim'], alpha=0.6)
    axes[1,0].set_xlabel('Missing Slice Index')
    axes[1,0].set_ylabel('SSIM')
    axes[1,0].set_title('SSIM by Slice Position')
    
    # 4. Metrics Correlation
    corr = metrics_df[['ssim', 'psnr', 'mae', 'mse']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[1,1])
    axes[1,1].set_title('Metrics Correlation')
    
    plt.tight_layout()
    plt.savefig("test_results/metrics_visualization.png", dpi=150)
    plt.show()
    
    print("✅ Visualization saved to test_results/metrics_visualization.png")

if __name__ == "__main__":
    visualize_test_results()