"""
Error Distribution Plots for Diameter Prediction by Trunk Diameter

This script generates error analysis plots showing how
prediction errors vary across different trunk diameter ranges.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from diameter_predictor import DiameterPredictor

# Set up matplotlib for better plots
rcParams['figure.figsize'] = (14, 10)
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 10


def load_and_predict(csv_path: str = 'combined_data.csv', 
                     model_path: str = 'diameter_predictor_model.joblib'):
    """
    Load data and make predictions for all examples.
    
    Returns:
        DataFrame with actual, predicted, and error columns
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Filter to rows with both diameter and height
    df = df.dropna(subset=['Диаметр', 'Высота'])
    
    # Load trained model
    predictor = DiameterPredictor()
    predictor.load(model_path)
    
    # Make predictions
    predictions = []
    for _, row in df.iterrows():
        pred = predictor.predict(
            composition=row['Состав'],
            element=row['Элемент леса'],
            height=row['Высота']
        )
        predictions.append(pred)
    
    df['predicted_diameter'] = predictions
    df['error'] = df['Диаметр'] - df['predicted_diameter']
    df['abs_error'] = df['error'].abs()
    df['relative_error'] = (df['error'] / df['Диаметр']) * 100  # in percent
    
    return df, predictor


def create_diameter_bins(df: pd.DataFrame, n_bins: int = 10):
    """
    Create diameter bins for analysis.
    """
    # Create bins based on diameter quantiles for more even distribution
    df['diameter_bin'] = pd.qcut(df['Диаметр'], q=n_bins, duplicates='drop')
    
    # Also create fixed-width bins
    min_d, max_d = df['Диаметр'].min(), df['Диаметр'].max()
    bin_edges = np.linspace(min_d, max_d, n_bins + 1)
    df['diameter_bin_fixed'] = pd.cut(df['Диаметр'], bins=bin_edges, include_lowest=True)
    
    return df


def plot_error_by_diameter(df: pd.DataFrame, output_path: str = 'error_by_diameter.png'):
    """
    Create comprehensive error analysis plots by diameter.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Color scheme
    main_color = '#2E86AB'
    accent_color = '#E94F37'
    
    # --- Plot 1: Scatter plot of error vs actual diameter ---
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['Диаметр'], df['error'], 
                          alpha=0.4, c=df['abs_error'], cmap='RdYlGn_r', 
                          s=30, edgecolor='none')
    ax1.axhline(y=0, color=accent_color, linestyle='--', linewidth=2, label='Zero error')
    
    # Add trend line
    z = np.polyfit(df['Диаметр'], df['error'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['Диаметр'].min(), df['Диаметр'].max(), 100)
    ax1.plot(x_trend, p(x_trend), color='navy', linewidth=2, linestyle='-', 
             label=f'Trend (slope={z[0]:.3f})')
    
    ax1.set_xlabel('Actual Diameter (cm)')
    ax1.set_ylabel('Prediction Error (cm)')
    ax1.set_title('Prediction Error vs Actual Diameter')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Absolute Error (cm)')
    
    # --- Plot 2: Mean error by diameter bins ---
    ax2 = axes[0, 1]
    
    # Calculate statistics per bin
    bin_stats = df.groupby('diameter_bin_fixed').agg({
        'error': ['mean', 'std', 'count'],
        'abs_error': 'mean',
        'Диаметр': 'mean'
    }).reset_index()
    bin_stats.columns = ['bin', 'mean_error', 'std_error', 'count', 'mae', 'mean_diameter']
    bin_stats = bin_stats.dropna()
    
    # Bar plot with error bars
    x_pos = range(len(bin_stats))
    bars = ax2.bar(x_pos, bin_stats['mean_error'], 
                   yerr=bin_stats['std_error'], 
                   capsize=4, color=main_color, alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
    
    # Color bars by sign
    for i, bar in enumerate(bars):
        if bin_stats.iloc[i]['mean_error'] < 0:
            bar.set_color('#E94F37')
        else:
            bar.set_color('#2E86AB')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{b.left:.0f}-{b.right:.0f}' for b in bin_stats['bin']], 
                        rotation=45, ha='right')
    ax2.set_xlabel('Diameter Range (cm)')
    ax2.set_ylabel('Mean Error ± Std (cm)')
    ax2.set_title('Mean Prediction Error by Diameter Range')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, bin_stats['count'])):
        height = bar.get_height()
        ax2.annotate(f'n={int(count)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8)
    
    # --- Plot 3: MAE and relative error by diameter ---
    ax3 = axes[1, 0]
    
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(x_pos, bin_stats['mae'], 'o-', color=main_color, 
                     linewidth=2, markersize=8, label='MAE (cm)')
    ax3.fill_between(x_pos, bin_stats['mae'], alpha=0.2, color=main_color)
    
    # Calculate relative MAE
    bin_stats['relative_mae'] = (bin_stats['mae'] / bin_stats['mean_diameter']) * 100
    line2 = ax3_twin.plot(x_pos, bin_stats['relative_mae'], 's--', color=accent_color, 
                          linewidth=2, markersize=8, label='Relative MAE (%)')
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{b.left:.0f}-{b.right:.0f}' for b in bin_stats['bin']], 
                        rotation=45, ha='right')
    ax3.set_xlabel('Diameter Range (cm)')
    ax3.set_ylabel('Mean Absolute Error (cm)', color=main_color)
    ax3_twin.set_ylabel('Relative MAE (%)', color=accent_color)
    ax3.set_title('Absolute and Relative Error by Diameter Range')
    ax3.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right')
    
    # --- Plot 4: Box plot of errors by diameter bins ---
    ax4 = axes[1, 1]
    
    # Prepare data for box plot
    box_data = [df[df['diameter_bin_fixed'] == b]['error'].values 
                for b in bin_stats['bin']]
    
    bp = ax4.boxplot(box_data, patch_artist=True)
    
    # Style the box plot
    for patch in bp['boxes']:
        patch.set_facecolor(main_color)
        patch.set_alpha(0.6)
    for whisker in bp['whiskers']:
        whisker.set_color('gray')
    for cap in bp['caps']:
        cap.set_color('gray')
    for median in bp['medians']:
        median.set_color(accent_color)
        median.set_linewidth(2)
    
    ax4.axhline(y=0, color=accent_color, linestyle='--', linewidth=2)
    ax4.set_xticklabels([f'{b.left:.0f}-{b.right:.0f}' for b in bin_stats['bin']], 
                        rotation=45, ha='right')
    ax4.set_xlabel('Diameter Range (cm)')
    ax4.set_ylabel('Prediction Error (cm)')
    ax4.set_title('Error Distribution by Diameter Range (Box Plot)')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Diameter Prediction Error Analysis by Trunk Diameter', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    print(f"Plot saved to: {output_path.replace('.png', '.pdf')}")
    
    return fig, bin_stats


def print_error_statistics_by_diameter(df: pd.DataFrame, bin_stats: pd.DataFrame):
    """
    Print detailed error statistics by diameter range.
    """
    print("\n" + "=" * 90)
    print("ERROR STATISTICS BY DIAMETER RANGE")
    print("=" * 90)
    
    print(f"\n{'Diameter Range':<18} {'Count':>8} {'Mean Err':>10} {'Std Err':>10} "
          f"{'MAE':>10} {'Rel MAE %':>10}")
    print("-" * 88)
    
    for _, row in bin_stats.iterrows():
        bin_label = f"{row['bin'].left:.0f}-{row['bin'].right:.0f} cm"
        rel_mae = (row['mae'] / row['mean_diameter']) * 100
        print(f"{bin_label:<18} {int(row['count']):>8} {row['mean_error']:>10.2f} "
              f"{row['std_error']:>10.2f} {row['mae']:>10.2f} {rel_mae:>10.1f}%")
    
    # Overall statistics
    print("-" * 88)
    overall_mean = df['error'].mean()
    overall_std = df['error'].std()
    overall_mae = df['abs_error'].mean()
    overall_rel_mae = (df['abs_error'] / df['Диаметр']).mean() * 100
    print(f"{'OVERALL':<18} {len(df):>8} {overall_mean:>10.2f} {overall_std:>10.2f} "
          f"{overall_mae:>10.2f} {overall_rel_mae:>10.1f}%")
    
    # Bias analysis
    print("\n" + "-" * 50)
    print("BIAS ANALYSIS:")
    print("-" * 50)
    
    # Small vs large diameter bias
    median_d = df['Диаметр'].median()
    small_d = df[df['Диаметр'] <= median_d]
    large_d = df[df['Диаметр'] > median_d]
    
    print(f"  Median diameter: {median_d:.1f} cm")
    print(f"  Small diameters (≤{median_d:.0f} cm): Mean error = {small_d['error'].mean():.2f} cm")
    print(f"  Large diameters (>{median_d:.0f} cm): Mean error = {large_d['error'].mean():.2f} cm")
    
    # Correlation between diameter and error
    corr = df['Диаметр'].corr(df['error'])
    print(f"\n  Correlation (diameter vs error): {corr:.4f}")
    
    if corr > 0.1:
        print("  → Model tends to UNDERESTIMATE larger diameters")
    elif corr < -0.1:
        print("  → Model tends to OVERESTIMATE larger diameters")
    else:
        print("  → No significant diameter-dependent bias")


def main():
    """Main function to generate error analysis by diameter."""
    print("=" * 90)
    print("DIAMETER PREDICTION ERROR ANALYSIS BY TRUNK DIAMETER")
    print("=" * 90)
    
    # Load data and make predictions
    print("\nLoading data and making predictions...")
    df, predictor = load_and_predict()
    
    print(f"Total samples: {len(df)}")
    print(f"Diameter range: {df['Диаметр'].min():.1f} - {df['Диаметр'].max():.1f} cm")
    print(f"Model used: {predictor.model_name}")
    
    # Create diameter bins
    df = create_diameter_bins(df, n_bins=10)
    
    # Create plots
    print("\n" + "-" * 50)
    print("Generating plots...")
    print("-" * 50)
    
    fig, bin_stats = plot_error_by_diameter(df)
    
    # Print statistics
    print_error_statistics_by_diameter(df, bin_stats)
    
    print("\n" + "=" * 90)
    print("ANALYSIS COMPLETE")
    print("=" * 90)
    print("\nGenerated files:")
    print("  - error_by_diameter.png (comprehensive error analysis)")
    print("  - error_by_diameter.pdf (vector format)")
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()

