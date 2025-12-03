"""
Error Distribution Plots for Diameter Prediction by Species

This script generates overlaid error distribution plots showing how
prediction errors are distributed around zero for each tree species.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from diameter_predictor import DiameterPredictor

# Set up matplotlib for better plots
rcParams['figure.figsize'] = (14, 8)
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
    
    return df, predictor


def plot_error_distributions(df: pd.DataFrame, output_path: str = 'error_distribution_by_species.png'):
    """
    Create overlaid error distribution plots for each species.
    """
    # Get unique species
    species = df['Элемент леса'].unique()
    n_species = len(species)
    
    # Create color palette
    colors = plt.cm.tab10(np.linspace(0, 1, n_species))
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # --- Plot 1: Overlaid Histograms ---
    ax1 = axes[0]
    
    for i, sp in enumerate(sorted(species)):
        sp_data = df[df['Элемент леса'] == sp]['error']
        ax1.hist(sp_data, bins=30, alpha=0.5, label=f'{sp} (n={len(sp_data)})', 
                 color=colors[i], edgecolor='white', linewidth=0.5)
    
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax1.set_xlabel('Prediction Error (Actual - Predicted Diameter, cm)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution by Species (Histograms)')
    ax1.legend(loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Kernel Density Estimation (smoother) ---
    ax2 = axes[1]
    
    from scipy import stats
    
    x_range = np.linspace(df['error'].min() - 1, df['error'].max() + 1, 500)
    
    for i, sp in enumerate(sorted(species)):
        sp_data = df[df['Элемент леса'] == sp]['error']
        if len(sp_data) > 1:
            try:
                kde = stats.gaussian_kde(sp_data)
                ax2.plot(x_range, kde(x_range), label=f'{sp} (n={len(sp_data)}, μ={sp_data.mean():.2f})', 
                        color=colors[i], linewidth=2)
                ax2.fill_between(x_range, kde(x_range), alpha=0.2, color=colors[i])
            except Exception:
                # If KDE fails, skip this species
                pass
    
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax2.set_xlabel('Prediction Error (Actual - Predicted Diameter, cm)')
    ax2.set_ylabel('Density')
    ax2.set_title('Error Distribution by Species (Kernel Density Estimation)')
    ax2.legend(loc='upper right', ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    print(f"Plot saved to: {output_path.replace('.png', '.pdf')}")
    
    return fig


def print_error_statistics(df: pd.DataFrame):
    """
    Print error statistics for each species.
    """
    print("\n" + "=" * 80)
    print("ERROR STATISTICS BY SPECIES")
    print("=" * 80)
    
    print(f"\n{'Species':<10} {'Count':>8} {'Mean Err':>10} {'Std Err':>10} {'MAE':>10} {'Min':>10} {'Max':>10}")
    print("-" * 78)
    
    for sp in sorted(df['Элемент леса'].unique()):
        sp_data = df[df['Элемент леса'] == sp]
        errors = sp_data['error']
        
        print(f"{sp:<10} {len(errors):>8} {errors.mean():>10.2f} {errors.std():>10.2f} "
              f"{errors.abs().mean():>10.2f} {errors.min():>10.2f} {errors.max():>10.2f}")
    
    # Overall statistics
    print("-" * 78)
    errors = df['error']
    print(f"{'OVERALL':<10} {len(errors):>8} {errors.mean():>10.2f} {errors.std():>10.2f} "
          f"{errors.abs().mean():>10.2f} {errors.min():>10.2f} {errors.max():>10.2f}")
    
    # R² by species
    print("\n" + "-" * 50)
    print("R² Score by Species:")
    print("-" * 50)
    
    from sklearn.metrics import r2_score
    
    for sp in sorted(df['Элемент леса'].unique()):
        sp_data = df[df['Элемент леса'] == sp]
        if len(sp_data) > 1:
            r2 = r2_score(sp_data['Диаметр'], sp_data['predicted_diameter'])
            print(f"  {sp}: R² = {r2:.4f}")
    
    overall_r2 = r2_score(df['Диаметр'], df['predicted_diameter'])
    print(f"  OVERALL: R² = {overall_r2:.4f}")


def create_detailed_species_plot(df: pd.DataFrame, output_path: str = 'error_by_species_detailed.png'):
    """
    Create a detailed multi-panel plot with one subplot per species.
    """
    species = sorted(df['Элемент леса'].unique())
    n_species = len(species)
    
    # Calculate grid dimensions
    n_cols = 4
    n_rows = int(np.ceil(n_species / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_species))
    
    for i, sp in enumerate(species):
        ax = axes[i]
        sp_data = df[df['Элемент леса'] == sp]['error']
        
        ax.hist(sp_data, bins=20, color=colors[i], edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.axvline(x=sp_data.mean(), color='blue', linestyle='-', linewidth=2, 
                   label=f'Mean: {sp_data.mean():.2f}')
        
        ax.set_title(f'{sp} (n={len(sp_data)})')
        ax.set_xlabel('Error (cm)')
        ax.set_ylabel('Frequency')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_species, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Diameter Prediction Error Distribution by Species\n(Red = Zero Error, Blue = Mean Error)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Detailed plot saved to: {output_path}")
    
    return fig


def main():
    """Main function to generate all error plots."""
    print("=" * 80)
    print("DIAMETER PREDICTION ERROR ANALYSIS BY SPECIES")
    print("=" * 80)
    
    # Load data and make predictions
    print("\nLoading data and making predictions...")
    df, predictor = load_and_predict()
    
    print(f"Total samples: {len(df)}")
    print(f"Unique species: {df['Элемент леса'].nunique()}")
    print(f"Model used: {predictor.model_name}")
    
    # Print statistics
    print_error_statistics(df)
    
    # Create overlaid plot
    print("\n" + "-" * 50)
    print("Generating plots...")
    print("-" * 50)
    
    fig1 = plot_error_distributions(df)
    fig2 = create_detailed_species_plot(df)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - error_distribution_by_species.png (overlaid histograms + KDE)")
    print("  - error_distribution_by_species.pdf (vector format)")
    print("  - error_by_species_detailed.png (individual species plots)")
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()

