import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path
import glob
import argparse


def plot_multiple_roc_curves(data_files=None, output_path="combined_roc_curve.png", colors=None, title=None):
    """
    Plot multiple ROC curves on a single figure from saved data files.
    
    Args:
        data_files: List of file paths containing ROC data. If None, all .pkl files in the roc_data directory are used.
        output_path: Path to save the combined ROC curve plot.
        colors: List of colors for ROC curves. If None, default colors are used.
        title: Title for the plot. If None, a default title is used.
    """
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(16, 14))
    
    # Add light shading to differentiate performance regions
    plt.fill_between([0, 1], [0, 1], [0, 0], color='lightgray', alpha=0.3, label='Worse than random')
    plt.fill_between([0, 1], [0, 1], [1, 1], color='lightgreen', alpha=0.3, label='Better than random')
    
    # Add a perfect prediction line
    plt.plot([0, 0, 1], [0, 1, 1], color='green', lw=2, linestyle=':', label='Perfect prediction (AUC = 1.0)')
    
    # If data_files not specified, use all pkl files in the roc_data directory
    if data_files is None:
        script_path = Path(os.path.dirname(os.path.abspath(__file__)))
        data_dir = script_path / "roc_data"
        data_files = glob.glob(str(data_dir / "*.pkl"))
    
    # Default colors if not provided
    if colors is None:
        colors = ['darkorange', 'green', 'blue', 'red', 'purple', 'brown', 'pink', 'gray']
    
    # Make sure we have enough colors
    while len(colors) < len(data_files):
        colors.extend(colors)
    
    for idx, file_path in enumerate(data_files):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            labels = data['labels']
            probs = data['probabilities']
            model_name = os.path.basename(file_path).replace('.pkl', '')
            
            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, color=colors[idx], lw=2, 
                     label=f'{model_name} (AUC = {roc_auc:.4f})')
                
            print(f"Loaded data from {file_path} - AUC: {roc_auc:.4f}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Plotting reference diagonal line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random prediction (AUC = 0.5)')
    
    # Add annotations to explain the plot
    plt.annotate('Perfect classifier', xy=(0.0, 1.0), xytext=(0.3, 0.8),
                 arrowprops=dict(facecolor='green', shrink=0.05))
                 
    plt.annotate('Random classifier', xy=(0.5, 0.5), xytext=(0.6, 0.4),
                 arrowprops=dict(facecolor='navy', shrink=0.05))
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    
    if title:
        plt.title(f"{title}\nUnderstanding ROC Curve Performance")
    else:
        plt.title('Comparison of ROC Curves\nUnderstanding Model Performance')
    
    plt.legend(loc="lower right")

    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add more detail to the graph
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined ROC curve saved as '{output_path}'")
    
    # Return the figure for potential further customization
    return plt.gcf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot multiple ROC curves from saved data')
    parser.add_argument('--files', nargs='+', help='List of ROC data files to plot')
    parser.add_argument('--output', default='combined_roc_curve.png', help='Output file path')
    parser.add_argument('--title', help='Title for the plot')
    args = parser.parse_args()
    
    # Plot the ROC curves
    plot_multiple_roc_curves(args.files, args.output, title=args.title)
    plt.show()
