"""
Visualization utilities for plotting predictions and analysis results.
"""
import logging
from typing import Optional, Union, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

def plot_predictions(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot actual vs predicted values.
    
    Args:
        df: DataFrame with datetime index
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
    
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get x-axis values (dates)
    x = df.index
    
    # Plot actual and predicted values
    ax.plot(x, y_true, 'b-', label='Actual')
    ax.plot(x, y_pred, 'r--', label='Predicted')
    
    # Add title and labels
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Actual vs Predicted Values')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Figure saved to {save_path}")
    
    return fig

def plot_feature_importance(
    feature_names: List[str],
    importance_values: np.ndarray,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        feature_names: List of feature names
        importance_values: Feature importance values
        top_n: Number of top features to show
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure
    """
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Get top_n features
    importance_df = importance_df.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot
    sns.barplot(
        x='Importance',
        y='Feature',
        data=importance_df,
        ax=ax,
        palette='viridis'
    )
    
    # Add title and labels
    ax.set_title(f'Top {top_n} Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Figure saved to {save_path}")
    
    return fig 