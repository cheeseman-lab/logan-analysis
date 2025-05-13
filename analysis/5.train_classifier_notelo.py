import os
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Standard library imports
import os
import json

# Data manipulation and analysis
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, label_binarize, LabelEncoder
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report as sklearn_classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Optional ML libraries (with try/except handling in your code)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Serialization
import dill

# Import 
from lib.aggregate.cell_classification import CellClassifier
from lib.aggregate.cell_data_utils import split_cell_data, DEFAULT_METADATA_COLS

# Abstract base classes
from abc import ABC, abstractmethod

def setup_publication_plot_style():
    """Set matplotlib parameters for publication-ready plots."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True,
        'figure.figsize': (10, 8),
        'figure.dpi': 300
    })
    
    # Create a distinct color palette different from the benchmark scripts
    # Using a blue to purple to red custom colormap without trying to register it
    colors = [(0.0, 0.4, 0.8), (0.6, 0.0, 0.8), (0.8, 0.0, 0.4)]  # Blue to purple to red
    cell_classifier_cmap = LinearSegmentedColormap.from_list('cell_classifier', colors, N=100)
    
    return cell_classifier_cmap

def create_run_directories():
    """Create directories for current run with timestamp."""
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    
    # Create base output directories
    base_output_dir = os.path.join('analysis_root', 'classifier_notelo')
    run_dir = os.path.join(base_output_dir, run_name)
    
    # Create subdirectories
    statistics_dir = os.path.join(run_dir, 'statistics')
    models_dir = os.path.join(run_dir, 'models')
    plots_dir = os.path.join(run_dir, 'plots')
    results_dir = os.path.join(run_dir, 'results')
    
    for directory in [statistics_dir, models_dir, plots_dir, results_dir]:
        os.makedirs(directory, exist_ok=True)
        
    print(f"Created output directories for run: {run_name}")
    
    return {
        'base': base_output_dir,
        'run': run_dir,
        'statistics': statistics_dir,
        'models': models_dir,
        'plots': plots_dir,
        'results': results_dir,
        'timestamp': timestamp
    }

def load_cellprofiler_data(file_paths):
    """
    Load and combine multiple CellProfiler parquet files
    
    Parameters:
    -----------
    file_paths : list
        List of file paths to parquet files
        
    Returns:
    --------
    pd.DataFrame
        Combined data from all files
    """
    all_data = []
    
    for file_path in file_paths:
        # Load data
        data = pd.read_parquet(file_path)
        all_data.append(data)
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    # Reset index
    combined_data.reset_index(drop=True, inplace=True)
    
    print(f"Loaded {len(combined_data)} cells from {len(file_paths)} files")
    return combined_data

def apply_cell_category_mapping(data, label_col, remove_phases=None, output_col='phase', 
                              category_col='category', mapping=None, 
                              default_category="Unknown", verbose=True):
    """
    Apply a flexible mapping to categorize cell labels into phases and categories,
    and optionally remove unwanted phases
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the cell data
    label_col : str
        Name of the column containing the original cell labels
    remove_phases : list or None
        List of phase names to remove (e.g., ["Discard", "Unsure"])
    output_col : str
        Name of the column to store the mapped phase names
    category_col : str
        Name of the column to store the simplified categories (e.g., Interphase/Mitotic)
    mapping : dict
        Custom mapping dictionary with the following format:
        {
            'label_to_phase': {label1: 'Phase1', label2: 'Phase2', ...},
            'phase_to_category': {'Phase1': 'Category1', 'Phase2': 'Category2', ...},
            'category_colors': {'Category1': '#color1', 'Category2': '#color2', ...},
            'phase_colors': {'Phase1': '#color1', 'Phase2': '#color2', ...}
        }
    default_category : str
        Default category to use for unmapped labels
    verbose : bool
        Whether to print summary of mapping
        
    Returns:
    --------
    pd.DataFrame
        Modified DataFrame with new columns for phase and category, with unwanted phases removed
    """
    # Create a copy to avoid modifying the original
    result_df = data.copy()
    
    # Check if label column exists
    if label_col not in result_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame")
    
    # Apply default mapping for cell cycle if none provided
    if mapping is None:
        raise ValueError("Mapping dictionary is required")
    
    # Ensure all required mapping keys exist
    required_keys = ['label_to_phase', 'phase_to_category']
    missing_keys = [key for key in required_keys if key not in mapping]
    if missing_keys:
        raise ValueError(f"Missing required mapping keys: {missing_keys}")
    
    # Apply label to phase mapping
    phase_mapping = mapping['label_to_phase']
    result_df[output_col] = result_df[label_col].map(
        lambda x: phase_mapping.get(x, default_category)
    )
    
    # Apply phase to category mapping
    category_mapping = mapping['phase_to_category']
    result_df[category_col] = result_df[output_col].map(
        lambda x: category_mapping.get(x, default_category)
    )
    
    # Store phase and category colors if provided
    if 'phase_colors' in mapping:
        # Store as attribute for later use in plotting
        result_df.attrs['phase_colors'] = mapping['phase_colors']
    
    if 'category_colors' in mapping:
        # Store as attribute for later use in plotting
        result_df.attrs['category_colors'] = mapping['category_colors']
    
    # Print summary if verbose
    if verbose:
        print("\nCategory Mapping Summary:")
        print(f"Original labels in {label_col}: {sorted(result_df[label_col].unique())}")
        print(f"Mapped phases in {output_col}: {sorted(result_df[output_col].unique())}")
        print(f"Mapped categories in {category_col}: {sorted(result_df[category_col].unique())}")
        
        # Display counts for phases and categories
        print("\nPhase Distribution:")
        phase_counts = result_df[output_col].value_counts()
        for phase, count in phase_counts.items():
            percentage = count / len(result_df) * 100
            print(f"  {phase}: {count} cells ({percentage:.1f}%)")
            
        print("\nCategory Distribution:")
        category_counts = result_df[category_col].value_counts()
        for category, count in category_counts.items():
            percentage = count / len(result_df) * 100
            print(f"  {category}: {count} cells ({percentage:.1f}%)")
    
    # Remove unwanted phases if specified
    if remove_phases:
        original_count = len(result_df)
        result_df = result_df[~result_df[output_col].isin(remove_phases)]
        removed_count = original_count - len(result_df)
        
        if verbose and removed_count > 0:
            print(f"\nRemoved {removed_count} cells with phases: {remove_phases}")
            print(f"Remaining cells: {len(result_df)}")
    
    return result_df


def select_features_from_split(features_df, feature_markers=None, exclude_markers=None, 
                              exclude_cols=None, remove_nan=True, verbose=True):
    """
    Select features from the features dataframe
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        Features dataframe from split_cell_data
    feature_markers : dict
        Dictionary of marker names to include mapped to True/False
        Example: {'DAPI': True, 'ACTIN': False}
    exclude_markers : list
        List of marker strings to exclude from features
    exclude_cols : list
        List of specific columns to exclude from features
    remove_nan : bool
        Whether to remove columns with NaN values
    verbose : bool
        Whether to print details about selected features
        
    Returns:
    --------
    list
        List of selected feature column names
    """
    # Default values
    if feature_markers is None:
        feature_markers = {'DAPI': True}  # By default, only include DAPI features
    
    if exclude_markers is None:
        exclude_markers = []
        if verbose:
            print("No markers to exclude")
    
    if exclude_cols is None:
        exclude_cols = []
        if verbose:
            print("No specific columns to exclude")
    
    # Get all columns as potential features
    all_feature_cols = features_df.columns.tolist()
    
    # Remove explicitly excluded columns
    feature_cols = [col for col in all_feature_cols 
                   if col not in exclude_cols and 
                   not any(col.startswith(ex) for ex in exclude_cols)]
    
    # Initialize feature lists by marker
    feature_sets = {marker: [] for marker in feature_markers}
    feature_sets['morphology'] = []  # For features not associated with any marker
    
    # Categorize features by marker
    for col in feature_cols:
        # Check if column belongs to any marker category
        assigned = False
        for marker in feature_markers:
            if marker in col and not any(ex in col for ex in exclude_markers):
                feature_sets[marker].append(col)
                assigned = True
                break
        
        # If not assigned to any marker, it's a morphology feature
        if not assigned and not any(ex in col for ex in exclude_markers):
            feature_sets['morphology'].append(col)
    
    # Combine selected feature sets based on user's choices
    selected_features = []
    for marker, include in feature_markers.items():
        if include and marker in feature_sets:
            selected_features.extend(feature_sets[marker])
    
    # Always include morphology features unless explicitly turned off
    if feature_markers.get('morphology', True):
        selected_features.extend(feature_sets['morphology'])
    
    # Remove duplicates and sort for consistency
    selected_features = list(set(selected_features))
    selected_features.sort()
    
    # Remove columns with NaN values if requested
    if remove_nan:
        if verbose:
            print(f"Number of selected features before NaN removal: {len(selected_features)}")
        selected_features = [col for col in selected_features if features_df[col].notna().all()]
        if verbose:
            print(f"Number of selected features after NaN removal: {len(selected_features)}")
    
    # Print feature information if verbose
    if verbose:
        print("\nFeature Selection Summary:")
        print(f"Total features selected: {len(selected_features)}")
        print("\nFeatures by category:")
        for category, features in feature_sets.items():
            if feature_markers.get(category, category == 'morphology'):
                included = [f for f in features if f in selected_features]
                print(f"  {category}: {len(included)} features")
                if len(included) > 0 and len(included) <= 5:
                    print(f"    Examples: {', '.join(included)}")
                elif len(included) > 5:
                    print(f"    Examples: {', '.join(included[:5])}...")
    
    return selected_features

def plot_distribution_statistics(data, target_col, split_col=None, output_dir='.', prefix='', 
                              exclude_values=None, target_order=None, figsize=(10, 7), 
                              palette=None, dpi=300):
    """
    Generate summary statistics and plots for any categorical data distribution
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data frame containing the target column
    target_col : str
        Name of the target column to analyze
    split_col : str or None
        Name of the column to use for splitting analysis, or None
    output_dir : str
        Directory to save plots
    prefix : str
        Prefix for output filenames
    exclude_values : list
        Values to exclude from the target column
    target_order : list
        Custom order for target column values in plots
    figsize : tuple
        Default figure size for plots
    palette : str, list, or colormap
        Color palette for plots. If None, will use default seaborn palette.
    dpi : int
        Resolution for saved figures
        
    Returns:
    --------
    dict
        Dictionary containing summary statistics
    """
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    if split_col and split_col not in data.columns:
        raise ValueError(f"Split column '{split_col}' not found in data")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle palette - use default seaborn palette if the custom one is unavailable
    if palette == 'cell_classifier':
        palette = None  # Use default seaborn palette instead
    
    # Filter data to exclude unwanted values
    filtered_data = data
    if exclude_values:
        filtered_data = data[~data[target_col].isin(exclude_values)]
    
    # Get unique values for target column
    if target_order is None:
        target_values = sorted(filtered_data[target_col].unique().tolist())
    else:
        # Use provided order but ensure all values are included
        available_values = set(filtered_data[target_col].unique())
        target_values = [v for v in target_order if v in available_values]
        # Add any missing values at the end
        missing_values = sorted(list(available_values - set(target_values)))
        target_values.extend(missing_values)
    
    # 1. Overall target distribution
    plt.figure(figsize=figsize)
    # Fix for FutureWarning - use hue parameter instead of direct palette
    ax = sns.countplot(
        data=filtered_data, 
        x=target_col,
        hue=target_col,  # Use hue instead of direct palette
        order=target_values,
        hue_order=target_values,
        palette=palette,  # Use the palette parameter directly
        legend=False  # Don't show the legend since it's redundant
    )
    
    # Add count labels
    for p in ax.patches:
        ax.annotate(
            f'{int(p.get_height())}', 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', 
            va='bottom', 
            fontsize=10,
            fontweight='bold'
        )
    
    plt.title(f'{target_col} Distribution', fontsize=16, fontweight='bold')
    plt.xlabel(target_col, fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add subtle grid lines for readability
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}{target_col}_distribution.png"), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 2. Distribution by split variable (if provided)
    if split_col:
        plt.figure(figsize=figsize)
        
        # Group by split column and target
        split_counts = filtered_data.groupby([split_col, target_col]).size().unstack(fill_value=0)
        
        # Calculate percentages
        split_percentages = split_counts.div(split_counts.sum(axis=1), axis=0) * 100
        
        # Plot stacked percentages with enhanced styling
        ax = split_percentages.plot(
            kind='bar', 
            stacked=True,
            figsize=figsize,
            colormap=palette if isinstance(palette, str) else None,
            color=palette if not isinstance(palette, str) else None
        )
        
        plt.title(f'{target_col} Distribution by {split_col}', fontsize=16, fontweight='bold')
        plt.xlabel(split_col, fontsize=14)
        plt.ylabel('Percentage (%)', fontsize=14)
        plt.xticks(rotation=0)
        
        # Enhance legend
        leg = plt.legend(title=target_col, title_fontsize=12, frameon=True, fancybox=True, framealpha=0.9, fontsize=10)
        leg.get_frame().set_edgecolor('lightgray')
        
        # Add value labels for bars > 5%
        for i, bar in enumerate(ax.patches):
            if bar.get_height() > 5:  # Only label bars with significant height
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_y() + bar.get_height()/2,
                    f'{bar.get_height():.1f}%',
                    ha='center',
                    va='center',
                    fontsize=9,
                    fontweight='bold',
                    color='white'
                )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}{target_col}_by_{split_col}.png"), dpi=dpi, bbox_inches='tight')
        plt.close()
        
        # For each target value, calculate percentage by split variable
        for target_value in target_values:
            if target_value in filtered_data[target_col].unique():
                value_percent = filtered_data.groupby(split_col)[target_col].apply(
                    lambda x: sum(x == target_value) / len(x) * 100
                ).reset_index()
                value_percent.columns = [split_col, f'{target_value} Percentage (%)']
                
                plt.figure(figsize=figsize)
                
                # Use a simple color for the bar plot
                ax = sns.barplot(
                    x=split_col, 
                    y=f'{target_value} Percentage (%)', 
                    data=value_percent,
                    color='#3498db',  # Use a default blue
                    alpha=0.8
                )
                
                # Add value labels on top of bars
                for i, v in enumerate(value_percent[f'{target_value} Percentage (%)']):
                    ax.text(
                        i, 
                        v + 0.5, 
                        f'{v:.1f}%', 
                        ha='center',
                        fontsize=10,
                        fontweight='bold'
                    )
                
                plt.title(f'{target_value} Percentage by {split_col}', fontsize=16, fontweight='bold')
                plt.ylabel(f'Percentage of {target_value}', fontsize=14)
                plt.xlabel(split_col, fontsize=14)
                plt.ylim(0, max(value_percent[f'{target_value} Percentage (%)']) * 1.2)  # Add some headroom
                
                # Add subtle grid lines
                ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{prefix}{target_value}_percent_by_{split_col}.png"), dpi=dpi, bbox_inches='tight')
                plt.close()
    
    # 3. Collect and return summary statistics
    stats = {
        'total_items': len(filtered_data),
        f'{target_col}_counts': filtered_data[target_col].value_counts().to_dict(),
        f'{target_col}_percentages': (filtered_data[target_col].value_counts(normalize=True) * 100).to_dict()
    }
    
    if split_col:
        stats[f'{target_col}_by_{split_col}'] = {
            split_val: filtered_data[filtered_data[split_col] == split_val][target_col].value_counts().to_dict()
            for split_val in filtered_data[split_col].unique()
        }
        
        # Include percentage of each target value by split variable
        for target_value in target_values:
            stats[f'{target_value}_percentage_by_{split_col}'] = filtered_data.groupby(split_col)[target_col].apply(
                lambda x: sum(x == target_value) / len(x) * 100
            ).to_dict()
    
    return stats

# Feature selection function
def enhance_feature_selection(features_df, target_series, selected_features=None, 
                             remove_low_variance=True, variance_threshold=0.01,
                             remove_correlated=True, correlation_threshold=0.95,
                             select_k_best=None, output_dir=None, prefix=None):
    """
    Advanced feature selection to improve model performance
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        Features dataframe
    target_series : pd.Series
        Target series (class labels)
    selected_features : list or None
        Pre-selected feature columns. If None, use all features in features_df
    remove_low_variance : bool
        Whether to remove features with low variance
    variance_threshold : float
        Threshold for variance filtering
    remove_correlated : bool
        Whether to remove highly correlated features
    correlation_threshold : float
        Threshold for correlation filtering (features with correlation above this will be reduced)
    select_k_best : int or None
        Number of top features to select based on statistical tests
    output_dir : str or None
        Directory to save feature selection artifacts
    prefix : str or None
        Prefix for artifact filenames
        
    Returns:
    --------
    list
        List of selected feature column names
    """
    from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    # Use provided features or all features
    if selected_features is None:
        selected_features = features_df.columns.tolist()
        
    # Start with selected features
    X = features_df[selected_features]
    y = target_series
    
    # Final selected features
    final_features = selected_features.copy()
    original_count = len(final_features)
    
    # Step 1: Remove low variance features
    if remove_low_variance:
        # Apply variance threshold
        selector = VarianceThreshold(threshold=variance_threshold)
        selector.fit(X)
        # Get mask of selected features
        support = selector.get_support()
        # Update feature list
        low_var_removed = [feat for i, feat in enumerate(final_features) if support[i]]
        
        print(f"Removed {len(final_features) - len(low_var_removed)} low variance features")
        final_features = low_var_removed
    
    # Step 2: Remove highly correlated features
    if remove_correlated and len(final_features) > 1:
        # Calculate correlation matrix
        X_filtered = features_df[final_features]
        corr_matrix = X_filtered.corr().abs()
        
        # Create a mask for the upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
        
        # Update feature list
        corr_removed = [feat for feat in final_features if feat not in to_drop]
        
        print(f"Removed {len(final_features) - len(corr_removed)} highly correlated features")
        final_features = corr_removed
        
        # Plot correlation matrix if output directory provided
        if output_dir and prefix:
            os.makedirs(output_dir, exist_ok=True)
            plt.figure(figsize=(15, 15))
            sns.heatmap(features_df[final_features].corr(), annot=False, cmap='cell_classifier', 
                      vmin=-1, vmax=1, square=True)
            plt.title('Feature Correlation Matrix After Filtering', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}_correlation_matrix.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    # Step 3: Select top K features based on ANOVA F-value
    if select_k_best and select_k_best < len(final_features):
        X_filtered = features_df[final_features]
        
        # Apply SelectKBest with f_classif (ANOVA F-value) for classification
        selector = SelectKBest(f_classif, k=select_k_best)
        selector.fit(X_filtered, y)
        
        # Get selected feature indices
        support = selector.get_support()
        
        # Get feature scores
        scores = selector.scores_
        
        # Extract selected features
        anova_selected = [feat for i, feat in enumerate(final_features) if support[i]]
        
        # Create a dataframe with feature names and scores
        feature_scores = pd.DataFrame({
            'Feature': final_features,
            'Score': scores
        })
        feature_scores = feature_scores.sort_values(by='Score', ascending=False)
        
        print(f"Selected top {len(anova_selected)} features based on ANOVA F-value")
        final_features = anova_selected
        
        # Plot feature importance if output directory provided
        if output_dir and prefix:
            num_features_to_plot = min(50, len(feature_scores))
            plt.figure(figsize=(12, max(8, num_features_to_plot/3)))  # Adjust height based on feature count
            
            # Color mapping based on score percentile
            max_score = feature_scores['Score'].max()
            feature_scores['Normalized_Score'] = feature_scores['Score'] / max_score
            
            # Create a color gradient for the bars
            colors = [plt.cm.viridis(score) for score in feature_scores.head(num_features_to_plot)['Normalized_Score']]
            
            # Plot with enhanced styling
            ax = sns.barplot(
                x='Score', 
                y='Feature', 
                data=feature_scores.head(num_features_to_plot),
                palette=colors
            )
            
            plt.title('Top Features by ANOVA F-value', fontsize=16, fontweight='bold')
            plt.xlabel('F-value Score', fontsize=14)
            plt.ylabel('Feature', fontsize=14)
            
            # Add value labels
            for i, v in enumerate(feature_scores.head(num_features_to_plot)['Score']):
                ax.text(v + max_score * 0.01, i, f'{v:.1f}', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}_top_features.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save feature scores
            feature_scores.to_csv(os.path.join(output_dir, f"{prefix}_feature_scores.csv"), index=False)
    
    print(f"Feature selection: {original_count} features -> {len(final_features)} features")
    
    return final_features


class CellClassifier(ABC):
    """Base class for cell classifiers."""

    @abstractmethod
    def classify_cells(self, metadata, features):
        """Classify cells based on feature data.

        Takes DataFrames with metadata and features for cells.
        Uses features to determine the class of each cell and the confidence of that classification.
        The class and confidence are added to the metadata DataFrame.
        Return the modified metadata and features DataFrames.
        """
        print("No classification method defined! Returning orginal cell data...")

    def save(self, filename):
        """Save the classifier to a file."""
        with open(filename, "wb") as f:
            dill.dump(self, f)

    @staticmethod
    def load(filename):
        """Load a classifier from a file."""
        with open(filename, "rb") as f:
            return dill.load(f)


class SciKitCellClassifier(CellClassifier):
    """Classifier for cell cycle phases using scikit-learn models with support for split data."""
    def __init__(self, model, features, class_mapping=None, binary=True, label_encoder=None, encoded_classes=None):
        self.model = model
        self.features = features
        self.class_mapping = class_mapping
        self.binary = binary
        self.label_encoder = label_encoder
        self.encoded_classes = encoded_classes
        
        # Get the classes from the model or label encoder
        if hasattr(model, 'classes_'):
            self.classes_ = model.classes_
        elif encoded_classes is not None:
            self.classes_ = encoded_classes
        else:
            # Default for binary classification
            self.classes_ = np.array(['Interphase', 'Mitotic']) if binary else None

        # Default phase to category mapping if not provided
        self.phase_to_category = {
            "Interphase": "Interphase",
            "Prometaphase": "Mitotic",
            "Metaphase": "Mitotic",
            "Anaphase": "Mitotic",
            "Telophase": "Mitotic"
        }
        
    def classify_cells(self, metadata_df, features_df):
        """
        Classify cells based on metadata and feature dataframes
        
        Parameters:
        -----------
        metadata_df : pd.DataFrame
            DataFrame containing cell metadata
        features_df : pd.DataFrame
            DataFrame containing cell features
                
        Returns:
        --------
        tuple
            (metadata_df, features_df): Updated DataFrames with classification results,
            with rows containing NaN values removed
        """
        # Check if all required features are present
        feature_cols = self.features
        missing_features = [f for f in feature_cols if f not in features_df.columns]
        
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Extract features
        X = features_df[feature_cols]
        
        # Filter out rows with NaN values in the feature columns
        valid_mask = ~X.isna().any(axis=1)
        if not valid_mask.all():
            removed_count = (~valid_mask).sum()
            print(f"Removing {removed_count} rows with NaN values in features")
            X = X.loc[valid_mask]
            metadata_df = metadata_df.loc[valid_mask]
            features_df = features_df.loc[valid_mask]
        
        if len(X) == 0:
            print("Warning: No valid rows found for classification")
            return metadata_df, features_df
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # If using label encoder, convert numeric predictions back to original labels
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)
        
        # Get confidence scores (maximum probability)
        confidences = np.max(probabilities, axis=1)
        
        # Apply class mapping if provided
        if self.class_mapping is not None:
            classes = [self.class_mapping.get(cls, cls) for cls in predictions]
        else:
            classes = predictions
        
        # Create result DataFrame
        result_data = pd.DataFrame(index=X.index)
        
        # For all classifiers, both binary and multiclass:
        # - 'class' column will always be the binary category (Interphase/Mitotic)
        # - For multiclass, add 'phase' column with the detailed classification
        
        if self.binary:
            # For binary classifiers, 'class' is already the category
            result_data["class"] = classes
        else:
            # For multiclass classifiers:
            # 1. Store original predictions as 'phase'
            result_data["phase"] = classes
            
            # 2. Derive 'class' from phases using phase_to_category mapping
            result_data["class"] = [self.phase_to_category.get(phase, "Other") for phase in classes]
        
        # Add confidence scores for all models
        result_data["confidence"] = confidences
        
        # Add the classification results to the metadata
        metadata_df = pd.concat([metadata_df, result_data], axis=1)
        
        return metadata_df, features_df


    # Add a method to set custom phase to category mapping
    def set_phase_to_category_mapping(self, mapping):
        """
        Set a custom mapping from detailed phases to categories
        
        Parameters:
        -----------
        mapping : dict
            Dictionary mapping from phase names to category names
        """
        self.phase_to_category = mapping
    
    @classmethod
    def from_training(cls, metadata_df, features_df, target_column, selected_features=None,
                    model_type='svc', scaler_type='standard', do_grid_search=True, class_mapping=None,
                    output_dir=None, prefix=None, plot_results=True, 
                    retrain_on_full_data=True, 
                    enhance_features=False,
                    remove_low_variance=True, variance_threshold=0.01,
                    remove_correlated=True, correlation_threshold=0.95,
                    select_k_best=None):
        """
        Create a classifier by training on split data
        
        Parameters:
        -----------
        metadata_df : pd.DataFrame
            Metadata dataframe with target labels
        features_df : pd.DataFrame
            Features dataframe
        target_column : str
            Column name with target labels in metadata_df
        selected_features : list or None
            List of feature columns to use. If None, use all features in features_df
        model_type : str
            Type of model to train ('svc', 'rf', 'lgb', 'xgb')
        scaler_type : str
            Type of scaler to use ('standard', 'robust', 'minmax', 'none')
        do_grid_search : bool
            Whether to perform grid search for hyperparameters
        class_mapping : dict or None
            Optional mapping from model output classes to desired output classes
        output_dir : str or None
            Directory to save training artifacts
        prefix : str or None
            Prefix for artifact filenames
        plot_results : bool
            Whether to generate and save evaluation plots
        retrain_on_full_data : bool
            Whether to retrain the model on the full dataset after finding optimal parameters
        enhance_features : bool
            Whether to apply enhanced feature selection
        remove_low_variance : bool
            Whether to remove features with low variance
        variance_threshold : float
            Threshold for variance filtering
        remove_correlated : bool
            Whether to remove highly correlated features
        correlation_threshold : float
            Threshold for correlation filtering
        select_k_best : int or None
            Number of top features to select based on statistical tests
            
        Returns:
        --------
        SciKitCellClassifier
            Trained classifier
        """       
        # Create output directory if needed
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create model-specific subdirectory
            model_dir = os.path.join(output_dir, prefix) if prefix else output_dir
            os.makedirs(model_dir, exist_ok=True)
            
            # Create feature selection subdirectory
            feature_dir = os.path.join(model_dir, 'features')
            os.makedirs(feature_dir, exist_ok=True)
            
            # Create evaluation subdirectory
            eval_dir = os.path.join(model_dir, 'evaluation')
            os.makedirs(eval_dir, exist_ok=True)
        else:
            model_dir = None
            feature_dir = None
            eval_dir = None
        
        # Check if target column exists in metadata
        if target_column not in metadata_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in metadata_df")
        
        # Use provided features or all features
        if selected_features is None:
            selected_features = features_df.columns.tolist()
        
        # Check if features exist in the dataframe
        missing_features = [f for f in selected_features if f not in features_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Apply enhanced feature selection if requested
        if enhance_features:            
            selected_features = enhance_feature_selection(
                features_df=features_df,
                target_series=metadata_df[target_column],
                selected_features=selected_features,
                remove_low_variance=remove_low_variance,
                variance_threshold=variance_threshold,
                remove_correlated=remove_correlated,
                correlation_threshold=correlation_threshold,
                select_k_best=select_k_best,
                output_dir=feature_dir,
                prefix=prefix
            )

            # Save selected features list
            if feature_dir:
                feature_path = os.path.join(feature_dir, f"{prefix}_features.txt")
                with open(feature_path, 'w') as f:
                    f.write('\n'.join(selected_features))
                print(f"Saved selected features to {feature_path}")
        
        # For XGBoost and LightGBM, we need a label encoder
        needs_label_encoder = model_type in ['xgb', 'lgb']
        label_encoder = None
        encoded_classes = None


        if needs_label_encoder:
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            # Fit the label encoder
            y = metadata_df[target_column]
            y_encoded = label_encoder.fit_transform(y)
            encoded_classes = label_encoder.classes_
        
        # Check if binary classification
        unique_classes = metadata_df[target_column].unique()
        is_binary = len(unique_classes) == 2
        
        # Set up scaler based on scaler_type
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'none':
            scaler = None
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # Create pipeline steps
        pipeline_steps = []
               
        # Add scaler if not None
        if scaler is not None:
            pipeline_steps.append((scaler_type + '_scaler', scaler))
        
        # Create model pipeline based on model_type
        if model_type == 'svc':
            # Set default param grid
            param_grid = {
                'svc__C': [0.1, 1, 10, 100],
                'svc__gamma': ['scale', 'auto', 0.1, 0.01],
                'svc__kernel': ['rbf', 'linear']
            }
            
            pipeline_steps.append(('svc', SVC(probability=True, class_weight='balanced')))
        
        elif model_type == 'rf':           
            param_grid = {
                'rf__n_estimators': [100, 200, 300],
                'rf__max_depth': [None, 10, 20, 30]
            }
            
            pipeline_steps.append(('rf', RandomForestClassifier(random_state=42, class_weight='balanced')))
        
        elif model_type == 'xgb':
            param_grid = {
                'xgb__n_estimators': [100, 200, 300],
                'xgb__max_depth': [3, 5, 7, 10],
                'xgb__learning_rate': [0.01, 0.1, 0.2]
            }
            
            # Calculate class weights for balancing
            from sklearn.utils.class_weight import compute_class_weight
            original_y = metadata_df[target_column]
            class_weights = compute_class_weight('balanced', classes=np.unique(original_y), y=original_y)
            class_weight_dict = {cls: weight for cls, weight in zip(np.unique(original_y), class_weights)}
            
            # Set the appropriate objective based on binary or multiclass
            if len(unique_classes) == 2:
                # For binary, we can use scale_pos_weight
                # Calculate ratio of negative to positive samples
                neg_pos_ratio = len(original_y[original_y != unique_classes[1]]) / len(original_y[original_y == unique_classes[1]])
                
                pipeline_steps.append(('xgb', XGBClassifier(
                    random_state=42, 
                    eval_metric='logloss',
                    objective='binary:logistic',
                    scale_pos_weight=neg_pos_ratio  # Balance classes in binary case
                )))
            else:
                pipeline_steps.append(('xgb', XGBClassifier(
                    random_state=42, 
                    eval_metric='mlogloss',
                    objective='multi:softprob',
                    num_class=len(unique_classes)
                )))

        elif model_type == 'lgb':
            param_grid = {
                'lgb__n_estimators': [100, 200, 300],
                'lgb__max_depth': [3, 5, 7, 10],
                'lgb__learning_rate': [0.01, 0.05, 0.1]
            }
            
            # Set the appropriate objective based on binary or multiclass
            if len(unique_classes) == 2:
                pipeline_steps.append(('lgb', LGBMClassifier(
                    random_state=42, 
                    verbose=-1,
                    objective='binary',
                    class_weight='balanced'  
                )))
            else:
                pipeline_steps.append(('lgb', LGBMClassifier(
                    random_state=42, 
                    verbose=-1,
                    objective='multiclass',
                    num_class=len(unique_classes),
                    class_weight='balanced'  
                )))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create the pipeline
        pipeline = Pipeline(pipeline_steps)
        
        # Extract features and target
        X = features_df[selected_features]
        y = metadata_df[target_column]

        # Apply label encoding if needed
        if needs_label_encoder:
            y = label_encoder.transform(y)
        
        # Ensure X and y have the same length
        if len(X) != len(y):
            raise ValueError(f"Features ({len(X)} rows) and target ({len(y)} rows) must have the same length")
               
        # Split data into training and test sets for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        
        # Record start time for training
        start_time = time.time()
        
        # Train model with or without grid search
        if do_grid_search:
            print(f"Performing grid search for {target_column} classification with {model_type}...")
            grid_search = GridSearchCV(
                pipeline, 
                param_grid, 
                cv=5, 
                scoring='balanced_accuracy',
                verbose=1,
                n_jobs=10
            )
            grid_search.fit(X_train, y_train)
            
            # Get best model and parameters
            best_params = grid_search.best_params_
            best_pipeline = grid_search.best_estimator_
            
            print("Best parameters:", best_params)
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        else:
            # Train with default parameters
            print(f"Training {model_type} model for {target_column} classification...")
            pipeline.fit(X_train, y_train)
            best_pipeline = pipeline
        
        # Record training time
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate model on test set
        y_pred = best_pipeline.predict(X_test)
        class_report = sklearn_classification_report(y_test, y_pred, output_dict=True)
        print("\nTest Set Classification Report:")
        print(sklearn_classification_report(y_test, y_pred))
        
        # Plot results if requested (using test set results)
        if plot_results and eval_dir is not None:
            cls._plot_evaluation_results(
                y_test, y_pred, 
                best_pipeline.predict_proba(X_test),
                class_names=unique_classes,
                output_dir=eval_dir,
                prefix=prefix if prefix is not None else model_type
            )
            
            # If it's a tree-based model, plot feature importance
            if model_type in ['rf', 'lgb', 'xgb'] and feature_dir is not None:
                cls._plot_feature_importance(
                    best_pipeline, selected_features,
                    output_dir=feature_dir,
                    prefix=prefix if prefix is not None else model_type
                )
                
        # Retrain on full dataset if requested
        if retrain_on_full_data:
            print("\nRetraining model on full dataset with optimal parameters...")
            # Create a new pipeline with the same steps and best parameters
            if do_grid_search:
                # Clone the best pipeline to get one with the same parameters
                final_pipeline = clone(best_pipeline)
            else:
                # If no grid search was done, just use the trained pipeline
                final_pipeline = best_pipeline
            
            # Record start time for full training
            full_start_time = time.time()
                
            # Fit on the full dataset, with sample weights for XGBoost multiclass
            if model_type == 'xgb' and len(unique_classes) > 2:
                # Create sample weights based on class weights
                sample_weights = np.array([class_weight_dict[label] for label in original_y])
                final_pipeline.fit(X, y, xgb__sample_weight=sample_weights)
            else:
                final_pipeline.fit(X, y)
            
            # Record full training time
            full_training_time = time.time() - full_start_time
            print(f"Full dataset training completed in {full_training_time:.2f} seconds")
            
            # Create and return classifier with the full-data model
            classifier = cls(
                model=final_pipeline,
                features=selected_features,
                class_mapping=class_mapping,
                binary=is_binary,
                label_encoder=label_encoder if needs_label_encoder else None
            )
            
            print("Full dataset training complete.")
        else:
            # Use the model trained on the training set
            classifier = cls(
                model=final_pipeline,
                features=selected_features,
                class_mapping=class_mapping,
                binary=is_binary,
                label_encoder=label_encoder if needs_label_encoder else None,
                encoded_classes=encoded_classes if needs_label_encoder else None
            )
    
        # Test classifier
        if len(X_test) >= 5:
            # Get corresponding metadata rows for testing
            test_metadata = metadata_df.loc[X_test.iloc[:5].index]
            metadata_result, features_result = classifier.classify_cells(test_metadata, X_test.iloc[:5])
            print("\nTest prediction examples:")
            print(metadata_result[["class", "confidence"]].head())
        
        # Save training artifacts if output_dir is provided
        if model_dir is not None and prefix is not None:
            # Save model with dill
            model_path = os.path.join(model_dir, f"{prefix}_model.dill")
            classifier.save(model_path)
            print(f"Model saved to: {model_path}")
            
            # Save feature list
            feature_path = os.path.join(model_dir, f"{prefix}_features.txt")
            with open(feature_path, 'w') as f:
                f.write('\n'.join(selected_features))
            
            # Save classification report as JSON
            report_path = os.path.join(model_dir, f"{prefix}_classification_report.json")
            with open(report_path, 'w') as f:
                json.dump(class_report, f, indent=4)
            
            # Save training configuration
            config = {
                "model_type": model_type,
                "scaler_type": scaler_type,
                "feature_count": len(selected_features),
                "do_grid_search": do_grid_search,
                "retrain_on_full_data": retrain_on_full_data,
                "target_column": target_column,
                "binary_classification": is_binary,
                "training_time": training_time,
                "full_training_time": full_training_time if retrain_on_full_data else None,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "selected_features": selected_features,
                "classes": unique_classes.tolist()
            }
            
            config_path = os.path.join(model_dir, f"{prefix}_config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
        
        return classifier

    def merge_predictions(self, metadata_df, predictions_df):
        """
        Merge predictions with metadata
        
        Parameters:
        -----------
        metadata_df : pd.DataFrame
            DataFrame containing cell metadata
        predictions_df : pd.DataFrame
            DataFrame with class and confidence from classify_cells
            
        Returns:
        --------
        pd.DataFrame
            Merged DataFrame with metadata and predictions
        """
        # Check if indices match
        if not metadata_df.index.equals(predictions_df.index):
            raise ValueError("Metadata and predictions must have matching indices")
        
        # Merge dataframes
        result = pd.concat([metadata_df, predictions_df], axis=1)
        
        return result
    
    @staticmethod
    def _plot_evaluation_results(y_true, y_pred, y_prob, class_names, 
                            output_dir='.', prefix='model'):
        """
        Plot confusion matrix and ROC curves for model evaluation
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_prob : array-like
            Predicted probabilities
        class_names : list
            Names of classes
        output_dir : str
            Directory to save plots
        prefix : str
            Prefix for plot filenames
        """        
        print("Class names:", class_names)
        print("Sample of y_true:", y_true[:5])
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate normalized confusion matrix as percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotation text with both count and percentage
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1f}%)"
        
        # Plot as heatmap with combined counts and percentages
        ax = sns.heatmap(cm, annot=annot, fmt='', 
                       xticklabels=class_names, yticklabels=class_names,
                       cmap='Blues', vmin=0, annot_kws={"size": 12})
        
        # Improve visual elements
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        
        # Rotate tick labels for better readability if needed
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Add gridlines to separate classes
        ax.set_xticklabels(ax.get_xticklabels(), weight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), weight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_confusion_matrix.png"), dpi=300, bbox_inches='tight')
        plt.close()
            
        # 2. Plot ROC curves
        plt.figure(figsize=(10, 8))
        
        # Check if we're dealing with encoded integer labels
        is_encoded = False
        if hasattr(y_true, 'dtype') and np.issubdtype(y_true.dtype, np.integer):
            is_encoded = True
        elif isinstance(y_true[0] if not hasattr(y_true, 'iloc') else y_true.iloc[0], (int, np.integer)):
            is_encoded = True
        
        # Get custom colormap for curves
        cmap = plt.cm.viridis
        
        # If binary classification
        if len(class_names) == 2:
            if is_encoded:
                # For integer-encoded labels, positive class is typically 1
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            else:
                # For string labels, check against the second class name
                fpr, tpr, _ = roc_curve((y_true == class_names[1]).astype(int), y_prob[:, 1])
                
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, 
                    label=f'ROC curve (area = {roc_auc:.2f})',
                    color=cmap(0.5), linewidth=3)
            
        else:  # Multiclass
            # One-hot encode the labels - handle both encoded and string labels
            if is_encoded:
                # For integer-encoded labels, create a mapping to indices
                num_classes = len(class_names)
                # y_bin will be a one-hot encoded matrix
                y_bin = np.zeros((len(y_true), num_classes))
                for i in range(len(y_true)):
                    y_bin[i, y_true[i]] = 1
            else:
                # For string labels, use sklearn's label_binarize
                y_bin = label_binarize(y_true, classes=class_names)
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i, class_name in enumerate(class_names):
                # Check if any positive samples exist for this class in the test set
                if np.sum(y_bin[:, i]) > 0:  # Only calculate ROC if positive samples exist
                    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    
                    # Get evenly spaced colors from the colormap
                    color_pos = i / (len(class_names) - 1) if len(class_names) > 1 else 0.5
                    
                    plt.plot(fpr[i], tpr[i], 
                            label=f'ROC {class_name} (area = {roc_auc[i]:.2f})',
                            linewidth=2, color=cmap(color_pos))
                else:
                    print(f"Warning: No positive samples for class {class_name} in test set, skipping ROC curve.")
        
        # Plot random chance line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        
        # Enhance visual appearance
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC)', fontsize=16, fontweight='bold')
        
        # Add styled legend
        leg = plt.legend(loc="lower right", frameon=True, fancybox=True, framealpha=0.9)
        leg.get_frame().set_edgecolor('lightgray')
        
        # Add grid for readability
        plt.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_roc_curve.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _plot_feature_importance(model, feature_names, top_n=20, 
                               output_dir='.', prefix='model'):
        """
        Plot feature importance for tree-based models
        
        Parameters:
        -----------
        model : sklearn.pipeline.Pipeline
            Trained model pipeline
        feature_names : list
            Names of features
        top_n : int
            Number of top features to display
        output_dir : str
            Directory to save plots
        prefix : str
            Prefix for plot filenames
        """
        # Check if model has feature importances
        has_importance = False
        
        # Try to get the actual model from the pipeline
        if hasattr(model, 'named_steps'):
            for name, step in model.named_steps.items():
                if hasattr(step, 'feature_importances_'):
                    estimator = step
                    has_importance = True
                    break
        elif hasattr(model, 'feature_importances_'):
            estimator = model
            has_importance = True
        
        if not has_importance:
            print("Model doesn't have feature importances. Skipping importance plot.")
            return
        
        # Get feature importances
        importances = estimator.feature_importances_
        
        # Sort feature importances
        indices = np.argsort(importances)[::-1]
        
        # Create DataFrame for easier handling
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
        
        # Select top N features
        if top_n is not None and top_n < len(feature_names):
            importance_df = importance_df.head(top_n)
        
        # Normalize importances for color mapping
        max_importance = importance_df['Importance'].max()
        importance_df['Normalized'] = importance_df['Importance'] / max_importance
        
        # Create color gradient
        colors = [plt.cm.viridis(val) for val in importance_df['Normalized']]
        
        # Plot feature importances with enhanced styling
        plt.figure(figsize=(12, max(8, len(importance_df) * 0.3)))  # Adjust height based on feature count
        
        # Plot horizontal bars
        ax = plt.barh(
            importance_df['Feature'], 
            importance_df['Importance'],
            color=colors,
            alpha=0.8,
            edgecolor='gray',
            linewidth=0.5
        )
        
        # Add importance value labels
        for i, v in enumerate(importance_df['Importance']):
            plt.text(
                v + max_importance * 0.01,  # Slight offset from end of bar
                i,
                f'{v:.4f}',
                va='center',
                fontsize=9,
                fontweight='bold'
            )
        
        # Enhance visual elements
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.title('Feature Importance', fontsize=16, fontweight='bold')
        
        # Add grid lines
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_feature_importance.png"), dpi=300, bbox_inches='tight')
        
        # Also save a CSV with the full feature importance ranking
        importance_df.drop('Normalized', axis=1).to_csv(
            os.path.join(output_dir, f"{prefix}_feature_importance.csv"),
            index=False
        )
        
        plt.close()


def main():
    # Initialize publication style and get the colormap
    cell_classifier_cmap = setup_publication_plot_style()

    # Create directories for this run
    dirs = create_run_directories()

    # Access directory paths
    statistics_dir = dirs['statistics']
    models_dir = dirs['models']
    plots_dir = dirs['plots']
    results_dir = dirs['results']
    
    print(f"Output will be saved to run directory: {dirs['run']}")

    # List of file paths
    file_paths = [
        "analysis_root/phenotype/parquets/P-2_W-A1__phenotype_cp.parquet", 
        "analysis_root/phenotype/parquets/P-2_W-A2__phenotype_cp.parquet",
        "analysis_root/phenotype/parquets/P-3_W-A3__phenotype_cp.parquet",
    ]

    # Create the mapping dictionary
    cell_cycle_mapping = {
        'label_to_phase': {
            1: "Interphase",
            2: "Discard",
            3: "Prometaphase",
            4: "Metaphase",
            5: "Anaphase",
            6: "Telophase",
            7: "Unsure"
        },
        'phase_to_category': {
            "Interphase": "Interphase",
            "Prometaphase": "Mitotic",
            "Metaphase": "Mitotic",
            "Anaphase": "Mitotic",
            "Telophase": "Other",
            "Discard": "Other",
            "Unsure": "Other"
        }
    }

    # 1. Load the data
    data = load_cellprofiler_data(file_paths)

    # 2. Apply cell category mapping to the data and remove unwanted phases
    data_with_mapping = apply_cell_category_mapping(
        data, 
        label_col='cell_label', 
        remove_phases=["Discard", "Unsure", "Telophase"],
        mapping=cell_cycle_mapping, 
        verbose=True
    )

    # 3. Split the data into metadata and features
    metadata_df, features_df = split_cell_data(data_with_mapping, 
                                            metadata_cols=['label', 'cell_label', 'phase', 'category'] + DEFAULT_METADATA_COLS)

    # 4. Select features from the feature dataframe
    selected_features = select_features_from_split(
        features_df,
        feature_markers={'DAPI': True, 'ACTIN': False, 'TUBULIN': False, 'VIMENTIN': False, 'morphology': True}, 
        exclude_markers=['ACTIN', 'TUBULIN', 'VIMENTIN'],
        exclude_cols=None,
        remove_nan=True,
        verbose=True
    )

    # 5. Plot distribution statistics
    stats = plot_distribution_statistics(
        metadata_df, 
        target_col='category',
        split_col='well',
        output_dir=statistics_dir,
        prefix='cell_category_',
        target_order=["Interphase", "Mitotic"],
        figsize=(12, 8),
        palette=None,  # Use default seaborn palette instead of custom colormap
        dpi=300
    )

    stats = plot_distribution_statistics(
        metadata_df, 
        target_col='phase',
        split_col=None,
        output_dir=statistics_dir,
        prefix='cell_phase_',
        target_order=["Interphase", "Prometaphase", "Metaphase", "Anaphase"],
        figsize=(12, 8),
        palette=None,  # Use default seaborn palette instead of custom colormap
        dpi=300
    )

    model_configs = [
        # 1) All model types with standard scaling
        ('rf_standard', 'rf', 'standard', None),
        ('svc_standard', 'svc', 'standard', None),
        ('xgb_standard', 'xgb', 'standard', None),
        ('lgb_standard', 'lgb', 'standard', None),
        
        # 2) XGBoost with different scalers
        ('xgb_robust', 'xgb', 'robust', None),
        ('xgb_minmax', 'xgb', 'minmax', None),
        ('xgb_none', 'xgb', 'none', None),
        
        # 3) XGBoost with none scaling and different feature selection strategies
        ('xgb_none_var', 'xgb', 'none', {'enhance': True, 'remove_low_variance': True, 'remove_correlated': False, 'select_k_best': None}),
        ('xgb_none_corr', 'xgb', 'none', {'enhance': True, 'remove_low_variance': False, 'remove_correlated': True, 'select_k_best': None}), 
        ('xgb_none_kbest100', 'xgb', 'none', {'enhance': True, 'remove_low_variance': False, 'remove_correlated': False, 'select_k_best': 100}),
        ('xgb_none_combined', 'xgb', 'none', {'enhance': True, 'remove_low_variance': True, 'remove_correlated': True, 'select_k_best': 100}),
    ]

    # Train models and collect results
    binary_results = []
    multiclass_results = []

    # Train binary classifiers (interphase vs mitotic)
    print("\n=== Training Binary Classifiers (Interphase vs Mitotic) ===\n")
    binary_results = []

    for config in model_configs:
        # Handle both 3-tuple (old style) and 4-tuple (new style with feature selection)
        if len(config) == 3:
            name, model_type, scaler_type = config
            feature_config = None
        else:
            name, model_type, scaler_type, feature_config = config
            
        model_name = f"binary_{name}"
        print(f"\n{'-'*50}")
        print(f"Training model: {model_name}")
        print(f"{'-'*50}")
        
        try:
            # Default feature enhancement parameters (off by default)
            enhance_params = {
                'enhance_features': False,
                'remove_low_variance': True, 
                'variance_threshold': 0.01,
                'remove_correlated': True, 
                'correlation_threshold': 0.95,
                'select_k_best': None
            }
            
            # Update with config-specific parameters if provided
            if feature_config:
                if feature_config.get('enhance'):
                    enhance_params['enhance_features'] = True
                if 'remove_low_variance' in feature_config:
                    enhance_params['remove_low_variance'] = feature_config['remove_low_variance']
                if 'remove_correlated' in feature_config:
                    enhance_params['remove_correlated'] = feature_config['remove_correlated']
                if 'select_k_best' in feature_config:
                    enhance_params['select_k_best'] = feature_config['select_k_best']
            
            # Train classifier with applicable parameters
            model = SciKitCellClassifier.from_training(
                metadata_df=metadata_df, 
                features_df=features_df,
                target_column='category',  
                selected_features=selected_features,
                model_type=model_type,          
                scaler_type=scaler_type,      
                do_grid_search=False,      
                output_dir=models_dir,
                prefix=model_name,
                plot_results=True,         
                retrain_on_full_data=True,
                # Feature enhancement parameters
                enhance_features=enhance_params['enhance_features'],
                remove_low_variance=enhance_params['remove_low_variance'],
                variance_threshold=enhance_params['variance_threshold'],
                remove_correlated=enhance_params['remove_correlated'],
                correlation_threshold=enhance_params['correlation_threshold'],
                select_k_best=enhance_params['select_k_best']
            )
            
            # Get performance metrics from the saved classification report
            with open(os.path.join(models_dir, model_name, f"{model_name}_classification_report.json"), 'r') as f:
                report_dict = json.load(f)
                
            # Check if we're using label encoding (for XGBoost and LightGBM)
            if model_type in ['xgb', 'lgb']:
                # For models using label encoding, the report uses numeric indices
                metrics = {
                    'model': model_name,
                    'model_type': model_type,
                    'scaler_type': scaler_type,
                    'feature_selection': 'none' if not feature_config else ('var' if feature_config.get('remove_low_variance') else '') + 
                                       ('corr' if feature_config.get('remove_correlated') else '') + 
                                       (f'k{feature_config.get("select_k_best")}' if feature_config.get('select_k_best') else ''),
                    'accuracy': report_dict['accuracy'],
                    'balanced_accuracy': report_dict.get('balanced_accuracy', 
                                                  (report_dict['0']['recall'] + 
                                                   report_dict['1']['recall']) / 2),
                    'interphase_precision': report_dict['0']['precision'],
                    'interphase_recall': report_dict['0']['recall'],
                    'interphase_f1': report_dict['0']['f1-score'],
                    'mitotic_precision': report_dict['1']['precision'],
                    'mitotic_recall': report_dict['1']['recall'],
                    'mitotic_f1': report_dict['1']['f1-score'],
                }
            else:
                # For models using original class names (SVC, RandomForest)
                metrics = {
                    'model': model_name,
                    'model_type': model_type,
                    'scaler_type': scaler_type,
                    'feature_selection': 'none' if not feature_config else ('var' if feature_config.get('remove_low_variance') else '') + 
                                       ('corr' if feature_config.get('remove_correlated') else '') + 
                                       (f'k{feature_config.get("select_k_best")}' if feature_config.get('select_k_best') else ''),
                    'accuracy': report_dict['accuracy'],
                    'balanced_accuracy': report_dict.get('balanced_accuracy', 
                                                  (report_dict['Interphase']['recall'] + 
                                                   report_dict['Mitotic']['recall']) / 2),
                    'interphase_precision': report_dict['Interphase']['precision'],
                    'interphase_recall': report_dict['Interphase']['recall'],
                    'interphase_f1': report_dict['Interphase']['f1-score'],
                    'mitotic_precision': report_dict['Mitotic']['precision'],
                    'mitotic_recall': report_dict['Mitotic']['recall'],
                    'mitotic_f1': report_dict['Mitotic']['f1-score'],
                }
            
            # Get the number of features actually used
            try:
                with open(os.path.join(models_dir, model_name, f"{model_name}_features.txt"), 'r') as f:
                    used_features = f.read().splitlines()
                    metrics['feature_count'] = len(used_features)
            except:
                # If file not found, use original feature count
                metrics['feature_count'] = len(selected_features)
                
            binary_results.append(metrics)
            print(f"Successfully trained {model_name}")
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue

        
    # Train multiclass classifiers (all phases)
    print("\n=== Training Multiclass Classifiers (All Cell Cycle Phases) ===\n")
    multiclass_results = []

    for config in model_configs:
        # Handle both 3-tuple (old style) and 4-tuple (new style with feature selection)
        if len(config) == 3:
            name, model_type, scaler_type = config
            feature_config = None
        else:
            name, model_type, scaler_type, feature_config = config
            
        model_name = f"multiclass_{name}"
        print(f"\n{'-'*50}")
        print(f"Training model: {model_name}")
        print(f"{'-'*50}")
        
        try:
            # Default feature enhancement parameters (off by default)
            enhance_params = {
                'enhance_features': False,
                'remove_low_variance': True, 
                'variance_threshold': 0.01,
                'remove_correlated': True, 
                'correlation_threshold': 0.95,
                'select_k_best': None
            }
            
            # Update with config-specific parameters if provided
            if feature_config:
                if feature_config.get('enhance'):
                    enhance_params['enhance_features'] = True
                if 'remove_low_variance' in feature_config:
                    enhance_params['remove_low_variance'] = feature_config['remove_low_variance']
                if 'remove_correlated' in feature_config:
                    enhance_params['remove_correlated'] = feature_config['remove_correlated']
                if 'select_k_best' in feature_config:
                    enhance_params['select_k_best'] = feature_config['select_k_best']
            
            # Train classifier with applicable parameters
            model = SciKitCellClassifier.from_training(
                metadata_df=metadata_df, 
                features_df=features_df,
                target_column='phase',     
                selected_features=selected_features,
                model_type=model_type,           
                scaler_type=scaler_type,       
                do_grid_search=False,      
                output_dir=models_dir,
                prefix=model_name,
                plot_results=True,         
                retrain_on_full_data=True,
                # Feature enhancement parameters
                enhance_features=enhance_params['enhance_features'],
                remove_low_variance=enhance_params['remove_low_variance'],
                variance_threshold=enhance_params['variance_threshold'],
                remove_correlated=enhance_params['remove_correlated'],
                correlation_threshold=enhance_params['correlation_threshold'],
                select_k_best=enhance_params['select_k_best']
            )
            # Get performance metrics from the saved classification report
            with open(os.path.join(models_dir, model_name, f"{model_name}_classification_report.json"), 'r') as f:
                report_dict = json.load(f)

            # Extract phases
            phases = ["Interphase", "Prometaphase", "Metaphase", "Anaphase"]

            # Create metrics dictionary with basic info
            metrics = {
                'model': model_name,
                'model_type': model_type,
                'scaler_type': scaler_type,
                'feature_selection': 'none' if not feature_config else ('var' if feature_config.get('remove_low_variance') else '') + 
                                ('corr' if feature_config.get('remove_correlated') else '') + 
                                (f'k{feature_config.get("select_k_best")}' if feature_config.get('select_k_best') else ''),
                'accuracy': report_dict['accuracy'],
                'macro_avg_precision': report_dict['macro avg']['precision'],
                'macro_avg_recall': report_dict['macro avg']['recall'],
                'macro_avg_f1': report_dict['macro avg']['f1-score'],
            }

            # Add per-phase metrics - handle both string labels and encoded labels
            if model_type in ['xgb', 'lgb']:
                # For LightGBM and XGBoost, the classes are encoded as integers
                # Map the numeric indices to phase names
                numeric_to_phase = {}
                
                # Try to get the mapping from the model's label encoder
                if hasattr(model, 'label_encoder') and model.label_encoder is not None:
                    if hasattr(model.label_encoder, 'classes_'):
                        for i, phase in enumerate(model.label_encoder.classes_):
                            numeric_to_phase[str(i)] = phase
                
                # If we couldn't get the mapping from the model, use a fixed mapping based on alphabetical order
                # This is needed because LightGBM and XGBoost use alphabetical ordering for class encoding
                if not numeric_to_phase:
                    # Sort phases alphabetically (this is how LabelEncoder typically assigns indices)
                    sorted_phases = sorted(phases)
                    for i, phase in enumerate(sorted_phases):
                        numeric_to_phase[str(i)] = phase
                
                # Now extract the metrics using the numeric keys
                for numeric, phase in numeric_to_phase.items():
                    if numeric in report_dict:
                        metrics[f'{phase.lower()}_precision'] = report_dict[numeric]['precision']
                        metrics[f'{phase.lower()}_recall'] = report_dict[numeric]['recall']
                        metrics[f'{phase.lower()}_f1'] = report_dict[numeric]['f1-score']
            else:
                # For models using original class names (SVC, RandomForest)
                for phase in phases:
                    if phase in report_dict:
                        metrics[f'{phase.lower()}_precision'] = report_dict[phase]['precision']
                        metrics[f'{phase.lower()}_recall'] = report_dict[phase]['recall']
                        metrics[f'{phase.lower()}_f1'] = report_dict[phase]['f1-score']
                        
            # Get the number of features actually used
            try:
                with open(os.path.join(models_dir, model_name, f"{model_name}_features.txt"), 'r') as f:
                    used_features = f.read().splitlines()
                    metrics['feature_count'] = len(used_features)
            except:
                # If file not found, use original feature count
                metrics['feature_count'] = len(selected_features)
            
            multiclass_results.append(metrics)
            print(f"Successfully trained {model_name}")
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue

    # Convert results to DataFrames
    binary_df = pd.DataFrame(binary_results)
    multiclass_df = pd.DataFrame(multiclass_results)

    # Save results to CSV
    if not binary_df.empty:
        binary_df.to_csv(os.path.join(results_dir, 'binary_classifier_results.csv'), index=False)
    
    if not multiclass_df.empty:
        multiclass_df.to_csv(os.path.join(results_dir, 'multiclass_classifier_results.csv'), index=False)

    # Generate comparison plots
    print("\n=== Generating Result Comparisons ===\n")

    # Binary classifier plots
    if not binary_df.empty:
        # 1. Binary classifier accuracy comparison
        plt.figure(figsize=(12, 8))
        
        # Use a customized color palette based on model_type
        model_types = binary_df['model_type'].unique()
        color_positions = np.linspace(0.1, 0.9, len(model_types))
        # Use standard colormap instead of custom one
        model_type_colors = {model: plt.cm.viridis(pos) for model, pos in zip(model_types, color_positions)}
        
        # Create the plot with enhanced styling
        binary_plot = sns.barplot(
            x='model', 
            y='accuracy', 
            hue='model_type', 
            data=binary_df,
            palette=model_type_colors,
            alpha=0.8
        )
        
        # Add value labels on top of bars
        for i, p in enumerate(binary_plot.patches):
            binary_plot.annotate(
                f'{p.get_height():.3f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )
        
        binary_plot.set_xticklabels(binary_plot.get_xticklabels(), rotation=45, ha='right')
        plt.title('Binary Classifier Accuracy Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=14)
        plt.xlabel('Model', fontsize=14)
        
        # Enhance legend appearance
        leg = plt.legend(title="Model Type", frameon=True, fancybox=True, framealpha=0.9)
        leg.get_title().set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'binary_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Binary classifier balanced accuracy comparison
        plt.figure(figsize=(12, 8))
        
        binary_bal_plot = sns.barplot(
            x='model', 
            y='balanced_accuracy', 
            hue='model_type', 
            data=binary_df,
            palette=model_type_colors,
            alpha=0.8
        )
        
        # Add value labels on top of bars
        for i, p in enumerate(binary_bal_plot.patches):
            binary_bal_plot.annotate(
                f'{p.get_height():.3f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )
        
        binary_bal_plot.set_xticklabels(binary_bal_plot.get_xticklabels(), rotation=45, ha='right')
        plt.title('Binary Classifier Balanced Accuracy Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Balanced Accuracy', fontsize=14)
        plt.xlabel('Model', fontsize=14)
        
        # Enhance legend appearance
        leg = plt.legend(title="Model Type", frameon=True, fancybox=True, framealpha=0.9)
        leg.get_title().set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'binary_balanced_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature count vs accuracy plot (for binary classifiers)
        if 'feature_count' in binary_df.columns:
            plt.figure(figsize=(12, 8))
            
            # Create scatter plot with model_type as colors
            model_types = binary_df['model_type'].unique()
            markers = ['o', 's', '^', 'd', 'v']  # Different marker shapes for model types
            
            # Plot each model type with different marker and color
            for i, model_type in enumerate(model_types):
                model_data = binary_df[binary_df['model_type'] == model_type]
                
                # Use a color from the viridis colormap
                color_pos = i / (len(model_types) - 1) if len(model_types) > 1 else 0.5
                color = plt.cm.viridis(color_pos)
                
                # Use a different marker shape for each model type (cycle through markers if needed)
                marker = markers[i % len(markers)]
                
                plt.scatter(
                    model_data['feature_count'], 
                    model_data['balanced_accuracy'],
                    label=model_type,
                    color=color,
                    marker=marker,
                    alpha=0.8,
                    s=150,
                    edgecolors='black',
                    linewidths=0.5
                )
            
            # Add model name labels
            for i, row in binary_df.iterrows():
                plt.annotate(
                    row['model'].replace('binary_', ''), 
                    (row['feature_count'], row['balanced_accuracy']),
                    xytext=(5, 0),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.8
                )
            
            # Add trendline (linear regression)
            from scipy.stats import linregress
            if len(binary_df) > 1:  # Need at least 2 points for regression
                slope, intercept, r_value, p_value, std_err = linregress(
                    binary_df['feature_count'], binary_df['balanced_accuracy']
                )
                x_range = np.array([binary_df['feature_count'].min(), binary_df['feature_count'].max()])
                plt.plot(
                    x_range, 
                    intercept + slope * x_range, 
                    'k--', 
                    alpha=0.6,
                    label=f'R² = {r_value**2:.3f}'
                )
            
            # Add legend with custom styling
            leg = plt.legend(title="Model Type", frameon=True, fancybox=True, framealpha=0.9, loc='best')
            leg.get_title().set_fontweight('bold')
            
            plt.xlabel('Number of Features', fontsize=14)
            plt.ylabel('Balanced Accuracy', fontsize=14)
            plt.title('Feature Count vs. Balanced Accuracy (Binary Classifiers)', fontsize=16, fontweight='bold')
            
            # Add grid
            plt.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'binary_feature_count_vs_accuracy.png'), dpi=300, bbox_inches='tight')
            plt.close()

    # Multiclass classifier plots
    if not multiclass_df.empty:
        # 4. Multiclass classifier accuracy comparison
        plt.figure(figsize=(12, 8))
        
        # Use a customized color palette based on model_type
        model_types = multiclass_df['model_type'].unique()
        color_positions = np.linspace(0.1, 0.9, len(model_types))
        model_type_colors = {model: plt.cm.viridis(pos) for model, pos in zip(model_types, color_positions)}
        
        multiclass_plot = sns.barplot(
            x='model', 
            y='accuracy', 
            hue='model_type', 
            data=multiclass_df,
            palette=model_type_colors,
            alpha=0.8
        )
        
        # Add value labels on top of bars
        for i, p in enumerate(multiclass_plot.patches):
            multiclass_plot.annotate(
                f'{p.get_height():.3f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )
        
        multiclass_plot.set_xticklabels(multiclass_plot.get_xticklabels(), rotation=45, ha='right')
        plt.title('Multiclass Classifier Accuracy Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=14)
        plt.xlabel('Model', fontsize=14)
        
        # Enhance legend appearance
        leg = plt.legend(title="Model Type", frameon=True, fancybox=True, framealpha=0.9)
        leg.get_title().set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'multiclass_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. F1 score heatmap for multiclass classifiers
        f1_cols = [col for col in multiclass_df.columns if col.endswith('_f1')]
        if f1_cols:
            f1_data = multiclass_df[['model'] + f1_cols].copy()
            
            # Clean up column names for display
            f1_data.columns = [col.replace('_f1', '') for col in f1_data.columns]
            
            # Pivot the data
            f1_pivot = f1_data.set_index('model')
            
            # Create heatmap with enhanced styling
            plt.figure(figsize=(14, 10))
            ax = sns.heatmap(
                f1_pivot, 
                annot=True, 
                cmap='viridis', 
                fmt='.3f',
                annot_kws={"fontsize": 10, "fontweight": "bold"},
                linewidths=0.5,
                linecolor='white',
                vmin=0.0,
                vmax=1.0
            )
            
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label('F1 Score', fontsize=12, fontweight='bold')
            
            plt.title('F1 Scores by Model and Cell Cycle Phase', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'multiclass_f1_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 6. Feature count vs accuracy plot (for multiclass classifiers)
        if 'feature_count' in multiclass_df.columns:
            plt.figure(figsize=(12, 8))
            
            # Create scatter plot with model_type as colors
            model_types = multiclass_df['model_type'].unique()
            markers = ['o', 's', '^', 'd', 'v']  # Different marker shapes for model types
            
            # Plot each model type with different marker and color
            for i, model_type in enumerate(model_types):
                model_data = multiclass_df[multiclass_df['model_type'] == model_type]
                
                # Use a color from the viridis colormap
                color_pos = i / (len(model_types) - 1) if len(model_types) > 1 else 0.5
                color = plt.cm.viridis(color_pos)
                
                # Use a different marker shape for each model type (cycle through markers if needed)
                marker = markers[i % len(markers)]
                
                plt.scatter(
                    model_data['feature_count'], 
                    model_data['accuracy'],
                    label=model_type,
                    color=color,
                    marker=marker,
                    alpha=0.8,
                    s=150,
                    edgecolors='black',
                    linewidths=0.5
                )
            
            # Add model name labels
            for i, row in multiclass_df.iterrows():
                plt.annotate(
                    row['model'].replace('multiclass_', ''), 
                    (row['feature_count'], row['accuracy']),
                    xytext=(5, 0),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.8
                )
            
            # Add trendline (linear regression)
            from scipy.stats import linregress
            if len(multiclass_df) > 1:  # Need at least 2 points for regression
                slope, intercept, r_value, p_value, std_err = linregress(
                    multiclass_df['feature_count'], multiclass_df['accuracy']
                )
                x_range = np.array([multiclass_df['feature_count'].min(), multiclass_df['feature_count'].max()])
                plt.plot(
                    x_range, 
                    intercept + slope * x_range, 
                    'k--', 
                    alpha=0.6,
                    label=f'R² = {r_value**2:.3f}'
                )
            
            # Add legend with custom styling
            leg = plt.legend(title="Model Type", frameon=True, fancybox=True, framealpha=0.9, loc='best')
            leg.get_title().set_fontweight('bold')
            
            plt.xlabel('Number of Features', fontsize=14)
            plt.ylabel('Accuracy', fontsize=14)
            plt.title('Feature Count vs. Accuracy (Multiclass Classifiers)', fontsize=16, fontweight='bold')
            
            # Add grid
            plt.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'multiclass_feature_count_vs_accuracy.png'), dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    main()