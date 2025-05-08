import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set plot style
sns.set(style="whitegrid")
plt.style.use('fivethirtyeight')

def load_data(file_path):
    """Load the UFC dataset."""
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded. Shape: {df.shape}")
    return df

def show_basic_info(df):
    """Display basic information about the dataset."""
    print("\n===== BASIC INFORMATION =====")
    print("\nDataset Preview:")
    print(df.head())

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

def check_missing_values(df):
    """Check for missing values in the dataset."""
    print("\n===== MISSING VALUES =====")
    missing_values = df.isnull().sum()
    missing_percent = (df.isnull().sum() / df.shape[0]) * 100
    missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
    print(missing_df[missing_df['Missing Values'] > 0])

def show_statistical_summary(df):
    """Display statistical summary of numerical columns."""
    print("\n===== STATISTICAL SUMMARY =====")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe())

def analyze_fight_outcome(df, output_dir):
    """Analyze and visualize fight outcome distribution."""
    print("\n===== FIGHT OUTCOME DISTRIBUTION =====")
    if 'fight_outcome' in df.columns:
        outcome_counts = df['fight_outcome'].value_counts()
        print("Fight Outcome Distribution:")
        print(outcome_counts)

        plt.figure(figsize=(8, 8))
        plt.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Distribution of Fight Outcomes')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.savefig(os.path.join(output_dir, 'fight_outcome_distribution.png'))
        plt.close()
    else:
        print("Coluna 'fight_outcome' não encontrada.")

def analyze_events(df, output_dir):
    """Analyze and visualize top events by number of fights."""
    print("\n===== EVENT DISTRIBUTION =====")
    if 'event' in df.columns:
        event_counts = df['event'].value_counts().head(10)
        print("Top 10 Events by Number of Fights:")
        print(event_counts)

        plt.figure(figsize=(12, 6))
        event_counts.plot(kind='bar')
        plt.title('Top 10 Events by Number of Fights')
        plt.xlabel('Event')
        plt.ylabel('Number of Fights')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_events.png'))
        plt.close()
    else:
        print("Coluna 'event' não encontrada.")

def analyze_fighter_age(df, output_dir):
    """Analyze and visualize fighter age distribution."""
    print("\n===== FIGHTER AGE DISTRIBUTION =====")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle('Age Distribution of Fighters')

    if 'fighter1_Age' in df.columns:
        sns.histplot(df['fighter1_Age'].dropna(), bins=20, kde=True, ax=axes[0])
        axes[0].set_title('Fighter 1 Age')
        axes[0].set_xlabel('Age')
    else:
        axes[0].set_title('Fighter 1 Age (Coluna não encontrada)')

    if 'fighter2_Age' in df.columns:
        sns.histplot(df['fighter2_Age'].dropna(), bins=20, kde=True, ax=axes[1])
        axes[1].set_title('Fighter 2 Age')
        axes[1].set_xlabel('Age')
    else:
        axes[1].set_title('Fighter 2 Age (Coluna não encontrada)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
    plt.savefig(os.path.join(output_dir, 'fighter_age_distribution.png'))
    plt.close()

def analyze_fighter_stance(df, output_dir):
    """Analyze and visualize fighter stance distribution."""
    print("\n===== FIGHTER STANCE DISTRIBUTION =====")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Stance Distribution of Fighters')

    if 'fighter1_Stance' in df.columns:
        stance1_counts = df['fighter1_Stance'].value_counts()
        sns.barplot(x=stance1_counts.index, y=stance1_counts.values, ax=axes[0])
        axes[0].set_title('Fighter 1 Stance')
        axes[0].set_ylabel('Count')
    else:
        axes[0].set_title('Fighter 1 Stance (Coluna não encontrada)')

    if 'fighter2_Stance' in df.columns:
        stance2_counts = df['fighter2_Stance'].value_counts()
        sns.barplot(x=stance2_counts.index, y=stance2_counts.values, ax=axes[1])
        axes[1].set_title('Fighter 2 Stance')
        axes[1].set_ylabel('Count')
    else:
        axes[1].set_title('Fighter 2 Stance (Coluna não encontrada)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'fighter_stance_distribution.png'))
    plt.close()

def analyze_correlations(df, output_dir):
    """Analyze and visualize correlation between numeric features."""
    print("\n===== CORRELATION ANALYSIS =====")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not df[numeric_cols].empty:
        correlation_matrix = df[numeric_cols].corr()
        plt.figure(figsize=(18, 15))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=.5)
        plt.title('Correlation Matrix of Numeric Features')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
        plt.close()

        # Print highly correlated pairs (threshold: > 0.7 or < -0.7)
        print("\nHighly Correlated Feature Pairs (abs > 0.7):")
        corr_pairs = correlation_matrix.unstack()
        sorted_pairs = corr_pairs.sort_values(kind="quicksort", ascending=False)
        strong_pairs = sorted_pairs[abs(sorted_pairs) > 0.7]
        strong_pairs = strong_pairs[strong_pairs != 1.0]  # Remove self-correlation
        print(strong_pairs.drop_duplicates())
    else:
        print("Nenhuma coluna numérica encontrada para análise de correlação.")

def run_complete_analysis(file_path, output_dir):
    """Run the complete exploratory data analysis pipeline."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_data(file_path)
    
    # Run analyses
    show_basic_info(df)
    check_missing_values(df)
    show_statistical_summary(df)
    analyze_fight_outcome(df, output_dir)
    analyze_events(df, output_dir)
    analyze_fighter_age(df, output_dir)
    analyze_fighter_stance(df, output_dir)
    analyze_correlations(df, output_dir)
    
    print("\nAnalysis complete. Plots saved in the 'output' directory.")
    return df