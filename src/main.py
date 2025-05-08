import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
#from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO  
#from IPython.display import Image  
#import pydotplus

# Set plot style
sns.set(style="whitegrid")
plt.style.use('fivethirtyeight')

# Load the datasets
print("Loading datasets...")
clean_ufc_fights = pd.read_csv('../assets/clean_ufc_all_fights.csv')

# Choose one dataset to focus on (the clean one seems more suitable for analysis)
df = clean_ufc_fights
print(f"Dataset loaded. Shape: {df.shape}")

# 1. Basic Information
print("\n===== BASIC INFORMATION =====")
print("\nDataset Preview:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# 2. Check for Missing Values
print("\n===== MISSING VALUES =====")
missing_values = df.isnull().sum()
missing_percent = (df.isnull().sum() / df.shape[0]) * 100
missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
print(missing_df[missing_df['Missing Values'] > 0])

# 3. Statistical Summary
print("\n===== STATISTICAL SUMMARY =====")
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(df[numeric_cols].describe())

# Create output directory if it doesn't exist
output_dir = '../output'
os.makedirs(output_dir, exist_ok=True)

# 4. Fight Outcome Analysis
print("\n===== FIGHT OUTCOME DISTRIBUTION =====")
if 'fight_outcome' in df.columns:
    outcome_counts = df['fight_outcome'].value_counts()
    print("Fight Outcome Distribution:")
    print(outcome_counts)

    plt.figure(figsize=(8, 8))
    plt.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Fight Outcomes')
    plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig(os.path.join(output_dir, 'fight_outcome_distribution.png'))
    # plt.show() # Uncomment to display plot immediately
    plt.close()
else:
    print("Coluna 'fight_outcome' não encontrada.")

# 5. Event Analysis
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
    # plt.show()
    plt.close()
else:
    print("Coluna 'event' não encontrada.")

# 6. Fighter Age Distribution
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

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.savefig(os.path.join(output_dir, 'fighter_age_distribution.png'))
# plt.show()
plt.close()

# 7. Fighter Stance Distribution
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
# plt.show()
plt.close()

# 8. Correlation Analysis (Numeric Features)
print("\n===== CORRELATION ANALYSIS =====")
if not df[numeric_cols].empty:
    correlation_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(18, 15))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=.5)
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    # plt.show()
    plt.close()

    # Print highly correlated pairs (example threshold: > 0.7 or < -0.7)
    print("\nHighly Correlated Feature Pairs (abs > 0.7):")
    corr_pairs = correlation_matrix.unstack()
    sorted_pairs = corr_pairs.sort_values(kind="quicksort", ascending=False)
    strong_pairs = sorted_pairs[abs(sorted_pairs) > 0.7]
    strong_pairs = strong_pairs[strong_pairs != 1.0] # Remove self-correlation
    print(strong_pairs.drop_duplicates())
else:
    print("Nenhuma coluna numérica encontrada para análise de correlação.")

print("\nAnalysis complete. Plots saved in the 'output' directory.")


print("RATATATATATAAAAAAAAAAAAAAAAAAAAAAAAAAA")

def create_model(type, df, test_size, neighbors):

    # Features: relevant columns of the data
    feature_cols = ['fighter1_Weight', 'fighter1_Reach','fighter1_SLpM','fighter1_StrAcc','fighter1_SApM',
                        'fighter1_StrDef','fighter1_TDAvg','fighter1_TDAcc','fighter1_TDDef','fighter1_SubAvg',
                        'fighter2_Weight','fighter2_Reach','fighter2_SLpM','fighter2_StrAcc',
                        'fighter2_SApM','fighter2_StrDef','fighter2_TDAvg','fighter2_TDAcc','fighter2_TDDef',
                        'fighter2_SubAvg','fighter1_Wins','fighter1_Losses','fighter1_Draws','fighter2_Wins',
                        'fighter2_Losses','fighter2_Draws','fighter1_Height_in','fighter2_Height_in','fighter1_Age',
                        'fighter2_Age']
    # Since fighter stance has not integer or float value, we need to replace the column with dummies
    df = pd.get_dummies(df, columns=['fighter1_Stance', 'fighter2_Stance'])
    # The target column (the one we want to predict) is which fighter won the fight, if fighter1 won, target = 1, else target = 0 (assuming no draws)
    df['target'] = df['fight_outcome'].apply(lambda x: 1 if x == 'fighter1' else (0 if x == 'fighter2' else np.nan))
    df = df.dropna(subset=['target'])
    stance_cols = [col for col in df.columns if col.startswith('fighter1_Stance_') or col.startswith('fighter2_Stance_')]
    # X axis are the feature columnss and Y axis is the target
    X = df[feature_cols + stance_cols]
    y = df['target']
    # Split the data in 2: One set for training and one set for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7)
    if(type == "decisionTree"):
        model = DecisionTreeClassifier(random_state=7)
    if(type == "K-nearestNeighbors"):
        model = KNeighborsClassifier(n_neighbors=neighbors)

    # Perform the trainig of the model using the train data sets

    model.fit(X_train, y_train)
    return model, X_test, y_test


def test_model(model, X_test, y_test):
    # Try to predict using the test data set
    y_pred = model.predict(X_test)

    # Evaluate the performance of the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot the confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fighter2', 'Fighter1'], yticklabels=['Fighter2', 'Fighter1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join("../output/", 'confusion_matrix.png'))



df = pd.read_csv('../assets/clean_ufc_all_fights.csv')

model, X_test, y_test = create_model("decisionTree", df, 0.2, 5)
test_model(model, X_test, y_test)

model, X_test, y_test = create_model("K-nearestNeighbors", df, 0.2, 5)
test_model(model, X_test, y_test)

