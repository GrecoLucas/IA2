import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os

def create_model(type, df, test_size, neighbors, n_estimators=100, c_value=1.0, random_state=42):
    # Features: relevant columns of the data
    feature_cols = ['fighter1_Weight', 'fighter1_Reach','fighter1_SLpM','fighter1_StrAcc','fighter1_SApM',
                    'fighter1_StrDef','fighter1_TDAvg','fighter1_TDAcc','fighter1_TDDef','fighter1_SubAvg',
                    'fighter2_Weight','fighter2_Reach','fighter2_SLpM','fighter2_StrAcc',
                    'fighter2_SApM','fighter2_StrDef','fighter2_TDAvg','fighter2_TDAcc','fighter2_TDDef',
                    'fighter2_SubAvg','fighter1_Wins','fighter1_Losses','fighter1_Draws','fighter2_Wins',
                    'fighter2_Losses','fighter2_Draws','fighter1_Height_in','fighter2_Height_in','fighter1_Age',
                    'fighter2_Age']
    
    df = df.copy()
    df = pd.get_dummies(df, columns=['fighter1_Stance', 'fighter2_Stance'])

    df['target'] = df['fight_outcome'].apply(lambda x: 1 if x == 'fighter1' else (0 if x == 'fighter2' else np.nan))
    df = df.dropna(subset=['target'])
    
    stance_cols = [col for col in df.columns if col.startswith('fighter1_Stance_') or col.startswith('fighter2_Stance_')]

    X = df[feature_cols + stance_cols]
    y = df['target']
    
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    if(type == "decisionTree"):
        model = DecisionTreeClassifier(random_state=random_state)
    elif(type == "K-nearestNeighbors"):
        model = KNeighborsClassifier(neighbors)
    elif(type == "randomForest"):
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    elif(type == "svm"):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = SVC(C=c_value, kernel='rbf', random_state=random_state, probability=True)
        model.fit(X_train, y_train)
        return model, X_test, y_test, scaler

    model.fit(X_train, y_train)
    return model, X_test, y_test

def test_model(model, X_test, y_test):
    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    t = end - start
    return accuracy, precision, recall, f1, t, y_pred

def plot_confusion_matrix(y_test, y_pred, model_name, output_dir):
    """Plot confusion matrix for a model."""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fighter 2 Wins', 'Fighter 1 Wins'],
                yticklabels=['Fighter 2 Wins', 'Fighter 1 Wins'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()

def plot_roc_curve(model, X_test, y_test, model_name, output_dir):
    """Plot ROC curve for a model."""
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            print(f"Cannot generate ROC curve for {model_name}: model doesn't support probability prediction")
            return
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'roc_curve_{model_name.lower().replace(" ", "_")}.png'))
        plt.close()
        
        return roc_auc
    except Exception as e:
        print(f"Error generating ROC curve for {model_name}: {e}")
        return None

def plot_learning_curve(estimator, X, y, title, output_dir, cv=5):
    """Plot learning curve for a model."""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=42)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title(f'Learning Curve - {title}')
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'learning_curve_{title.lower().replace(" ", "_")}.png'))
    plt.close()

def evaluate_with_cv(model, X, y, cv=5):
    """Evaluate model with cross-validation."""
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return cv_scores.mean(), cv_scores.std()

def make_graph_decision_tree(measure, xAxis, yAxis, output_dir):
    """Create a graph showing the effect of test size proportion on model performance."""
    plt.figure(figsize=(8, 8))
    plt.plot(xAxis, yAxis, marker='o', linestyle='-', color='b')
    plt.title(f'Effects of training data size in {measure}')
    plt.xlabel('Percentage of Data used for Testing')
    plt.ylabel(f'{measure}')
    plt.grid(True)
    plt.xticks(xAxis)

    plt.ylim(max(min(yAxis)-0.05, 0), max(yAxis) + 0.05)  
    plt.savefig(os.path.join(output_dir, f'{measure}_decision_trees.png'))
    plt.close()

def make_graph_k_neighbors(measure, xAxis, yAxis, output_dir):
    """Create a graph showing the effect of number of neighbors on model performance."""
    plt.figure(figsize=(8, 8))
    plt.plot(xAxis, yAxis, marker='o', linestyle='-', color='b')
    plt.title(f'Effects of number of neighbors in {measure}')
    plt.xlabel('Number of Neighbors')
    plt.ylabel(f'{measure}')
    plt.grid(True)
    plt.xticks(xAxis)

    plt.ylim(max(min(yAxis)-0.05, 0), max(yAxis) + 0.05)  
    plt.savefig(os.path.join(output_dir, f'{measure}_number_neighbors.png'))
    plt.close()

def make_graph_models(measure, models, yAxis, output_dir):
    """Create a graph comparing different models."""
    plt.figure(figsize=(8, 8))
    plt.bar(models, yAxis, color='b')
    plt.title(f'Comparison between models: {measure}')
    plt.xlabel('Model')
    plt.ylabel(f'{measure}')
    plt.grid(True, axis='y') 
    plt.xticks(rotation=45)

    plt.ylim(max(min(yAxis)-0.05, 0), max(yAxis) + 0.05)  
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{measure}_models.png'))
    plt.close()

def make_graph_random_forest(measure, xAxis, yAxis, output_dir):
    """Create a graph showing the effect of number of estimators on RF performance."""
    plt.figure(figsize=(8, 8))
    plt.plot(xAxis, yAxis, marker='o', linestyle='-', color='g')
    plt.title(f'Effects of number of estimators in {measure}')
    plt.xlabel('Number of Estimators')
    plt.ylabel(f'{measure}')
    plt.grid(True)
    plt.xticks(xAxis)

    plt.ylim(max(min(yAxis)-0.05, 0), max(yAxis) + 0.05)  
    plt.savefig(os.path.join(output_dir, f'{measure}_random_forest.png'))
    plt.close()

def make_graph_svm(measure, xAxis, yAxis, output_dir):
    """Create a graph showing the effect of C parameter on SVM performance."""
    plt.figure(figsize=(8, 8))
    plt.plot(xAxis, yAxis, marker='o', linestyle='-', color='r')
    plt.title(f'Effects of C parameter in {measure} (SVM)')
    plt.xlabel('C Parameter Value')
    plt.ylabel(f'{measure}')
    plt.xscale('log') 
    plt.grid(True)

    plt.ylim(max(min(yAxis)-0.05, 0), max(yAxis) + 0.05)  
    plt.savefig(os.path.join(output_dir, f'{measure}_svm.png'))
    plt.close()

def compare_decision_tree(df, output_dir):
    accuracy = []
    precision = []
    recall = []
    f1 = []
    train_time = []
    test_time = []
    cv_scores = []
    
    for i in range(9, 0, -1):
        start = time.time()
        model, X_test, y_test = create_model("decisionTree", df, i/10, 0)
        end = time.time()
        train_time += [end-start]
        acc, prec, rec, f1_, time_, y_pred = test_model(model, X_test, y_test)
        accuracy += [acc]
        precision += [prec]
        recall += [rec]
        f1 += [f1_]
        test_time += [time_]
        
        # Cross-validation
        feature_cols = ['fighter1_Weight', 'fighter1_Reach','fighter1_SLpM','fighter1_StrAcc','fighter1_SApM',
                        'fighter1_StrDef','fighter1_TDAvg','fighter1_TDAcc','fighter1_TDDef','fighter1_SubAvg',
                        'fighter2_Weight','fighter2_Reach','fighter2_SLpM','fighter2_StrAcc',
                        'fighter2_SApM','fighter2_StrDef','fighter2_TDAvg','fighter2_TDAcc','fighter2_TDDef',
                        'fighter2_SubAvg','fighter1_Wins','fighter1_Losses','fighter1_Draws','fighter2_Wins',
                        'fighter2_Losses','fighter2_Draws','fighter1_Height_in','fighter2_Height_in','fighter1_Age',
                        'fighter2_Age']
        
        df_temp = df.copy()
        df_temp = pd.get_dummies(df_temp, columns=['fighter1_Stance', 'fighter2_Stance'])
        df_temp['target'] = df_temp['fight_outcome'].apply(lambda x: 1 if x == 'fighter1' else (0 if x == 'fighter2' else np.nan))
        df_temp = df_temp.dropna(subset=['target'])
        
        stance_cols = [col for col in df_temp.columns if col.startswith('fighter1_Stance_') or col.startswith('fighter2_Stance_')]
        X_full = df_temp[feature_cols + stance_cols].fillna(df_temp[feature_cols + stance_cols].median())
        y_full = df_temp['target']
        
        cv_mean, cv_std = evaluate_with_cv(DecisionTreeClassifier(random_state=42), X_full, y_full)
        cv_scores.append(cv_mean)
        
        # Generate confusion matrix for the best test size (smallest one)
        if i == 1:
            plot_confusion_matrix(y_test, y_pred, "Decision Tree", output_dir)
            plot_roc_curve(model, X_test, y_test, "Decision Tree", output_dir)
            plot_learning_curve(DecisionTreeClassifier(random_state=42), X_full, y_full, "Decision Tree", output_dir)
    
    train_sizes = [x/10 for x in range(9,0,-1)]
    
    # Print results
    print("\n----- DECISION TREE RESULTS BY TEST SIZE -----")
    results_df = pd.DataFrame({
        'Test Size': train_sizes,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'CV Score': cv_scores,
        'Training Time': train_time,
        'Testing Time': test_time
    })
    print(results_df.round(4))
    
    make_graph_decision_tree("Accuracy", train_sizes, accuracy, output_dir)
    make_graph_decision_tree("Precision", train_sizes, precision, output_dir)
    make_graph_decision_tree("Recall", train_sizes, recall, output_dir)
    make_graph_decision_tree("F1 Score", train_sizes, f1, output_dir)
    make_graph_decision_tree("CV Score", train_sizes, cv_scores, output_dir)
    make_graph_decision_tree("Training Time", train_sizes, train_time, output_dir)
    make_graph_decision_tree("Testing Time", train_sizes, test_time, output_dir)

def compare_k_nearest_neighbors(df, output_dir):
    """Compare performance of K-Nearest Neighbors models with different numbers of neighbors."""
    accuracy = []
    precision = []
    recall = []
    f1 = []
    train_time = []
    test_time = []
    cv_scores = []
    
    # Prepare data once for cross-validation
    feature_cols = ['fighter1_Weight', 'fighter1_Reach','fighter1_SLpM','fighter1_StrAcc','fighter1_SApM',
                    'fighter1_StrDef','fighter1_TDAvg','fighter1_TDAcc','fighter1_TDDef','fighter1_SubAvg',
                    'fighter2_Weight','fighter2_Reach','fighter2_SLpM','fighter2_StrAcc',
                    'fighter2_SApM','fighter2_StrDef','fighter2_TDAvg','fighter2_TDAcc','fighter2_TDDef',
                    'fighter2_SubAvg','fighter1_Wins','fighter1_Losses','fighter1_Draws','fighter2_Wins',
                    'fighter2_Losses','fighter2_Draws','fighter1_Height_in','fighter2_Height_in','fighter1_Age',
                    'fighter2_Age']
    
    df_temp = df.copy()
    df_temp = pd.get_dummies(df_temp, columns=['fighter1_Stance', 'fighter2_Stance'])
    df_temp['target'] = df_temp['fight_outcome'].apply(lambda x: 1 if x == 'fighter1' else (0 if x == 'fighter2' else np.nan))
    df_temp = df_temp.dropna(subset=['target'])
    
    stance_cols = [col for col in df_temp.columns if col.startswith('fighter1_Stance_') or col.startswith('fighter2_Stance_')]
    X_full = df_temp[feature_cols + stance_cols].fillna(df_temp[feature_cols + stance_cols].median())
    y_full = df_temp['target']
    
    best_k = 0
    best_accuracy = 0
    best_model_data = None
    
    for i in range(1, 31, 1):
        start = time.time()
        model, X_test, y_test = create_model("K-nearestNeighbors", df, 0.1, i)
        end = time.time()
        train_time += [end-start]
        acc, prec, rec, f1_, time_, y_pred = test_model(model, X_test, y_test)
        accuracy += [acc]
        precision += [prec]
        recall += [rec]
        f1 += [f1_]
        test_time += [time_]
        
        # Cross-validation
        cv_mean, cv_std = evaluate_with_cv(KNeighborsClassifier(i), X_full, y_full)
        cv_scores.append(cv_mean)
        
        # Track best model
        if acc > best_accuracy:
            best_accuracy = acc
            best_k = i
            best_model_data = (model, X_test, y_test, y_pred)
    
    num_neigs = [x for x in range(1,31,1)]
    
    # Generate confusion matrix and ROC curve for best k
    if best_model_data:
        model, X_test, y_test, y_pred = best_model_data
        plot_confusion_matrix(y_test, y_pred, f"KNN (k={best_k})", output_dir)
        plot_roc_curve(model, X_test, y_test, f"KNN (k={best_k})", output_dir)
        plot_learning_curve(KNeighborsClassifier(best_k), X_full, y_full, f"KNN (k={best_k})", output_dir)
    
    # Print results
    print("\n----- KNN RESULTS BY NUMBER OF NEIGHBORS -----")
    k_samples = [1, 5, 10, 15, 20, 25, 30]
    k_indices = [k-1 for k in k_samples]
    results_df = pd.DataFrame({
        'K Neighbors': [num_neigs[i] for i in k_indices],
        'Accuracy': [accuracy[i] for i in k_indices],
        'Precision': [precision[i] for i in k_indices],
        'Recall': [recall[i] for i in k_indices],
        'F1 Score': [f1[i] for i in k_indices],
        'CV Score': [cv_scores[i] for i in k_indices],
        'Training Time': [train_time[i] for i in k_indices],
        'Testing Time': [test_time[i] for i in k_indices]
    })
    print(results_df.round(4))
    
    best_k_idx = np.argmax(accuracy)
    print(f"\nBest K value found: {best_k_idx + 1}")
    print(f"Maximum accuracy: {accuracy[best_k_idx]:.4f}")
    print(f"Precision: {precision[best_k_idx]:.4f}")
    print(f"Recall: {recall[best_k_idx]:.4f}")
    print(f"F1 Score: {f1[best_k_idx]:.4f}")
    
    make_graph_k_neighbors("Accuracy", num_neigs, accuracy, output_dir)
    make_graph_k_neighbors("Precision", num_neigs, precision, output_dir)
    make_graph_k_neighbors("Recall", num_neigs, recall, output_dir)
    make_graph_k_neighbors("F1 Score", num_neigs, f1, output_dir)
    make_graph_k_neighbors("CV Score", num_neigs, cv_scores, output_dir)
    make_graph_k_neighbors("Training Time", num_neigs, train_time, output_dir)
    make_graph_k_neighbors("Testing Time", num_neigs, test_time, output_dir)

def compare_random_forest(df, output_dir):
    accuracy = []
    precision = []
    recall = []
    f1 = []
    train_time = []
    test_time = []
    cv_scores = []
    n_estimators_values = [10, 50, 100, 150, 200]
    
    # Prepare data for cross-validation
    feature_cols = ['fighter1_Weight', 'fighter1_Reach','fighter1_SLpM','fighter1_StrAcc','fighter1_SApM',
                    'fighter1_StrDef','fighter1_TDAvg','fighter1_TDAcc','fighter1_TDDef','fighter1_SubAvg',
                    'fighter2_Weight','fighter2_Reach','fighter2_SLpM','fighter2_StrAcc',
                    'fighter2_SApM','fighter2_StrDef','fighter2_TDAvg','fighter2_TDAcc','fighter2_TDDef',
                    'fighter2_SubAvg','fighter1_Wins','fighter1_Losses','fighter1_Draws','fighter2_Wins',
                    'fighter2_Losses','fighter2_Draws','fighter1_Height_in','fighter2_Height_in','fighter1_Age',
                    'fighter2_Age']
    
    df_temp = df.copy()
    df_temp = pd.get_dummies(df_temp, columns=['fighter1_Stance', 'fighter2_Stance'])
    df_temp['target'] = df_temp['fight_outcome'].apply(lambda x: 1 if x == 'fighter1' else (0 if x == 'fighter2' else np.nan))
    df_temp = df_temp.dropna(subset=['target'])
    
    stance_cols = [col for col in df_temp.columns if col.startswith('fighter1_Stance_') or col.startswith('fighter2_Stance_')]
    X_full = df_temp[feature_cols + stance_cols].fillna(df_temp[feature_cols + stance_cols].median())
    y_full = df_temp['target']
    
    best_n = 0
    best_accuracy = 0
    best_model_data = None
    
    for n_estimators in n_estimators_values:
        start = time.time()
        model, X_test, y_test = create_model("randomForest", df, 0.1, 0, n_estimators=n_estimators)
        end = time.time()
        train_time += [end-start]
        acc, prec, rec, f1_, time_, y_pred = test_model(model, X_test, y_test)
        accuracy += [acc]
        precision += [prec]
        recall += [rec]
        f1 += [f1_]
        test_time += [time_]
        
        # Cross-validation
        cv_mean, cv_std = evaluate_with_cv(RandomForestClassifier(n_estimators=n_estimators, random_state=42), X_full, y_full)
        cv_scores.append(cv_mean)
        
        # Track best model
        if acc > best_accuracy:
            best_accuracy = acc
            best_n = n_estimators
            best_model_data = (model, X_test, y_test, y_pred)
    
    # Generate confusion matrix and ROC curve for best n_estimators
    if best_model_data:
        model, X_test, y_test, y_pred = best_model_data
        plot_confusion_matrix(y_test, y_pred, f"Random Forest (n={best_n})", output_dir)
        plot_roc_curve(model, X_test, y_test, f"Random Forest (n={best_n})", output_dir)
        plot_learning_curve(RandomForestClassifier(n_estimators=best_n, random_state=42), X_full, y_full, f"Random Forest (n={best_n})", output_dir)
    
    # Print results
    print("\n----- RANDOM FOREST RESULTS BY NUMBER OF ESTIMATORS -----")
    results_df = pd.DataFrame({
        'N Estimators': n_estimators_values,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'CV Score': cv_scores,
        'Training Time': train_time,
        'Testing Time': test_time
    })
    print(results_df.round(4))
    
    best_n_idx = np.argmax(accuracy)
    print(f"\nBest number of estimators: {n_estimators_values[best_n_idx]}")
    print(f"Maximum accuracy: {accuracy[best_n_idx]:.4f}")
    print(f"Precision: {precision[best_n_idx]:.4f}")
    print(f"Recall: {recall[best_n_idx]:.4f}")
    print(f"F1 Score: {f1[best_n_idx]:.4f}")
    
    make_graph_random_forest("Accuracy", n_estimators_values, accuracy, output_dir)
    make_graph_random_forest("Precision", n_estimators_values, precision, output_dir)
    make_graph_random_forest("Recall", n_estimators_values, recall, output_dir)
    make_graph_random_forest("F1 Score", n_estimators_values, f1, output_dir)
    make_graph_random_forest("CV Score", n_estimators_values, cv_scores, output_dir)
    make_graph_random_forest("Training Time", n_estimators_values, train_time, output_dir)
    make_graph_random_forest("Testing Time", n_estimators_values, test_time, output_dir)

def compare_svm(df, output_dir):
    """Compare SVM performance with different C values."""
    accuracy = []
    precision = []
    recall = []
    f1 = []
    train_time = []
    test_time = []
    cv_scores = []
    c_values = [0.1, 0.5, 1, 5, 10, 100]
    
    # Prepare data for cross-validation
    feature_cols = ['fighter1_Weight', 'fighter1_Reach','fighter1_SLpM','fighter1_StrAcc','fighter1_SApM',
                    'fighter1_StrDef','fighter1_TDAvg','fighter1_TDAcc','fighter1_TDDef','fighter1_SubAvg',
                    'fighter2_Weight','fighter2_Reach','fighter2_SLpM','fighter2_StrAcc',
                    'fighter2_SApM','fighter2_StrDef','fighter2_TDAvg','fighter2_TDAcc','fighter2_TDDef',
                    'fighter2_SubAvg','fighter1_Wins','fighter1_Losses','fighter1_Draws','fighter2_Wins',
                    'fighter2_Losses','fighter2_Draws','fighter1_Height_in','fighter2_Height_in','fighter1_Age',
                    'fighter2_Age']
    
    df_temp = df.copy()
    df_temp = pd.get_dummies(df_temp, columns=['fighter1_Stance', 'fighter2_Stance'])
    df_temp['target'] = df_temp['fight_outcome'].apply(lambda x: 1 if x == 'fighter1' else (0 if x == 'fighter2' else np.nan))
    df_temp = df_temp.dropna(subset=['target'])
    
    stance_cols = [col for col in df_temp.columns if col.startswith('fighter1_Stance_') or col.startswith('fighter2_Stance_')]
    X_full = df_temp[feature_cols + stance_cols].fillna(df_temp[feature_cols + stance_cols].median())
    y_full = df_temp['target']
    
    best_c = 0
    best_accuracy = 0
    best_model_data = None
    
    for c in c_values:
        start = time.time()
        result = create_model("svm", df, 0.1, 0, c_value=c)
        if len(result) == 4: 
            model, X_test, y_test, scaler = result
        else:
            model, X_test, y_test = result
        end = time.time()
        train_time += [end-start]
        acc, prec, rec, f1_, time_, y_pred = test_model(model, X_test, y_test)
        accuracy += [acc]
        precision += [prec]
        recall += [rec]
        f1 += [f1_]
        test_time += [time_]
        
        # Cross-validation with scaling
        scaler_cv = StandardScaler()
        X_full_scaled = scaler_cv.fit_transform(X_full)
        cv_mean, cv_std = evaluate_with_cv(SVC(C=c, kernel='rbf', random_state=42), X_full_scaled, y_full)
        cv_scores.append(cv_mean)
        
        # Track best model
        if acc > best_accuracy:
            best_accuracy = acc
            best_c = c
            best_model_data = (model, X_test, y_test, y_pred)
    
    # Generate confusion matrix and ROC curve for best C
    if best_model_data:
        model, X_test, y_test, y_pred = best_model_data
        plot_confusion_matrix(y_test, y_pred, f"SVM (C={best_c})", output_dir)
        plot_roc_curve(model, X_test, y_test, f"SVM (C={best_c})", output_dir)
        
        # For learning curve, we need to create a pipeline with scaling
        from sklearn.pipeline import Pipeline
        svm_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(C=best_c, kernel='rbf', random_state=42))
        ])
        plot_learning_curve(svm_pipeline, X_full, y_full, f"SVM (C={best_c})", output_dir)
    
    print("\n----- SVM RESULTS BY C VALUE -----")
    results_df = pd.DataFrame({
        'C Value': c_values,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'CV Score': cv_scores,
        'Training Time': train_time,
        'Testing Time': test_time
    })
    print(results_df.round(4))
    
    best_c_idx = np.argmax(accuracy)
    print(f"\nBest C value: {c_values[best_c_idx]}")
    print(f"Maximum accuracy: {accuracy[best_c_idx]:.4f}")
    print(f"Precision: {precision[best_c_idx]:.4f}")
    print(f"Recall: {recall[best_c_idx]:.4f}")
    print(f"F1 Score: {f1[best_c_idx]:.4f}")
    
    make_graph_svm("Accuracy", c_values, accuracy, output_dir)
    make_graph_svm("Precision", c_values, precision, output_dir)
    make_graph_svm("Recall", c_values, recall, output_dir)
    make_graph_svm("F1 Score", c_values, f1, output_dir)
    make_graph_svm("CV Score", c_values, cv_scores, output_dir)
    make_graph_svm("Training Time", c_values, train_time, output_dir)
    make_graph_svm("Testing Time", c_values, test_time, output_dir)

def compare_models(df, n_neighbors, output_dir, n_estimators=100, c_value=1.0):
    accuracy = []
    precision = []
    recall = []
    f1 = []
    train_time = []
    test_time = []
    cv_scores = []
    models = ["Decision Tree","K-Nearest Neighbors", "SVM", "Random Forest"]
    model_types = ["decisionTree","K-nearestNeighbors", "svm", "randomForest"]
    
    # Prepare data for cross-validation
    feature_cols = ['fighter1_Weight', 'fighter1_Reach','fighter1_SLpM','fighter1_StrAcc','fighter1_SApM',
                    'fighter1_StrDef','fighter1_TDAvg','fighter1_TDAcc','fighter1_TDDef','fighter1_SubAvg',
                    'fighter2_Weight','fighter2_Reach','fighter2_SLpM','fighter2_StrAcc',
                    'fighter2_SApM','fighter2_StrDef','fighter2_TDAvg','fighter2_TDAcc','fighter2_TDDef',
                    'fighter2_SubAvg','fighter1_Wins','fighter1_Losses','fighter1_Draws','fighter2_Wins',
                    'fighter2_Losses','fighter2_Draws','fighter1_Height_in','fighter2_Height_in','fighter1_Age',
                    'fighter2_Age']
    
    df_temp = df.copy()
    df_temp = pd.get_dummies(df_temp, columns=['fighter1_Stance', 'fighter2_Stance'])
    df_temp['target'] = df_temp['fight_outcome'].apply(lambda x: 1 if x == 'fighter1' else (0 if x == 'fighter2' else np.nan))
    df_temp = df_temp.dropna(subset=['target'])
    
    stance_cols = [col for col in df_temp.columns if col.startswith('fighter1_Stance_') or col.startswith('fighter2_Stance_')]
    X_full = df_temp[feature_cols + stance_cols].fillna(df_temp[feature_cols + stance_cols].median())
    y_full = df_temp['target']
    
    for i, model_type in enumerate(model_types):
        acc_model = []
        prec_model = []
        recall_model = []
        f1_model = []
        train_time_model = []
        test_time_model = []
        
        for j in range(10):
            start = time.time()
            result = create_model(model_type, df, 0.1, n_neighbors, n_estimators=n_estimators, c_value=c_value)
            if len(result) == 4:
                model, X_test, y_test, scaler = result
            else:
                model, X_test, y_test = result
            end = time.time()
            train_time_model += [end-start]
            acc, prec, rec, f1_, time_, y_pred = test_model(model, X_test, y_test)
            acc_model += [acc]
            prec_model += [prec]
            recall_model += [rec]
            f1_model += [f1_]
            test_time_model += [time_]
        
        accuracy += [sum(acc_model)/len(acc_model)]
        precision += [sum(prec_model)/len(prec_model)]
        recall += [sum(recall_model)/len(recall_model)]
        f1 += [sum(f1_model)/len(f1_model)]
        train_time += [sum(train_time_model)/len(train_time_model)]
        test_time += [sum(test_time_model)/len(test_time_model)]
        
        # Cross-validation
        if model_type == "decisionTree":
            cv_model = DecisionTreeClassifier(random_state=42)
            cv_mean, cv_std = evaluate_with_cv(cv_model, X_full, y_full)
        elif model_type == "K-nearestNeighbors":
            cv_model = KNeighborsClassifier(n_neighbors)
            cv_mean, cv_std = evaluate_with_cv(cv_model, X_full, y_full)
        elif model_type == "randomForest":
            cv_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            cv_mean, cv_std = evaluate_with_cv(cv_model, X_full, y_full)
        elif model_type == "svm":
            from sklearn.pipeline import Pipeline
            cv_model = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(C=c_value, kernel='rbf', random_state=42))
            ])
            cv_mean, cv_std = evaluate_with_cv(cv_model, X_full, y_full)
        
        cv_scores.append(cv_mean)
        
        # Generate confusion matrix for each model
        plot_confusion_matrix(y_test, y_pred, models[i], output_dir)
        
        # Generate ROC curve for each model
        if len(result) == 4:
            plot_roc_curve(model, X_test, y_test, models[i], output_dir)
        else:
            plot_roc_curve(model, X_test, y_test, models[i], output_dir)
    
    # Print comparison results
    print("\n----- COMPARISON BETWEEN DECISION TREE, KNN, RANDOM FOREST AND SVM -----")
    results_df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'CV Score': cv_scores,
        'Training Time': train_time,
        'Testing Time': test_time
    })
    print(results_df.round(4))
    
    # Identify and print best model
    best_model_idx = np.argmax(accuracy)
    print(f"\nBest model: {models[best_model_idx]}")
    print(f"Best model accuracy: {accuracy[best_model_idx]:.4f}")
    print(f"Best model CV score: {cv_scores[best_model_idx]:.4f}")

    make_graph_models("Accuracy", models, accuracy, output_dir)
    make_graph_models("Precision", models, precision, output_dir)
    make_graph_models("Recall", models, recall, output_dir)
    make_graph_models("F1 Score", models, f1, output_dir)
    make_graph_models("CV Score", models, cv_scores, output_dir)
    make_graph_models("Training Time", models, train_time, output_dir)
    make_graph_models("Testing Time", models, test_time, output_dir)

def run_model_comparisons(df, output_dir):
    """
    Run all model comparison analyses.
    
    Args:
        df (DataFrame): Dataset containing UFC fights
        output_dir (str): Directory to save output graphs
    """
    # Create subdirectories for each model
    decision_tree_dir = os.path.join(output_dir, "decision_tree")
    knn_dir = os.path.join(output_dir, "knn")
    random_forest_dir = os.path.join(output_dir, "random_forest")
    svm_dir = os.path.join(output_dir, "svm")
    comparison_dir = os.path.join(output_dir, "model_comparison")
    
    # Create directories if they don't exist
    for directory in [decision_tree_dir, knn_dir, random_forest_dir, svm_dir, comparison_dir]:
        os.makedirs(directory, exist_ok=True)
    
    print("\n===== COMPARING DECISION TREE MODELS =====")
    compare_decision_tree(df, decision_tree_dir)
    
    print("\n===== COMPARING K-NEAREST NEIGHBORS MODELS =====")
    compare_k_nearest_neighbors(df, knn_dir)
    
    print("\n===== COMPARING RANDOM FOREST MODELS =====")
    compare_random_forest(df, random_forest_dir)
    
    print("\n===== COMPARING SVM MODELS =====")
    compare_svm(df, svm_dir)
    
    print("\n===== COMPARING DIFFERENT MODEL TYPES =====")
    # Use the best parameters found
    compare_models(df, 5, comparison_dir, n_estimators=100, c_value=10)
    
    print("\nModel comparison complete. Graphs saved in subdirectories:")
    print(f"- Decision Tree: {decision_tree_dir}")
    print(f"- KNN: {knn_dir}")
    print(f"- Random Forest: {random_forest_dir}")
    print(f"- SVM: {svm_dir}")
    print(f"- Model Comparison: {comparison_dir}")