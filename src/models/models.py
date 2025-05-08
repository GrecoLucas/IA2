import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

def create_model(type, df, test_size, neighbors):
    # Features: relevant columns of the data
    feature_cols = ['fighter1_Weight', 'fighter1_Reach','fighter1_SLpM','fighter1_StrAcc','fighter1_SApM',
                    'fighter1_StrDef','fighter1_TDAvg','fighter1_TDAcc','fighter1_TDDef','fighter1_SubAvg',
                    'fighter2_Weight','fighter2_Reach','fighter2_SLpM','fighter2_StrAcc',
                    'fighter2_SApM','fighter2_StrDef','fighter2_TDAvg','fighter2_TDAcc','fighter2_TDDef',
                    'fighter2_SubAvg','fighter1_Wins','fighter1_Losses','fighter1_Draws','fighter2_Wins',
                    'fighter2_Losses','fighter2_Draws','fighter1_Height_in','fighter2_Height_in','fighter1_Age',
                    'fighter2_Age']
    df = pd.get_dummies(df, columns=['fighter1_Stance', 'fighter2_Stance'])

    df['target'] = df['fight_outcome'].apply(lambda x: 1 if x == 'fighter1' else (0 if x == 'fighter2' else np.nan))
    df = df.dropna(subset=['target'])
    stance_cols = [col for col in df.columns if col.startswith('fighter1_Stance_') or col.startswith('fighter2_Stance_')]

    X = df[feature_cols + stance_cols]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    if(type == "decisionTree"):
        model = DecisionTreeClassifier()
    if(type == "K-nearestNeighbors"):
        model = KNeighborsClassifier(n_neighbors=neighbors)


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
    return accuracy, precision, recall, f1, t


def make_graph_decision_tree(measure, xAxis, yAxis, output_dir):
    """Create a graph showing the effect of test size proportion on model performance."""
    plt.figure(figsize=(8, 8))
    plt.plot(xAxis, yAxis, marker='o', linestyle='-', color='b')
    plt.title(f'Efects of training data size in {measure}')
    plt.xlabel('Percentage of Data used for Testing')
    plt.ylabel(f'{measure}')
    plt.grid(True)
    plt.xticks(xAxis)

    plt.ylim(max(min(yAxis)-0.05, 0), max(yAxis) + 0.05)  
    plt.savefig(os.path.join(output_dir, f'{measure}_decision_trees.png'))


def make_graph_k_neighbors(measure, xAxis, yAxis, output_dir):
    """Create a graph showing the effect of number of neighbors on model performance."""
    plt.figure(figsize=(8, 8))
    plt.plot(xAxis, yAxis, marker='o', linestyle='-', color='b')
    plt.title(f'Efects of number of neighbors in {measure}')
    plt.xlabel('Number of Neighbors')
    plt.ylabel(f'{measure}')
    plt.grid(True)
    plt.xticks(xAxis)

    plt.ylim(max(min(yAxis)-0.05, 0), max(yAxis) + 0.05)  
    plt.savefig(os.path.join(output_dir, f'{measure}_number_neighbors.png'))


def make_graph_models(measure, models, yAxis, output_dir):
    """Create a graph comparing different models."""
    plt.figure(figsize=(8, 8))
    plt.bar(models, yAxis, color='b')
    plt.title(f'Comparison between models: {measure}')
    plt.xlabel('Model')
    plt.ylabel(f'{measure}')
    plt.grid(True, axis='y') 
    plt.xticks(models)

    plt.ylim(max(min(yAxis)-0.05, 0), max(yAxis) + 0.05)  
    plt.savefig(os.path.join(output_dir, f'{measure}_models.png'))


def compare_decision_tree(df, output_dir):
    accuracy = []
    precision = []
    recall = []
    f1 = []
    train_time = []
    test_time = []
    for i in range(9, 0, -1):
        start = time.time()
        model, X_test, y_test = create_model("decisionTree", df, i/10, 0)
        end = time.time()
        train_time += [end-start]
        acc, prec, rec, f1_, time_ = test_model(model, X_test, y_test)
        accuracy += [acc]
        precision += [prec]
        recall += [rec]
        f1 += [f1_]
        test_time += [time_]
    train_sizes = [x/10 for x in range(9,0,-1)]
    
    # Imprime os resultados no terminal
    print("\n----- DECISION TREE RESULTS BY TEST SIZE -----")
    results_df = pd.DataFrame({
        'Test Size': train_sizes,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Training Time': train_time,
        'Testing Time': test_time
    })
    print(results_df.round(4))
    
    make_graph_decision_tree("Accuracy", train_sizes, accuracy, output_dir)
    make_graph_decision_tree("Precision", train_sizes, precision, output_dir)
    make_graph_decision_tree("Recall", train_sizes, recall, output_dir)
    make_graph_decision_tree("F1 Score", train_sizes, f1, output_dir)
    make_graph_decision_tree("Training Time", train_sizes, train_time, output_dir)
    make_graph_decision_tree("Testing Time", train_sizes, test_time, output_dir)


def compare_k_nearest_neighbors(df, output_dir):
    """
    Compare performance of K-Nearest Neighbors models with different numbers of neighbors.
    
    Args:
        df (DataFrame): Dataset containing UFC fights
        output_dir (str): Directory to save output graphs
    """
    accuracy = []
    precision = []
    recall = []
    f1 = []
    train_time = []
    test_time = []
    for i in range(1, 31, 1):
        start = time.time()
        model, X_test, y_test = create_model("K-nearestNeighbors", df, 0.1, i)
        end = time.time()
        train_time += [end-start]
        acc, prec, rec, f1_, time_ = test_model(model, X_test, y_test)
        accuracy += [acc]
        precision += [prec]
        recall += [rec]
        f1 += [f1_]
        test_time += [time_]
    num_neigs = [x for x in range(1,31,1)]
    
    # Imprime os resultados no terminal
    print("\n----- KNN RESULTS BY NUMBER OF NEIGHBORS -----")
    # Imprimir apenas alguns valores de k para não sobrecarregar o terminal
    k_samples = [1, 5, 10, 15, 20, 25, 30]
    k_indices = [k-1 for k in k_samples]
    results_df = pd.DataFrame({
        'K Neighbors': [num_neigs[i] for i in k_indices],
        'Accuracy': [accuracy[i] for i in k_indices],
        'Precision': [precision[i] for i in k_indices],
        'Recall': [recall[i] for i in k_indices],
        'F1 Score': [f1[i] for i in k_indices],
        'Training Time': [train_time[i] for i in k_indices],
        'Testing Time': [test_time[i] for i in k_indices]
    })
    print(results_df.round(4))
    
    # Encontra e exibe o melhor valor de k
    best_k_idx = np.argmax(accuracy)
    print(f"\nMelhor valor de K encontrado: {best_k_idx + 1}")
    print(f"Acurácia máxima: {accuracy[best_k_idx]:.4f}")
    print(f"Precisão: {precision[best_k_idx]:.4f}")
    print(f"Recall: {recall[best_k_idx]:.4f}")
    print(f"F1 Score: {f1[best_k_idx]:.4f}")
    
    make_graph_k_neighbors("Accuracy", num_neigs, accuracy, output_dir)
    make_graph_k_neighbors("Precision", num_neigs, precision, output_dir)
    make_graph_k_neighbors("Recall", num_neigs, recall, output_dir)
    make_graph_k_neighbors("F1 Score", num_neigs, f1, output_dir)
    make_graph_k_neighbors("Training Time", num_neigs, train_time, output_dir)
    make_graph_k_neighbors("Testing Time", num_neigs, test_time, output_dir)


def compare_models(df, n_neighbors, output_dir):
    accuracy = []
    precision = []
    recall = []
    f1 = []
    train_time = []
    test_time = []
    models = ["decisionTree","K-nearestNeighbors"]
    for i in models:
        acc_model = []
        prec_model= []
        recall_model = []
        f1_model = []
        train_time_model = []
        test_time_model = []
        for j in range(10):
            start = time.time()
            model, X_test, y_test = create_model(i, df, 0.1, n_neighbors)
            end = time.time()
            train_time_model += [end-start]
            acc, prec, rec, f1_, time_ = test_model(model, X_test, y_test)
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
    
    # Imprimir resultados da comparação no terminal
    print("\n----- COMPARISON BETWEEN DECISION TREE AND KNN -----")
    results_df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Training Time': train_time,
        'Testing Time': test_time
    })
    print(results_df.round(4))
    
    # Identificar e imprimir o melhor modelo
    best_model_idx = np.argmax(accuracy)
    print(f"\nMelhor modelo: {models[best_model_idx]}")
    print(f"Diferença de acurácia: {abs(accuracy[0] - accuracy[1]):.4f}")
    print(f"Diferença de tempo de treinamento: {abs(train_time[0] - train_time[1]):.4f} segundos")
    
    make_graph_models("Accuracy", models, accuracy, output_dir)
    make_graph_models("Precision", models, precision, output_dir)
    make_graph_models("Recall", models, recall, output_dir)
    make_graph_models("F1 Score", models, f1, output_dir)
    make_graph_models("Training Time", models, train_time, output_dir)
    make_graph_models("Testing Time", models, test_time, output_dir)


def run_model_comparisons(df, output_dir):
    """
    Run all model comparison analyses.
    
    Args:
        df (DataFrame): Dataset containing UFC fights
        output_dir (str): Directory to save output graphs
    """
    print("\n===== COMPARING DECISION TREE MODELS =====")
    compare_decision_tree(df, output_dir)
    
    print("\n===== COMPARING K-NEAREST NEIGHBORS MODELS =====")
    compare_k_nearest_neighbors(df, output_dir)
    
    print("\n===== COMPARING DIFFERENT MODEL TYPES =====")
    compare_models(df, 5, output_dir)
    
    print("\nModel comparison complete. Plots saved in the output directory.")