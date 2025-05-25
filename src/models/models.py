import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier


def create_model(type, df, test_size, neighbors, n_estimators=100, random_state=42):
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
        model = KNeighborsClassifier(neighbors)
    if(type == "randomForest"):
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    if type == "gradientBoosting":
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
    
    if type == "stacking":
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)),
            ('knn', KNeighborsClassifier(neighbors)),
            ('dt', DecisionTreeClassifier())
        ]
        final_estimator = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
        model = StackingClassifier(estimators=estimators, final_estimator=final_estimator, passthrough=True)


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
    plt.close()

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

def compare_random_forest(df, output_dir):
    accuracy = []
    precision = []
    recall = []
    f1 = []
    train_time = []
    test_time = []
    n_estimators_values = [10, 50, 100, 150, 200]
    
    for n_estimators in n_estimators_values:
        start = time.time()
        model, X_test, y_test = create_model("randomForest", df, 0.1, 0, n_estimators=n_estimators)
        end = time.time()
        train_time += [end-start]
        acc, prec, rec, f1_, time_ = test_model(model, X_test, y_test)
        accuracy += [acc]
        precision += [prec]
        recall += [rec]
        f1 += [f1_]
        test_time += [time_]
    
    # Imprime os resultados no terminal
    print("\n----- RESULTADOS DO RANDOM FOREST POR NÚMERO DE ESTIMADORES -----")
    results_df = pd.DataFrame({
        'N Estimators': n_estimators_values,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Training Time': train_time,
        'Testing Time': test_time
    })
    print(results_df.round(4))
    
    # Encontra e exibe o melhor número de estimadores
    best_n_idx = np.argmax(accuracy)
    print(f"\nMelhor número de estimadores: {n_estimators_values[best_n_idx]}")
    print(f"Acurácia máxima: {accuracy[best_n_idx]:.4f}")
    print(f"Precisão: {precision[best_n_idx]:.4f}")
    print(f"Recall: {recall[best_n_idx]:.4f}")
    print(f"F1 Score: {f1[best_n_idx]:.4f}")
    
    # Cria gráficos
    make_graph_random_forest("Accuracy", n_estimators_values, accuracy, output_dir)
    make_graph_random_forest("Precision", n_estimators_values, precision, output_dir)
    make_graph_random_forest("Recall", n_estimators_values, recall, output_dir)
    make_graph_random_forest("F1 Score", n_estimators_values, f1, output_dir)
    make_graph_random_forest("Training Time", n_estimators_values, train_time, output_dir)
    make_graph_random_forest("Testing Time", n_estimators_values, test_time, output_dir)

def compare_gradient_boosting(df, output_dir):
    accuracy, precision, recall, f1, train_time, test_time = [], [], [], [], [], []
    n_estimators_values = [10, 50, 100, 150, 200]

    for n_estimators in n_estimators_values:
        start = time.time()
        model, X_test, y_test = create_model("gradientBoosting", df, 0.1, 0, n_estimators=n_estimators)
        end = time.time()
        train_time.append(end - start)
        acc, prec, rec, f1_, time_ = test_model(model, X_test, y_test)
        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)
        f1.append(f1_)
        test_time.append(time_)

    # Print summary
    print("\n----- GRADIENT BOOSTING RESULTS BY N ESTIMATORS -----")
    results_df = pd.DataFrame({
        'N Estimators': n_estimators_values,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Training Time': train_time,
        'Testing Time': test_time
    })
    print(results_df.round(4))

    # Create graphs
    make_graph_gradient_boost("Accuracy", n_estimators_values, accuracy, output_dir)
    make_graph_gradient_boost("Precision", n_estimators_values, precision, output_dir)
    make_graph_gradient_boost("Recall", n_estimators_values, recall, output_dir)
    make_graph_gradient_boost("F1 Score", n_estimators_values, f1, output_dir)
    make_graph_gradient_boost("Training Time", n_estimators_values, train_time, output_dir)
    make_graph_gradient_boost("Testing Time", n_estimators_values, test_time, output_dir)

def compare_stacking_model(df, output_dir):
    accuracy, precision, recall, f1, train_time, test_time = [], [], [], [], [], []
    n_estimators_values = [10, 50, 100, 150, 200]

    for n_estimators in n_estimators_values:
        start = time.time()
        model, X_test, y_test = create_model("stacking", df, 0.1, neighbors=5, n_estimators=n_estimators)
        end = time.time()
        train_time.append(end - start)
        acc, prec, rec, f1_, time_ = test_model(model, X_test, y_test)
        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)
        f1.append(f1_)
        test_time.append(time_)

    print("\n----- STACKING MODEL RESULTS BY N ESTIMATORS -----")
    results_df = pd.DataFrame({
        'N Estimators': n_estimators_values,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Training Time': train_time,
        'Testing Time': test_time
    })
    print(results_df.round(4))

    # Reuse graph function (Gradient Boosting's for now)
    make_graph_stacking("Accuracy", n_estimators_values, accuracy, output_dir)
    make_graph_stacking("Precision", n_estimators_values, precision, output_dir)
    make_graph_stacking("Recall", n_estimators_values, recall, output_dir)
    make_graph_stacking("F1 Score", n_estimators_values, f1, output_dir)
    make_graph_stacking("Training Time", n_estimators_values, train_time, output_dir)
    make_graph_stacking("Testing Time", n_estimators_values, test_time, output_dir)


def make_graph_random_forest(measure, xAxis, yAxis, output_dir):
    """Create a graph showing the effect of number of estimators on RF performance."""
    plt.figure(figsize=(8, 8))
    plt.plot(xAxis, yAxis, marker='o', linestyle='-', color='b')
    plt.title(f'Efeitos do número de estimadores em {measure}')
    plt.xlabel('Número de Estimadores')
    plt.ylabel(f'{measure}')
    plt.grid(True)
    plt.xticks(xAxis)

    plt.ylim(max(min(yAxis)-0.05, 0), max(yAxis) + 0.05)  
    plt.savefig(os.path.join(output_dir, f'{measure}_random_forest.png'))

def make_graph_gradient_boost(measure, xAxis, yAxis, output_dir):
    """Create a graph showing the effect of number of estimators on RF performance."""
    plt.figure(figsize=(8, 8))
    plt.plot(xAxis, yAxis, marker='o', linestyle='-', color='b')
    plt.title(f'Efeitos do número de estimadores em {measure}')
    plt.xlabel('Número de Estimadores')
    plt.ylabel(f'{measure}')
    plt.grid(True)
    plt.xticks(xAxis)

    plt.ylim(max(min(yAxis)-0.05, 0), max(yAxis) + 0.05)  
    plt.savefig(os.path.join(output_dir, f'{measure}_gradient_boost.png'))


def make_graph_stacking(measure, xAxis, yAxis, output_dir):
    """Create a graph showing the effect of number of estimators on RF performance."""
    plt.figure(figsize=(8, 8))
    plt.plot(xAxis, yAxis, marker='o', linestyle='-', color='b')
    plt.title(f'Efeitos do número de estimadores em {measure}')
    plt.xlabel('Número de Estimadores')
    plt.ylabel(f'{measure}')
    plt.grid(True)
    plt.xticks(xAxis)

    plt.ylim(max(min(yAxis)-0.05, 0), max(yAxis) + 0.05)  
    plt.savefig(os.path.join(output_dir, f'{measure}_stacking.png'))




def compare_models(df, n_neighbors, output_dir, n_estimators=100):
    accuracy = []
    precision = []
    recall = []
    f1 = []
    train_time = []
    test_time = []
    models = ["decisionTree","K-nearestNeighbors", "randomForest", "gradientBoosting", "stacking"]
    for i in models:
        acc_model = []
        prec_model= []
        recall_model = []
        f1_model = []
        train_time_model = []
        test_time_model = []
        for j in range(10):
            start = time.time()
            model, X_test, y_test = create_model(i, df, 0.1, n_neighbors, n_estimators=n_estimators)
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
    print("\n----- COMPARAÇÃO ENTRE DECISION TREE, KNN E RANDOM FOREST -----")
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
    print(f"Acurácia do melhor modelo: {accuracy[best_model_idx]:.4f}")

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
    # Criar subdiretórios para cada modelo
    decision_tree_dir = os.path.join(output_dir, "decision_tree")
    knn_dir = os.path.join(output_dir, "knn")
    random_forest_dir = os.path.join(output_dir, "random_forest")
    gradient_boost_dir = os.path.join(output_dir, "gradient_boost")
    stacking_dir = os.path.join(output_dir, "stacking")    
    comparison_dir = os.path.join(output_dir, "model_comparison")


    # Criar diretórios se não existirem
    for directory in [decision_tree_dir, knn_dir, random_forest_dir, gradient_boost_dir, comparison_dir, stacking_dir]:
        os.makedirs(directory, exist_ok=True)
    
    print("\n===== COMPARANDO MODELOS DECISION TREE =====")
    compare_decision_tree(df, decision_tree_dir)
    
    print("\n===== COMPARANDO MODELOS K-NEAREST NEIGHBORS =====")
    compare_k_nearest_neighbors(df, knn_dir)
    
    print("\n===== COMPARANDO MODELOS RANDOM FOREST =====")
    compare_random_forest(df, random_forest_dir)

    print("\n===== COMPARANDO MODELOS GRADIENT BOOSTING =====")
    compare_gradient_boosting(df, gradient_boost_dir)

    print("\n===== COMPARANDO MODELO STACKING =====")
    compare_stacking_model(df, stacking_dir)

    print("\n===== COMPARANDO DIFERENTES TIPOS DE MODELOS =====")
    # Use o melhor n_neighbors encontrado para KNN e melhor n_estimators para RF
    compare_models(df, 5, comparison_dir, n_estimators=100)
    
    print("\nComparação de modelos completa. Gráficos salvos em subdiretórios:")
    print(f"- Árvore de decisão: {decision_tree_dir}")
    print(f"- KNN: {knn_dir}")
    print(f"- Random Forest: {random_forest_dir}")
    print(f"- Gradient boosting: {gradient_boost_dir}")
    print(f"- Stacking: {stacking_dir}")
    print(f"- Comparação entre modelos: {comparison_dir}")