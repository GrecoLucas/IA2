import os
import sys
import pandas as pd

from analysis.analysis import load_data, run_complete_analysis
from models.models import run_model_comparisons

# Opções: "analysis", "model", "both"
RUN_MODE = "model"  # Altere conforme necessário

def main():
    # Definição de caminhos
    asset_path = '../assets/clean_ufc_all_fights.csv'
    output_dir = '../output'
    
    # Garante que o diretório de saída existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Carrega os dados (sempre necessário)
    print("\nCarregando dados...")
    df = load_data(asset_path)
    
    # Executa o modo selecionado
    if RUN_MODE == "analysis" or RUN_MODE == "both":
        print("\n==== INICIANDO ANÁLISE EXPLORATÓRIA DE DADOS DO UFC ====")
        run_complete_analysis(asset_path, output_dir)
        print("\n==== ANÁLISE EXPLORATÓRIA COMPLETA! ====")
    
    if RUN_MODE == "model" or RUN_MODE == "both":
        print("\n==== INICIANDO MODELAGEM PREDITIVA DE LUTAS DO UFC ====")
        run_model_comparisons(df, output_dir)
        print("\n==== MODELAGEM PREDITIVA COMPLETA! ====")
    
    print(f"Todos os resultados foram salvos no diretório: {output_dir}")

if __name__ == "__main__":
    main()