# avaliar_modelo.py
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import joblib
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import configuracao as cfg # Importa as configurações

# Reutiliza a função de extração do script de treino
from train_classifier import extrair_caracteristicas 

def main():
    """
    Função principal que carrega o modelo treinado e o avalia
    no conjunto de dados de validação.
    """
    print("Carregando o classificador SVM e os rótulos...")
    try:
        classificador = joblib.load(cfg.CAMINHO_MODELO_SVM)
        nomes_classes = joblib.load(cfg.CAMINHO_ROTULOS_CLASSES)
        print("Modelos carregados com sucesso.")
    except FileNotFoundError:
        print(f"ERRO: Modelo não encontrado. Execute 'treinar_classificador.py' primeiro.")
        return

    # Extrai as características e rótulos do dataset de VALIDAÇÃO
    X_val, y_val, _ = extrair_caracteristicas(cfg.PASTA_VALIDACAO)
    
    if len(X_val) == 0:
        print("\nERRO: Nenhuma característica foi extraída do conjunto de validação.")
        return

    print(f"\nTotal de amostras de validação extraídas: {len(X_val)}")
    print("Avaliando o desempenho do modelo no conjunto de validação...")
    
    # Faz as predições no conjunto de validação
    predicoes = classificador.predict(X_val)
    
    # Calcula e exibe as métricas de desempenho
    acuracia = accuracy_score(y_val, predicoes)
    print(f"\n>> Acurácia (na validação): {acuracia:.4f} <<")
    
    print("\nRelatório de Classificação (na validação):")
    print(classification_report(y_val, predicoes, target_names=nomes_classes, zero_division=0))

if __name__ == "__main__":
    main()