# treinar_classificador.py
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
from tqdm import tqdm
import configuracao as cfg # Importa as configurações

MODELO_YOLO_EXTRATOR = 'yolov8n-cls.pt'

def extrair_caracteristicas(caminho_dataset: Path):
    """
    Percorre as pastas do dataset, extrai o vetor de características (embedding)
    de cada imagem usando o YOLO e retorna os dados prontos para o treinamento.
    """
    caracteristicas = []
    rotulos = []
    
    # Encontra os nomes das classes (nomes das pastas) e os ordena
    nomes_classes = sorted([pasta.name for pasta in caminho_dataset.iterdir() if pasta.is_dir()])
    
    print(f"Extraindo características do dataset em: {caminho_dataset}")
    print(f"Classes encontradas: {nomes_classes}")

    # Carrega o modelo YOLO que será usado para extrair as características
    extrator_yolo = YOLO(MODELO_YOLO_EXTRATOR)
    
    for indice_classe, nome_classe in enumerate(nomes_classes):
        pasta_classe = caminho_dataset / nome_classe
        
        # Procura por imagens com diferentes extensões
        padroes_imagens = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
        caminhos_imagens = []
        for padrao in padroes_imagens:
            caminhos_imagens.extend(list(pasta_classe.glob(padrao)))
            
        print(f"\nProcessando classe '{nome_classe}'...")
        
        for caminho_imagem in tqdm(caminhos_imagens, desc=f"Classe {nome_classe}"):
            try:
                # O método 'embed' retorna uma lista de tensores com as características
                resultado_tensor = extrator_yolo.embed([str(caminho_imagem)], verbose=False)
                
                if resultado_tensor:
                    # Converte o tensor para um array NumPy 1D
                    vetor_caracteristicas = resultado_tensor[0].cpu().numpy().flatten()
                    caracteristicas.append(vetor_caracteristicas)
                    rotulos.append(indice_classe)
            except Exception as e:
                print(f"Erro ao processar a imagem {caminho_imagem}: {e}")
                
    return np.array(caracteristicas), np.array(rotulos), nomes_classes

def main():
    """
    Função principal que orquestra o processo de treinamento.
    """
    # Extrai as características e rótulos do dataset de treino
    X_treino, y_treino, nomes_classes = extrair_caracteristicas(cfg.PASTA_TREINO)

    if len(X_treino) == 0:
        print("\nERRO: Nenhuma característica foi extraída. Verifique seu dataset.")
        return

    print(f"\nTotal de amostras de treino extraídas: {len(X_treino)}")
    print("Iniciando o treinamento do classificador SVM...")
    
    # Cria e treina o classificador SVM (Support Vector Machine)
    classificador_svm = SVC(kernel='linear', probability=True, C=1.0)
    classificador_svm.fit(X_treino, y_treino)
    
    print("Treinamento concluído com sucesso.")
    
    # Salva o classificador treinado e a lista de nomes das classes
    joblib.dump(classificador_svm, cfg.CAMINHO_MODELO_SVM)
    joblib.dump(nomes_classes, cfg.CAMINHO_ROTULOS_CLASSES)
    print(f"Classificador SVM salvo em: {cfg.CAMINHO_MODELO_SVM}")
    print(f"Rótulos das classes salvos em: {cfg.CAMINHO_ROTULOS_CLASSES}")

    # Avalia o modelo nos próprios dados de treino para ter uma primeira impressão
    print("\nAvaliando o desempenho nos dados de treino...")
    predicoes = classificador_svm.predict(X_treino)
    
    acuracia = accuracy_score(y_treino, predicoes)
    print(f"\nAcurácia (no treino): {acuracia:.4f}")
    
    print("\nRelatório de Classificação (no treino):")
    print(classification_report(y_treino, predicoes, target_names=nomes_classes, zero_division=0))

if __name__ == "__main__":
    main()