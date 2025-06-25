# Visualizador e Classificador de Imagens

Este projeto é uma aplicação em Python construída com `FreeSimpleGUI` que utiliza um classificador customizado para identificar o conteúdo de imagens carregadas de arquivos locais ou capturadas pela webcam.

## Funcionalidades

- **Carregamento de Imagens**: Permite carregar imagens dos formatos PNG, JPG e JPEG.
- **Captura via Webcam**: Captura imagens estáticas da webcam do usuário.
- **Classificação Inteligente**: Utiliza um modelo híbrido para identificar o conteúdo da imagem.
- **Interface Intuitiva**: Exibe a imagem e o resultado da classificação de forma clara.
- **Ferramentas Básicas**: Inclui funcionalidades de rotação, redimensionamento (exemplo) e salvamento da imagem processada.


## O Classificador

O sistema de classificação utiliza uma abordagem híbrida, combinando o poder do Deep Learning para extração de características com a eficiência do Machine Learning clássico para a classificação final.

- **Extrator de Características**: Um modelo `YOLOv8n-cls` pré-treinado na robusta base de dados ImageNet é utilizado para converter cada imagem em um vetor de características (*embedding*) de alta dimensão. Este vetor captura os atributos semânticos essenciais da imagem.

- **Algoritmo de Classificação**: Um classificador **Support Vector Machine (SVM)** com kernel linear, implementado com a biblioteca Scikit-learn, é treinado usando os vetores de características extraídos pelo YOLO. Esta abordagem é extremamente rápida e eficaz para aprender a separar as classes a partir dos dados já processados.

### Categorias

O modelo foi treinado para identificar e diferenciar as seguintes categorias de animais:
- Cachorros (`dog`)
- Gatos (`cat`)
- Cobras (`snake`)

### Dataset

O treinamento foi realizado com um dataset customizado contendo **mais de 15.000 imagens** (12.261 para treino e 3.502 para validação), divididas entre as categorias acima. O dataset principal foi obtido do Kaggle.

### Desempenho do Classificador

A performance do modelo foi medida no conjunto de validação, que contém imagens que o modelo não viu durante o treinamento. Esta é a medida mais representativa de seu desempenho em cenários reais.

- **Acurácia (Validação)**: **98.29%**
- **Precisão Média (Validação)**: **98%**
- **Recall Médio (Validação)**: **98%**

Abaixo estão os relatórios detalhados para os conjuntos de validação e treino.

<details>
<summary><strong>Clique para ver os Resultados no Conjunto de Validação (Métricas Reais)</strong></summary>

```
>> Acurácia (na validação): 0.9829 <<

Relatório de Classificação (na validação):
              precision    recall  f1-score   support

         cat       0.99      0.99      0.99      2266
         dog       0.97      0.97      0.97       949
       snake       1.00      1.00      1.00       287

    accuracy                           0.98      3502
   macro avg       0.99      0.98      0.98      3502
weighted avg       0.98      0.98      0.98      3502
```
</details>

<details>
<summary><strong>Clique para ver os Resultados no Conjunto de Treino</strong></summary>

```
Acurácia (no treino): 0.9885

Relatório de Classificação (no treino):
              precision    recall  f1-score   support

         cat       0.99      0.99      0.99      7933
         dog       0.99      0.97      0.98      3322
       snake       1.00      1.00      1.00      1006

    accuracy                           0.99     12261
   macro avg       0.99      0.99      0.99     12261
weighted avg       0.99      0.99      0.99     12261
```
</details>


## Guia de Execução

Siga os passos abaixo para configurar e executar o projeto em sua máquina local.

### 1. Configuração do Ambiente

Primeiro, clone o repositório e instale as dependências necessárias.

```bash
# Clone o repositório
git clone [https://github.com/danielppd/classificador_imagens.git](https://github.com/danielppd/classificador_imagens.git)
cd classificador_imagens

# Crie e ative um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale as bibliotecas necessárias
pip install -r requirements.txt
```
*(Nota: Certifique-se de ter um arquivo `requirements.txt` com as bibliotecas: `ultralytics`, `scikit-learn`, `joblib`, `tqdm`, `opencv-python`, `FreeSimpleGUI`)*

### 2. Preparação do Dataset (Para Treinar seu Próprio Modelo)

Este passo é **necessário apenas se você quiser treinar o modelo do zero** com suas próprias imagens.

1.  Crie uma pasta chamada `imagens_originais` na raiz do projeto.
2.  Dentro de `imagens_originais`, crie uma subpasta para cada categoria (ex: `gatos`, `cachorros`).
3.  Coloque todas as suas imagens brutas nas pastas de suas respectivas categorias.
4.  Execute o script de preparação para dividir os dados automaticamente:
    ```bash
    python preparar_dataset.py
    ```
    Este comando criará a pasta `dataset` com as subpastas `train`, `val` e `test` já populadas.

### 3. Treinamento e Avaliação (Para Treinar seu Próprio Modelo)

Após preparar o dataset no passo anterior, você pode treinar e avaliar o modelo.

```bash
# Treina o classificador e salva os arquivos .joblib
python train_classifier.py

# Avalia o modelo no conjunto de validação para obter as métricas de desempenho
python avaliar_modelo.py
```

### 4. Execução da Aplicação Principal

Com os modelos já treinados e salvos (`classifier_svm.joblib` e `class_labels.joblib`), inicie a interface gráfica.

```bash
python visualizador_classificador.py
```

---

## Como Utilizar a Aplicação

1.  **Carregar Imagem**: Clique para selecionar um arquivo de imagem do seu computador.
2.  **Capturar da Webcam**: Clique para tirar uma foto usando sua webcam.
3.  **Classificar Imagem**: Com uma imagem carregada, clique neste botão para que o modelo faça a predição. O resultado aparecerá no painel de controle.
4.  **Resetar Imagem**: Restaura a imagem processada para o seu estado original.
5.  **Salvar Resultado**: Permite salvar a imagem atualmente exibida.