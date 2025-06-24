import cv2
import FreeSimpleGUI as sg # ATENÇÃO: Use FreeSimpleGUI
import joblib
from ultralytics import YOLO
from layout import layout # Importa o layout simplificado

# carregar os modelos
try:
    extrator_caracteristicas = YOLO('yolov8n-cls.pt')
    classificador = joblib.load('classifier_svm.joblib')
    nomes_classes = joblib.load('class_labels.joblib')
except FileNotFoundError:
    sg.popup_error('Arquivos de modelo (.joblib) não encontrados! Execute "treinar_classificador.py" primeiro.')
    exit()

# janela da GUI
janela = sg.Window('Visualizador de Imagens', layout)
imagem_atual = None # Variável para guardar a imagem em formato OpenCV

while True:
    evento, valores = janela.read()
    if evento in (sg.WIN_CLOSED, 'Exit'):
        break

    if evento == 'Carregar Imagem':
        caminho_imagem = sg.popup_get_file('Escolha uma imagem')
        if caminho_imagem:
            imagem_atual = cv2.imread(caminho_imagem)
            # Converte a imagem para o formato que o PySimpleGUI pode exibir
            dados_imagem = cv2.imencode('.png', imagem_atual)[1].tobytes()
            janela['-IMAGEM-'].update(data=dados_imagem)
            janela['-RESULTADO_TEXTO-'].update('Resultado da Classificação:') # Limpa o resultado anterior

    if evento == 'Capturar da Webcam':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            sg.popup_error("Não foi possível acessar a webcam.")
        else:
            ret, frame = cap.read()
            cap.release()
            if ret:
                imagem_atual = frame
                dados_imagem = cv2.imencode('.png', imagem_atual)[1].tobytes()
                janela['-IMAGEM-'].update(data=dados_imagem)
                janela['-RESULTADO_TEXTO-'].update('Resultado da Classificação:')

    if evento == 'Classificar Imagem':
        if imagem_atual is not None:
            # 1. Extrair características da imagem atual
            resultado_tensor = extrator_caracteristicas.embed(source=[imagem_atual], verbose=False)
            vetor_caracteristicas = resultado_tensor[0].cpu().numpy().flatten().reshape(1, -1)
            
            # 2. Prever com o classificador SVM
            indice_predicao = classificador.predict(vetor_caracteristicas)[0]
            classe_predita = nomes_classes[indice_predicao]
            
            # 3. Obter a confiança
            probabilidades = classificador.predict_proba(vetor_caracteristicas)[0]
            confianca = probabilidades[indice_predicao] * 100
            
            # 4. Atualizar a interface
            texto_final = f'Resultado: {classe_predita} ({confianca:.2f}%)'
            janela['-RESULTADO_TEXTO-'].update(texto_final)
        else:
            sg.popup('Por favor, carregue uma imagem primeiro.')

janela.close()