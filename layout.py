import FreeSimpleGUI as sg # ATENÇÃO: Use FreeSimpleGUI

layout = [
    [sg.Text('Visualizador e Classificador', font=('Any', 16))],
    [sg.Button('Carregar Imagem'), sg.Button('Capturar da Webcam'), sg.Button('Classificar Imagem', button_color=('white', 'green'))],
    [sg.Text('Resultado da Classificação:', key='-RESULTADO_TEXTO-', font=('Any', 12))],
    [sg.Image(key='-IMAGEM-')]
]