import shutil
import random
from pathlib import Path
from tqdm import tqdm
import configuracao as cfg # Importa as configurações

# --- Proporções da Divisão ---
# 70% para treino, 20% para validação, 10% para teste
PROPORCAO_TREINO = 0.7
PROPORCAO_VALIDACAO = 0.2

def dividir_e_copiar_arquivos():
    """
    Função principal que organiza e divide o dataset.
    Ele irá apagar o diretório de dataset antigo para garantir uma divisão limpa.
    """
    print("Iniciando a preparação e divisão do dataset...")

    # Verifica se a pasta de imagens originais existe
    if not cfg.PASTA_IMAGENS_ORIGINAIS.exists():
        print(f"ERRO: A pasta de origem '{cfg.PASTA_IMAGENS_ORIGINAIS}' não foi encontrada.")
        print("Por favor, crie esta pasta e coloque as imagens das suas categorias dentro dela.")
        return

    # --- Limpeza do Dataset Antigo ---
    # Para garantir que não haja arquivos antigos misturados, apagamos a pasta 'dataset' se ela existir
    if cfg.PASTA_DATASET.exists():
        print(f"A pasta de dataset '{cfg.PASTA_DATASET}' já existe e será recriada do zero.")
        shutil.rmtree(cfg.PASTA_DATASET)
    
    print(f"Criando nova estrutura de pastas em '{cfg.PASTA_DATASET}'...")
    cfg.PASTA_DATASET.mkdir(parents=True)

    # Encontra as pastas de cada classe (cachorros, gatos, etc.)
    pastas_classes = [pasta for pasta in cfg.PASTA_IMAGENS_ORIGINAIS.iterdir() if pasta.is_dir()]

    # Percorre cada classe para fazer a divisão
    for pasta_classe in pastas_classes:
        nome_classe = pasta_classe.name
        print(f"\nProcessando a classe: '{nome_classe}'")
        
        # Lista todas as imagens da classe atual
        padroes_imagens = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
        imagens = []
        for padrao in padroes_imagens:
            imagens.extend(list(pasta_classe.glob(padrao)))
        
        if not imagens:
            print(f"  Aviso: Nenhuma imagem encontrada para a classe '{nome_classe}'. Pulando.")
            continue

        # Embaralha a lista de imagens para garantir aleatoriedade
        random.shuffle(imagens)
        
        # Calcula os pontos de corte para a divisão
        total_imagens = len(imagens)
        ponto_corte_treino = int(total_imagens * PROPORCAO_TREINO)
        ponto_corte_validacao = ponto_corte_treino + int(total_imagens * PROPORCAO_VALIDACAO)
        
        # Divide a lista de imagens
        imagens_treino = imagens[:ponto_corte_treino]
        imagens_validacao = imagens[ponto_corte_treino:ponto_corte_validacao]
        imagens_teste = imagens[ponto_corte_validacao:]
        
        print(f"  - Total: {total_imagens} | Treino: {len(imagens_treino)} | Validação: {len(imagens_validacao)} | Teste: {len(imagens_teste)}")

        # Define os conjuntos de dados e seus respectivos diretórios de destino
        conjuntos = {
            'train': imagens_treino,
            'val': imagens_validacao,
            'test': imagens_teste
        }
        
        # Cria as subpastas e copia os arquivos
        for nome_conjunto, lista_imagens in conjuntos.items():
            pasta_destino = cfg.PASTA_DATASET / nome_conjunto / nome_classe
            pasta_destino.mkdir(parents=True, exist_ok=True)
            
            for caminho_imagem in tqdm(lista_imagens, desc=f"  Copiando para {nome_conjunto}/{nome_classe}"):
                shutil.copy2(caminho_imagem, pasta_destino)

    print("\n----------------------------------------------------")
    print("Divisão e organização do dataset concluída com sucesso!")
    print("----------------------------------------------------")


if __name__ == "__main__":
    dividir_e_copiar_arquivos()