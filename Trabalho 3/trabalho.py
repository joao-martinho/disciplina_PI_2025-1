# Felipe Bona, João Martinho

# Importação das bibliotecas necessárias
import cv2  # OpenCV para processamento de vídeo e imagens
from ultralytics import YOLO  # YOLO para detecção de objetos
import pandas as pd  # Pandas para manipulação de dados e exportação para CSV

def processar_video(caminho_video, modelo, caminho_saida, classes_a_rastrear):
    """
    Processa um vídeo, detecta e conta objetos das classes especificadas.
    
    Args:
        caminho_video (str): Caminho para o arquivo de vídeo de entrada
        modelo (YOLO): Modelo YOLO para detecção de objetos
        caminho_saida (str): Caminho para salvar o vídeo processado
        classes_a_rastrear (list): Lista de classes de objetos a serem rastreadas
        
    Returns:
        dict: Dicionário com a contagem total de objetos por classe
    """
    
    # Abre o vídeo de entrada
    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {caminho_video}")
        return None

    # Obtém as propriedades do vídeo (largura, altura, FPS)
    largura_quadro = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura_quadro = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Configura o vídeo de saída
    saida = cv2.VideoWriter(caminho_saida, cv2.VideoWriter_fourcc(*'mp4v'), fps, (largura_quadro, altura_quadro))

    # Inicializa contadores
    contagem_objetos = {cls: 0 for cls in classes_a_rastrear}  # Contagem total
    contagens_quadro = []  # Armazena contagens por quadro

    # Processa cada quadro do vídeo
    while cap.isOpened():
        ret, quadro = cap.read()
        if not ret:
            break  # Sai do loop quando o vídeo terminar

        # Executa a detecção e rastreamento de objetos
        resultados = modelo.track(quadro, persist=True)
        
        # Inicializa contador para o quadro atual
        contagens_quadro_atual = {cls: 0 for cls in classes_a_rastrear}

        # Processa cada detecção no quadro
        for resultado in resultados:
            for caixa in resultado.boxes:
                nome_cls = modelo.names[int(caixa.cls)]  # Obtém o nome da classe
                
                # Verifica se a classe está na lista de classes a rastrear
                if nome_cls in classes_a_rastrear:
                    contagens_quadro_atual[nome_cls] += 1

                    # Desenha a caixa delimitadora e o rótulo
                    x1, y1, x2, y2 = map(int, caixa.xyxy[0])
                    cv2.rectangle(quadro, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(quadro, f"{nome_cls}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Atualiza contagens totais
        for cls in classes_a_rastrear:
            contagem_objetos[cls] += contagens_quadro_atual[cls]

        # Armazena contagens do quadro e escreve no vídeo de saída
        contagens_quadro.append(contagens_quadro_atual)
        saida.write(quadro)

    # Libera recursos
    cap.release()
    saida.release()

    # Salva contagens por quadro em arquivo CSV
    df = pd.DataFrame(contagens_quadro)
    df.to_csv("contagem_objetos_por_quadro.csv", index=False)

    return contagem_objetos

def principal():
    """
    Função principal que configura e executa o processamento do vídeo.
    """
    # Carrega o modelo YOLO pré-treinado
    modelo = YOLO('yolov8n.pt')
    
    # Configura caminhos e parâmetros
    caminho_video = r'D:\dowloads\exemplo.mp4'  # Substitua pelo caminho do seu vídeo
    caminho_saida = 'output_detected.mp4'  # Nome do arquivo de saída
    classes_a_rastrear = ['car', 'truck', 'bus', 'van']  # Classes de veículos a detectar

    # Processa o vídeo e obtém contagens
    contagens = processar_video(caminho_video, modelo, caminho_saida, classes_a_rastrear)
    
    # Exibe resultados
    print("\nContagem total de objetos no vídeo:")
    for cls, contagem in contagens.items():
        print(f"{cls}: {contagem}")

# Ponto de entrada do programa
if __name__ == "__main__":
    principal()
