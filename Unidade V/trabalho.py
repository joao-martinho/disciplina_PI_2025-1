# Felipe Bona, João Martinho

import cv2
from ultralytics import YOLO
import pandas as pd

def processar_video(caminho_video, modelo, caminho_saida, classes_para_rastrear):
    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {caminho_video}")
        return None

    largura_quadro = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura_quadro = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(caminho_saida, cv2.VideoWriter_fourcc(*'mp4v'), fps, (largura_quadro, altura_quadro))

    contagem_objetos = {cls: 0 for cls in classes_para_rastrear}
    contagem_quadros = []

    while cap.isOpened():
        ret, quadro = cap.read()
        if not ret:
            break

        resultados = modelo.track(quadro, persist=True)
        contagem_quadro_atual = {cls: 0 for cls in classes_para_rastrear}

        for resultado in resultados:
            for caixa in resultado.boxes:
                nome_cls = modelo.names[int(caixa.cls)]
                if nome_cls in classes_para_rastrear:
                    contagem_quadro_atual[nome_cls] += 1

                    x1, y1, x2, y2 = map(int, caixa.xyxy[0])
                    cv2.rectangle(quadro, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(quadro, f"{nome_cls}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        for cls in classes_para_rastrear:
            contagem_objetos[cls] += contagem_quadro_atual[cls]

        contagem_quadros.append(contagem_quadro_atual)
        out.write(quadro)

    cap.release()
    out.release()
    
    df = pd.DataFrame(contagem_quadros)
    df.to_csv("contagem_objetos_por_quadro.csv", index=False)
    
    return contagem_objetos

def principal():
    modelo = YOLO('yolov8n.pt')  
    caminho_video = 'UA-DETRAC/test/MVI_39031.mp4'  
    caminho_saida = 'saida_detectada.mp4'
    classes_para_rastrear = ['carro', 'caminhao', 'onibus', 'van']

    contagens = processar_video(caminho_video, modelo, caminho_saida, classes_para_rastrear)
    print("\nContagem total de objetos no vídeo:")
    for cls, contagem in contagens.items():
        print(f"{cls}: {contagem}")

if __name__ == "__main__":
    principal()

