# Felipe Bona, João Martinho

import cv2
from ultralytics import YOLO
import pandas as pd

def processar_video(caminho_video, modelo, caminho_saida, classes_a_rastrear):
    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {caminho_video}")
        return None

    largura_quadro = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura_quadro = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    saida = cv2.VideoWriter(caminho_saida, cv2.VideoWriter_fourcc(*'mp4v'), fps, (largura_quadro, altura_quadro))

    contagem_objetos = {cls: 0 for cls in classes_a_rastrear}
    contagens_quadro = []

    while cap.isOpened():
        ret, quadro = cap.read()
        if not ret:
            break

        resultados = modelo.track(quadro, persist=True)
        contagens_quadro_atual = {cls: 0 for cls in classes_a_rastrear}

        for resultado in resultados:
            for caixa in resultado.boxes:
                nome_cls = modelo.names[int(caixa.cls)]
                if nome_cls in classes_a_rastrear:
                    contagens_quadro_atual[nome_cls] += 1

                    x1, y1, x2, y2 = map(int, caixa.xyxy[0])
                    cv2.rectangle(quadro, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(quadro, f"{nome_cls}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        for cls in classes_a_rastrear:
            contagem_objetos[cls] += contagens_quadro_atual[cls]

        contagens_quadro.append(contagens_quadro_atual)
        saida.write(quadro)

    cap.release()
    saida.release()

    df = pd.DataFrame(contagens_quadro)
    df.to_csv("contagem_objetos_por_quadro.csv", index=False)

    return contagem_objetos

def principal():
    modelo = YOLO('yolov8n.pt')
    caminho_video = r'D:\dowloads\exemplo.mp4'
    caminho_saida = 'output_detected.mp4'
    classes_a_rastrear = ['car', 'truck', 'bus', 'van']

    contagens = processar_video(caminho_video, modelo, caminho_saida, classes_a_rastrear)
    print("\nContagem total de objetos no vídeo:")
    for cls, contagem in contagens.items():
        print(f"{cls}: {contagem}")

if __name__ == "__main__":
    principal()
