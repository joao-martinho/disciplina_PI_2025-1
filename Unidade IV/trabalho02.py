# Felipe Bona, João Martinho

import cv2
import numpy as np

def detectar_iris(caminho_imagem, caminho_saida):
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        print("Erro ao carregar a imagem :(")
        return
    
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    cinza_desfocado = cv2.medianBlur(cinza, 5)
    
    circulos = cv2.HoughCircles(cinza_desfocado, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=30, maxRadius=100)
    
    if circulos is not None:
        circulos = np.uint16(np.around(circulos))
        (x, y, raio) = circulos[0, 0]
        
        mascara = np.zeros_like(cinza)
        cv2.circle(mascara, (x, y), raio, 255, -1)
        
        iris_isolada = cv2.bitwise_and(imagem, imagem, mask=mascara)
        
        y1, y2 = max(0, y-raio), min(imagem.shape[0], y+raio)
        x1, x2 = max(0, x-raio), min(imagem.shape[1], x+raio)
        iris_recortada = iris_isolada[y1:y2, x1:x2]
        
        cv2.imwrite(caminho_saida, iris_recortada)
        print(f"Íris detectada e salva em {caminho_saida}")
    else:
        print("Nenhuma íris detectada na imagem :(")

if __name__ == "__main__":
    caminho_entrada = input("Digite o caminho para a imagem (a partir do direrório atual): ")
    caminho_saida = caminho_entrada + "_iris_isolada.png"
    detectar_iris(caminho_entrada, caminho_saida)
