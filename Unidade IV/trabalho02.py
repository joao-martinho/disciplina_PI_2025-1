# Felipe Bona, João Martinho

import cv2
import numpy as np

def processar_imagem(caminho_imagem):
    imagem_original = cv2.imread(caminho_imagem)
    if imagem_original is None:
        print("Não foi possível carregar a imagem :(")
        return

    imagem_cinza = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)
    imagem_cinza = cv2.medianBlur(imagem_cinza, 9)
    imagem_cinza = cv2.equalizeHist(imagem_cinza)

    circulos = cv2.HoughCircles(
        imagem_cinza,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=100,
        param2=40,
        minRadius=30,
        maxRadius=150
    )

    if circulos is not None:
        circulos = np.uint16(np.around(circulos))
        melhor_circulo = circulos[0][0]
        centro_x, centro_y, raio = melhor_circulo[0], melhor_circulo[1], melhor_circulo[2]

        mascara_iris = np.zeros(imagem_original.shape[:2], dtype=np.uint8)
        cv2.circle(mascara_iris, (centro_x, centro_y), raio, 255, -1)

        mascara_pupila = np.zeros(imagem_original.shape[:2], dtype=np.uint8)
        cv2.circle(mascara_pupila, (centro_x, centro_y), int(raio*0.35), 255, -1)

        mascara_final = cv2.subtract(mascara_iris, mascara_pupila)
        
        iris_isolada = cv2.bitwise_and(imagem_original, imagem_original, mask=mascara_final)

        fundo_branco = np.full_like(imagem_original, 255)
        resultado = cv2.bitwise_or(fundo_branco, fundo_branco, mask=cv2.bitwise_not(mascara_final))
        resultado = cv2.add(resultado, iris_isolada)

        nome_saida = caminho_imagem.split('.')[0] + "_iris_isolada.png"
        cv2.imwrite(nome_saida, resultado)
    else:
        print("Não foi possível detectar a íris na imagem :(")

if __name__ == "__main__":
    caminho = input("Digite o caminho da imagem: ")
    processar_imagem(caminho)
