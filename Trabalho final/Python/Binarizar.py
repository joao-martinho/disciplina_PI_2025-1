import cv2
import os
import numpy as np

def binarizar_imagem_automatica():
    caminho_imagem = input("Digite o caminho da imagem que deseja binarizar: ")

    if not os.path.isfile(caminho_imagem):
        print("Arquivo não encontrado!")
        return

    caminho_saida = input("Digite o caminho para salvar a imagem binarizada (com extensão .jpg ou .png): ")

    imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
    if imagem is None:
        print("Não foi possível carregar a imagem!")
        return

    limiar = np.mean(imagem)
    print(f"Limiar automático calculado: {limiar:.2f}")

    _, imagem_binarizada = cv2.threshold(imagem, limiar, 255, cv2.THRESH_BINARY)

    cv2.imwrite(caminho_saida, imagem_binarizada)
    print(f"Imagem binarizada salva com sucesso em: {caminho_saida}")

if __name__ == "__main__":
    binarizar_imagem_automatica()
