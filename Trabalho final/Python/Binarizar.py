import cv2
import os
import numpy as np


def carregar_imagem_em_cinza(caminho):
    imagem = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    if imagem is None:
        raise ValueError("Não foi possível carregar a imagem!")
    return imagem


def calcular_limiar_automatico(imagem):
    return np.mean(imagem)


def binarizar_imagem(imagem, limiar):
    _, binarizada = cv2.threshold(imagem, limiar, 255, cv2.THRESH_BINARY)
    return binarizada


def binarizar_imagem_automatica():
    caminho_imagem = input("Digite o caminho da imagem que deseja binarizar: ").strip()

    if not os.path.isfile(caminho_imagem):
        print("Arquivo não encontrado!")
        return

    caminho_saida = input("Digite o caminho para salvar a imagem binarizada (com extensão .jpg ou .png): ").strip()
    if not caminho_saida.lower().endswith((".jpg", ".jpeg", ".png")):
        print("Extensão inválida! Use .jpg ou .png")
        return

    try:
        imagem = carregar_imagem_em_cinza(caminho_imagem)
        limiar = calcular_limiar_automatico(imagem)
        print(f"Limiar automático calculado: {limiar:.2f}")

        binarizada = binarizar_imagem(imagem, limiar)
        cv2.imwrite(caminho_saida, binarizada)
        print(f"Imagem binarizada salva com sucesso em: {caminho_saida}")
    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")


if __name__ == "__main__":
    binarizar_imagem_automatica()
