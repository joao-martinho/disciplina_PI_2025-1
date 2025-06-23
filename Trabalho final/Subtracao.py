import cv2
import numpy as np
import os

def carregar_imagem(caminho):
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    imagem = cv2.imread(caminho)
    if imagem is None:
        raise ValueError(f"Não foi possível ler a imagem: {caminho}")
    return imagem

def subtrair_imagens(imagem_antes, imagem_depois):
    if imagem_antes.shape != imagem_depois.shape:
        imagem_depois = cv2.resize(imagem_depois, (imagem_antes.shape[1], imagem_antes.shape[0]))

    imagem_diferenca = cv2.absdiff(imagem_antes, imagem_depois)
    imagem_diferenca_cinza = cv2.cvtColor(imagem_diferenca, cv2.COLOR_BGR2GRAY)
    _, imagem_binaria = cv2.threshold(imagem_diferenca_cinza, 50, 255, cv2.THRESH_BINARY)

    return imagem_diferenca, imagem_binaria

def salvar_imagem(imagem, caminho_saida):
    try:
        sucesso = cv2.imwrite(caminho_saida, imagem)
        if not sucesso:
            raise ValueError("Falha ao salvar a imagem")
    except Exception as e:
        print(f"Erro ao salvar imagem: {e}")

def main():
    caminho_antes = input("Digite o caminho da imagem 'antes': ").strip()
    caminho_depois = input("Digite o caminho da imagem 'depois': ").strip()

    try:
        imagem_antes = carregar_imagem(caminho_antes)
        imagem_depois = carregar_imagem(caminho_depois)

        imagem_diferenca, imagem_binaria = subtrair_imagens(imagem_antes, imagem_depois)

        salvar_imagem(imagem_diferenca, caminho_antes + "imagem_sub.png")
        salvar_imagem(imagem_binaria, caminho_antes + "_sub_bin.png")

    except Exception as e:
        print(f"Erro: {str(e)}")

if __name__ == "__main__":
    main()
