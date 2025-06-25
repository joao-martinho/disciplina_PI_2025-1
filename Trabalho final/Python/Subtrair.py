import cv2
import numpy as np
import os


def carregar_imagem(caminho):
    if not os.path.isfile(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    imagem = cv2.imread(caminho)
    if imagem is None:
        raise ValueError(f"Não foi possível carregar a imagem: {caminho}")
    return imagem


def redimensionar_para_compatibilidade(imagem_base, imagem_alvo):
    if imagem_base.shape != imagem_alvo.shape:
        return cv2.resize(imagem_alvo, (imagem_base.shape[1], imagem_base.shape[0]))
    return imagem_alvo


def subtrair_imagens(imagem_antes, imagem_depois):
    imagem_depois = redimensionar_para_compatibilidade(imagem_antes, imagem_depois)
    diferenca = cv2.absdiff(imagem_antes, imagem_depois)
    return diferenca


def salvar_imagem(imagem, caminho_saida):
    if not caminho_saida.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise ValueError("A extensão do arquivo de saída deve ser .png, .jpg ou .jpeg")
    if not cv2.imwrite(caminho_saida, imagem):
        raise IOError(f"Erro ao salvar imagem: {caminho_saida}")


def gerar_caminho_saida(caminho_original, sufixo="_sub"):
    base, ext = os.path.splitext(caminho_original)
    return f"{base}{sufixo}{ext}"


def main():
    caminho_antes = input("Digite o caminho da imagem 'antes': ").strip()
    caminho_depois = input("Digite o caminho da imagem 'depois': ").strip()

    try:
        imagem_antes = carregar_imagem(caminho_antes)
        imagem_depois = carregar_imagem(caminho_depois)

        imagem_dif = subtrair_imagens(imagem_antes, imagem_depois)

        caminho_saida_dif = gerar_caminho_saida(caminho_depois, "_sub")
        salvar_imagem(imagem_dif, caminho_saida_dif)

        print(f"Imagem de diferença salva: {caminho_saida_dif}")

    except Exception as e:
        print(f"Erro: {e}")


if __name__ == "__main__":
    main()
