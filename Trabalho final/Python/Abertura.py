import cv2
import numpy as np
import os


def carregar_imagem_em_cinza(caminho):
    if not os.path.isfile(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    imagem = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    if imagem is None:
        raise ValueError(f"Erro ao carregar a imagem: {caminho}")
    return imagem


def aplicar_filtro_morfologico(imagem, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(imagem, cv2.MORPH_OPEN, kernel)


def salvar_imagem(imagem, caminho_saida):
    if not caminho_saida.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise ValueError("A extensão do arquivo deve ser .png, .jpg ou .jpeg")
    if not cv2.imwrite(caminho_saida, imagem):
        raise IOError(f"Erro ao salvar a imagem em: {caminho_saida}")


def solicitar_kernel_size():
    while True:
        entrada = input("Digite o tamanho do kernel (1 a 5): ").strip()
        try:
            valor = int(entrada)
            if 1 <= valor <= 5:
                return valor
            else:
                print("Por favor, insira um número entre 1 e 5.")
        except ValueError:
            print("Entrada inválida. Digite um número inteiro entre 1 e 5.")


def main():
    caminho_entrada = input("Caminho da imagem binária de entrada: ").strip('" ')
    caminho_saida = input("Caminho para salvar a imagem filtrada: ").strip('" ')
    kernel_size = solicitar_kernel_size()

    try:
        imagem = carregar_imagem_em_cinza(caminho_entrada)
        imagem_filtrada = aplicar_filtro_morfologico(imagem, kernel_size)
        salvar_imagem(imagem_filtrada, caminho_saida)

        print(f"Imagem filtrada salva com sucesso em: {caminho_saida} (Kernel = {kernel_size}x{kernel_size})")
    except Exception as e:
        print(f"Erro: {e}")


if __name__ == "__main__":
    main()
