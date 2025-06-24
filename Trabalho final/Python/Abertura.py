import cv2
import numpy as np
import sys

def filtrar_regioes_pequenas(caminho_entrada, caminho_saida, kernel_size=3):
    img = cv2.imread(caminho_entrada, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro: Não foi possível carregar a imagem. Verifique o caminho.")
        sys.exit(1)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    img_filtrada = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    cv2.imwrite(caminho_saida, img_filtrada)
    print(f"Imagem filtrada salva em: {caminho_saida} (Kernel = {kernel_size}x{kernel_size})")

if __name__ == "__main__":
    caminho_entrada = input("Caminho da imagem binária de entrada: ").strip('"')
    caminho_saida = input("Caminho para salvar a imagem filtrada: ").strip('"')

    filtrar_regioes_pequenas(caminho_entrada, caminho_saida, kernel_size=3)
