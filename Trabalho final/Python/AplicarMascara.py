import cv2
import numpy as np

def carregar_imagem(caminho):
    imagem = cv2.imread(caminho, cv2.IMREAD_COLOR)
    if imagem is None:
        raise ValueError(f"Não foi possível carregar a imagem em {caminho}")
    return imagem

def processar_imagem_binaria(imagem_binaria):
    if len(imagem_binaria.shape) == 3:
        imagem_binaria = cv2.cvtColor(imagem_binaria, cv2.COLOR_BGR2GRAY)
    _, mascara_binaria = cv2.threshold(imagem_binaria, 127, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascaras = []
    for contorno in contornos:
        mascara = np.zeros_like(imagem_binaria)
        cv2.drawContours(mascara, [contorno], -1, 255, -1)
        mascaras.append(mascara)
    return mascaras

def aplicar_mascaras(imagem, mascaras):
    resultado = imagem.copy()
    for mascara in mascaras:
        regiao_mascarada = cv2.bitwise_and(imagem, imagem, mask=mascara)
        hsv = cv2.cvtColor(regiao_mascarada, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = 255  # Saturação máxima
        hsv[:,:,2] = np.where(mascara > 0, 255, hsv[:,:,2])  # Valor máximo nas áreas mascaradas
        regiao_saturada = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        resultado = cv2.addWeighted(resultado, 1, regiao_saturada, 0.7, 0)
    return resultado

def salvar_imagem(imagem, caminho_original):
    nome_base = caminho_original.split('/')[-1].split('\\')[-1].split('.')[0]
    caminho_saida = f"{nome_base}_mask.png"
    cv2.imwrite(caminho_saida, imagem)

def main():
    caminho_binaria = input("Digite o caminho da imagem binária: ")
    caminho_segunda = input("Digite o caminho da segunda imagem: ")

    imagem_binaria = carregar_imagem(caminho_binaria)
    segunda_imagem = carregar_imagem(caminho_segunda)

    mascaras = processar_imagem_binaria(imagem_binaria)
    imagem_resultante = aplicar_mascaras(segunda_imagem, mascaras)

    salvar_imagem(imagem_resultante, caminho_segunda)

if __name__ == "__main__":
    main()
