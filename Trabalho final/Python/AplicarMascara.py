import cv2
import numpy as np
import os


def carregar_imagem(caminho):
    if not os.path.isfile(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
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
        regiao = cv2.bitwise_and(imagem, imagem, mask=mascara)
        hsv = cv2.cvtColor(regiao, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = 255  # saturação máxima
        hsv[:, :, 2] = np.where(mascara > 0, 255, hsv[:, :, 2])  # brilho máximo na região

        regiao_realcada = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        resultado = cv2.addWeighted(resultado, 1, regiao_realcada, 0.7, 0)

    return resultado


def gerar_caminho_saida(caminho_entrada, sufixo="_mask", extensao=".png"):
    base = os.path.splitext(os.path.basename(caminho_entrada))[0]
    pasta = os.path.dirname(caminho_entrada) or "."
    return os.path.join(pasta, f"{base}{sufixo}{extensao}")


def salvar_imagem(imagem, caminho_saida):
    if not cv2.imwrite(caminho_saida, imagem):
        raise IOError(f"Erro ao salvar a imagem em: {caminho_saida}")
    print(f"Imagem salva com sucesso: {caminho_saida}")


def main():
    caminho_binaria = input("Digite o caminho da imagem binária: ").strip('" ')
    caminho_base = input("Digite o caminho da segunda imagem (a ser mascarada): ").strip('" ')

    try:
        imagem_binaria = carregar_imagem(caminho_binaria)
        imagem_base = carregar_imagem(caminho_base)

        mascaras = processar_imagem_binaria(imagem_binaria)
        imagem_resultante = aplicar_mascaras(imagem_base, mascaras)

        caminho_saida = gerar_caminho_saida(caminho_base)
        salvar_imagem(imagem_resultante, caminho_saida)

    except Exception as e:
        print(f"Erro: {e}")


if __name__ == "__main__":
    main()
