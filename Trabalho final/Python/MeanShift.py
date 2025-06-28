from PIL import Image
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import os


def redimensionar_imagem(imagem, max_pixels=1_000_000):
    altura, largura = imagem.shape[:2]
    total_pixels = altura * largura
    if total_pixels <= max_pixels:
        return imagem

    escala = np.sqrt(max_pixels / total_pixels)
    nova_tam = (int(largura * escala), int(altura * escala))
    return np.array(Image.fromarray(imagem).resize(nova_tam, Image.LANCZOS))


def to_data(img):
    for row in img:
        for pixel in row:
            yield tuple(pixel)


def mean_shift(data, bandwidth):
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(data)
    labels = ms.labels_
    centers = ms.cluster_centers_
    return labels, centers


def make_shifted_img(shape, labels, centers):
    img = []
    color = (centers[la] for la in labels)
    for i in range(shape[0]):
        img.append([])
        for j in range(shape[1]):
            img[i].append([int(c) for c in next(color)])
    return np.array(img, dtype='uint8')


def segmentar_imagem_mean_shift(caminho_imagem, caminho_saida, quantil=0.1, amostras=500):
    try:
        imagem = Image.open(caminho_imagem).convert('RGB')
        array_img = np.array(imagem)
        array_img = redimensionar_imagem(array_img)

        dados = tuple(to_data(array_img))

        largura_banda = estimate_bandwidth(dados, quantile=quantil, n_samples=amostras)
        largura_banda = max(largura_banda, 0.1)

        print(f"Quantil usado: {quantil}, Largura de banda estimada: {largura_banda:.2f}")

        rotulos, centros = mean_shift(dados, largura_banda)

        num_clusters = len(np.unique(rotulos))
        print(f"Número de clusters encontrados: {num_clusters}")

        img_segmentada = make_shifted_img(array_img.shape[:2], rotulos, centros)

        Image.fromarray(img_segmentada).save(caminho_saida)
        print(f"Imagem segmentada salva em: {caminho_saida}")

        return num_clusters

    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")
        return 0


def principal():
    caminho_imagem = input("Digite o caminho completo da imagem a ser processada: ")

    if not os.path.isfile(caminho_imagem):
        print("Arquivo não encontrado!")
        return

    nome_base, extensao = os.path.splitext(caminho_imagem)
    caminho_saida = f"{nome_base}_mean_shift{extensao}"

    try:
        quantil = float(input("Digite o parâmetro quantil (0.01-0.2, padrão 0.1): ").strip() or 0.1)
    except ValueError:
        print("Valor inválido, usando padrão 0.1")
        quantil = 0.1

    quantil = np.clip(quantil, 0.01, 0.2)

    num_clusters = segmentar_imagem_mean_shift(caminho_imagem, caminho_saida, quantil)
    print(f"\nSegmentação concluída com {num_clusters} cores distintas.")


if __name__ == "__main__":
    principal()
