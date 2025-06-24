from PIL import Image
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import os

def segmentar_imagem_mean_shift(caminho_imagem, caminho_saida, quantil=0.1, amostras=500):
    try:
        imagem_original = Image.open(caminho_imagem).convert('RGB')
        array_imagem = np.array(imagem_original)

        altura, largura = array_imagem.shape[:2]

        if altura * largura > 1000000:
            fator_escala = 1000000 / (altura * largura)
            nova_altura = int(altura * np.sqrt(fator_escala))
            nova_largura = int(largura * np.sqrt(fator_escala))
            array_imagem = np.array(imagem_original.resize((nova_largura, nova_altura), Image.LANCZOS))
            altura, largura = array_imagem.shape[:2]

        imagem_plana = array_imagem.reshape((-1, 3))

        largura_banda = estimate_bandwidth(imagem_plana, quantile=quantil, n_samples=amostras)
        if largura_banda <= 0:
            largura_banda = 0.1

        print(f"Quantil usado: {quantil}, Largura de banda estimada: {largura_banda:.2f}")

        ms = MeanShift(bandwidth=largura_banda, bin_seeding=True)
        ms.fit(imagem_plana)

        rotulos = ms.labels_
        centros_cluster = ms.cluster_centers_

        num_clusters = len(np.unique(rotulos))
        print(f"Número de clusters encontrados: {num_clusters}")

        imagem_segmentada = centros_cluster[rotulos].reshape(array_imagem.shape)
        imagem_segmentada = np.clip(imagem_segmentada, 0, 255).astype(np.uint8)

        imagem_resultante = Image.fromarray(imagem_segmentada)
        imagem_resultante.save(caminho_saida)

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

    quantil = input("Digite o parâmetro quantil (0.01-0.2, padrão 0.1): ")
    try:
        quantil = float(quantil) if quantil.strip() else 0.1
        quantil = np.clip(quantil, 0.01, 0.2)
    except ValueError:
        print("Valor inválido, usando padrão 0.1")
        quantil = 0.1

    num_clusters = segmentar_imagem_mean_shift(caminho_imagem, caminho_saida, quantil)

    print(f"\nSegmentação concluída com {num_clusters} cores distintas.")

if __name__ == "__main__":
    principal()
