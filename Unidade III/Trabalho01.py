# Felipe Bona, João Martinho

import os
import sys
from datetime import datetime

import imageio
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, generate_binary_structure


class ProcessadorDeImagens:
    def __init__(self, caminho_entrada):
        self.caminho_entrada = caminho_entrada
        self.diretorio_saida = f"saida_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.imagem_original = None
        self.imagem_atual = None
        self.eh_3d = False

    def carregar_imagem(self):
        try:
            if not os.path.isfile(self.caminho_entrada):
                raise FileNotFoundError(f"ERRO: arquivo não encontrado.")

            if self.caminho_entrada.lower().endswith(('.nii', '.nii.gz')):
                img = nib.load(self.caminho_entrada)
                dados = img.get_fdata()
                if dados.ndim == 2:
                    dados = dados[..., np.newaxis] 
                self.imagem_original = dados
                self.eh_3d = True
            else:
                img = Image.open(self.caminho_entrada)
                if img.mode != 'L':
                    img = img.convert('L')
                self.imagem_original = np.array(img)
                self.eh_3d = False

            self.imagem_original = (self.imagem_original - np.min(self.imagem_original)) / (
                    np.max(self.imagem_original) - np.min(self.imagem_original) + 1e-8)
            self.imagem_atual = self.imagem_original.copy()

            print(f"Imagem {'3D' if self.eh_3d else '2D'} carregada. Dimensões: {self.imagem_original.shape}")
            return True

        except Exception as e:
            print(f"ERRO AO CARREGAR: {str(e)}")
            return False

    def salvar_imagem(self, etapa):
        try:
            os.makedirs(self.diretorio_saida, exist_ok=True)
            caminho_saida = os.path.join(self.diretorio_saida, f"etapa_{etapa}")

            if self.eh_3d:
                nib.save(nib.Nifti1Image(self.imagem_atual, np.eye(4)), f"{caminho_saida}.nii.gz")
            else:
                img_2d = (np.squeeze(self.imagem_atual) * 255).astype(np.uint8)
                imageio.imwrite(f"{caminho_saida}.png", img_2d)

            print(f"Etapa {etapa} salva em: {caminho_saida}")
            return True
        except Exception as e:
            print(f"ERRO AO SALVAR: {str(e)}")
            return False

    def binarizar(self, limiar=0.5):
        try:
            self.imagem_atual = (self.imagem_atual > limiar).astype(np.uint8)
            return True
        except Exception as e:
            print(f"ERRO NA BINARIZAÇÃO: {str(e)}")
            return False

    def erodir(self):
        try:
            dim = 3 if self.eh_3d else 2
            estrutura = generate_binary_structure(dim, 2)

            binaria_img = (self.imagem_atual > 0.5).astype(np.uint8)
            self.imagem_atual = binary_erosion(binaria_img, structure=estrutura)
            return True
        except Exception as e:
            print(f"ERRO NA EROSÃO: {str(e)}")
            return False

    def detectar_contornos(self):
        try:
            binaria_img = (self.imagem_atual > 0.5).astype(np.uint8)

            dim = 3 if self.eh_3d else 2
            estrutura = generate_binary_structure(dim, 2)
            erodida = binary_erosion(binaria_img, structure=estrutura)

            self.imagem_atual = binaria_img - erodida
            return True
        except Exception as e:
            print(f"ERRO NOS CONTORNOS: {str(e)}")
            return False


def main():
    print("=== PROCESSADOR DE IMAGENS 3D ===")

    if len(sys.argv) > 1:
        caminho_entrada = sys.argv[1]
    else:
        caminho_entrada = input("Digite o caminho da imagem (absoluto): ").strip('"')

    processador = ProcessadorDeImagens(caminho_entrada)

    if not processador.carregar_imagem():
        input("\nPressione enter para sair...")
        sys.exit(1)

    etapas = [
        ("1. Binarização", processador.binarizar),
        ("2. Erosão", processador.erodir),
        ("3. Detecção de contornos", processador.detectar_contornos)
    ]

    for nome, operacao in etapas:
        print(f"\n{nome}")
        if not operacao():
            print("INTERROMPIDO: erro na operação")
            input("Pressione enter para sair...")
            sys.exit(1)
        processador.salvar_imagem(nome[0])

    print("\nSucesso!")
    print(f"Resultados em: {os.path.abspath(processador.diretorio_saida)}")
    input("Pressione enter para sair...")


if __name__ == "__main__":
    main()

