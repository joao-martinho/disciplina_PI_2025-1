# Felipe Karmann, João Martinho 
import os
import sys
from datetime import datetime

import imageio
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, generate_binary_structure


class ImageProcessor:
    def __init__(self, input_path):
        self.input_path = input_path
        self.output_dir = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.original_image = None
        self.current_image = None
        self.is_3d = False

    def load_image(self):
        """Carrega imagens 2D ou 3D com verificação robusta"""
        try:
            # Verifica se o arquivo existe
            if not os.path.isfile(self.input_path):
                raise FileNotFoundError(f"Arquivo não encontrado: {self.input_path}")

            # Processamento diferente para NIfTI vs imagens comuns
            if self.input_path.lower().endswith(('.nii', '.nii.gz')):
                img = nib.load(self.input_path)
                data = img.get_fdata()
                if data.ndim == 2:
                    data = data[..., np.newaxis]  # Adiciona dimensão Z se for 2D
                self.original_image = data
                self.is_3d = True
            else:
                img = Image.open(self.input_path)
                if img.mode != 'L':
                    img = img.convert('L')  # Converte para escala de cinza
                self.original_image = np.array(img)
                self.is_3d = False

            # Normalização para [0, 1]
            self.original_image = (self.original_image - np.min(self.original_image)) / (
                    np.max(self.original_image) - np.min(self.original_image) + 1e-8)
            self.current_image = self.original_image.copy()

            print(f"Imagem {'3D' if self.is_3d else '2D'} carregada. Dimensões: {self.original_image.shape}")
            return True

        except Exception as e:
            print(f"ERRO AO CARREGAR: {str(e)}")
            return False

    def save_image(self, etapa):
        """Salva imagens com tratamento adequado para 2D/3D"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            output_path = os.path.join(self.output_dir, f"etapa_{etapa}")

            if self.is_3d:
                nib.save(nib.Nifti1Image(self.current_image, np.eye(4)), f"{output_path}.nii.gz")
            else:
                img_2d = (np.squeeze(self.current_image) * 255).astype(np.uint8)
                imageio.imwrite(f"{output_path}.png", img_2d)

            print(f"Etapa {etapa} salva em: {output_path}")
            return True
        except Exception as e:
            print(f"ERRO AO SALVAR: {str(e)}")
            return False

    def binarize(self, threshold=0.5):
        """Binarização com limiar adaptativo"""
        try:
            self.current_image = (self.current_image > threshold).astype(np.uint8)
            return True
        except Exception as e:
            print(f"ERRO NA BINARIZAÇÃO: {str(e)}")
            return False

    def erode(self):
        """Erosão com estrutura adequada à dimensionalidade"""
        try:
            dim = 3 if self.is_3d else 2
            structure = generate_binary_structure(dim, 2)  # Conexão total

            # Garante que é binário
            binary_img = (self.current_image > 0.5).astype(np.uint8)
            self.current_image = binary_erosion(binary_img, structure=structure)
            return True
        except Exception as e:
            print(f"ERRO NA EROSÃO: {str(e)}")
            return False

    def detect_edges(self):
        """Detecção de bordas por diferença"""
        try:
            binary_img = (self.current_image > 0.5).astype(np.uint8)

            dim = 3 if self.is_3d else 2
            structure = generate_binary_structure(dim, 2)
            eroded = binary_erosion(binary_img, structure=structure)

            self.current_image = binary_img - eroded
            return True
        except Exception as e:
            print(f"ERRO NAS BORDAS: {str(e)}")
            return False


def main():
    print("=== PROCESSADOR DE IMAGENS 2D/3D ===")

    # Obter caminho da imagem
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = input("Arraste a imagem para aqui ou digite o caminho: ").strip('"')

    # Processamento
    processor = ImageProcessor(input_path)

    if not processor.load_image():
        input("\nPressione Enter para sair...")
        sys.exit(1)

    steps = [
        ("1. Binarização", processor.binarize),
        ("2. Erosão", processor.erode),
        ("3. Detecção de Bordas", processor.detect_edges)
    ]

    for name, operation in steps:
        print(f"\n{name}")
        if not operation():
            print("INTERROMPIDO - Erro na operação")
            input("Pressione Enter para sair...")
            sys.exit(1)
        processor.save_image(name[0])

    print("\nPROCESSO CONCLUÍDO COM SUCESSO!")
    print(f"Resultados em: {os.path.abspath(processor.output_dir)}")
    input("Pressione Enter para sair...")


if __name__ == "__main__":
    main()
