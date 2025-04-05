# Felipe Karmann, João Martinho 
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, generate_binary_structure
import os
import sys
from datetime import datetime


class ImageProcessor3D:
    def __init__(self, input_path):
        self.input_path = input_path
        self.output_dir = "output_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        self.original_image = None
        self.current_image = None
        self.header = None
        self.affine = None

    def load_image(self):
        try:
            img = nib.load(self.input_path)
            self.original_image = img.get_fdata()
            self.current_image = self.original_image.copy()
            self.header = img.header
            self.affine = img.affine
            print(f"Imagem carregada com sucesso. Dimensões: {self.original_image.shape}")
            return True
        except Exception as e:
            print(f"Erro ao carregar a imagem: {str(e)}")
            return False

    def create_output_dir(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Diretório de saída criado: {os.path.abspath(self.output_dir)}")
            return True
        except Exception as e:
            print(f"Erro ao criar diretório de saída: {str(e)}")
            return False

    def save_current_image(self, etapa):
        output_path = os.path.join(self.output_dir, f"resultado_etapa{etapa}.nii.gz")
        try:
            new_img = nib.Nifti1Image(self.current_image, self.affine, self.header)
            nib.save(new_img, output_path)
            print(f"Imagem da etapa {etapa} salva com sucesso em: {output_path}")
            return True
        except Exception as e:
            print(f"Erro ao salvar imagem da etapa {etapa}: {str(e)}")
            return False

    def binarize_image(self, threshold=0.5):
        try:
            max_val = np.max(self.current_image)
            threshold_value = threshold * max_val
            self.current_image = (self.current_image > threshold_value).astype(np.uint8)
            print("Imagem binarizada com sucesso.")
            return True
        except Exception as e:
            print(f"Erro ao binarizar imagem: {str(e)}")
            return False

    def apply_erosion(self):
        try:
            structure = generate_binary_structure(3, 3)  # Rank 3, conexão máxima
            self.current_image = binary_erosion(self.current_image, structure=structure).astype(np.uint8)
            print("Erosão aplicada com sucesso.")
            return True
        except Exception as e:
            print(f"Erro ao aplicar erosão: {str(e)}")
            return False

    def detect_edges(self):
        try:
            binary_image = (self.current_image > 0).astype(np.uint8)

            structure = generate_binary_structure(3, 3)
            eroded = binary_erosion(binary_image, structure=structure).astype(np.uint8)

            self.current_image = binary_image - eroded
            print("Bordas detectadas com sucesso.")
            return True
        except Exception as e:
            print(f"Erro ao detectar bordas: {str(e)}")
            return False

    def process_pipeline(self):
        if not self.load_image():
            return False

        if not self.create_output_dir():
            return False

        # Etapa 1: Binarização
        if not self.binarize_image():
            return False
        if not self.save_current_image(1):
            return False

        # Etapa 2: Erosão
        if not self.apply_erosion():
            return False
        if not self.save_current_image(2):
            return False

        # Etapa 3: Detecção de bordas
        if not self.detect_edges():
            return False
        if not self.save_current_image(3):
            return False

        return True


def main():
    print("Processador de Imagem 3D - Erosão em 3 Etapas")
    print("--------------------------------------------")

    if len(sys.argv) == 2:
        input_path = sys.argv[1]
    else:
        input_path = input("Digite o caminho para a imagem 3D (formato NIfTI .nii ou .nii.gz): ").strip('"')

    if not os.path.isfile(input_path):
        print(f"\nErro: Arquivo não encontrado - {input_path}")
        print("Verifique se:")
        print("1. O caminho está correto")
        print("2. O arquivo tem extensão .nii ou .nii.gz")
        input("Pressione Enter para sair...")
        sys.exit(1)

    processor = ImageProcessor3D(input_path)

    print("\nIniciando processamento de imagem 3D...")
    print("=====================================")

    if processor.process_pipeline():
        print("\nProcessamento concluído com sucesso!")
        print(f"Resultados salvos em: {os.path.abspath(processor.output_dir)}")
    else:
        print("\nO processamento falhou. Verifique as mensagens de erro acima.")

    input("Pressione Enter para sair...")


if __name__ == "__main__":
    main()
