# pip install torch torchvision pillow numpy

import argparse
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import os

cores_solo = {
    'área_urbana': (255, 0, 0),
    'solo_exposto': (139, 69, 19),
    'matas': (34, 139, 34),
    'pasto': (255, 255, 0)
}

def carregar_imagem(caminho_imagem):
    imagem = Image.open(caminho_imagem).convert('RGB')
    transformacao = transforms.Compose([
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return imagem, transformacao(imagem).unsqueeze(0)

def segmentar(modelo, imagem_tensor):
    modelo.eval()
    with torch.no_grad():
        saida = modelo(imagem_tensor)['out'][0]
    return torch.argmax(saida, dim=0).byte().cpu().numpy()

def mapear_classes(mascara):
    mascara_colorida = np.zeros((mascara.shape[0], mascara.shape[1], 3), dtype=np.uint8)
    for i in range(mascara.shape[0]):
        for j in range(mascara.shape[1]):
            classe = mascara[i, j]
            if classe in [0, 1, 2]:
                cor = cores_solo['solo_exposto']
            elif classe in [3, 4, 5]:
                cor = cores_solo['pasto']
            elif classe in [6, 7, 8, 9, 10]:
                cor = cores_solo['matas']
            elif classe in [11, 12, 13, 14, 15, 16, 17, 18]:
                cor = cores_solo['área_urbana']
            else:
                cor = (0, 0, 0)
            mascara_colorida[i, j] = cor
    return mascara_colorida

def salvar_imagem(mascara_colorida, caminho_original):
    nome = os.path.splitext(os.path.basename(caminho_original))[0]
    novo_nome = f"{nome}_processada.png"
    imagem = Image.fromarray(mascara_colorida)
    imagem.save(novo_nome)

def main():
    parser = argparse.ArgumentParser(description='Segmentação de tipos de solo em imagem de satélite.')
    parser.add_argument('caminho_imagem', type=str, help='Caminho para a imagem a ser processada')
    args = parser.parse_args()

    imagem_original, imagem_tensor = carregar_imagem(args.caminho_imagem)
    modelo = models.segmentation.deeplabv3_resnet101(pretrained=True)
    mascara = segmentar(modelo, imagem_tensor)
    mascara_colorida = mapear_classes(mascara)
    salvar_imagem(mascara_colorida, args.caminho_imagem)

if __name__ == '__main__':
    main()

