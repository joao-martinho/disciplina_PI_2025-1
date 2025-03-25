import cv2
import numpy as np
import matplotlib.pyplot as plt

# exercício 1

image = cv2.imread('Bola.jpg')

if image is None:
    print("Erro ao carregar a imagem!")
else:
    cv2.imshow('Imagem Colorida', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('imagem_colorida_salva.jpg', image)
    print("Imagem colorida salva com sucesso!")

# exercício 2

if image is None:
    print("Erro ao carregar a imagem!")
else:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Imagem Colorida', image)
    cv2.imshow('Imagem em Cinza', gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('imagem_cinza.jpg', gray_image)
    print("Imagem em tons de cinza salva com sucesso!")

# exercício 3

if image is None:
    print("Erro ao carregar a imagem!")
else:
    b, g, r = cv2.split(image)

    zeros = np.zeros(image.shape[:2], dtype="uint8")

    blue_channel = cv2.merge([b, zeros, zeros])
    green_channel = cv2.merge([zeros, g, zeros])
    red_channel = cv2.merge([zeros, zeros, r])

    cv2.imshow('Canal Vermelho (R)', red_channel)
    cv2.imshow('Canal Verde (G)', green_channel)
    cv2.imshow('Canal Azul (B)', blue_channel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('canal_vermelho.jpg', red_channel)
    cv2.imwrite('canal_verde.jpg', green_channel)
    cv2.imwrite('canal_azul.jpg', blue_channel)
    print("Canais RGB salvos com sucesso!")

# exercício 4

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if gray_image is None:
    print("Erro ao carregar a imagem!")
else:
    blurred = cv2.blur(gray_image, (5, 5))

    median = cv2.medianBlur(gray_image, 5)

    cv2.imshow('Imagem Original', gray_image)
    cv2.imshow('Filtro de Media (Blur)', blurred)
    cv2.imshow('Filtro de Mediana', median)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('filtro_media.jpg', blurred)
    cv2.imwrite('filtro_mediana.jpg', median)
    print("Imagens com filtros aplicados salvas com sucesso!")

# exercício 5

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if gray_image is None:
    print("Erro ao carregar a imagem!")
else:
    _, thresh1 = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    thresh2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

    cv2.imshow('Imagem Original', gray_image)
    cv2.imshow('Limiarizacao Simples', thresh1)
    cv2.imshow('Limiarizacao Adaptativa', thresh2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('limiarizacao_simples.jpg', thresh1)
    cv2.imwrite('limiarizacao_adaptativa.jpg', thresh2)
    print("Imagens com limiarização salvas com sucesso!")

# exercício 6

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if gray_image is None:
    print("Erro ao carregar a imagem!")
else:
    hist_original = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    equalized = cv2.equalizeHist(gray_image)

    hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0, 256])

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Imagem Original')

    plt.subplot(2, 2, 2)
    plt.plot(hist_original)
    plt.title('Histograma Original')

    plt.subplot(2, 2, 3)
    plt.imshow(equalized, cmap='gray')
    plt.title('Imagem Equalizada')

    plt.subplot(2, 2, 4)
    plt.plot(hist_equalized)
    plt.title('Histograma Equalizado')

    plt.tight_layout()
    plt.show()

    cv2.imwrite('imagem_equalizada.jpg', equalized)
    print("Imagem equalizada salva com sucesso!")

