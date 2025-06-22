import os
import sys
from datetime import datetime
import cv2
import numpy as np

class ProcessadorRiscoTerreno:
    def __init__(self):
        self.mapeamento_cores = {
            'matas': {'cor': (0, 255, 0), 'descricao': 'Matas (Baixo risco)'},
            'urbana': {'cor': (255, 0, 0), 'descricao': 'Area Urbana (Medio risco)'},
            'pastagem': {'cor': (0, 255, 255), 'descricao': 'Pastagem (Alto risco)'},
            'solo_exposto': {'cor': (0, 0, 255), 'descricao': 'Solo Exposto (Alto risco)'}
        }

        self.faixas_cores = {
            'matas': {'minimo': np.array([35, 80, 20]), 'maximo': np.array([85, 255, 120])},
            'urbana': {'minimo': np.array([0, 0, 120]), 'maximo': np.array([180, 60, 255])},
            'solo_exposto': {'minimo': np.array([8, 40, 80]), 'maximo': np.array([25, 180, 220])},
            'pastagem': {'minimo': np.array([35, 20, 80]), 'maximo': np.array([85, 120, 180])}
        }

        self.transparencia = 0.4

    def calcular_risco_enchentes(self, estatisticas):
        percentual_urbano = estatisticas.get('urbana', {}).get('porcentagem', 0)
        percentual_solo = estatisticas.get('solo_exposto', {}).get('porcentagem', 0)
        percentual_matas = estatisticas.get('matas', {}).get('porcentagem', 0)
        percentual_pastagem = estatisticas.get('pastagem', {}).get('porcentagem', 0)

        pontuacao = (percentual_urbano * 0.7) + (percentual_solo * 0.9) + (percentual_pastagem * 0.4) - (percentual_matas * 0.6)
        pontuacao = max(0, min(100, pontuacao))

        if pontuacao > 70:
            return "Alta probabilidade de enchentes"
        elif pontuacao > 40:
            return "Média probabilidade de enchentes"
        elif pontuacao > 15:
            return "Baixa probabilidade de enchentes"
        else:
            return "Muito baixa probabilidade de enchentes"

    def gerar_relatorio_texto(self, estatisticas, nome_imagem, dimensoes):
        risco_enchentes = self.calcular_risco_enchentes(estatisticas)
        alertas = []

        if estatisticas.get('solo_exposto', {}).get('porcentagem', 0) > 8:
            alertas.append("⚠ Solo exposto acima do limite seguro (8%)")

        relatorio = f"""
{'='*60}
RELATÓRIO DE ANÁLISE - {nome_imagem}
{'='*60}
Dimensões: {dimensoes[1]}x{dimensoes[0]} pixels

ÁREAS DETECTADAS:"""

        for area in ['matas', 'urbana', 'pastagem', 'solo_exposto']:
            dados = estatisticas.get(area, {})
            relatorio += f"\n- {self.mapeamento_cores[area]['descricao']}: {dados.get('porcentagem', 0):.2f}%"

        relatorio += "\n\nALERTAS:"
        if alertas:
            for alerta in alertas:
                relatorio += f"\n{alerta}"
        else:
            relatorio += "\nNenhum alerta crítico detectado"

        relatorio += f"\n\nRISCO DE ENCHENTES: {risco_enchentes}"
        relatorio += f"\n{'='*60}\n"

        return relatorio

    def carregar_imagem(self, caminho):
        if not os.path.exists(caminho):
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

        imagem = cv2.imread(caminho)
        if imagem is None:
            raise ValueError(f"Não foi possível ler a imagem: {caminho}")

        print(f"Imagem carregada: {imagem.shape[1]}x{imagem.shape[0]} pixels")
        return imagem

    def criar_mascaras(self, imagem_hsv):
        mascaras = {}

        for tipo_terreno in self.faixas_cores:
            minimo = self.faixas_cores[tipo_terreno]['minimo']
            maximo = self.faixas_cores[tipo_terreno]['maximo']
            mascaras[tipo_terreno] = cv2.inRange(imagem_hsv, minimo, maximo)

        kernel = np.ones((2, 2), np.uint8)
        for tipo_terreno in mascaras:
            mascaras[tipo_terreno] = cv2.morphologyEx(mascaras[tipo_terreno], cv2.MORPH_CLOSE, kernel)

        ordem_prioridade = ['matas', 'urbana', 'solo_exposto', 'pastagem']

        for i, tipo_terreno in enumerate(ordem_prioridade):
            if tipo_terreno in mascaras:
                for maior_prioridade in ordem_prioridade[:i]:
                    if maior_prioridade in mascaras:
                        mascaras[tipo_terreno] = cv2.bitwise_and(mascaras[tipo_terreno],
                                                          cv2.bitwise_not(mascaras[maior_prioridade]))

        return mascaras

    def calcular_areas_risco(self, mascaras, dimensoes):
        estatisticas = {}
        total_pixels = dimensoes[0] * dimensoes[1]

        tipos_terreno = ['matas', 'urbana', 'pastagem', 'solo_exposto']

        for tipo_terreno in tipos_terreno:
            if tipo_terreno in mascaras:
                pixels_area = cv2.countNonZero(mascaras[tipo_terreno])
            else:
                pixels_area = 0

            estatisticas[tipo_terreno] = {
                'pixels': pixels_area,
                'porcentagem': (pixels_area / total_pixels) * 100
            }

        return estatisticas

    def aplicar_mascaras(self, imagem, mascaras):
        imagem_processada = imagem.copy().astype(np.float32)

        for tipo_terreno, mascara in mascaras.items():
            if cv2.countNonZero(mascara) > 0:
                cor = np.array(self.mapeamento_cores[tipo_terreno]['cor'], dtype=np.float32)

                for i in range(3):
                    imagem_processada[mascara > 0, i] = (
                            self.transparencia * cor[i] + (1 - self.transparencia) * imagem_processada[mascara > 0, i]
                    )

        return imagem_processada.astype(np.uint8)

    def salvar_imagem(self, imagem, caminho_original):
        try:
            nome_arquivo, ext = os.path.splitext(caminho_original)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            caminho_saida = f"{nome_arquivo}_processada_{timestamp}.png"

            sucesso = cv2.imwrite(caminho_saida, imagem)
            if not sucesso:
                raise ValueError("Falha ao salvar a imagem")

            return caminho_saida
        except Exception as e:
            print(f"Erro ao salvar imagem: {e}")
            caminho_reserva = f"processada_{timestamp}.png"
            cv2.imwrite(caminho_reserva, imagem)
            return caminho_reserva

    def processar_imagem(self, caminho_imagem):
        print(f"\nProcessando: {os.path.basename(caminho_imagem)}")

        try:
            imagem = self.carregar_imagem(caminho_imagem)
            print("✔ Imagem carregada")

            altura, largura = imagem.shape[:2]
            if largura > 2000 or altura > 2000:
                escala = min(2000 / largura, 2000 / altura)
                nova_largura = int(largura * escala)
                nova_altura = int(altura * escala)
                imagem = cv2.resize(imagem, (nova_largura, nova_altura))
                print(f"✔ Imagem redimensionada para {nova_largura}x{nova_altura}")

            imagem_filtrada = cv2.bilateralFilter(imagem, 9, 75, 75)
            hsv = cv2.cvtColor(imagem_filtrada, cv2.COLOR_BGR2HSV)
            print("✔ Conversão HSV concluída")

            mascaras = self.criar_mascaras(hsv)
            print("✔ Máscaras criadas")

            estatisticas = self.calcular_areas_risco(mascaras, imagem.shape[:2])

            imagem_processada = self.aplicar_mascaras(imagem, mascaras)
            print("✔ Máscaras aplicadas")

            relatorio = self.gerar_relatorio_texto(estatisticas, os.path.basename(caminho_imagem), imagem.shape[:2])
            print(relatorio)

            caminho_saida = self.salvar_imagem(imagem_processada, caminho_imagem)
            print(f"✔ Imagem salva em: {caminho_saida}")

            return caminho_saida, estatisticas

        except Exception as e:
            print(f"✖ Erro: {str(e)}")
            raise

def obter_entrada_usuario():
    print("\n" + "=" * 60)
    print("ANÁLISE DE RISCO DE DESMORONAMENTO E ENCHENTES")
    print("=" * 60 + "\n")
    print("CONFIGURAÇÕES DE CORES:")
    print("• Verde escuro → Matas (baixo risco)")
    print("• Verde claro → Pastagem (alto risco)")
    print("• Cinza/Branco → Área urbana (médio risco)")
    print("• Laranja/Bege → Solo exposto (alto risco)")
    print("")

    while True:
        print("1. Processar uma imagem")
        print("2. Processar todas as imagens em um diretório")
        print("3. Sair")

        opcao = input("\nEscolha uma opção (1-3): ").strip()

        if opcao == "1":
            caminho = input("\nDigite o caminho da imagem: ").strip().replace('"', '')
            if os.path.isfile(caminho):
                return [caminho]
            print(f"\n✖ Arquivo não encontrado: {caminho}")

        elif opcao == "2":
            caminho_dir = input("\nDigite o caminho do diretório: ").strip().replace('"', '')
            if os.path.isdir(caminho_dir):
                extensoes_validas = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
                imagens = []
                for arquivo in os.listdir(caminho_dir):
                    if os.path.splitext(arquivo)[1].lower() in extensoes_validas:
                        imagens.append(os.path.join(caminho_dir, arquivo))

                if imagens:
                    print(f"Encontradas {len(imagens)} imagens para processar.")
                    return imagens
                print("\n✖ Nenhuma imagem encontrada no diretório.")
            else:
                print(f"\n✖ Diretório não encontrado: {caminho_dir}")

        elif opcao == "3":
            print("\nEncerrando...")
            sys.exit(0)

        else:
            print("\n✖ Opção inválida. Tente novamente.")

def principal():
    try:
        print("Inicializando processador de imagens...")
        processador = ProcessadorRiscoTerreno()

        caminhos_imagens = obter_entrada_usuario()

        if not caminhos_imagens:
            print("Nenhuma imagem para processar.")
            return

        print(f"\nIniciando processamento de {len(caminhos_imagens)} imagem(ns)...")
        sucesso = 0

        for i, caminho in enumerate(caminhos_imagens, 1):
            try:
                print(f"\n[{i}/{len(caminhos_imagens)}] Processando: {os.path.basename(caminho)}")
                caminho_saida, estatisticas = processador.processar_imagem(caminho)
                sucesso += 1

            except Exception as e:
                print(f"\n✖ Falha ao processar {os.path.basename(caminho)}: {str(e)}")
                continue

        print(f"\nProcessamento concluído!")
        print(f"✔ {sucesso}/{len(caminhos_imagens)} imagens processadas com sucesso")

    except KeyboardInterrupt:
        print("\nProcessamento interrompido pelo usuário.")
    except Exception as e:
        print(f"\nErro inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        input("\nPressione Enter para sair...")

if __name__ == "__main__":
    principal()
