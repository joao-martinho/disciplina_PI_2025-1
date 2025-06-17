# Importações necessárias
import os
import sys
from datetime import datetime
import cv2
import numpy as np

class FloodAndLandslideProcessor:
    """
    Classe para processamento de imagens de satélite para identificação de:
    - Áreas propensas a desmoronamento
    - Áreas afetadas por enchentes
    """

    def __init__(self):
        # Configurações unificadas de cores (BGR format)
        self.color_mappings = {
            'matas': {
                'color': (36, 120, 36),
                'description': 'Matas (Baixo risco)',
                'reference': np.array([36, 120, 36]),
                'threshold': 60  # Ajustado para maior precisão
            },
            'urbana': {
                'color': (200, 200, 200),
                'description': 'Area Urbana (Medio risco)',
                'reference': np.array([200, 200, 200]),
                'threshold': 70
            },
            'pastagem': {
                'color': (100, 255, 100),
                'description': 'Pastagem (Alto risco)',
                'reference': np.array([100, 255, 100]),
                'threshold': 65
            },
            'solo_exposto': {
                'color': (0, 140, 255),
                'description': 'Solo Exposto (Alto risco)',
                'reference': np.array([0, 140, 255]),
                'threshold': 65
            },
            'enchente': {
                'color': (139, 69, 19),
                'description': 'Area Alagada (Enchente)',
                'reference': np.array([139, 69, 19]),
                'threshold': 75
            }
        }
        self.alpha = 0.4

    def load_image(self, image_path):
        """Carrega a imagem e verifica se é válida"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Não foi possível ler a imagem: {image_path}")

        print(f"Imagem carregada: {image.shape[1]}x{image.shape[0]} pixels")
        return image

    def color_distance(self, pixel, reference_color):
        """Calcula a distância euclidiana entre cores usando vetorização"""
        return np.sqrt(np.sum((pixel.astype(np.float32) - reference_color.astype(np.float32)) ** 2))
        

    def create_masks(self, image):
        """Cria máscaras usando subtração de cores vetorizada"""
        height, width = image.shape[:2]
        
        # Converte imagem para float32 para maior precisão
        img_float = image.astype(np.float32)
        
        # Reshape a imagem para processar todos os pixels de uma vez
        pixels = img_float.reshape(-1, 3)
        
        # Prepara array para armazenar todas as distâncias de cores
        distances = np.zeros((len(self.color_mappings), pixels.shape[0]), dtype=np.float32)
        
        # Calcula distâncias para todas as cores de referência de uma vez
        for idx, (terrain, config) in enumerate(self.color_mappings.items()):
            ref_color = config['reference'].astype(np.float32)
            
            # Calcula distância euclidiana vetorizada
            diff = pixels - ref_color
            distances[idx] = np.sqrt(np.sum(diff * diff, axis=1))
        
        # Normaliza as distâncias
        max_dist = np.max(distances)
        distances = distances / max_dist if max_dist > 0 else distances
        
        # Encontra a classe mais próxima para cada pixel
        best_matches = np.argmin(distances, axis=0)
        
        # Cria máscaras para cada classe
        masks = {}
        for idx, (terrain, config) in enumerate(self.color_mappings.items()):
            # Cria máscara binária
            mask = (best_matches == idx).reshape(height, width).astype(np.uint8) * 255
            
            # Aplica threshold baseado na distância
            dist_mask = distances[idx].reshape(height, width)
            threshold_mask = (dist_mask < (config['threshold'] / 255.0)).astype(np.uint8) * 255
            
            # Combina as máscaras
            final_mask = cv2.bitwise_and(mask, threshold_mask)
            
            # Limpa ruído
            kernel = np.ones((3,3), np.uint8)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
            
            masks[terrain] = final_mask
        
        return masks

    def calculate_risk_areas(self, masks, image_size):
        """Calcula a porcentagem de cada tipo de área na imagem"""
        stats = {}
        total_pixels = image_size[0] * image_size[1]

        # Garante que todos os tipos de terreno estejam nas estatísticas
        all_terrain_types = ['matas', 'urbana', 'pastagem', 'solo_exposto', 'enchente']

        for terrain_type in all_terrain_types:
            if terrain_type in masks:
                area_pixels = cv2.countNonZero(masks[terrain_type])
            else:
                area_pixels = 0

            stats[terrain_type] = {
                'pixels': area_pixels,
                'percentage': (area_pixels / total_pixels) * 100
            }

        return stats

    def apply_masks(self, image, masks):
        """Aplica as máscaras na imagem original com transparência"""
        processed_image = image.copy().astype(np.float32)

        for terrain_type, mask in masks.items():
            if cv2.countNonZero(mask) > 0:
                color = np.array(self.color_mappings[terrain_type]['color'], dtype=np.float32)
                mask_3d = np.stack([mask] * 3, axis=2) / 255.0
                overlay = color.reshape(1, 1, 3) * mask_3d
                processed_image = processed_image * (1 - self.alpha * mask_3d) + overlay * self.alpha

        return processed_image.astype(np.uint8)

    def add_legend_and_stats(self, image, stats):
        """Adiciona legenda e estatísticas à imagem processada"""
        height, width = image.shape[:2]

        # Identifica alertas
        alerts = []
        if stats.get('enchente', {}).get('percentage', 0) > 3:
            alerts.append("ALERTA: Areas de enchente detectadas")
        if stats.get('solo_exposto', {}).get('percentage', 0) > 8:
            alerts.append("ALERTA: Muito solo exposto risco de erosao")

        # Calcula dimensões da legenda
        items_count = 5  # Total fixo de itens na legenda
        base_legend_height = 50 + (items_count * 25)  # 50 para título e margens + 25 por item
        alert_height = len(alerts) * 30  # 30 pixels por alerta
        total_height = base_legend_height + alert_height
        legend_width = 450

        # Ajusta para imagens pequenas
        if height < total_height + 20:
            scale_factor = (height - 20) / total_height
            base_legend_height = int(base_legend_height * scale_factor)
            alert_height = int(alert_height * scale_factor)
            total_height = base_legend_height + alert_height
        
        if width < legend_width + 20:
            legend_width = width - 20

        # Cria retângulo de fundo único para legenda e alertas
        cv2.rectangle(image, (10, height - total_height - 10),
                      (legend_width, height - 10), (255, 255, 255), -1)
        cv2.rectangle(image, (10, height - total_height - 10),
                      (legend_width, height - 10), (0, 0, 0), 2)

        # Adiciona título
        cv2.putText(image, "ANALISE DE RISCO",
                    (20, height - total_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Adiciona itens da legenda (todos os 5 tipos)
        y_offset = height - total_height + 55
        legend_order = ['matas', 'urbana', 'pastagem', 'solo_exposto', 'enchente']

        for terrain_type in legend_order:
            color = self.color_mappings[terrain_type]['color']
            desc = self.color_mappings[terrain_type]['description']
            percentage = stats.get(terrain_type, {}).get('percentage', 0)

            # Adiciona quadrado colorido
            cv2.rectangle(image, (20, y_offset - 10), (40, y_offset + 5), color, -1)
            cv2.rectangle(image, (20, y_offset - 10), (40, y_offset + 5), (0, 0, 0), 1)

            # Adiciona texto
            stats_text = f"{desc}: {percentage:.2f}%"
            cv2.putText(image, stats_text, (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            y_offset += 25

        # Adiciona alertas abaixo da legenda
        if alerts:
            alert_y = height - total_height + base_legend_height + 10
            for alert in alerts:
                cv2.putText(image, alert,
                            (20, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                alert_y += 30

    def save_image(self, image, original_path):
        """Salva a imagem processada em alta qualidade"""
        try:
            filename, ext = os.path.splitext(original_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{filename}_processada_{timestamp}.png"
            
            # Salva com máxima qualidade
            success = cv2.imwrite(output_path, image, [
                cv2.IMWRITE_PNG_COMPRESSION, 0,  # Sem compressão PNG
                cv2.IMWRITE_JPEG_QUALITY, 100    # Máxima qualidade JPEG
            ])
            
            if not success:
                raise ValueError("Falha ao salvar a imagem")
            
            return output_path
        except Exception as e:
            print(f"Erro ao salvar imagem: {e}")
            fallback_path = f"processada_{timestamp}.png"
            cv2.imwrite(fallback_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            return fallback_path

    def process_image(self, image_path):
        """Processa uma imagem mantendo alta resolução"""
        try:
            # Carrega a imagem em alta resolução
            image = self.load_image(image_path)
            print("✔ Imagem carregada")
            
            # Cria cópia para processamento
            processing_image = image.copy()
            
            # Melhoria na qualidade sem perda de resolução
            processing_image = cv2.bilateralFilter(processing_image, 9, 75, 75)
            
            # Ajuste de cor e contraste preservando detalhes
            lab = cv2.cvtColor(processing_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge((l,a,b))
            processing_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Processamento mantendo resolução original
            masks = self.create_masks(processing_image)
            print("✔ Máscaras criadas")
            
            # Debug masks em alta resolução
            debug_dir = os.path.join(os.path.dirname(image_path), "debug_masks")
            os.makedirs(debug_dir, exist_ok=True)
            
            for terrain, mask in masks.items():
                debug_path = os.path.join(debug_dir, f"mask_{terrain}.png")
                cv2.imwrite(debug_path, mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            # Calcula estatísticas usando máscara em resolução total
            stats = self.calculate_risk_areas(masks, image.shape[:2])
            
            # Aplica máscaras na imagem original
            processed_image = self.apply_masks(image, masks)
            
            # Adiciona legenda mantendo qualidade
            self.add_legend_and_stats(processed_image, stats)
            
            # Salva em alta qualidade
            output_path = self.save_image(processed_image, image_path)
            print(f"✔ Imagem salva em alta resolução: {output_path}")
            
            return output_path, stats

        except Exception as e:
            print(f"✖ Erro: {str(e)}")
            raise

def get_user_input():
    """Obtém entrada do usuário de forma interativa"""
    print("\n" + "=" * 60)
    print("ANÁLISE DE RISCO DE DESMORONAMENTO E DETECÇÃO DE ENCHENTES")
    print("=" * 60 + "\n")
    print("CONFIGURAÇÕES DE CORES:")
    print("• Verde escuro → Matas (baixo risco)")
    print("• Verde claro → Pastagem (alto risco)")
    print("• Cinza/Branco → Área urbana (médio risco)")
    print("• Laranja/Bege → Solo exposto (alto risco)")
    print("• Marrom/Azul/Preto → Água/Enchente")
    print("")

    while True:
        print("1. Processar uma imagem")
        print("2. Processar todas as imagens em um diretório")
        print("3. Sair")

        choice = input("\nEscolha uma opção (1-3): ").strip()

        if choice == "1":
            path = input("\nDigite o caminho da imagem: ").strip().replace('"', '')
            if os.path.isfile(path):
                return [path]
            print(f"\n✖ Arquivo não encontrado: {path}")

        elif choice == "2":
            dir_path = input("\nDigite o caminho do diretório: ").strip().replace('"', '')
            if os.path.isdir(dir_path):
                valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
                images = []
                for f in os.listdir(dir_path):
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        images.append(os.path.join(dir_path, f))

                if images:
                    print(f"Encontradas {len(images)} imagens para processar.")
                    return images
                print("\n✖ Nenhuma imagem encontrada no diretório.")
            else:
                print(f"\n✖ Diretório não encontrado: {dir_path}")

        elif choice == "3":
            print("\nEncerrando...")
            sys.exit(0)

        else:
            print("\n✖ Opção inválida. Tente novamente.")

def main():
    """Função principal que orquestra todo o processamento"""
    try:
        print("Inicializando processador de imagens...")
        processor = FloodAndLandslideProcessor()

        image_paths = get_user_input()

        if not image_paths:
            print("Nenhuma imagem para processar.")
            return

        print(f"\nIniciando processamento de {len(image_paths)} imagem(ns)...")
        successful = 0

        for i, path in enumerate(image_paths, 1):
            try:
                print(f"\n[{i}/{len(image_paths)}] Processando: {os.path.basename(path)}")
                output_path, stats = processor.process_image(path)

                print("\nEstatísticas da imagem:")
                for area, data in stats.items():
                    desc = processor.color_mappings[area]['description']
                    print(f"- {desc}: {data['percentage']:.2f}%")

                successful += 1

            except Exception as e:
                print(f"\n✖ Falha ao processar {os.path.basename(path)}: {str(e)}")
                continue

        print(f"\nProcessamento concluído!")
        print(f"✔ {successful}/{len(image_paths)} imagens processadas com sucesso")

    except KeyboardInterrupt:
        print("\nProcessamento interrompido pelo usuário.")
    except Exception as e:
        print(f"\nErro inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        input("\nPressione Enter para sair...")

if __name__ == "__main__":
    main()
