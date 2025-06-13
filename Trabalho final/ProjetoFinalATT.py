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
        # Configurações de cores e descrições para cada tipo de área
        self.color_mappings = {
            'matas': {'color': (0, 255, 0), 'description': 'Matas (Baixo risco)'},
            'urbana': {'color': (255, 0, 0), 'description': 'Area Urbana (Medio risco)'},
            'pastagem': {'color': (0, 255, 255), 'description': 'Pastagem (Alto risco)'},
            'solo_exposto': {'color': (0, 0, 255), 'description': 'Solo Exposto (Alto risco)'},
            'enchente': {'color': (128, 0, 128), 'description': 'Area Alagada (Enchente)'}
        }

        # Intervalos de cores no espaço HSV
        self.color_ranges = {
            'matas': {'lower': np.array([35, 80, 20]), 'upper': np.array([85, 255, 120])},
            'urbana': {'lower': np.array([0, 0, 120]), 'upper': np.array([180, 60, 255])},
            'solo_exposto': {'lower': np.array([8, 40, 80]), 'upper': np.array([25, 180, 220])},
            'pastagem': {'lower': np.array([35, 20, 80]), 'upper': np.array([85, 120, 180])},
            'enchente': {'lower': np.array([0, 30, 0]), 'upper': np.array([30, 255, 80])}
        }

        # Intervalos adicionais para água (azul)
        self.water_blue_range = {'lower': np.array([80, 40, 20]), 'upper': np.array([140, 255, 220])}

        self.alpha = 0.4  # Transparência das áreas destacadas

    def load_image(self, image_path):
        """Carrega a imagem e verifica se é válida"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Não foi possível ler a imagem: {image_path}")

        print(f"Imagem carregada: {image.shape[1]}x{image.shape[0]} pixels")
        return image

    def enhance_water_detection(self, hsv_image):
        """Melhora a detecção de água aplicando múltiplas técnicas"""
        # Canal Value (V) do HSV - detecta pixels muito escuros
        v_channel = hsv_image[:, :, 2]
        _, black_mask = cv2.threshold(v_channel, 40, 255, cv2.THRESH_BINARY_INV)

        # Máscara para tons de azul (água)
        blue_mask = cv2.inRange(hsv_image, self.water_blue_range['lower'], self.water_blue_range['upper'])

        # Máscara para tons de marrom (água/lama)
        brown_mask = cv2.inRange(hsv_image, self.color_ranges['enchente']['lower'],
                                 self.color_ranges['enchente']['upper'])

        # Combina todas as máscaras de água
        combined_water_mask = cv2.bitwise_or(black_mask, blue_mask)
        combined_water_mask = cv2.bitwise_or(combined_water_mask, brown_mask)

        # Operações morfológicas para limpar a máscara
        kernel = np.ones((3, 3), np.uint8)
        combined_water_mask = cv2.morphologyEx(combined_water_mask, cv2.MORPH_CLOSE, kernel)
        combined_water_mask = cv2.morphologyEx(combined_water_mask, cv2.MORPH_OPEN, kernel)

        # Remove ruídos pequenos
        kernel_big = np.ones((5, 5), np.uint8)
        combined_water_mask = cv2.morphologyEx(combined_water_mask, cv2.MORPH_OPEN, kernel_big)

        return combined_water_mask

    def create_masks(self, hsv_image):
        """Cria máscaras binárias para todos os tipos de áreas definidas"""
        masks = {}

        # Cria máscaras para todos os tipos de terreno
        for terrain_type in self.color_ranges:
            lower = self.color_ranges[terrain_type]['lower']
            upper = self.color_ranges[terrain_type]['upper']
            masks[terrain_type] = cv2.inRange(hsv_image, lower, upper)

        # Aplica detecção aprimorada de água
        enhanced_water = self.enhance_water_detection(hsv_image)
        masks['enchente'] = cv2.bitwise_or(masks['enchente'], enhanced_water)

        # Filtro de ruído
        kernel = np.ones((2, 2), np.uint8)
        for terrain_type in masks:
            masks[terrain_type] = cv2.morphologyEx(masks[terrain_type], cv2.MORPH_CLOSE, kernel)

        # Prioridade: enchente > matas > urbana > solo_exposto > pastagem
        priority_order = ['enchente', 'matas', 'urbana', 'solo_exposto', 'pastagem']

        for i, terrain_type in enumerate(priority_order):
            if terrain_type in masks:
                for higher_priority in priority_order[:i]:
                    if higher_priority in masks:
                        masks[terrain_type] = cv2.bitwise_and(masks[terrain_type],
                                                          cv2.bitwise_not(masks[higher_priority]))

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

                for i in range(3):
                    processed_image[mask > 0, i] = (
                            self.alpha * color[i] + (1 - self.alpha) * processed_image[mask > 0, i]
                    )

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
        """Salva a imagem processada com timestamp no nome"""
        try:
            filename, ext = os.path.splitext(original_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{filename}_processada_{timestamp}.png"

            success = cv2.imwrite(output_path, image)
            if not success:
                raise ValueError("Falha ao salvar a imagem")

            return output_path
        except Exception as e:
            print(f"Erro ao salvar imagem: {e}")
            fallback_path = f"processada_{timestamp}.png"
            cv2.imwrite(fallback_path, image)
            return fallback_path

    def process_image(self, image_path):
        """Processa uma imagem completa, executando todas as etapas"""
        print(f"\nProcessando: {os.path.basename(image_path)}")

        try:
            # 1. Carrega a imagem
            image = self.load_image(image_path)
            print("✔ Imagem carregada")

            # 2. Redimensiona se muito grande
            height, width = image.shape[:2]
            if width > 2000 or height > 2000:
                scale = min(2000 / width, 2000 / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
                print(f"✔ Imagem redimensionada para {new_width}x{new_height}")

            # 3. Converte para HSV e aplica filtro
            image_filtered = cv2.bilateralFilter(image, 9, 75, 75)
            hsv = cv2.cvtColor(image_filtered, cv2.COLOR_BGR2HSV)
            print("✔ Conversão HSV concluída")

            # 4. Cria máscaras
            masks = self.create_masks(hsv)
            print("✔ Máscaras criadas")

            # 5. Calcula estatísticas
            stats = self.calculate_risk_areas(masks, image.shape[:2])

            # 6. Aplica máscaras
            processed_image = self.apply_masks(image, masks)
            print("✔ Máscaras aplicadas")

            # 7. Adiciona legenda
            self.add_legend_and_stats(processed_image, stats)
            print("✔ Legenda e estatísticas adicionadas")

            # 8. Salva a imagem
            output_path = self.save_image(processed_image, image_path)
            print(f"✔ Imagem salva em: {output_path}")

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
