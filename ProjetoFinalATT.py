import cv2
import numpy as np
import os
import sys
from datetime import datetime

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
            'urbana': {'color': (255, 0, 0), 'description': 'Área Urbana (Médio risco)'},
            'pastagem': {'color': (0, 255, 255), 'description': 'Pastagem (Alto risco)'},
            'solo_exposto': {'color': (0, 0, 255), 'description': 'Solo Exposto (Alto risco)'},
            'enchente': {'color': (128, 0, 128), 'description': 'Área Alagada (Enchente)'}
        }
        
        # Intervalos de cores no espaço HSV ajustados para melhor detecção
        self.color_ranges = {
            'matas': {'lower': np.array([35, 40, 40]), 'upper': np.array([85, 255, 255])},
            'urbana': {'lower': np.array([0, 0, 100]), 'upper': np.array([180, 50, 255])},  # Cores neutras/cinzas
            'solo_exposto': {'lower': np.array([10, 50, 50]), 'upper': np.array([25, 255, 200])},  # Marrom/bege
            'pastagem': {'lower': np.array([25, 30, 30]), 'upper': np.array([35, 255, 200])},  # Verde amarelado
            'enchente': {'lower': np.array([100, 50, 20]), 'upper': np.array([130, 255, 150])}  # Azul escuro para água
        }
        
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
        """
        Melhora a detecção de água aplicando:
        1. Limiarização na banda de valor (V) - água tem baixa luminosidade
        2. Detecção de saturação baixa (água pode aparecer desaturada)
        3. Operações morfológicas para limpeza
        """
        # Canal Value (V) do HSV - água geralmente tem valores baixos
        v_channel = hsv_image[:,:,2]
        s_channel = hsv_image[:,:,1]
        
        # Máscara para pixels escuros (possível água)
        _, dark_mask = cv2.threshold(v_channel, 80, 255, cv2.THRESH_BINARY_INV)
        
        # Máscara para pixels com baixa saturação (água pode ser cinza)
        _, low_sat_mask = cv2.threshold(s_channel, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Combina as duas condições
        water_mask = cv2.bitwise_or(dark_mask, low_sat_mask)
        
        # Operações morfológicas para limpar a máscara
        kernel = np.ones((3,3), np.uint8)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
        
        # Remove ruídos pequenos
        kernel_big = np.ones((5,5), np.uint8)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel_big)
        
        return water_mask

    def create_masks(self, hsv_image):
        """Cria máscaras para todos os tipos de áreas"""
        masks = {}
        
        # Detecção padrão por cor
        for terrain_type in self.color_ranges:
            lower = self.color_ranges[terrain_type]['lower']
            upper = self.color_ranges[terrain_type]['upper']
            masks[terrain_type] = cv2.inRange(hsv_image, lower, upper)
        
        # Detecção aprimorada de água (CORRIGIDO: usa OR em vez de AND)
        enhanced_water = self.enhance_water_detection(hsv_image)
        masks['enchente'] = cv2.bitwise_or(masks['enchente'], enhanced_water)
        
        # Aplica filtro de ruído em todas as máscaras
        kernel = np.ones((2,2), np.uint8)
        for terrain_type in masks:
            masks[terrain_type] = cv2.morphologyEx(masks[terrain_type], cv2.MORPH_CLOSE, kernel)
        
        return masks

    def calculate_risk_areas(self, masks, image_size):
        """Calcula a porcentagem de cada tipo de área na imagem"""
        stats = {}
        total_pixels = image_size[0] * image_size[1]
        
        for name, mask in masks.items():
            area_pixels = cv2.countNonZero(mask)
            stats[name] = {
                'pixels': area_pixels,
                'percentage': (area_pixels / total_pixels) * 100
            }
        
        return stats

    def apply_masks(self, image, masks):
        """Aplica as máscaras na imagem original (CORRIGIDO)"""
        processed_image = image.copy().astype(np.float32)  # Converte para float para evitar overflow
        
        for terrain_type, mask in masks.items():
            if cv2.countNonZero(mask) > 0:  # Só aplica se houver pixels na máscara
                color = np.array(self.color_mappings[terrain_type]['color'], dtype=np.float32)
                
                # Aplica a cor com transparência
                for i in range(3):  # Para cada canal de cor (B, G, R)
                    processed_image[mask > 0, i] = (
                        self.alpha * color[i] + (1 - self.alpha) * processed_image[mask > 0, i]
                    )
        
        return processed_image.astype(np.uint8)  # Converte de volta para uint8

    def add_legend_and_stats(self, image, stats):
        """Adiciona legenda e estatísticas à imagem"""
        height, width = image.shape[:2]
        
        # Calcula altura necessária para a legenda
        legend_height = 220
        legend_width = 450
        
        # Garante que a legenda caiba na imagem
        if height < legend_height + 20:
            legend_height = height - 20
        if width < legend_width + 20:
            legend_width = width - 20
        
        # Fundo da legenda com bordas
        cv2.rectangle(image, (10, height-legend_height), 
                     (legend_width, height-10), (255, 255, 255), -1)
        cv2.rectangle(image, (10, height-legend_height), 
                     (legend_width, height-10), (0, 0, 0), 2)
        
        # Título
        cv2.putText(image, "ANALISE DE RISCO - DESMORONAMENTO/ENCHENTE", 
                   (20, height-legend_height+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Itens da legenda
        y_offset = height-legend_height+55
        for terrain_type in self.color_mappings:
            if y_offset > height - 30:  # Para não sair da imagem
                break
                
            color = self.color_mappings[terrain_type]['color']
            desc = self.color_mappings[terrain_type]['description']
            
            # Marcador de cor
            cv2.rectangle(image, (20, y_offset-10), (40, y_offset+5), color, -1)
            cv2.rectangle(image, (20, y_offset-10), (40, y_offset+5), (0, 0, 0), 1)
            
            # Texto e estatísticas
            percentage = stats.get(terrain_type, {}).get('percentage', 0)
            stats_text = f"{desc}: {percentage:.2f}%"
            cv2.putText(image, stats_text, (50, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            y_offset += 25
        
        # Adiciona avisos se áreas de risco forem significativas
        warning_y = height - 40
        if stats.get('enchente', {}).get('percentage', 0) > 3:
            cv2.putText(image, "ALERTA: Areas de enchente detectadas!", 
                       (20, warning_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            warning_y -= 20
        
        if stats.get('solo_exposto', {}).get('percentage', 0) > 8:
            cv2.putText(image, "ALERTA: Muito solo exposto - risco de erosao!", 
                       (20, warning_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def save_image(self, image, original_path):
        """Salva a imagem processada com timestamp"""
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
            # Tenta salvar em diretório atual como fallback
            fallback_path = f"processada_{timestamp}.png"
            cv2.imwrite(fallback_path, image)
            return fallback_path

    def process_image(self, image_path):
        """Processa uma imagem completa"""
        print(f"\nProcessando: {os.path.basename(image_path)}")
        
        try:
            # 1. Carrega a imagem
            image = self.load_image(image_path)
            print("✔ Imagem carregada")
            
            # 2. Redimensiona se muito grande (para performance)
            height, width = image.shape[:2]
            if width > 2000 or height > 2000:
                scale = min(2000/width, 2000/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
                print(f"✔ Imagem redimensionada para {new_width}x{new_height}")
            
            # 3. Converte para HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            print("✔ Conversão HSV concluída")
            
            # 4. Aplica filtro de ruído
            image_filtered = cv2.bilateralFilter(image, 9, 75, 75)
            hsv = cv2.cvtColor(image_filtered, cv2.COLOR_BGR2HSV)
            
            # 5. Cria máscaras
            masks = self.create_masks(hsv)
            print("✔ Máscaras criadas")
            
            # 6. Calcula estatísticas
            stats = self.calculate_risk_areas(masks, image.shape[:2])
            
            # 7. Aplica máscaras
            processed_image = self.apply_masks(image, masks)
            print("✔ Máscaras aplicadas")
            
            # 8. Adiciona legenda
            self.add_legend_and_stats(processed_image, stats)
            print("✔ Legenda e estatísticas adicionadas")
            
            # 9. Salva resultado
            output_path = self.save_image(processed_image, image_path)
            print(f"✔ Imagem salva em: {output_path}")
            
            return output_path, stats
            
        except Exception as e:
            print(f"✖ Erro: {str(e)}")
            raise

def get_user_input():
    """Obtém entrada do usuário de forma interativa"""
    print("\n" + "="*60)
    print("ANÁLISE DE RISCO DE DESMORONAMENTO E DETECÇÃO DE ENCHENTES")
    print("="*60 + "\n")
    
    while True:
        print("1. Processar uma imagem")
        print("2. Processar todas as imagens em um diretório")
        print("3. Sair")
        
        choice = input("\nEscolha uma opção (1-3): ").strip()
        
        if choice == "1":
            path = input("\nDigite o caminho da imagem: ").strip().replace('"', '')  # Remove aspas
            if os.path.isfile(path):
                return [path]
            print(f"\n✖ Arquivo não encontrado: {path}")
            print("Certifique-se de que o caminho está correto.")
            
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
                print("Formatos suportados: JPG, JPEG, PNG, BMP, TIFF")
            else:
                print(f"\n✖ Diretório não encontrado: {dir_path}")
                
        elif choice == "3":
            print("\nEncerrando...")
            sys.exit(0)
            
        else:
            print("\n✖ Opção inválida. Tente novamente.")

def main():
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
                
                # Exibe estatísticas no console
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
