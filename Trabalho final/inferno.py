# Importações necessárias
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Tentar importar segmentation_models_pytorch
try:
    import segmentation_models_pytorch as smp

    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("⚠️  segmentation_models_pytorch não encontrado. Usando modelo alternativo.")


class FloodAndLandslideAIProcessor:
    """
    Classe para processamento de imagens de satélite usando modelo de deep learning
    para identificação de:
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

        # Mapeamento de classes do modelo para nossas categorias
        self.class_mapping = {
            'vegetation': 'matas',
            'urban': 'urbana',
            'grassland': 'pastagem',
            'bare_soil': 'solo_exposto',
            'water': 'enchente'
        }

        self.alpha = 0.4  # Transparência das áreas destacadas
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None

        print(f"Usando dispositivo: {self.device}")
        self.load_model()

    def load_model(self):
        """Carrega o modelo pré-treinado de segmentação"""
        try:
            if SMP_AVAILABLE:
                self.load_smp_model()
            else:
                self.load_alternative_model()
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            print("Usando método de fallback...")
            self.load_fallback_model()

    def load_smp_model(self):
        """Carrega modelo usando segmentation_models_pytorch"""
        print("Carregando modelo DeepLabV3+ com ResNet-101...")

        # Criar modelo
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            in_channels=3,
            classes=5,  # 5 classes: matas, urbana, pastagem, solo_exposto, enchente
        )

        # Modo de avaliação
        self.model.eval()
        self.model.to(self.device)

        # Transformações de pré-processamento
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print("✔ Modelo DeepLabV3+ carregado com sucesso")

    def load_alternative_model(self):
        """Carrega modelo alternativo usando torchvision"""
        try:
            from torchvision.models.segmentation import deeplabv3_resnet101
            print("Carregando modelo DeepLabV3 do torchvision...")

            self.model = deeplabv3_resnet101(pretrained=True)
            self.model.eval()
            self.model.to(self.device)

            # Transformações de pré-processamento
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            print("✔ Modelo DeepLabV3 (torchvision) carregado com sucesso")

        except Exception as e:
            print(f"Erro ao carregar modelo torchvision: {e}")
            raise

    def load_fallback_model(self):
        """Modelo de fallback usando detecção por cores (método original)"""
        print("Usando método de fallback (detecção por cores)...")
        self.model = None

        # Intervalos de cores no espaço HSV (método original)
        self.color_ranges = {
            'matas': {'lower': np.array([35, 80, 20]), 'upper': np.array([85, 255, 120])},
            'urbana': {'lower': np.array([0, 0, 120]), 'upper': np.array([180, 60, 255])},
            'solo_exposto': {'lower': np.array([8, 40, 80]), 'upper': np.array([25, 180, 220])},
            'pastagem': {'lower': np.array([35, 20, 80]), 'upper': np.array([85, 120, 180])},
            'enchente': {'lower': np.array([0, 30, 0]), 'upper': np.array([30, 255, 80])}
        }

    def load_image(self, image_path):
        """Carrega a imagem e verifica se é válida"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Não foi possível ler a imagem: {image_path}")

        print(f"Imagem carregada: {image.shape[1]}x{image.shape[0]} pixels")
        return image

    def preprocess_image(self, image):
        """Pré-processa a imagem para o modelo"""
        # Converte BGR para RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Converte para PIL Image
        pil_image = Image.fromarray(image_rgb)

        # Aplica transformações
        if self.transform:
            tensor_image = self.transform(pil_image)
            tensor_image = tensor_image.unsqueeze(0)  # Adiciona batch dimension
            return tensor_image.to(self.device)

        return None

    def predict_with_model(self, image):
        """Faz predição usando o modelo de deep learning"""
        if self.model is None:
            return self.fallback_color_detection(image)

        try:
            # Pré-processa a imagem
            tensor_image = self.preprocess_image(image)

            with torch.no_grad():
                if SMP_AVAILABLE:
                    # Para segmentation_models_pytorch
                    output = self.model(tensor_image)
                else:
                    # Para torchvision models
                    output = self.model(tensor_image)['out']

                # Aplica softmax e pega a classe com maior probabilidade
                predictions = F.softmax(output, dim=1)
                predicted_classes = torch.argmax(predictions, dim=1)

                # Converte para numpy e redimensiona para o tamanho original
                prediction_map = predicted_classes.cpu().numpy()[0]

                # Redimensiona para o tamanho original da imagem
                original_height, original_width = image.shape[:2]
                prediction_map = cv2.resize(
                    prediction_map.astype(np.uint8),
                    (original_width, original_height),
                    interpolation=cv2.INTER_NEAREST
                )

                return self.convert_predictions_to_masks(prediction_map, image.shape[:2])

        except Exception as e:
            print(f"Erro na predição: {e}")
            print("Usando método de fallback...")
            return self.fallback_color_detection(image)

    def convert_predictions_to_masks(self, prediction_map, image_shape):
        """Converte o mapa de predições em máscaras binárias"""
        masks = {}

        # Mapeia classes numéricas para tipos de terreno
        class_to_terrain = {
            0: 'matas',
            1: 'urbana',
            2: 'pastagem',
            3: 'solo_exposto',
            4: 'enchente'
        }

        for class_id, terrain_type in class_to_terrain.items():
            masks[terrain_type] = (prediction_map == class_id).astype(np.uint8) * 255

        # Pós-processamento para limpar as máscaras
        kernel = np.ones((3, 3), np.uint8)
        for terrain_type in masks:
            # Remove ruído e suaviza bordas
            masks[terrain_type] = cv2.morphologyEx(masks[terrain_type], cv2.MORPH_CLOSE, kernel)
            masks[terrain_type] = cv2.morphologyEx(masks[terrain_type], cv2.MORPH_OPEN, kernel)

        return masks

    def fallback_color_detection(self, image):
        """Método de fallback usando detecção por cores"""
        print("Usando detecção por cores (fallback)...")

        # Aplica filtro e converte para HSV
        image_filtered = cv2.bilateralFilter(image, 9, 75, 75)
        hsv = cv2.cvtColor(image_filtered, cv2.COLOR_BGR2HSV)

        masks = {}

        # Cria máscaras para todos os tipos de terreno
        for terrain_type in self.color_ranges:
            lower = self.color_ranges[terrain_type]['lower']
            upper = self.color_ranges[terrain_type]['upper']
            masks[terrain_type] = cv2.inRange(hsv, lower, upper)

        # Detecção aprimorada de água
        enhanced_water = self.enhance_water_detection(hsv)
        masks['enchente'] = cv2.bitwise_or(masks['enchente'], enhanced_water)

        # Filtro de ruído
        kernel = np.ones((2, 2), np.uint8)
        for terrain_type in masks:
            masks[terrain_type] = cv2.morphologyEx(masks[terrain_type], cv2.MORPH_CLOSE, kernel)

        return masks

    def enhance_water_detection(self, hsv_image):
        """Melhora a detecção de água (método auxiliar para fallback)"""
        v_channel = hsv_image[:, :, 2]
        _, black_mask = cv2.threshold(v_channel, 40, 255, cv2.THRESH_BINARY_INV)

        water_blue_range = {'lower': np.array([80, 40, 20]), 'upper': np.array([140, 255, 220])}
        blue_mask = cv2.inRange(hsv_image, water_blue_range['lower'], water_blue_range['upper'])

        brown_mask = cv2.inRange(hsv_image, self.color_ranges['enchente']['lower'],
                                 self.color_ranges['enchente']['upper'])

        combined_water_mask = cv2.bitwise_or(black_mask, blue_mask)
        combined_water_mask = cv2.bitwise_or(combined_water_mask, brown_mask)

        kernel = np.ones((3, 3), np.uint8)
        combined_water_mask = cv2.morphologyEx(combined_water_mask, cv2.MORPH_CLOSE, kernel)
        combined_water_mask = cv2.morphologyEx(combined_water_mask, cv2.MORPH_OPEN, kernel)

        return combined_water_mask

    def calculate_risk_areas(self, masks, image_size):
        """Calcula a porcentagem de cada tipo de área na imagem"""
        stats = {}
        total_pixels = image_size[0] * image_size[1]

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

        # Adiciona informação sobre o método usado
        method_text = "Metodo: IA Deep Learning" if self.model is not None else "Metodo: Deteccao por Cores"
        alerts.append(method_text)

        # Calcula dimensões da legenda
        items_count = 5
        base_legend_height = 50 + (items_count * 25)
        alert_height = len(alerts) * 30
        total_height = base_legend_height + alert_height
        legend_width = 450

        # Ajustes para imagens pequenas
        if height < total_height + 20:
            scale_factor = (height - 20) / total_height
            base_legend_height = int(base_legend_height * scale_factor)
            alert_height = int(alert_height * scale_factor)
            total_height = base_legend_height + alert_height

        if width < legend_width + 20:
            legend_width = width - 20

        # Cria retângulo de fundo
        cv2.rectangle(image, (10, height - total_height - 10),
                      (legend_width, height - 10), (255, 255, 255), -1)
        cv2.rectangle(image, (10, height - total_height - 10),
                      (legend_width, height - 10), (0, 0, 0), 2)

        # Adiciona título
        cv2.putText(image, "ANALISE DE RISCO - IA",
                    (20, height - total_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Adiciona itens da legenda
        y_offset = height - total_height + 55
        legend_order = ['matas', 'urbana', 'pastagem', 'solo_exposto', 'enchente']

        for terrain_type in legend_order:
            color = self.color_mappings[terrain_type]['color']
            desc = self.color_mappings[terrain_type]['description']
            percentage = stats.get(terrain_type, {}).get('percentage', 0)

            cv2.rectangle(image, (20, y_offset - 10), (40, y_offset + 5), color, -1)
            cv2.rectangle(image, (20, y_offset - 10), (40, y_offset + 5), (0, 0, 0), 1)

            stats_text = f"{desc}: {percentage:.2f}%"
            cv2.putText(image, stats_text, (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            y_offset += 25

        # Adiciona alertas
        if alerts:
            alert_y = height - total_height + base_legend_height + 10
            for alert in alerts:
                color = (0, 0, 255) if "ALERTA" in alert else (0, 100, 0)
                cv2.putText(image, alert,
                            (20, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                alert_y += 30

    def save_image(self, image, original_path):
        """Salva a imagem processada com timestamp no nome"""
        try:
            filename, ext = os.path.splitext(original_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            method = "AI" if self.model is not None else "COLOR"
            output_path = f"{filename}_processada_{method}_{timestamp}.png"

            success = cv2.imwrite(output_path, image)
            if not success:
                raise ValueError("Falha ao salvar a imagem")

            return output_path
        except Exception as e:
            print(f"Erro ao salvar imagem: {e}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            method = "AI" if self.model is not None else "COLOR"
            fallback_path = f"processada_{method}_{timestamp}.png"
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

            # 3. Faz predição usando modelo de IA ou fallback
            masks = self.predict_with_model(image)
            print("✔ Análise de terreno concluída")

            # 4. Calcula estatísticas
            stats = self.calculate_risk_areas(masks, image.shape[:2])

            # 5. Aplica máscaras
            processed_image = self.apply_masks(image, masks)
            print("✔ Máscaras aplicadas")

            # 6. Adiciona legenda
            self.add_legend_and_stats(processed_image, stats)
            print("✔ Legenda e estatísticas adicionadas")

            # 7. Salva a imagem
            output_path = self.save_image(processed_image, image_path)
            print(f"✔ Imagem salva em: {output_path}")

            return output_path, stats

        except Exception as e:
            print(f"✖ Erro: {str(e)}")
            raise


def install_dependencies():
    """Instala dependências necessárias"""
    print("Verificando dependências...")

    try:
        import torch
        print("✔ PyTorch encontrado")
    except ImportError:
        print("⚠️  PyTorch não encontrado. Instale com:")
        print("pip install torch torchvision")
        return False

    try:
        import segmentation_models_pytorch
        print("✔ segmentation_models_pytorch encontrado")
    except ImportError:
        print("⚠️  segmentation_models_pytorch não encontrado.")
        print("Para melhor performance, instale com:")
        print("pip install segmentation-models-pytorch")
        print("Continuando com modelo alternativo...")

    return True


def get_user_input():
    """Obtém entrada do usuário de forma interativa"""
    print("\n" + "=" * 70)
    print("ANÁLISE DE RISCO COM INTELIGÊNCIA ARTIFICIAL")
    print("=" * 70 + "\n")
    print("FUNCIONALIDADES:")
    print("• Usa modelo de Deep Learning para segmentação semântica")
    print("• Identifica automaticamente tipos de terreno")
    print("• Detecta áreas de risco de desmoronamento e enchentes")
    print("• Fallback para detecção por cores se modelo não disponível")
    print("")

    while True:
        print("1. Processar uma imagem")
        print("2. Processar todas as imagens em um diretório")
        print("3. Verificar dependências")
        print("4. Sair")

        choice = input("\nEscolha uma opção (1-4): ").strip()

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
            install_dependencies()

        elif choice == "4":
            print("\nEncerrando...")
            sys.exit(0)

        else:
            print("\n✖ Opção inválida. Tente novamente.")


def main():
    """Função principal que orquestra todo o processamento"""
    try:
        print("Inicializando processador de imagens com IA...")

        # Verifica dependências básicas
        if not install_dependencies():
            print("\nDependências necessárias não encontradas.")
            input("Pressione Enter para sair...")
            return

        processor = FloodAndLandslideAIProcessor()

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
