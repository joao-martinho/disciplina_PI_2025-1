# Importações necessárias
import os
import sys
from datetime import datetime

import cv2
import numpy as np
from sklearn.cluster import KMeans


class FloodAndLandslideProcessor:
    """
    Classe para processamento de imagens de satélite para identificação de:
    - Áreas propensas a desmoronamento
    - Áreas afetadas por enchentes
    """

    def __init__(self):
        self.color_mappings = {
            'agua_enchente': {
                'color': (139, 69, 19),  # Marrom escuro para enchente
                'description': 'Area Alagada/Enchente - CRITICO',
                'hsv_ranges': [
                    ([100, 30, 30], [130, 255, 255]),  # Água escura
                    ([0, 40, 40], [30, 255, 255]),     # Água barrenta
                    ([0, 0, 0], [180, 50, 200])        # Reflexos
                ],
                'rgb_ranges': [
                    ([60, 40, 0], [180, 150, 100]),    # Água turva
                    ([20, 20, 50], [100, 100, 200])    # Água profunda
                ]
            },
            'vegetacao_densa': {
                'color': (34, 139, 34),  # Verde escuro
                'description': 'Vegetacao Densa (Baixo risco)',
                'hsv_ranges': [
                    ([35, 40, 40], [85, 255, 255])  # Tons de verde
                ],
                'rgb_ranges': [
                    ([20, 60, 20], [80, 180, 80])  # Verde natural
                ]
            },
            'area_urbana': {
                'color': (169, 169, 169),  # Cinza
                'description': 'Area Urbana (Medio risco)',
                'hsv_ranges': [
                    ([0, 0, 100], [180, 30, 255])  # Tons acinzetados
                ],
                'rgb_ranges': [
                    ([100, 100, 100], [220, 220, 220])  # Concreto/asfalto
                ]
            },
            'solo_exposto': {
                'color': (0, 69, 255),  # Laranja mais vivo
                'description': 'Solo Exposto - Alto risco',
                'hsv_ranges': [
                    ([5, 50, 80], [25, 255, 255])  # Tons alaranjados/terrosos
                ],
                'rgb_ranges': [
                    ([120, 80, 40], [220, 160, 100])  # Solo exposto
                ]
            },
            'vegetacao_esparsa': {
                'color': (50, 205, 50),  # Verde claro
                'description': 'Vegetacao Esparsa (Medio-Alto risco)',
                'hsv_ranges': [
                    ([25, 30, 60], [65, 180, 200])  # Verde claro/pastagem
                ],
                'rgb_ranges': [
                    ([60, 100, 60], [140, 200, 120])  # Pastagem/campo
                ]
            }
        }
        self.alpha = 0.6  # Transparência para visualização

    def load_image(self, image_path):
        """Carrega a imagem e verifica se é válida"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Arquivo nao encontrado: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Nao foi possivel ler a imagem: {image_path}")

        print(f"Imagem carregada: {image.shape[1]}x{image.shape[0]} pixels")
        return image

    def analyze_dominant_colors(self, image, n_colors=8):
        """Analisa as cores dominantes na imagem para calibração automática"""
        small_image = cv2.resize(image, (200, 200))
        pixels = small_image.reshape((-1, 3))

        # Remove pixels muito escuros ou muito claros
        valid_pixels = pixels[
            (np.sum(pixels, axis=1) > 50) &
            (np.sum(pixels, axis=1) < 700)
            ]

        if len(valid_pixels) > 0:
            kmeans = KMeans(n_clusters=min(n_colors, len(valid_pixels)), random_state=42, n_init=10)
            kmeans.fit(valid_pixels)
            colors = kmeans.cluster_centers_.astype(int)

            print("Cores dominantes detectadas (BGR):")
            for i, color in enumerate(colors):
                print(f"  Cor {i + 1}: {color}")

            return colors
        return []

    def create_enhanced_masks(self, image):
        """Cria máscaras usando múltiplos critérios de detecção"""
        height, width = image.shape[:2]
        masks = {}

        # Converte para diferentes espaços de cores
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Analisa cores dominantes
        dominant_colors = self.analyze_dominant_colors(image)

        for terrain_type, config in self.color_mappings.items():
            mask = np.zeros((height, width), dtype=np.uint8)

            # Método 1: Detecção por faixas HSV
            if 'hsv_ranges' in config:
                for hsv_range in config['hsv_ranges']:
                    lower, upper = hsv_range
                    temp_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                    mask = cv2.bitwise_or(mask, temp_mask)

            # Método 2: Detecção por faixas RGB
            if 'rgb_ranges' in config:
                for rgb_range in config['rgb_ranges']:
                    lower, upper = rgb_range
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    temp_mask = cv2.inRange(rgb_image, np.array(lower), np.array(upper))
                    mask = cv2.bitwise_or(mask, temp_mask)

            # Processamento específico para detecção de enchentes
            if terrain_type == 'agua_enchente':
                mask = self.detect_flood_water(image, hsv, lab)

            # Limpeza e refinamento da máscara
            if cv2.countNonZero(mask) > 0:
                # Remove ruído
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                # Remove áreas muito pequenas
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                min_area = (height * width) * 0.0001  # 0.01% da imagem
                for contour in contours:
                    if cv2.contourArea(contour) < min_area:
                        cv2.fillPoly(mask, [contour], 0)

            masks[terrain_type] = mask

        return masks

    def detect_flood_water(self, image, hsv, lab):
        """Detecção aprimorada para água de enchente"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Água escura (rios e enchentes)
        dark_water = cv2.inRange(hsv, np.array([100, 30, 30]), np.array([130, 255, 255]))
        
        # Água barrenta (típica de enchentes)
        brown_water = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([30, 255, 255]))
        
        # Reflexos e áreas alagadas
        light_water = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 50, 255]))
        
        # Combina as detecções
        combined = cv2.bitwise_or(dark_water, brown_water)
        combined = cv2.bitwise_or(combined, light_water)
        
        # Análise de textura para confirmar água
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # Áreas com pouca textura são mais prováveis de ser água
        kernel = np.ones((15, 15), np.uint8)
        smooth_areas = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        smooth_areas = cv2.bitwise_not(smooth_areas)
        
        # Combina com a detecção por cor
        mask = cv2.bitwise_and(combined, smooth_areas)
        
        return mask

    def calculate_risk_assessment(self, masks, image_size):
        """Calcula avaliação de risco detalhada"""
        stats = {}
        total_pixels = image_size[0] * image_size[1]

        for terrain_type, mask in masks.items():
            area_pixels = cv2.countNonZero(mask)
            percentage = (area_pixels / total_pixels) * 100

            stats[terrain_type] = {
                'pixels': area_pixels,
                'percentage': percentage,
                'description': self.color_mappings[terrain_type]['description']
            }

        # Calcula níveis de risco
        risk_level = self.calculate_risk_level(stats)
        stats['risk_assessment'] = risk_level

        return stats

    def calculate_risk_level(self, stats):
        """Cálculo aprimorado de risco para Defesa Civil"""
        flood_pct = stats.get('agua_enchente', {}).get('percentage', 0)
        exposed_soil_pct = stats.get('solo_exposto', {}).get('percentage', 0)
        sparse_veg_pct = stats.get('vegetacao_esparsa', {}).get('percentage', 0)
        dense_veg_pct = stats.get('vegetacao_densa', {}).get('percentage', 0)

        risk_score = 0
        alerts = []

        # Análise de enchentes
        if flood_pct > 10:
            alerts.append("EMERGENCIA MAXIMA: Grande área alagada detectada!")
            risk_score += 100
        elif flood_pct > 5:
            alerts.append("ALERTA VERMELHO: Enchente significativa em curso")
            risk_score += 70
        elif flood_pct > 2:
            alerts.append("ALERTA AMARELO: Possível início de alagamento")
            risk_score += 40

        # Análise de deslizamentos
        if exposed_soil_pct > 20:
            alerts.append("RISCO CRÍTICO: Alta susceptibilidade a deslizamentos")
            risk_score += 60
        elif exposed_soil_pct > 10:
            alerts.append("ALERTA: Risco elevado de deslizamentos")
            risk_score += 40

        # Vegetação
        if sparse_veg_pct > 40 and dense_veg_pct < 15:
            alerts.append("ALERTA: Cobertura vegetal crítica - Risco de erosão")
            risk_score += 30

        # Determina nível final de risco
        if risk_score >= 70:
            level = "CRITICO - AÇÃO IMEDIATA NECESSÁRIA"
        elif risk_score >= 50:
            level = "ALTO - MONITORAMENTO URGENTE"
        elif risk_score >= 30:
            level = "MODERADO - ATENÇÃO REQUERIDA"
        else:
            level = "BAIXO - MONITORAMENTO NORMAL"

        return {
            'level': level,
            'score': risk_score,
            'alerts': alerts
        }

    def apply_enhanced_masks(self, image, masks):
        """Aplica máscaras com visualização melhorada"""
        result = image.copy().astype(np.float32)

        for terrain_type, mask in masks.items():
            if cv2.countNonZero(mask) > 0:
                color = np.array(self.color_mappings[terrain_type]['color'], dtype=np.float32)

                mask_3d = np.stack([mask] * 3, axis=2) / 255.0
                overlay = color.reshape(1, 1, 3) * mask_3d

                result = result * (1 - self.alpha * mask_3d) + overlay * self.alpha

        return result.astype(np.uint8)

    def add_comprehensive_legend(self, image, stats):
        """Adiciona legenda profissional com melhor formatação"""
        height, width = image.shape[:2]
        risk_info = stats.get('risk_assessment', {})

        # Ordem e espaçamento fixos para legendas
        terrain_order = ['agua_enchente', 'solo_exposto', 'vegetacao_esparsa', 
                        'area_urbana', 'vegetacao_densa']

        # Calcula dimensões da legenda
        legend_items = len(self.color_mappings)
        alerts = risk_info.get('alerts', [])
        
        # Aumenta espaçamento
        base_height = 100
        legend_height = legend_items * 35  # Aumentado de 30 para 35
        alert_height = len(alerts) * 40    # Aumentado de 35 para 40
        total_height = base_height + legend_height + alert_height + 50
        
        # Largura ajustada
        legend_width = min(600, width - 40)  # Aumentada e com mais margem

        # Background com maior opacidade
        overlay = image.copy()
        cv2.rectangle(overlay, (20, height - total_height - 20),
                     (legend_width + 20, height - 10), (255, 255, 255), -1)

        # Faixa de risco no topo
        risk_level = risk_info.get('level', 'N/A')
        risk_colors = {
            'CRITICO': ((0, 0, 180), (0, 0, 255)),
            'ALTO': ((0, 140, 255), (0, 165, 255)),
            'MODERADO': ((0, 215, 255), (0, 255, 255)),
            'BAIXO': ((0, 255, 0), (102, 255, 102))
        }

        # Limpa texto removendo acentos e caracteres especiais
        risk_level = risk_level.replace('Ç', 'C').replace('Ã', 'A').replace('-', ' ')
        
        # Texto mais compacto e sem caracteres especiais
        subtitle = "ANALISE DE AREAS DE RISCO DEFESA CIVIL"
        
        # Faixa de risco com gradiente
        cv2.rectangle(overlay, (20, height - total_height - 20),
                     (legend_width + 20, height - total_height + 30), risk_color, -1)

        # Aplica transparência
        cv2.addWeighted(overlay, 0.9, image, 0.1, 0, image)

        # Bordas mais finas e elegantes
        cv2.rectangle(image, (20, height - total_height - 20),
                     (legend_width + 20, height - 10), (0, 0, 0), 1)

        # Título com nível de risco
        title = f"NIVEL DE RISCO: {risk_level}"
        cv2.putText(image, title, (35, height - total_height + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Subtítulo
        cv2.putText(image, subtitle, (35, height - total_height + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Legendas em grade organizada
        y_pos = height - total_height + 60
        for terrain_type in terrain_order:
            if terrain_type in stats and terrain_type in self.color_mappings:
                config = self.color_mappings[terrain_type]
                
                # Remove caracteres especiais da descrição
                desc = config['description'].replace('/', ' ').replace('(', '').replace(')', '')
                desc = desc.replace('í', 'i').replace('á', 'a').replace('ã', 'a').replace('-', ' ')
                pct = stats[terrain_type]['percentage']
                text = f"{desc}: {pct:.1f}%"
                
                # Quadrado colorido com borda fina
                color = config['color']
                cv2.rectangle(image, (35, y_pos - 8), (55, y_pos + 8), color, -1)
                cv2.rectangle(image, (35, y_pos - 8), (55, y_pos + 8), (0, 0, 0), 1)

                # Texto mais compacto e profissional
                cv2.putText(image, text, (65, y_pos + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 3)
                cv2.putText(image, text, (65, y_pos + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

                y_pos += 35

        # Alertas sem caracteres especiais
        if alerts:
            y_pos += 15
            cv2.putText(image, "ALERTAS CRITICOS:", (35, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_pos += 25
            
            for i, alert in enumerate(alerts):
                # Remove caracteres especiais dos alertas
                alert = alert.replace('á', 'a').replace('í', 'i').replace('ã', 'a')
                alert = alert.replace('é', 'e').replace('ç', 'c').replace('ó', 'o')
                
                # Fundo semi-transparente para alertas
                alert_overlay = image.copy()
                cv2.rectangle(alert_overlay, (35, y_pos - 12),
                            (legend_width, y_pos + 12), (255, 255, 255), -1)
                cv2.addWeighted(alert_overlay, 0.7, image, 0.3, 0, image)
                
                # Texto do alerta com contorno
                cv2.putText(image, alert, (40, y_pos + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
                cv2.putText(image, alert, (40, y_pos + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                y_pos += 40

        # Adiciona data/hora
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
        cv2.putText(image, timestamp, (legend_width - 120, height - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    def save_results(self, image, original_path, stats):
        """Salva imagem processada e relatório"""
        try:
            # Salva imagem
            filename, ext = os.path.splitext(original_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            output_path = f"{filename}_analise_{timestamp}.png"
            success = cv2.imwrite(output_path, image, [
                cv2.IMWRITE_PNG_COMPRESSION, 1,
                cv2.IMWRITE_JPEG_QUALITY, 95
            ])

            if not success:
                raise ValueError("Falha ao salvar imagem")

            # Relatório em texto
            report_path = f"{filename}_relatorio_{timestamp}.txt"
            self.save_report(report_path, stats, original_path)

            return output_path, report_path

        except Exception as e:
            print(f"Erro ao salvar: {e}")
            fallback_path = f"analise_{timestamp}.png"
            cv2.imwrite(fallback_path, image)
            return fallback_path, None

    def save_report(self, report_path, stats, original_path):
        """Relatório profissional para Defesa Civil"""
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("RELATÓRIO DE ANÁLISE DE RISCO - DEFESA CIVIL DE BLUMENAU\n")
                f.write("=" * 80 + "\n\n")

                # Informações do documento
                f.write("INFORMAÇÕES DO DOCUMENTO\n")
                f.write("-" * 50 + "\n")
                f.write(f"Data/Hora da Análise: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
                f.write(f"Imagem Analisada: {os.path.basename(original_path)}\n")
                f.write(f"Código de Referência: {datetime.now().strftime('%Y%m%d%H%M')}\n\n")

                # Sumário executivo
                risk_info = stats.get('risk_assessment', {})
                f.write("SUMÁRIO EXECUTIVO\n")
                f.write("-" * 50 + "\n")
                f.write(f"Nível de Risco: {risk_info.get('level', 'N/A')}\n")
                f.write(f"Índice de Risco: {risk_info.get('score', 0)}/100\n\n")

                # Alertas críticos
                alerts = risk_info.get('alerts', [])
                if alerts:
                    f.write("ALERTAS CRÍTICOS\n")
                    f.write("-" * 50 + "\n")
                    for alert in alerts:
                        f.write(f"• {alert}\n")
                    f.write("\n")

                # Análise detalhada
                f.write("ANÁLISE DETALHADA POR ÁREA\n")
                f.write("-" * 50 + "\n")
                
                # Ordem específica para o relatório
                terrain_order = ['agua_enchente', 'solo_exposto', 'vegetacao_esparsa', 
                               'area_urbana', 'vegetacao_densa']
                
                for terrain_type in terrain_order:
                    if terrain_type in stats and terrain_type != 'risk_assessment':
                        data = stats[terrain_type]
                        f.write(f"\n{data.get('description', terrain_type).upper()}\n")
                        f.write(f"  • Área Afetada: {data['percentage']:.2f}%\n")
                        f.write(f"  • Pixels Detectados: {data['pixels']:,}\n")

                # Recomendações
                f.write("\nRECOMENDAÇÕES E AÇÕES NECESSÁRIAS\n")
                f.write("-" * 50 + "\n")

                flood_pct = stats.get('agua_enchente', {}).get('percentage', 0)
                if flood_pct > 1:
                    f.write("\nPARA ÁREAS ALAGADAS:\n")
                    f.write("1. Ativar protocolo de emergência para enchentes\n")
                    f.write("2. Evacuar imediatamente áreas críticas\n")
                    f.write("3. Acionar equipes de resgate e socorro\n")
                    f.write("4. Monitorar níveis dos rios a cada hora\n")
                    f.write("5. Estabelecer rotas de fuga e pontos de encontro\n")

                exposed_soil = stats.get('solo_exposto', {}).get('percentage', 0)
                if exposed_soil > 10:
                    f.write("\nPARA ÁREAS DE RISCO DE DESLIZAMENTO:\n")
                    f.write("1. Realizar vistoria técnica imediata\n")
                    f.write("2. Implementar medidas de contenção emergencial\n")
                    f.write("3. Avaliar necessidade de evacuação preventiva\n")
                    f.write("4. Instalar sistemas de monitoramento\n")
                    f.write("5. Restringir acesso às áreas críticas\n")

                # Observações finais
                f.write("\nOBSERVAÇÕES IMPORTANTES\n")
                f.write("-" * 50 + "\n")
                f.write("• Este relatório é gerado automaticamente e deve ser validado por equipe técnica\n")
                f.write("• As porcentagens indicadas são aproximações baseadas em análise de imagem\n")
                f.write("• Recomenda-se inspeção in loco para confirmação dos dados\n")
                f.write("• Em caso de dúvidas, contatar a coordenação da Defesa Civil\n")

                # Rodapé
                f.write("\n" + "=" * 80 + "\n")
                f.write("Documento gerado pelo Sistema Automatizado de Monitoramento\n")
                f.write("Defesa Civil de Blumenau - Análise de Riscos\n")
                f.write("Contato de Emergência: [INSERIR TELEFONE DE EMERGÊNCIA]\n")
                f.write("=" * 80 + "\n")

        except Exception as e:
            print(f"Erro ao salvar relatório: {e}")

    def process_image(self, image_path):
        """Processa uma imagem com análise completa"""
        try:
            print(f"\nAnalisando: {os.path.basename(image_path)}")

            image = self.load_image(image_path)
            print("✓ Imagem carregada")

            processed = image.copy()
            processed = cv2.bilateralFilter(processed, 9, 75, 75)

            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            processed = cv2.merge((l, a, b))
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)

            masks = self.create_enhanced_masks(processed)
            print("✓ Analise de areas concluida")

            stats = self.calculate_risk_assessment(masks, image.shape[:2])
            print("✓ Avaliacao de risco calculada")

            result_image = self.apply_enhanced_masks(image, masks)
            self.add_comprehensive_legend(result_image, stats)

            img_path, report_path = self.save_results(result_image, image_path, stats)
            print(f"✓ Resultados salvos: {os.path.basename(img_path)}")

            return img_path, report_path, stats

        except Exception as e:
            print(f"✗ Erro no processamento: {str(e)}")
            raise


def get_user_input():
    """Interface do usuário"""
    print("\n" + "=" * 70)
    print("SISTEMA DE ANALISE DE ENCHENTES E DESLIZAMENTOS")
    print("=" * 70)

    print("\nRECURSOS DISPONIVEIS:")
    print("• Deteccao automatica de enchentes")
    print("• Analise de risco de deslizamentos")
    print("• Mapeamento de tipos de terreno")
    print("• Relatorios detalhados")
    print("• Alertas automaticos")

    while True:
        print("\n" + "-" * 50)
        print("OPCOES:")
        print("1. Analisar uma imagem")
        print("2. Analisar pasta de imagens")
        print("3. Sair")

        choice = input("\nEscolha uma opcao (1-3): ").strip()

        if choice == "1":
            path = input("\nCaminho da imagem: ").strip().replace('"', '')
            if os.path.isfile(path):
                return [path]
            print(f"Arquivo nao encontrado: {path}")

        elif choice == "2":
            dir_path = input("\nCaminho da pasta: ").strip().replace('"', '')
            if os.path.isdir(dir_path):
                valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
                images = []
                for f in os.listdir(dir_path):
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        images.append(os.path.join(dir_path, f))

                if images:
                    print(f"{len(images)} imagens encontradas")
                    return images
                print("Nenhuma imagem valida encontrada")
            else:
                print(f"Pasta nao encontrada: {dir_path}")

        elif choice == "3":
            print("\nEncerrando sistema...")
            sys.exit(0)
        else:
            print("Opcao invalida")


def main():
    """Função principal"""
    try:
        print("Inicializando sistema...")
        processor = FloodAndLandslideProcessor()

        image_paths = get_user_input()

        if not image_paths:
            print("Nenhuma imagem para processar")
            return

        print(f"\nProcessando {len(image_paths)} imagem(ns)...")
        successful = 0
        high_risk_count = 0

        for i, path in enumerate(image_paths, 1):
            try:
                print(f"\n{'=' * 50}")
                print(f"[{i}/{len(image_paths)}] {os.path.basename(path)}")
                print("=" * 50)

                img_path, report_path, stats = processor.process_image(path)

                risk_info = stats.get('risk_assessment', {})
                risk_level = risk_info.get('level', 'N/A')

                print(f"\nRESUMO DA ANALISE:")
                print(f"   Nivel de Risco: {risk_level}")

                if risk_level in ['CRITICO', 'ALTO']:
                    high_risk_count += 1
                    print("   ATENCAO NECESSARIA!")

                alerts = risk_info.get('alerts', [])
                if alerts:
                    print("   ALERTAS:")
                    for alert in alerts[:3]:
                        print(f"     • {alert}")

                successful += 1

            except Exception as e:
                print(f"Erro: {str(e)}")
                continue

        print(f"\n{'=' * 60}")
        print("RESUMO FINAL")
        print("=" * 60)
        print(f"Processadas: {successful}/{len(image_paths)} imagens")

        if high_risk_count > 0:
            print(f"Areas de alto risco: {high_risk_count}")
            print("   Verifique os relatorios detalhados!")
        else:
            print("Nenhuma area de alto risco detectada")

        print(f"\nArquivos salvos no mesmo diretorio das imagens originais")

    except KeyboardInterrupt:
        print("\nProcessamento interrompido pelo usuario")
    except Exception as e:
        print(f"\nErro inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        input("\nPressione Enter para sair...")


if __name__ == "__main__":
    main()
