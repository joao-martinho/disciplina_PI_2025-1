# Detecção e Rastreamento de Objetos com YOLO

## 📌 Passo 1: Importação das Bibliotecas

- Utiliza `cv2` (OpenCV) para processamento de vídeo e imagens.
- Usa `YOLO` da biblioteca `ultralytics` para detecção de objetos.
- Utiliza `pandas` para gerar o relatório em formato CSV.

---

## 🎯 Passo 2: Função de Processamento do Vídeo

1. **Abertura do Vídeo**  
   O vídeo de entrada é carregado. Caso haja erro na abertura, o processo é interrompido.

2. **Extração de Propriedades do Vídeo**  
   São obtidas largura, altura e FPS do vídeo original.

3. **Criação do Vídeo de Saída**  
   Um novo vídeo é preparado, onde os resultados das detecções serão desenhados.

4. **Inicialização das Estruturas de Contagem**  
   - Um dicionário para contagem total de cada classe desejada.  
   - Uma lista para armazenar a contagem por frame.

5. **Processamento Frame a Frame**  
   - Cada frame é lido individualmente.
   - O modelo YOLO é utilizado para detectar objetos com `persist=True`, garantindo rastreamento entre frames.
   - As detecções são analisadas e filtradas com base nas classes desejadas.
   - Caixas verdes são desenhadas ao redor dos objetos detectados, com rótulos.
   - A contagem por classe é atualizada.
   - O frame processado é salvo no vídeo de saída.

6. **Liberação de Recursos**  
   Ao final, os arquivos de vídeo são fechados corretamente.

7. **Geração do CSV**  
   As contagens por frame são salvas em um arquivo `object_counts_per_frame.csv`.

8. **Retorno dos Resultados**  
   A função retorna a contagem total de objetos detectados por classe.

---

## 🚀 Passo 3: Função Principal (main)

1. **Carregamento do Modelo YOLO**  
   Usa o modelo pré-treinado `yolov8n.pt`, capaz de detectar até 80 classes diferentes.

2. **Definição de Caminhos**  
   - Caminho para o vídeo de entrada.  
   - Caminho para salvar o vídeo processado com detecções.

3. **Definição de Classes de Interesse**  
   O rastreamento é feito apenas para veículos: `car`, `truck`, `bus`, `van`.

4. **Chamada da Função de Processamento**  
   A função `process_video()` é executada com os parâmetros definidos.

5. **Exibição dos Resultados no Console**  
   A contagem total de objetos detectados é impressa.

---

## 🧠 Passo 4: Ponto de Entrada do Programa

O script é executado a partir do bloco:

```python
if __name__ == "__main__":
    main()

