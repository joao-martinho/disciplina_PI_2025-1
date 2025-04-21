import cv2
from ultralytics import YOLO
import pandas as pd

def process_video(video_path, model, output_path, classes_to_track):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    object_counts = {cls: 0 for cls in classes_to_track}
    frame_counts = []  # Para armazenar contagens por frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True)
        current_frame_counts = {cls: 0 for cls in classes_to_track}

        for result in results:
            for box in result.boxes:
                cls_name = model.names[int(box.cls)]
                if cls_name in classes_to_track:
                    current_frame_counts[cls_name] += 1

                    # Desenhar caixas
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{cls_name}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Atualizar contagens totais
        for cls in classes_to_track:
            object_counts[cls] += current_frame_counts[cls]

        frame_counts.append(current_frame_counts)
        out.write(frame)

    cap.release()
    out.release()
    
    # Salvar contagens em CSV
    df = pd.DataFrame(frame_counts)
    df.to_csv("object_counts_per_frame.csv", index=False)
    
    return object_counts

def main():
    model = YOLO('yolov8n.pt')  
    video_path = 'UA-DETRAC/test/MVI_39031.mp4'  # Exemplo real
    output_path = 'output_detected.mp4'
    classes_to_track = ['car', 'truck', 'bus', 'van']

    counts = process_video(video_path, model, output_path, classes_to_track)
    print("\nContagem total de objetos no vídeo:")
    for cls, count in counts.items():
        print(f"{cls}: {count}")

if __name__ == "__main__":
    main()
