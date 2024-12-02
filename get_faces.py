import cv2
import os
from tqdm import tqdm
import face_recognition

def save_faces_from_video(video_path, output_folder, frame_skip=5, margin=20):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Não foi possível abrir o vídeo.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processando vídeo: {video_path}")
    for frame_idx in tqdm(range(total_frames), desc="Detectando rostos"):
        ret, frame = cap.read()

        if not ret or frame_idx % frame_skip != 0:
            continue

        # Converte o frame para RGB (face_recognition utiliza RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detecta os rostos no frame
        face_locations = face_recognition.face_locations(rgb_frame)

        for face_idx, (top, right, bottom, left) in enumerate(face_locations):
            top = max(0, top - margin)
            right = min(frame.shape[1], right + margin)
            bottom = min(frame.shape[0], bottom + margin)
            left = max(0, left - margin)

            face_image = frame[top:bottom, left:right]

            file_name = f"frame_{frame_idx}_face_{face_idx}.jpg"
            file_path = os.path.join(output_folder, file_name)

            cv2.imwrite(file_path, face_image)

    cap.release()
    print(f"Processamento concluído. Rostos salvos em: {output_folder}")

input_video_path = "Unlocking Facial Recognition_ Diverse Activities Analysis.mp4"

output_folder = "images_teste"

save_faces_from_video(input_video_path, output_folder, frame_skip=30, margin=50)
