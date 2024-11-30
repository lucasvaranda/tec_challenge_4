import cv2
import mediapipe as mp
import os
from tqdm import tqdm
import face_recognition
import numpy as np
from deepface import DeepFace

def load_images_from_folder(folder):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(folder, filename)
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                face_encoding = face_encodings[0]
                name = os.path.splitext(filename)[0][:-1]
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)
    return known_face_encodings, known_face_names

def detect_pose_and_faces(video_path, output_path, pose_functions, known_face_encodings, known_face_names):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao executar o vídeo")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in tqdm(range(total_frames), desc='Processando vídeo'):
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            for pose_function in pose_functions:
                action_detected = pose_function(results.pose_landmarks.landmark, mp_pose)
                if action_detected:
                    cv2.putText(frame, action_detected, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                    break

        # Face Recognition and Emotion Analysis
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_names = []
        THRESHOLD = 0.5

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Desconhecido"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and face_distances[best_match_index] < THRESHOLD:
                name = known_face_names[best_match_index]
            face_names.append(name)

        result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)

        for face in result:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            dominant_emotion = face['dominant_emotion']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                if x <= left <= x + w and y <= top <= y + h:
                    cv2.putText(frame, name, (x + 6, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        out.write(frame)

        # cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Load known faces
image_folder = 'images'
known_face_encodings, known_face_names = load_images_from_folder(image_folder)

# Prepare video paths
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'Unlocking Facial Recognition_ Diverse Activities Analysis Compressed.mp4')
output_video_path = os.path.join(script_dir, 'output_video_recognize.mp4')

# Define pose functions
# Poses
def is_hands_on_face(landmarks, mp_pose):
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    left_hand_near_face = left_wrist.y < left_shoulder.y and left_wrist.x < left_shoulder.x
    right_hand_near_face = right_wrist.y < right_shoulder.y and right_wrist.x > right_shoulder.x

    if left_hand_near_face or right_hand_near_face:
        return "Maos no rosto"
    return None

def is_hand_wave(landmarks, mp_pose):
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]

    if right_wrist.y < right_shoulder.y and right_wrist.x < right_ear.x and right_wrist.y > right_ear.y:
        return "Mao direita levantada"
    elif left_wrist.y < left_shoulder.y and left_wrist.x > left_ear.x and left_wrist.y > left_ear.y:
        return "Mao esquerda levantada"
    return None

def analyze_arms_pose(landmarks, mp_pose):
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    left_arm_down = left_elbow.y < left_shoulder.y
    right_arm_down = right_elbow.y < right_shoulder.y
    left_arm_up = left_elbow.y > left_shoulder.y
    right_arm_up = right_elbow.y > right_shoulder.y

    if left_arm_up and right_arm_up:
        return "Bracos abaixados"
    elif left_arm_up:
        return "Braco direito para cima"
    elif right_arm_up:
        return "Braco esquerdo para cima"
    elif left_arm_down and right_arm_down:
        return "Bracos para cima"
    else:
        return None

pose_functions = [is_hands_on_face, is_hand_wave, analyze_arms_pose]

# Run the detection
detect_pose_and_faces(input_video_path, output_video_path, pose_functions, known_face_encodings, known_face_names)
