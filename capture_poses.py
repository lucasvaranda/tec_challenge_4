import cv2
import mediapipe as mp
import os
from tqdm import tqdm

def detect_pose(video_path, pose_functions):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Erro ao executar o vídeo")
        return
    
    while True:
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


        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

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

script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'Unlocking Facial Recognition_ Diverse Activities Analysis.mp4')

detect_pose(input_video_path, pose_functions)