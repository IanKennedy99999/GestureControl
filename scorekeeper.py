import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Initialize scores for two players
player_left_score = 0
player_right_score = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Convert the image to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # Process the image and draw landmarks
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            # Considering nose landmark as a central point to distinguish sides
            nose = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
            frame_center = image.shape[1] / 2
            
            # Check which side of the frame the nose landmark is on
            if nose.x < frame_center:  # Left side
                left_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
                left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                right_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
                right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                
                # Check if either arm is raised
                if left_wrist.y < left_shoulder.y or right_wrist.y < right_shoulder.y:
                    player_left_score += 1
                    print(f"Player Left Score: {player_left_score}")
                    
            else:  # Right side
                left_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
                left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                right_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
                right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                
                # Check if either arm is raised
                if left_wrist.y < left_shoulder.y or right_wrist.y < right_shoulder.y:
                    player_right_score += 1
                    print(f"Player Right Score: {player_right_score}")

    cv2.imshow('MediaPipe Holistic', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
