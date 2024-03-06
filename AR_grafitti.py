import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe hand model.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# Drawing setup
color = (255, 0, 0)  # Start with blue color

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Create an overlay for drawings to maintain them on screen
_, overlay = cap.read()
overlay.fill(0)

last_point = None  # Initialize last_point to None

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    # Flip the image and convert color space from BGR to RGB
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    # Variable to count open hands
    open_hands_count = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Pinching gesture detection setup
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))

            # Fist gesture for changing color and showing color picker
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            fist_distance = np.linalg.norm(np.array([wrist.x, wrist.y]) - np.array([middle_finger_mcp.x, middle_finger_mcp.y]))
            if fist_distance < 0.05:  # Threshold for fist gesture
                color = (np.random.randint(256), np.random.randint(256), np.random.randint(256))
                # Draw color picker GUI
                wrist_pos = (int(wrist.x * image.shape[1]), int(wrist.y * image.shape[0]))
                cv2.circle(overlay, wrist_pos, 20, color, -1)

            # Pinching gesture for drawing
            if distance < 0.04:  # Threshold for pinching gesture
                current_point = (int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0]))
                if last_point is not None:
                    cv2.line(overlay, last_point, current_point, color, 5)
                last_point = current_point
            else:
                last_point = None  # Reset last_point when not pinching

            # Check for open palm gesture
            thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
            if (np.linalg.norm(np.array([thumb_cmc.x, thumb_cmc.y]) - np.array([index_mcp.x, index_mcp.y])) > 0.1 and
                np.linalg.norm(np.array([thumb_cmc.x, thumb_cmc.y]) - np.array([pinky_mcp.x, pinky_mcp.y])) > 0.1):
                open_hands_count += 1
            
    # Clear the overlay if two open palms are detected
    if open_hands_count == 2:
        overlay.fill(0)

    # Combine the overlay with the current frame
    combined_image = cv2.addWeighted(overlay, 1, image, 1, 0)
    
    cv2.imshow('AR Art and Graffiti', combined_image)
    
    if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()