import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe hand model.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)

# Drawing setup
color = (255, 0, 0)  # Start with blue color

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Create an overlay for drawings to maintain them on screen
_, overlay = cap.read()
overlay.fill(0)

last_point = None  # Initialize last_point to None
thumb_up_detected_previously = False  # To track if thumb-up was detected in the previous frame

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
    thumb_up_detected = False  # Reset thumb-up detection flag for the current frame

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Retrieve landmarks for the thumb and index finger tips
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            # Detect thumb-up gesture based on position of thumb tip relative to thumb IP and wrist
            if thumb_tip.y < thumb_ip.y and thumb_tip.y < wrist.y:
                thumb_up_detected = True
                if not thumb_up_detected_previously:  # Change color only if thumb-up wasn't detected previously
                    color = (np.random.randint(256), np.random.randint(256), np.random.randint(256))

            # Update thumb-up detected previously flag
            thumb_up_detected_previously = thumb_up_detected

            # Pinching gesture for drawing (consider loosening criteria if necessary)
            distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
            if distance < 0.1:  # Adjust threshold if necessary
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
        thumb_up_detected_previously = False  # Reset thumb-up detection status when overlay is cleared

    # Combine the overlay with the current frame
    combined_image = cv2.addWeighted(overlay, 1, image, 1, 0)

    cv2.imshow('AR Art and Graffiti', combined_image)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
