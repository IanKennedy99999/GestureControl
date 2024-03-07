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
peace_sign_detected_previously = False  # To track if the peace sign was detected in the previous frame

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip the image and convert color space from BGR to RGB
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    open_hands_count = 0  # Reset open hands count for every frame

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Retrieve landmarks for key finger positions
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

            # Detect open palms for clearing
            fingers_extended = [index_tip.y < index_mcp.y, middle_tip.y < middle_mcp.y, ring_tip.y < ring_mcp.y, pinky_tip.y < pinky_mcp.y]
            if all(fingers_extended):
                open_hands_count += 1

            # Detect peace sign gesture and change color
            if index_tip.y < index_mcp.y and middle_tip.y < middle_mcp.y and not all(fingers_extended):  # Peace sign detected
                if not peace_sign_detected_previously:
                    color = (np.random.randint(256), np.random.randint(256), np.random.randint(256))
                    peace_sign_detected_previously = True
                    # Show the color being changed when peace sign is up
                cv2.circle(image, (int((index_tip.x + middle_tip.x) / 2 * image.shape[1]), int((index_tip.y + middle_tip.y) / 2 * image.shape[0])), 20, color, cv2.FILLED)
            else:
                peace_sign_detected_previously = False

            # Detect pinching for drawing
            distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
            if distance < 0.1:  # Adjust threshold if necessary
                current_point = (int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0]))
                if last_point is not None:
                    cv2.line(overlay, last_point, current_point, color, 5)
                last_point = current_point
            else:
                last_point = None

    # Clear the overlay if two open palms are detected
    if open_hands_count == 2:
        overlay.fill(0)

    # Combine the overlay with the current frame
    combined_image = cv2.addWeighted(overlay, 1, image, 1, 0)

    cv2.imshow('Drawing', combined_image)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
