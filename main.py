import cv2
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

#load images
smile_img = cv2.imread('smile.jpg', cv2.IMREAD_UNCHANGED)
thumbsup_img = cv2.imread('thumbsup.png', cv2.IMREAD_UNCHANGED)
rest_img = cv2.imread('rest.png', cv2.IMREAD_UNCHANGED)
thinking_img = cv2.imread('thinking.jpg', cv2.IMREAD_UNCHANGED)

if smile_img is None or thumbsup_img is None or rest_img is None:
    raise ValueError("One or more overlay images couldn't be loaded!")

def overlay_image(background, overlay, x, y, scale=1.0):
    if overlay is None:
        return background
    h, w = overlay.shape[:2]
    overlay = cv2.resize(overlay, (int(w * scale), int(h * scale)))
    h, w = overlay.shape[:2]
    if x + w > background.shape[1] or y + h > background.shape[0]:
        return background

    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[y:y+h, x:x+w, c] = (
                alpha * overlay[:, :, c] + (1 - alpha) * background[y:y+h, x:x+w, c]
            )
    else:
        background[y:y+h, x:x+w] = overlay
    return background

def mouth_aspect_ratio(landmarks, img_w, img_h):
    left = landmarks[61]
    right = landmarks[291]
    top = landmarks[13]
    bottom = landmarks[14]

    left = np.array([left.x, left.y])
    right = np.array([right.x, right.y])
    top = np.array([top.x, top.y])
    bottom = np.array([bottom.x, bottom.y])

    width = np.linalg.norm(right - left)
    height = np.linalg.norm(top - bottom)
    if height == 0:
        return 0
    return width / height


def detect_thumbs_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    mcp = [2, 5, 9, 13, 17]
    thumb_up = hand_landmarks.landmark[tips[0]].y < hand_landmarks.landmark[mcp[0]].y
    others_down = all(
        hand_landmarks.landmark[tips[i]].y > hand_landmarks.landmark[mcp[i]].y
        for i in range(1, 5)
    )
    return thumb_up and others_down

def detect_thinking(hand_landmarks, face_landmarks):
    chin = face_landmarks[152]
    chin_xy = np.array([chin.x, chin.y])

    fingertips = [4, 8, 12]  
    for tip_idx in fingertips:
        tip = hand_landmarks.landmark[tip_idx]
        tip_xy = np.array([tip.x, tip.y])
        distance = np.linalg.norm(tip_xy - chin_xy)
        if distance < 0.05: 
            return True
    return False


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Cannot access webcam')

smile_counter = 0
thumbs_counter = 0
thinking_counter = 0

with mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
     mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5) as hands:

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        img_h, img_w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_result = face_mesh.process(rgb)
        hands_result = hands.process(rgb)

        current_emoji = rest_img

       #detect thinking
        thinking_detected = False
        if hands_result.multi_hand_landmarks and face_result.multi_face_landmarks:
            face_landmarks = face_result.multi_face_landmarks[0].landmark
            for hand_landmarks in hands_result.multi_hand_landmarks:
                if detect_thinking(hand_landmarks, face_landmarks):
                    thinking_detected = True
                    break

        if thinking_detected:
            thinking_counter += 1
        else:
            thinking_counter = 0

        if thinking_counter > 3:
                current_emoji = thinking_img

        # detect thumbs up
        if hands_result.multi_hand_landmarks:
            thumbs_detected = False
            for hand_landmarks in hands_result.multi_hand_landmarks:
                if detect_thumbs_up(hand_landmarks):
                    thumbs_detected = True
                    break
                
            if thumbs_detected:
                thumbs_counter += 1
            else: 
                thumbs_counter = 0

            if thumbs_counter > 3:
                current_emoji = thumbsup_img

        # detect smile
        if face_result.multi_face_landmarks:
            for face_landmarks in face_result.multi_face_landmarks:
                mar = mouth_aspect_ratio(face_landmarks.landmark, img_w, img_h)
                print(f"MAR: {mar:.2f}")
                if mar < 30:
                    smile_counter += 1
                else:
                    smile_counter = 0

                if smile_counter > 3:
                    current_emoji = smile_img
                break

       #shows 2 windows, one webcame and one with the images
        cv2.imshow('Webcam Feed', frame)

        display_emoji = cv2.resize(current_emoji, (400, 400))
        cv2.imshow('Detected Emoji', display_emoji)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
