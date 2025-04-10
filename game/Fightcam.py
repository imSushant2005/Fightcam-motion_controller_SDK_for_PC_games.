import cv2
import mediapipe as mp
import pydirectinput
import time

# Initialize Pose Tracking
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.91, min_tracking_confidence=0.91)
mp_draw = mp.solutions.drawing_utils

# Open Webcam with Optimized Settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

# Motion Tracking Variables
action_delay = 0.05  # Reduced delay for faster response
last_action_time = time.time()

# Cooldown variables
last_punch_time = 0
last_kick_time = 0
last_move_time = 0
last_crouch_time = 0
last_jump_time = 0

# Body Movement Thresholds
PUNCH_THRESHOLD = 0.07
KICK_THRESHOLD = 0.07
MOVE_THRESHOLD = 0.07
JUMP_THRESHOLD = 0.08
CROUCH_THRESHOLD = 0.08

# Detect whether the player is standing still or moving
previous_center_x = None
previous_hip_y = None
smoothed_hip_y = None

# Player Calibration (Stores player's neutral pose for better detection)
calibrated_hip_y = None
calibrated_stance = None
calibration_done = False

# Function to dynamically adjust thresholds based on player's body

def calibrate_player(landmarks):
    global calibrated_hip_y, calibrated_stance, calibration_done
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    calibrated_hip_y = (left_hip.y + right_hip.y) / 2
    calibrated_stance = (left_hip.x + right_hip.x) / 2
    print("Calibration complete! Adjusting thresholds...")
    calibration_done = True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        current_time = time.time()

        # Perform calibration if not done yet
        if not calibration_done:
            calibrate_player(landmarks)
            continue

        # Extract key landmarks
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]

        # Movement Detection
        center_x = (left_hip.x + right_hip.x) / 2
        if previous_center_x is None:
            previous_center_x = center_x

        if abs(center_x - calibrated_stance) > MOVE_THRESHOLD and current_time - last_move_time > action_delay:
            if center_x < calibrated_stance - 0.04:
                pydirectinput.press("left")
                print("Moving Left!")
            elif center_x > calibrated_stance + 0.04:
                pydirectinput.press("right")
                print("Moving Right!")
            last_move_time = current_time

        previous_center_x = center_x

        # Punch Detection
        if current_time - last_punch_time > action_delay:
            if abs(left_wrist.x - left_elbow.x) > PUNCH_THRESHOLD and left_wrist.y < left_elbow.y:
                pydirectinput.press("x")
                print("Left Punch!")
                last_punch_time = current_time
            if abs(right_wrist.x - right_elbow.x) > PUNCH_THRESHOLD and right_wrist.y < right_elbow.y:
                pydirectinput.press("a")
                print("Right Punch!")
                last_punch_time = current_time

        # Kick Detection
        if current_time - last_kick_time > action_delay:
            if abs(left_knee.x - left_hip.x) > KICK_THRESHOLD and left_knee.y > left_hip.y:
                pydirectinput.press("z")
                print("Left Kick!")
                last_kick_time = current_time
            if abs(right_knee.x - right_hip.x) > KICK_THRESHOLD and right_knee.y > right_hip.y:
                pydirectinput.press("s")
                print("Right Kick!")
                last_kick_time = current_time

        # Crouch & Jump Detection with Smoother Tracking
        smoothed_hip_y = (smoothed_hip_y * 0.84) + (left_hip.y * 0.14) if smoothed_hip_y else left_hip.y
        if current_time - last_crouch_time > action_delay and (smoothed_hip_y - calibrated_hip_y > CROUCH_THRESHOLD * 1.2):
            pydirectinput.press("down")
            print("Crouching!")
            last_crouch_time = current_time
        if current_time - last_jump_time > action_delay and (calibrated_hip_y - smoothed_hip_y > JUMP_THRESHOLD * 1.1):
            pydirectinput.press("up")
            print("Jumping!")
            last_jump_time = current_time

        # Draw Landmarks Clearly
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                               mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3), 
                               mp_draw.DrawingSpec(color=(0,0,255), thickness=2))

    cv2.imshow("FightCam- Tekken motion fight", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
