import cv2
import mediapipe as mp

HAND_DETECTOR_PATH = '/Users/zchosed/HandCad/hand_detection_models/hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_DETECTOR_PATH),
    num_hands=2,
    running_mode=VisionRunningMode.VIDEO)
with HandLandmarker.create_from_options(options) as landmarker:

    frame_timestamp_ms = 0

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Waiting for camera frame...")
            continue  # Skip the rest of this loop iteration and try again

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        hand_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 10

        if hand_landmarker_result.hand_landmarks:
            for hand_landmarks in hand_landmarker_result.hand_landmarks:
                    
                    # --- EXTRACTING POSITION DATA ---
                    # Let's grab the position of the TIP of the index finger (Landmark #8)
                    index_finger_tip = hand_landmarks[0]
                    
                    # MediaPipe coordinates are normalized between 0.0 and 1.0. 
                    # Multiply them by your image dimensions to get pixel coordinates.
                    h, w, _ = frame.shape
                    pixel_x = int(index_finger_tip.x * w)
                    pixel_y = int(index_finger_tip.y * h)
                    
                    # Draw a custom bright circle on the index fingertip
                    cv2.circle(frame, (pixel_x, pixel_y), 12, (255, 0, 0), cv2.FILLED)

            # Display the feed
        cv2.imshow('Hand Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
cap.release()
cv2.destroyAllWindows()
