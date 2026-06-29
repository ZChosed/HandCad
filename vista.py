import os
import cv2
import pyvista as pv
import mediapipe as mp


# Mute noisy background framework logs
os.environ["GLOG_minloglevel"] = "3"
os.environ["OPENCV_VIDEOIO_LOG_LEVEL"] = "ERROR"

# MP setup
HAND_DETECTOR_PATH = '/Users/zchosed/HandCad/hand_detection_models/hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_DETECTOR_PATH),
    num_hands=2,
    running_mode=VisionRunningMode.VIDEO)

frame_timestamp_ms = 0

# OpenCV Video Capture setup
cap = cv2.VideoCapture(0)

# Setup PyVista Engine
plotter = pv.Plotter()
plotter.add_title("HandCAD (Native macOS Loop)", font_size=10)
plotter.set_background("white")

sphere_mesh = pv.Sphere(radius=1.0, center=(0, 0, 0))
actor = plotter.add_mesh(sphere_mesh, color="cyan", show_edges=True)

# Counter variable to simulate tracking movement values (for testing)
# Once MediaPipe is added back, this function will use your coordinates!
state = {"angle": 0.0}

def close_all_applications():
    print("Exiting HandCAD cleanly...")
    plotter.close()
    cap.release()
    cv2.destroyAllWindows()
    os._exit()

def update_scene_callback(step):
    global frame_timestamp_ms
    ret, frame = cap.read()
    if not ret or frame is None:
        return

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    hand_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    frame_timestamp_ms += 10

    x_3d, y_3d, z_3d = 0.0, 0.0, 0.0
    
    if hand_landmarker_result.hand_landmarks:
            for hand_landmarks in hand_landmarker_result.hand_landmarks:
                    
                    index_tip = hand_landmarks[8]
                    # h, w, _ = frame.shape
                    # pixel_x = int(index_finger_tip.x * w)
                    # pixel_y = int(index_finger_tip.y * h)
                    
                    # cv2.circle(frame, (pixel_x, pixel_y), 12, (255, 0, 0), cv2.FILLED)

                    x_3d = (index_tip.x) * 1.0
                    y_3d = (1 - index_tip.y) * 1.0
                    z_3d = index_tip.z * -1.0

    # cv2.imshow('Hand Tracker', frame)
    updated_sphere = pv.Sphere(radius=0.5, center=(x_3d, y_3d, z_3d))
    actor.mapper.dataset.copy_from(updated_sphere)
    plotter.render()

with HandLandmarker.create_from_options(options) as landmarker:
    plotter.add_timer_event(max_steps=1000000, duration=30, callback=update_scene_callback)
    plotter.add_key_event('q', close_all_applications)
    print("Starting native macOS event window...")
    plotter.show()