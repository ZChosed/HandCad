import os
import argparse
import cv2
import pyvista as pv
import mediapipe as mp


# Mute noisy background framework logs
os.environ["GLOG_minloglevel"] = "3"
os.environ["OPENCV_VIDEOIO_LOG_LEVEL"] = "ERROR"

parser = argparse.ArgumentParser()
parser.add_argument("--phone", action="store_true", help="Use phone (index 0) instead of webcam (index 1)")
args = parser.parse_args()
CAMERA_INDEX = 0 if args.phone else 1

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
cap = cv2.VideoCapture(CAMERA_INDEX)

# Setup PyVista Engine
plotter = pv.Plotter()
plotter.add_title("HandCAD (Native macOS Loop)", font_size=10)
plotter.set_background("white")

SCENE_SCALE = 10.0  # maps normalized [0,1] coords to [-5, 5] world units
FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
MAX_HANDS = 2
fingertip_meshes = []
for _ in range(MAX_HANDS * len(FINGERTIP_INDICES)):
    mesh = pv.Sphere(radius=0.3, center=(0, 0, 0))
    fingertip_meshes.append(mesh)
    plotter.add_mesh(mesh, color="cyan", show_edges=True)

plotter.camera.position = (0, 0, 20)
plotter.camera.focal_point = (0, 0, 0)
plotter.camera.up = (0, 1, 0)
plotter.camera_set = True  # prevent show() from auto-fitting and overriding camera

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

    tips = []
    if hand_landmarker_result.hand_landmarks:
        for hand_landmarks in hand_landmarker_result.hand_landmarks:
            for idx in FINGERTIP_INDICES:
                lm = hand_landmarks[idx]
                # MediaPipe: x/y normalized [0,1], y=0 is top. Convert to centered Cartesian.
                tips.append(((lm.x - 0.5) * SCENE_SCALE, (0.5 - lm.y) * SCENE_SCALE, 0.0))

    for i, mesh in enumerate(fingertip_meshes):
        center = tips[i] if i < len(tips) else (0.0, 0.0, 0.0)
        mesh.copy_from(pv.Sphere(radius=0.3, center=center))

    plotter.render()

with HandLandmarker.create_from_options(options) as landmarker:
    plotter.add_timer_event(max_steps=1000000, duration=30, callback=update_scene_callback)
    plotter.add_key_event('q', close_all_applications)
    print("Starting native macOS event window...")
    plotter.show()