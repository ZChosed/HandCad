import cv2

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Waiting for camera frame...")
        continue  # Skip the rest of this loop iteration and try again

    frame = cv2.resize(frame, (640,480))
    cv2.imshow('Webcam Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()