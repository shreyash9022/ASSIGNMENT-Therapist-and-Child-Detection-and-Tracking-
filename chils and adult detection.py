import cv2
import torch
import numpy as np
from sort import Sort
import matplotlib.pyplot as plt
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [0]
tracker = Sort()
def process_frame(frame):
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()
    tracking_input = []
    for *bbox, conf, cls in detections:
        x1, y1, x2, y2 = bbox
        tracking_input.append([x1, y1, x2, y2, conf, cls])
    tracking_input = np.array(tracking_input)
    tracked_objects = tracker.update(tracking_input)
    return tracked_objects
cap = cv2.VideoCapture('input_video.mp4')
output = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (int(cap.get(3)), int(cap.get(4))))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    tracked_objects = process_frame(frame)
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id, class_id = obj
        label = 'Child' if class_id == 0 else 'Adult'
        color = (0, 255, 0) if label == 'Child' else (255, 0, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f'{label} ID: {int(track_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    color, 2)
    output.write(frame)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
output.release()
cv2.destroyAllWindows()
