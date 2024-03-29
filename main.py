from ultralytics import YOLO
import supervision as sv
import numpy as np

# Car truck bus , cars trucks *buses

model = YOLO('yolov8x.pt')

VIDEO_PATH = './vids/vid.mp4'

video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)

def process_frame(frame: np.ndarray, _) -> np.ndarray:
    results = model(frame, imgsz=1280)[0]
    
    detections = sv.Detections.from_ultralytics(results)

    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

    # labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]
    # frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    frame = box_annotator.annotate(scene=frame, detections=detections)

    return frame

sv.process_video(source_path=VIDEO_PATH, target_path=f"result.mp4", callback=process_frame)

