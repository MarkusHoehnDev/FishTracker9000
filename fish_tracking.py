import cv2
import torch
import time
from ultralytics import YOLO


def process_video(video_path):
    # Load the YOLO model and move to GPU if available
    device = torch.device("mps")
    model = YOLO("yolov8s.pt").to(device)

    # Retrieve class names directly from the model
    class_names = model.names

    # Open the video stream
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video stream resolution: {width}x{height}")

    fps = 0
    prev_time = time.time()

    # Loop through the video frames
    while cap.isOpened():
        print(device)
        success, frame = cap.read()

        if success:
            # Calculate the FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Run YOLO tracking on the frame
            results = model.track(frame, persist=True, tracker="bytetrack.yaml")
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Add the FPS display on the frame
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the annotated frame
            cv2.imshow("YOLO Tracking", annotated_frame)

            # Loop through the results and print bounding boxes, class names, confidence, and tracking ID
            for result in results:
                if result.boxes:
                    for box in result.boxes:
                        coords = box.xyxy[0].cpu().numpy()
                        xmin, ymin, xmax, ymax = coords
                        obj_class = int(box.cls.cpu().numpy().item())
                        class_name = class_names.get(obj_class)
                        confidence = float(box.conf.cpu().numpy().item())
                        track_id = int(box.id.cpu().numpy().item()) if box.id is not None else "No ID" 
                        print(f"Object: {class_name}, Confidence: {confidence:.2f}, BBox: [{xmin}, {ymin}, {xmax}, {ymax}], ID: {track_id}")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

# Process the video
process_video(0)
