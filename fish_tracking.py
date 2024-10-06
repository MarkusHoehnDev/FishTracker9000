import cv2
import torch
import numpy as np
from ultralytics import YOLO

def process_video(video_path):
    # Load the YOLO model
    if torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    model = YOLO("fish.pt").to(device)  # Load your YOLO model

    # Open the video stream
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video stream resolution: {width}x{height}")

    # Initialize heatmap as zero array
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO tracking on the full frame
            results = model.track(frame, persist=True, tracker="botsort.yaml")

            # Loop through the results and extract bounding boxes
            for result in results:
                if result.boxes:
                    for box in result.boxes:
                        # Extract bounding box coordinates (xmin, ymin, xmax, ymax)
                        coords = box.xyxy[0].cpu().numpy()
                        xmin, ymin, xmax, ymax = coords

                        # Calculate the center of the bounding box
                        center = (int((xmin + xmax) // 2), int((ymin + ymax) // 2))
                        radius = min(int(xmax - xmin), int(ymax - ymin)) // 2

                        # Add heat to the heatmap by drawing circles at object locations
                        cv2.circle(heatmap, center, radius, 1, thickness=-1)

            # Normalize and apply colormap to the heatmap (orange for warm, blue for cold)
            heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)

            # Blend the heatmap with the original frame
            overlay_frame = cv2.addWeighted(frame, 0.5, heatmap_colored, 0.5, 0)

            # Display the frame with the heatmap overlay
            cv2.imshow("YOLO Tracking with Heatmap", overlay_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

# Define video path
video_path = 0  # Use 0 for webcam, or path to a video file for video

# Process the video
process_video(video_path)
