import cv2
import torch
from ultralytics import YOLO, solutions
import requests
import time
import numpy as np

toggleStates = {
    "boundingBoxes": False,
    "movementPatterns": False,
    "heatmap": False
}

RASPBERRY_PI_API = 'http://10.9.208.223:5000/sensors'

def fetch_sensor_data():
    try:
        response = requests.get(RASPBERRY_PI_API)
        data = response.json()
        print(f"Temperature: {data['temperature']}")
        print(f"TDS: {data['tds']} ")
    except Exception as e:
        print(f"Error fetching sensor data: {e}")

def process_video(video_path):
    # Load the YOLO model
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = YOLO("fish.pt").to(device)

    # Retrieve class names directly from the model
    class_names = model.names

    # Open the video stream
    cap = cv2.VideoCapture(video_path)
    heatmap_obj = solutions.Heatmap(colormap=cv2.COLORMAP_PARULA, shape="circle", names=model.names)


    # Get video resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video stream resolution: {width}x{height}")

    # Loop through video frames
    while cap.isOpened():
        success, frame = cap.read()

        # Save frame for reference (optional for finding ROI)
        cv2.imwrite('frame.jpg', frame)

        if success:
            # Define the regions of interest (ROI) for YOLO processing
            inner_x, inner_y, inner_width, inner_height = 514, 515, 894, 503
            white_x, white_y, white_width, white_height = 514, 515, 38, 145

            # Create an untouched copy of the original frame
            original_frame = frame.copy()

            # Create a modified copy for YOLO processing
            frame_copy = frame.copy()

            # Fill the white chunk inside the inner rectangle to mask the sensors from being detected as fish lol
            cv2.rectangle(frame_copy, (white_x, white_y), 
                          (white_x + white_width, white_y + white_height), 
                          (255, 255, 255), -1)  # fill with white

            # Crop the inner rectangle for YOLO processing
            cropped_frame = frame_copy[inner_y:inner_y + inner_height, 
                                       inner_x:inner_x + inner_width]

            # Run YOLO tracking on the cropped frame
            results = model.track(cropped_frame, persist=True, tracker="botsort.yaml")

            # Loop through the results and extract details
            for result in results:
                if result.boxes:
                    for box in result.boxes:
                        # Extract bounding box coordinates
                        coords = box.xyxy[0].cpu().numpy()
                        xmin, ymin, xmax, ymax = coords
                        c_curr = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))

                        # Get class name and confidence
                        obj_class = int(box.cls.cpu().numpy().item())
                        class_name = class_names.get(obj_class)
                        confidence = float(box.conf.cpu().numpy().item())

                        # Get tracking ID if available
                        track_id = int(box.id.cpu().numpy().item()) if box.id is not None else None

                        if track_id is not None:
                            print(f"Object: {class_name}, Confidence: {confidence:.2f}, "
                                  f"BBox: [{xmin}, {ymin}, {xmax}, {ymax}], ID: {track_id}")
                            
                            # If movement patterns toggle is active
                            if toggleStates["movementPatterns"]:
                                pattern = get_patterns(c_curr, track_id)
                                pre_p = c_curr

                                # Draw movement patterns on the cropped frame
                                for p in pattern[-50::5]:  # Skip every 5th frame
                                    cv2.circle(cropped_frame, p, 3, (0, 255, 0), -1)
                                    if pre_p != c_curr:
                                        cv2.line(cropped_frame, pre_p, p, (0, 255, 0), 1)
                                    pre_p = p

            # Visualize the results on the cropped frame
            if toggleStates["boundingBoxes"] and results[0].boxes:
                # If bounding boxes are enabled, draw them
                annotated_cropped_frame = results[0].plot(labels=False, probs=False)
            else:
                # Otherwise, retain the unannotated cropped frame
                annotated_cropped_frame = cropped_frame

            # Check for heatmap toggle
            if toggleStates["heatmap"]:
                # Apply the heatmap on the current frame, regardless of bounding boxes
                annotated_cropped_frame = np.ascontiguousarray(annotated_cropped_frame)
                annotated_cropped_frame = heatmap_obj.generate_heatmap(annotated_cropped_frame, results)


            # Overlay the annotated cropped frame back onto the original full frame
            frame[inner_y:inner_y + inner_height, inner_x:inner_x + inner_width] = annotated_cropped_frame

            # Restore the white chunk area from the original (untouched) frame
            frame[white_y:white_y + white_height + 1, white_x:white_x + white_width + 1] = original_frame[white_y:white_y + white_height + 1, white_x:white_x + white_width + 1]

            # Draw dotted rectangle around the inner rectangle (optional)
            draw_dotted_rectangle(frame, (inner_x, inner_y), 
                                  (inner_x + inner_width, inner_y + inner_height), 
                                  color=(0, 0, 255), thickness=1, gap=5)

            # Display the full frame with annotations
            cv2.imshow("YOLO Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


# Function to store movement patterns of the tracked fish
dict_tracks = {"Fish": {}}

def get_patterns(center, track_id):
    # Ensure the track ID is a string to prevent issues with dict keys
    track_id = str(track_id)

    # Check if this track_id already has a stored pattern
    if track_id in dict_tracks["Fish"]:
        dict_tracks["Fish"][track_id].append(center)
    else:
        dict_tracks["Fish"][track_id] = [center]

    # Keep only the last 30 positions, remove older ones
    if len(dict_tracks["Fish"][track_id]) > 60:
        del dict_tracks["Fish"][track_id][:10]

    return dict_tracks["Fish"][track_id]

def draw_dotted_rectangle(img, pt1, pt2, color, thickness=1, gap=5):
    x1, y1 = pt1
    x2, y2 = pt2

    # Draw top edge
    for x in range(x1, x2, gap*2):
        cv2.line(img, (x, y1), (min(x+gap, x2), y1), color, thickness)
    # Draw bottom edge
    for x in range(x1, x2, gap*2):
        cv2.line(img, (x, y2), (min(x+gap, x2), y2), color, thickness)
    # Draw left edge
    for y in range(y1, y2, gap*2):
        cv2.line(img, (x1, y), (x1, min(y+gap, y2)), color, thickness)
    # Draw right edge
    for y in range(y1, y2, gap*2):
        cv2.line(img, (x2, y), (x2, min(y+gap, y2)), color, thickness)

process_video(0)

if __name__ == "__main__":
    # Fetch sensor data every 1 second
    while True:
        fetch_sensor_data()
        time.sleep(1)