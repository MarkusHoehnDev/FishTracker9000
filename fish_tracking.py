import cv2
import torch
from ultralytics import YOLO

def process_video(video_path):
    # Load the YOLO model
    if torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    model = YOLO("fish.pt").to(device)  # Load your YOLO model

    # Retrieve class names directly from the model
    class_names = model.names

    # Open the video stream
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video stream resolution: {width}x{height}")

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        # do not remove it helps for finding ROI
        cv2.imwrite('frame.jpg', frame) 

        if success:
            # Define the cropping region (ROI)
            x_start = 497  # X-coordinate of the top-left corner of the crop
            y_start = 477  # Y-coordinate of the top-left corner of the crop
            new_width = 878  # Width of the crop
            new_height = 435  # Height of the crop

            # Crop the frame
            cropped_frame = frame[y_start:y_start+new_height, x_start:x_start+new_width]

            rect_x, rect_y = 0, 0
            rect_width, rect_height = 50, 105

            # Draw a white rectangle for visualization
            cv2.rectangle(cropped_frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 255, 255), -1)

            # Run YOLO tracking on the cropped frame
            results = model.track(cropped_frame, persist=True, tracker="botsort.yaml")

            # Loop through the results and extract bounding boxes, class names, confidence, and tracking ID
            for result in results:
                if result.boxes:
                    for box in result.boxes:
                        # Extract bounding box coordinates (xmin, ymin, xmax, ymax)
                        coords = box.xyxy[0].cpu().numpy()
                        xmin, ymin, xmax, ymax = coords

                        # Calculate the center of the bounding box (c_curr)
                        c_curr = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))

                        # Extract the class index and map it to the class name using model.names
                        obj_class = int(box.cls.cpu().numpy().item())
                        class_name = class_names.get(obj_class)

                        # Extract the confidence score
                        confidence = float(box.conf.cpu().numpy().item())

                        # Extract the tracking ID (if available)
                        track_id = int(box.id.cpu().numpy().item()) if box.id is not None else None

                        # If track_id is None, skip drawing and storing movement patterns
                        if track_id is not None:
                            # Print the details with class name and track ID
                            print(f"Object: {class_name}, Confidence: {confidence:.2f}, BBox: [{xmin}, {ymin}, {xmax}, {ymax}], ID: {track_id}")

                            # Get the moving patterns of the tracked fish
                            pattern = get_patterns(c_curr, track_id)  # Store pattern based on track_id
                            pre_p = c_curr

                            # Draw the movement patterns on the cropped frame
                            for p in pattern[-50::5]:  # Skip every 5th frame to avoid clutter
                                cv2.circle(cropped_frame, p, 3, (0, 255, 0), -1)  # Draw small circles at each point
                                if pre_p != c_curr:
                                    cv2.line(cropped_frame, pre_p, p, (0, 255, 0), 1)  # Draw lines connecting points
                                pre_p = p

            # Visualize the results on the cropped frame
            annotated_cropped_frame = results[0].plot()

            # Overlay the annotated cropped frame back onto the original frame
            frame[y_start:y_start+new_height, x_start:x_start+new_width] = annotated_cropped_frame

            # Draw a dotted rectangle around the crop area on the original frame
            draw_dotted_rectangle(frame, (x_start, y_start), (x_start + new_width, y_start + new_height), color=(0, 0, 255), thickness=1, gap=5)

            # Display the full frame with annotations
            cv2.imshow("YOLO Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close the display window
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

# Define video path
video_path = 0  # Use 0 for webcam, or path to a video file for video

# Process the video
process_video(video_path)