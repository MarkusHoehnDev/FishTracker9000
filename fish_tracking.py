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

        if success:

            cv2.imwrite('frame.jpg', frame) 

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

            # Run YOLO tracking on the frame, persisting tracks between frames
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

                            # Draw the movement patterns on the frame
                            for p in pattern[-20::5]:  # Skip every 5th frame to avoid clutter
                                cv2.circle(cropped_frame, p, 3, (0, 255, 0), -1)  # Draw small circles at each point
                                if pre_p != c_curr:
                                    cv2.line(cropped_frame, pre_p, p, (0, 255, 0), 1)  # Draw lines connecting points
                                pre_p = p

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLO Tracking", annotated_frame)

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
    if len(dict_tracks["Fish"][track_id]) > 30:
        del dict_tracks["Fish"][track_id][:10]
        
    return dict_tracks["Fish"][track_id]

# Define video path
video_path = 0  # Use 0 for webcam, or path to a video file for video

# Process the video
process_video(video_path)
