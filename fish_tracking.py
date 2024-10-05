import cv2
import torch
from ultralytics import YOLO

def process_video(video_path):
    # Load the YOLO model
    if torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    model = YOLO("fish.pt").to(device) # yolov8s = 15 fps
 
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
            # Define the cropping region (ROI)
            x_start = 497  # X-coordinate of the top-left corner of the crop
            y_start = 477  # Y-coordinate of the top-left corner of the crop
            new_width = 878  # Width of the crop
            new_height = 435  # Height of the crop

            # Crop the frame
            cropped_frame = frame[y_start:y_start+new_height, x_start:x_start+new_width]

            rect_x, rect_y = 0, 0 

            rect_width, rect_height = 50, 105

            cv2.rectangle(cropped_frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 255, 255), -1)

            cv2.imwrite('frame.jpg', cropped_frame) 

            

            # Run YOLO tracking on the frame, persisting tracks between frames
            results = model.track(cropped_frame, persist=True, tracker="botsort.yaml")
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLO Tracking", annotated_frame)

            # Loop through the results and print bounding boxes, class names, confidence, and tracking ID
            for result in results:
                if result.boxes:
                    for box in result.boxes:
                        # Extract bounding box coordinates (xmin, ymin, xmax, ymax)
                        coords = box.xyxy[0].cpu().numpy()
                        xmin, ymin, xmax, ymax = coords

                        # Extract the class index and map it to the class name using model.names
                        obj_class = int(box.cls.cpu().numpy().item())
                        class_name = class_names.get(obj_class)

                        # Extract the confidence score
                        confidence = float(box.conf.cpu().numpy().item())

                        # Extract the tracking ID (if available)
                        track_id = int(box.id.cpu().numpy().item()) if box.id is not None else "No ID"

                        # Print the details with class name
                        print(f"Object: {class_name}, Confidence: {confidence:.2f}, BBox: [{xmin}, {ymin}, {xmax}, {ymax}], ID: {track_id}")

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
