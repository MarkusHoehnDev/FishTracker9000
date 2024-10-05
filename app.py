import torch
import cv2
from ultralytics import YOLO
from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)

socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    socketio.start_background_task(process_video, 0)

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

            # Run YOLO tracking on the frame, persisting tracks between frames
            results = model.track(cropped_frame, persist=True, tracker="botsort.yaml")
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            

            # Loop through the results and emit details
            for result in results:
                if result.boxes:
                    socketio.emit('')
                    for box in result.boxes:
                        # Extract bounding box coordinates (xmin, ymin, xmax, ymax) and convert to regular float
                        coords = box.xyxy[0].cpu().numpy()
                        xmin, ymin, xmax, ymax = [float(coord) for coord in coords]

                        # Extract the class index and map it to the class name using model.names
                        obj_class = int(box.cls.cpu().numpy().item())
                        class_name = class_names.get(obj_class)

                        # Extract the confidence score and convert to regular float
                        confidence = float(box.conf.cpu().numpy().item())

                        # Extract the tracking ID (if available)
                        track_id = int(box.id.cpu().numpy().item()) if box.id is not None else "No ID"

                        # Print the details with class name
                        print(f"Object: {class_name}, Confidence: {confidence:.2f}, BBox: [{xmin}, {ymin}, {xmax}, {ymax}], ID: {track_id}")

                        # Emit all details to the client as a dictionary, ensuring all types are JSON-serializable
                        socketio.emit('detection', {
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': [xmin, ymin, xmax, ymax],
                            'track_id': track_id
                        })
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    socketio.run(app, debug=True, port=8000)