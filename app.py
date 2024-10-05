import os
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

@socketio.on('message')
def handle_message(data):
    print('Received message:', data)
    socketio.emit('response', 'Server received your message: ' + data)

def process_video(video_stream):
    # Load the YOLO model
    model = YOLO("yolo11m.pt")

    # Retrieve class names directly from the model
    class_names = model.names

    cap = cv2.VideoCapture(video_stream)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video stream resolution: {width}x{height}")

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, tracker="bytetrack.yaml")
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Loop through the results and emit details
            for result in results:
                if result.boxes:
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

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    socketio.run(app, debug=True, port=8000)