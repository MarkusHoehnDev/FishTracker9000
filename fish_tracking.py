import os, requests, time, cv2, torch, numpy as np
from tkinter import Tk, Canvas, Button, Label
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from ultralytics import YOLO, solutions
from PIL import Image, ImageTk


# Defaults
toggleStates = {
    "boundingBoxes": True,
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

def process_video(video_path, window, video_label):
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

    def update_frame():
        success, frame = cap.read()
        if success:
            # Processing code (same as before)
            inner_x, inner_y, inner_width, inner_height = 514, 515, 894, 503
            white_x, white_y, white_width, white_height = 514, 515, 38, 145

            original_frame = frame.copy()
            frame_copy = frame.copy()
            cv2.rectangle(frame_copy, (white_x, white_y), 
                          (white_x + white_width, white_y + white_height), 
                          (255, 255, 255), -1)  # fill with white

            cropped_frame = frame_copy[inner_y:inner_y + inner_height, 
                                       inner_x:inner_x + inner_width]

            results = model.track(cropped_frame, persist=True, tracker="botsort.yaml")

            for result in results:
                if result.boxes:
                    for box in result.boxes:
                        coords = box.xyxy[0].cpu().numpy()
                        xmin, ymin, xmax, ymax = coords
                        c_curr = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))

                        obj_class = int(box.cls.cpu().numpy().item())
                        class_name = class_names.get(obj_class)
                        confidence = float(box.conf.cpu().numpy().item())
                        track_id = int(box.id.cpu().numpy().item()) if box.id is not None else None

                        if track_id is not None:
                            print(f"Object: {class_name}, Confidence: {confidence:.2f}, "
                                  f"BBox: [{xmin}, {ymin}, {xmax}, {ymax}], ID: {track_id}")
                            
                            if toggleStates["movementPatterns"]:
                                pattern = get_patterns(c_curr, track_id)
                                pre_p = c_curr
                                for p in pattern[-50::5]:
                                    cv2.circle(cropped_frame, p, 3, (0, 255, 0), -1)
                                    if pre_p != c_curr:
                                        cv2.line(cropped_frame, pre_p, p, (0, 255, 0), 1)
                                    pre_p = p

            annotated_cropped_frame = results[0].plot(labels=False, probs=False) if toggleStates["boundingBoxes"] else cropped_frame
            if toggleStates["heatmap"]:
                annotated_cropped_frame = np.ascontiguousarray(annotated_cropped_frame)
                annotated_cropped_frame = heatmap_obj.generate_heatmap(annotated_cropped_frame, results)

            frame[inner_y:inner_y + inner_height, inner_x:inner_x + inner_width] = annotated_cropped_frame
            frame[white_y:white_y + white_height + 1, white_x:white_x + white_width + 1] = original_frame[white_y:white_y + white_height + 1, white_x:white_x + white_width + 1]
            draw_dotted_rectangle(frame, (inner_x, inner_y), 
                                  (inner_x + inner_width, inner_y + inner_height), 
                                  color=(0, 0, 255), thickness=1, gap=5)

            # Convert the frame to RGB for Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the video_label widget
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

        # Schedule the next frame update
        window.after(1, update_frame)

    # Start the frame update loop
    update_frame()

dict_tracks = {"Fish": {}}

def get_patterns(center, track_id):
    track_id = str(track_id)
    if track_id in dict_tracks["Fish"]:
        dict_tracks["Fish"][track_id].append(center)
    else:
        dict_tracks["Fish"][track_id] = [center]
    if len(dict_tracks["Fish"][track_id]) > 60:
        del dict_tracks["Fish"][track_id][:10]
    return dict_tracks["Fish"][track_id]

def draw_dotted_rectangle(img, pt1, pt2, color, thickness=1, gap=5):
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, gap*2):
        cv2.line(img, (x, y1), (min(x+gap, x2), y1), color, thickness)
    for x in range(x1, x2, gap*2):
        cv2.line(img, (x, y2), (min(x+gap, x2), y2), color, thickness)
    for y in range(y1, y2, gap*2):
        cv2.line(img, (x1, y), (x1, min(y+gap, y2)), color, thickness)
    for y in range(y1, y2, gap*2):
        cv2.line(img, (x2, y), (x2, min(y+gap, y2)), color, thickness)

def gui(window):
    window.tk.call('font', 'create', 'JetBrains Mono', '-family', 'JetBrains Mono', '-size', 10, '-weight', 'bold')

    # Get current screen width and height
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    # Ensure the window maximizes to full screen
    window.attributes('-fullscreen', True)

    # Set window size to the screen's resolution
    window.geometry(f"{screen_width}x{screen_height}")
    window.configure(bg="#FFFFFF")

    canvas = Canvas(
        window,
        bg="black",
        height=screen_height,
        width=screen_width,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )

    canvas.place(x=0, y=0)

    # Function to update toggle states
    def toggle_state(button_name):
        global toggleStates
        
        # Update the clicked button's state
        if button_name == "boundingBoxes":
            toggleStates["boundingBoxes"] = not toggleStates["boundingBoxes"]
        elif button_name == "movementPatterns":
            toggleStates["movementPatterns"] = not toggleStates["movementPatterns"]
        elif button_name == "heatmap":
            toggleStates["heatmap"] = not toggleStates["heatmap"]
        

        # Update the button styles
        update_button_styles()



    # Function to update button styles based on the toggle states
    def update_button_styles():
        if toggleStates["boundingBoxes"]:
            bounding_box_button.config(bg="green", fg="green")
        else:
            bounding_box_button.config(bg="red", fg="red")
        
        if toggleStates["movementPatterns"]:
            movement_patterns_button.config(bg="green", fg="green")
        else:
            movement_patterns_button.config(bg="red", fg="red")
        
        if toggleStates["heatmap"]:
            heatmap_button.config(bg="green", fg="green")
        else:
            heatmap_button.config(bg="red", fg="red")

    # Create buttons inside the rounded rectangle
    # The rectangle's coordinates are defined as (66, 73) to (screen_width - 66, screen_height - 73)
    rect_x_start = 100
    rect_y_start = screen_height - 113
    button_width = 300
    button_height = 100
    button_spacing = 20  # Space between buttons

    bounding_box_button = Button(
        window, text="Bounding Boxes", font=("JetBrains Mono", 25), command=lambda: toggle_state("boundingBoxes"),
        bg="green" if toggleStates["boundingBoxes"] else "red", fg="green" if toggleStates["boundingBoxes"] else "red"
    )
    bounding_box_button.place(x=rect_x_start + 20, y=rect_y_start - button_height - 20, width=button_width, height=button_height)

    movement_patterns_button = Button(
        window, text="Movement Patterns", font=("JetBrains Mono", 25), command=lambda: toggle_state("movementPatterns"),
        bg="green" if toggleStates["movementPatterns"] else "red", fg="green" if toggleStates["movementPatterns"] else "red"
    )
    movement_patterns_button.place(x=rect_x_start + button_width + button_spacing + 20, y=rect_y_start - button_height - 20, width=button_width, height=100)

    heatmap_button = Button(
        window, text="Heatmap", font=("JetBrains Mono", 25), command=lambda: toggle_state("heatmap"),
        bg="green" if toggleStates["heatmap"] else "red", fg="green" if toggleStates["heatmap"] else "red"
    )
    heatmap_button.place(x=rect_x_start + 2 * (button_width + button_spacing) + 20, y=rect_y_start - button_height - 20, width=button_width, height=100)

    # Initial button style setup
    update_button_styles()


    def create_rounded_rectangle(canvas, x1, y1, x2, y2, r=30, **kwargs):
        """Create a rounded rectangle on the canvas with smoother corners."""
        points = [
            x1 + r, y1,
            x2 - r, y1,
            x2, y1,
            x2, y1 + r,
            x2, y2 - r,
            x2, y2,
            x2 - r, y2,
            x1 + r, y2,
            x1, y2,
            x1, y2 - r,
            x1, y1 + r,
            x1, y1
        ]
        return canvas.create_polygon(points, **kwargs, smooth=True)

    # Rounded background rectangle
    create_rounded_rectangle(canvas, 0, 0, screen_width, screen_height, r=50, fill="#000000", outline="")

    # Adjust other elements based on the new screen resolution
    create_rounded_rectangle(canvas, 66, 73, screen_width - 66, screen_height - 73, r=50, fill="#1E2120", outline="")

    canvas.create_text(
        123.0,
        114.0,
        anchor="nw",
        text="fishtracker9000",
        fill="#FFFFFF",
        font=("JetBrains Mono", 40 * -1)
    )

    canvas.create_text(
        123.0,
        171.0,
        anchor="nw",
        text="dashboard",
        fill="#6B6A6A",
        font=("JetBrains Mono", 24 * -1)
    )

    # Rounded smaller rectangles for graphs
    create_rounded_rectangle(canvas, screen_width - 564, screen_height - 413, screen_width - 165, screen_height - 113, r=30, fill="#2F3235", outline="")
    create_rounded_rectangle(canvas, screen_width - 564, 114, screen_width - 165, 414, r=30, fill="#2F3235", outline="")

    # Graph styling
    plt.rcParams.update({
        "font.family": "JetBrains Mono",
        "font.size": 6,
        "axes.edgecolor": "#1E2120",
        "axes.linewidth": 0.6,
        "axes.labelcolor": "#FFFFFF",
        "xtick.color": "#FFFFFF",
        "ytick.color": "#FFFFFF",
        "text.color": "#FFFFFF",
        "figure.facecolor": "#1E2120"
    })

    # Create a matplotlib figure for the temperature graph
    fig_temp, ax_temp = plt.subplots()
    line_temp, = ax_temp.plot([], [], color='#FF7F0E', lw=2)
    ax_temp.set_title('Temperature (°C)', pad=0)

    # Create a matplotlib figure for the TDS graph
    fig_tds, ax_tds = plt.subplots()
    line_tds, = ax_tds.plot([], [], color='#1F77B4', lw=2)
    ax_tds.set_title('TDS (ppm)', pad=0)

    # Create a canvas to embed the temperature graph
    canvas_temp = FigureCanvasTkAgg(fig_temp, master=window)
    canvas_temp.get_tk_widget().place(x=screen_width - 564, y=114, width=500, height=300)

    # Create a canvas to embed the TDS graph
    canvas_tds = FigureCanvasTkAgg(fig_tds, master=window)
    canvas_tds.get_tk_widget().place(x=screen_width - 564, y=screen_height - 413, width=500, height=300)

    temperature_data = []
    tds_data = []

    def fetch_sensor_data():
        global temperature_data, tds_data
        try:
            response = requests.get(RASPBERRY_PI_API)
            data = response.json()
            temperature = float(data['temperature'].replace('°C', ''))  # Strip the '°C' from the value
            print(temperature)
            tds = float(data['tds'].replace('ppm', ''))  # Strip the 'ppm' from the value

            temperature_data.append(temperature)
            tds_data.append(tds)
            
            if len(temperature_data) > 10:
                del temperature_data[:1]  # Keep only the last 100 elements
            
            if len(tds_data) > 10:
                del tds_data[:1]  # Keep only the last 100 elements

            update_graphs()
        except Exception as e:
            print(f"Error fetching sensor data: {e}")

    def update_graphs():
        # Update temperature graph
        line_temp.set_data(range(len(temperature_data)), temperature_data)
        ax_temp.relim()  # Recalculate limits for y-axis
        ax_temp.autoscale_view()  # Autoscale only the y-axis
        ax_temp.get_xaxis().set_ticks([])  # Hide x-axis ticks and labels
        canvas_temp.draw()

        # Update TDS graph
        line_tds.set_data(range(len(tds_data)), tds_data)
        ax_tds.relim()  # Recalculate limits for y-axis
        ax_tds.autoscale_view()  # Autoscale only the y-axis
        ax_tds.get_xaxis().set_ticks([])  # Hide x-axis ticks and labels
        canvas_tds.draw()

    def update_data():
        while True:
            fetch_sensor_data()
            time.sleep(0.5)  # Rapid updates every 0.5 seconds

    window.mainloop()


root = Tk()
root.title("YOLO Fish Tracking")
controller = Tk()
controller.title("fishtracker9000")

# Create a label to display the video
video_label = Label(root)
video_label.pack()

# Start processing the video (0 for webcam, or provide video path)
process_video(0, root, video_label)
gui(controller)

# Fetch sensor data every second in a separate thread
root.after(1000, fetch_sensor_data)

# Start the Tkinter main loop
root.mainloop()