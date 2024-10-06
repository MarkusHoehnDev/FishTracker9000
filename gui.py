import os
import requests
import time
from tkinter import Tk, Canvas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from tkinter import Button

RASPBERRY_PI_API = 'http://10.9.208.223:5000/sensors'

# Global toggle states
toggleStates = {
    "boundingBoxes": True,
    "movementPatterns": False,
    "heatmap": False
}

# Initialize window
window = Tk()
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
    

    print(toggleStates)
    # Update the button styles
    update_button_styles()



# Function to update button styles based on the toggle states
def update_button_styles():
    if toggleStates["boundingBoxes"]:
        bounding_box_button.config(bg="green", fg="black")
    else:
        bounding_box_button.config(bg="grey", fg="green")
    
    if toggleStates["movementPatterns"]:
        movement_patterns_button.config(bg="green", fg="green")
    else:
        movement_patterns_button.config(bg="grey", fg="black")
    
    if toggleStates["heatmap"]:
        heatmap_button.config(bg="green", fg="green")
    else:
        heatmap_button.config(bg="grey", fg="black")

# Create buttons inside the rounded rectangle
# The rectangle's coordinates are defined as (66, 73) to (screen_width - 66, screen_height - 73)
rect_x_start = 100
rect_y_start = screen_height - 113
button_width = 300
button_height = 100
button_spacing = 20  # Space between buttons

bounding_box_button = Button(
    window, text="Bounding Boxes", font=("JetBrains Mono", 25), command=lambda: toggle_state("boundingBoxes"),
    bg="green" if toggleStates["boundingBoxes"] else "grey", fg="red" if toggleStates["boundingBoxes"] else "black"
)
bounding_box_button.place(x=rect_x_start + 20, y=rect_y_start - button_height - 20, width=button_width, height=button_height)

movement_patterns_button = Button(
    window, text="Movement Patterns", font=("JetBrains Mono", 25), command=lambda: toggle_state("movementPatterns"),
    bg="grey" if not toggleStates["movementPatterns"] else "green", fg="black" if not toggleStates["movementPatterns"] else "red"
)
movement_patterns_button.place(x=rect_x_start + button_width + button_spacing + 20, y=rect_y_start - button_height - 20, width=button_width, height=100)

heatmap_button = Button(
    window, text="Heatmap", font=("JetBrains Mono", 25), command=lambda: toggle_state("heatmap"),
    bg="grey" if not toggleStates["heatmap"] else "green", fg="black" if not toggleStates["heatmap"] else "red"
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

# Start the data fetching and updating process in a separate thread
threading.Thread(target=update_data, daemon=True).start()

window.mainloop()