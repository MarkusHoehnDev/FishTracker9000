import os
import requests
import time
from tkinter import Tk, Canvas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

RASPBERRY_PI_API = 'http://10.9.208.223:5000/sensors'

# Function to download and save the JetBrains Mono font
def download_font():
    font_url = "https://github.com/JetBrains/JetBrainsMono/raw/master/fonts/ttf/JetBrainsMono-Regular.ttf"
    font_path = "JetBrainsMono-Regular.ttf"
    
    # Download the font file
    response = requests.get(font_url)
    
    with open(font_path, "wb") as font_file:
        font_file.write(response.content)

# Check if the font is already downloaded, if not, download it
if not os.path.exists("JetBrainsMono-Regular.ttf"):
    download_font()

# Load the font into Tkinter
font_dir = os.path.join(os.getcwd(), "JetBrainsMono-Regular.ttf")
Tk().tk.call('font', 'create', 'JetBrains Mono', '-family', 'JetBrains Mono', '-size', 10, '-weight', 'bold')

# Initialize window
window = Tk()

window.geometry("1440x900")
window.configure(bg="#FFFFFF")

canvas = Canvas(
    window,
    bg="#FFFFFF",
    height=900,
    width=1440,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)

canvas.place(x=0, y=0)

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
create_rounded_rectangle(canvas, 0, 0, 1440, 900, r=50, fill="#000000", outline="")

# Rounded main panel
create_rounded_rectangle(canvas, 66, 73, 1374, 827, r=50, fill="#1E2120", outline="")

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
create_rounded_rectangle(canvas, 876, 487, 1275, 787, r=30, fill="#2F3235", outline="")
create_rounded_rectangle(canvas, 876, 114, 1275, 414, r=30, fill="#2F3235", outline="")

# Graph styling
plt.rcParams.update({
    "font.family": "JetBrains Mono",
    "font.size": 6,
    "axes.edgecolor": "#2F3235",
    "axes.linewidth": 0.6,
    "axes.labelcolor": "#FFFFFF",
    "xtick.color": "#FFFFFF",
    "ytick.color": "#FFFFFF",
    "text.color": "#FFFFFF",
    "figure.facecolor": "#2F3235"
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
canvas_temp.get_tk_widget().place(x=876, y=114, width=450, height=300)

# Create a canvas to embed the TDS graph
canvas_tds = FigureCanvasTkAgg(fig_tds, master=window)
canvas_tds.get_tk_widget().place(x=876, y=487, width=450, height=300)

temperature_data = []
tds_data = []

def fetch_sensor_data():
    global temperature_data, tds_data
    try:
        response = requests.get(RASPBERRY_PI_API)
        data = response.json()
        temperature = float(data['temperature'].replace('°C', ''))  # Strip the '°C' from the value
        tds = float(data['tds'].replace('ppm', ''))  # Strip the 'ppm' from the value

        temperature_data.append(temperature)
        tds_data.append(tds)
        
        if len(temperature_data) > 100:
            temperature_data.pop(0)
        if len(tds_data) > 100:
            tds_data.pop(0)
        
        update_graphs()
    except Exception as e:
        print(f"Error fetching sensor data: {e}")

def update_graphs():
    # Update temperature graph
    line_temp.set_data(range(len(temperature_data)), temperature_data)
    ax_temp.relim()
    ax_temp.autoscale_view()
    canvas_temp.draw()

    # Update TDS graph
    line_tds.set_data(range(len(tds_data)), tds_data)
    ax_tds.relim()
    ax_tds.autoscale_view()
    canvas_tds.draw()

def update_data():
    while True:
        fetch_sensor_data()
        time.sleep(0.5)  # Rapid updates every 0.5 seconds

# Start the data fetching and updating process in a separate thread
threading.Thread(target=update_data, daemon=True).start()

window.resizable(False, False)
window.mainloop()