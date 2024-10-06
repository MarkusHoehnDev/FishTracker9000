from flask import Flask, jsonify
import serial
import time

# Initialize Flask app
app = Flask(__name__)

# Initialize serial communication with Arduino for TDS sensor
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)  # Replace ACM0 if needed
time.sleep(2)  # Allow time for serial connection to initialize

# Function to read raw temperature data from the DS18B20 sensor
def read_temp_raw():
    device_folder = '/sys/bus/w1/devices/'  # Path to w1 devices
    device_file = device_folder + '28-0000008315c9/w1_slave'  # Replace with your actual sensor ID
    with open(device_file, 'r') as f:
        lines = f.readlines()
    return lines

# Function to parse the temperature data from the raw output
def read_temp():
    lines = read_temp_raw()

    # Wait for a valid reading (YES means successful CRC check)
    while lines[0].strip()[-3:] != 'YES':
        time.sleep(0.2)
        lines = read_temp_raw()

    # The temperature is in the second line after 't='
    equals_pos = lines[1].find('t=')
    if equals_pos != -1:
        temp_string = lines[1][equals_pos + 2:]
        temp_c = float(temp_string) / 1000.0
        return temp_c

# Function to read TDS data from Arduino over serial
def read_tds():
    if ser.in_waiting > 0:
        tds_data = ser.readline().decode('utf-8').strip()
        return tds_data
    return None

# Flask route to get both temperature and TDS data
@app.route('/sensors', methods=['GET'])
def get_sensor_data():
    # Read temperature from the Raspberry Pi sensor
    temperature = read_temp()

    # Read TDS data from the Arduino
    tds = read_tds()

    # Combine the data into a JSON response
    return jsonify({
        'temperature': f"{temperature:.2f} °C",
        'tds': f"{tds} ppm" if tds else "TDS data not available"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

