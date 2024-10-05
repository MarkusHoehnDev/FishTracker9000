import cv2

# Load the saved frame
image_path = 'first_frame.jpg'
img = cv2.imread(image_path)

# Initialize variables for cropping
cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0

# Mouse callback function to record cropping area
def mouse_crop(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, cropping

    # Record starting (x, y) coordinates on left mouse button down
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    # Update the ending coordinates as the mouse moves
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            x_end, y_end = x, y

    # Finalize the cropping region on left mouse button up
    elif event == cv2.EVENT_LBUTTONUP:
        x_end, y_end = x, y
        cropping = False
        # Draw the rectangle over the image
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.imshow("image", img)

# Display the image and set the mouse callback
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)

while True:
    cv2.imshow("image", img)
    key = cv2.waitKey(1) & 0xFF

    # Break the loop if 'q' is pressed
    if key == ord("q"):
        break

# Close all windows
cv2.destroyAllWindows()

# Print the selected coordinates
print(f"Selected crop coordinates: x_start={x_start}, y_start={y_start}, x_end={x_end}, y_end={y_end}")
