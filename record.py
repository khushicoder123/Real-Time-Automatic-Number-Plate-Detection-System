import cv2
import os
import time

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, 'sample.mp4')  # Define the output file path

# Create a VideoCapture object to capture video from the camera
cap = cv2.VideoCapture(0)  # 0 for the default camera, you can change it if you have multiple cameras

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for MP4 format
out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))  # Change the resolution if needed

# Record for 5 seconds
start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Write the frame to the output video file
    out.write(frame)

    # Check if 5 seconds have elapsed
    if time.time() - start_time >= 5:
        break

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()
