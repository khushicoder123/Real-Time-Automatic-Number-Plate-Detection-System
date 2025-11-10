
from ultralytics import YOLO
import cv2
import csv
import numpy as np
from scipy.interpolate import interp1d
import util
from sort import *
from util import get_car, read_license_plate, write_csv, detect_color

import os
import time


'''# Get the current directory
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
'''

# Load YOLO models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./best.pt')

# Load video
cap = cv2.VideoCapture('./sample1.mp4')

# Initialize SORT tracker
mot_tracker = Sort()

# Define class labels for cars and motorcycles
car_label = 2
motorcycle_label = 3

# Read frames
results = {}
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}

        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) == car_label or int(class_id) == motorcycle_label:
                detections_.append([x1, y1, x2, y2, score, int(class_id)])

        # Track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # Crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # Process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    print("License plate text:", license_plate_text)  # Debug print

                    # Detect color of the car
                    car_color = detect_color(frame, (xcar1, ycar1, xcar2, ycar2))

                    # Save results
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2],
                                                          'color': car_color},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}
                    print("Results:", results)  # Debug print
                else:
                    print("No license plate text found")  # Debug print

# Write results to CSV file
write_csv(results, './test.csv')





#filling the missing data


def interpolate_bounding_boxes(data):
    # Extract necessary data columns from input data
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])
    car_colors = [row['car_color'] for row in data]  # Extract car colors
    object_types = [row['object_type'] for row in data]  # Extract object types

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    for car_id in unique_car_ids:

        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]

        # Filter data for a specific car ID
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = car_frame_numbers[0]
        last_frame_number = car_frame_numbers[-1]

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i - 1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0,
                                           kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['frame_nmr'] = str(frame_number)
            row['car_id'] = str(car_id)
            row['car_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
            row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

            if str(frame_number) not in frame_numbers_:
                # Imputed row, set the following fields to '0'
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
                row['car_color'] = '0'  # Add car color field
                row['object_type'] = '0'  # Add object type field
            else:
                # Original row, retrieve values from the input data if available
                original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(
                    float(p['car_id'])) == int(float(car_id))][0]
                row['license_plate_bbox_score'] = original_row[
                    'license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else '0'
                row['license_number'] = original_row['license_number'] if 'license_number' in original_row else '0'
                row['license_number_score'] = original_row[
                    'license_number_score'] if 'license_number_score' in original_row else '0'
                row['car_color'] = car_colors[i]  # Add car color value
                row['object_type'] = object_types[i]  # Add object type value

            interpolated_data.append(row)

    return interpolated_data



# Load the CSV file
with open('test.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Interpolate missing data
interpolated_data = interpolate_bounding_boxes(data)

# Write updated data to a new CSV file
header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score',
          'license_number', 'license_number_score', 'car_color', 'object_type']

with open('test_interpolated.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)
