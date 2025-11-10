import cv2
import os
import time
import csv
import numpy as np
from scipy.interpolate import interp1d
from ultralytics import YOLO
from sort import Sort
from util import get_car, read_license_plate, write_csv, detect_color
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder='templates')

# Load YOLO models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./best.pt')

# Initialize SORT tracker
mot_tracker = Sort()

# Define class labels for cars and motorcycles
car_label = 2
motorcycle_label = 3

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video_ajax():
    # Load video
    cap = cv2.VideoCapture('sample1.mp4')

    # Initialize SORT tracker
    mot_tracker = Sort()

    # Define class labels for cars and motorcycles
    car_label = 2
    motorcycle_label = 3

    # Read colors from CSV file
    original_colors = {}
    csv_file = os.path.join(os.path.dirname(__file__), 'orignalcolors.csv')
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            # Assuming license plate number is in the 1st column and color is in the 2nd column starting from the 2nd row
            original_colors[row[0]] = row[1]

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

                        # Compare detected color with original color from CSV
                        if license_plate_text in original_colors and car_color != original_colors[license_plate_text]:
                            # Save results only if the detected color doesn't match the original color
                            results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2],
                                                                'color': car_color},
                                                          'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                            'text': license_plate_text,
                                                                            'bbox_score': score,
                                                                            'text_score': license_plate_text_score,
                                                                             'original_color': original_colors[license_plate_text]}}
                            print("Results:", results)  # Debug print
                    else:
                        print("No license plate text found")  # Debug print

    # Write results to CSV file
    write_csv(results, './result.csv')

    # Return results as JSON response
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
