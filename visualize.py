import cv2
import numpy as np
import pandas as pd


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=2, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


results = pd.read_csv('./test_interpolated.csv')

# load video
video_path = 'sample.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

license_plate = {}
for car_id in np.unique(results['car_id']):
    max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    license_plate[car_id] = {'license_crop': None,
                             'license_plate_number': results[(results['car_id'] == car_id) &
                                                             (results['license_number_score'] == max_)]['license_number'].iloc[0]}
    cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                             (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
    ret, frame = cap.read()

    bbox_str = results[(results['car_id'] == car_id) &
                       (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0]
    bbox = list(map(float, bbox_str.split()))
    x1, y1, x2, y2 = map(int, bbox)

    license_crop = frame[y1:y2, x1:x2, :]
    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

    license_plate[car_id]['license_crop'] = license_crop


frame_nmr = -1

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# read frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
        for row_indx in range(len(df_)):
            # Parse bounding box coordinates
            car_bbox = list(map(float, df_.iloc[row_indx]['car_bbox'].split()))
            car_x1, car_y1, car_x2, car_y2 = map(int, car_bbox)

            # Draw car
            draw_border(frame, (car_x1, car_y1), (car_x2, car_y2), (0, 255, 0), 25, line_length_x=200,
                        line_length_y=200)

            # Parse license plate bounding box coordinates
            license_bbox = list(map(float, df_.iloc[row_indx]['license_plate_bbox'].split()))
            x1, y1, x2, y2 = map(int, license_bbox)

            # Draw license plate bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 12)

            # Crop license plate
            license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']
            H, W, _ = license_crop.shape
            try:
                # Resize license plate crop to fit the region
                license_crop_resized = cv2.resize(license_crop, (x2 - x1, y2 - y1))

                frame[y1:y2, x1:x2, :] = license_crop_resized

                (text_width, text_height), _ = cv2.getTextSize(
                    license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4.3,
                    17)
                cv2.putText(frame,
                            license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                            (int((x2 + x1 - text_width) / 2), int(y1 - text_height - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2.0,  # Reduce the font scale to make the text smaller
                            (0, 0, 0),
                            5)  # Reduce the thickness of the text

            except Exception as e:
                print(e)

        out.write(frame)

# Release resources
cap.release()
out.release()
