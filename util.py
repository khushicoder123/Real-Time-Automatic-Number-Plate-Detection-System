import string
import easyocr

import cv2
import numpy as np

def detect_color(frame, bbox):
    """
    Detect the color of the car in the region defined by the bounding box.

    Args:
        frame (numpy.ndarray): The input frame.
        bbox (tuple): A tuple containing the bounding box coordinates (x1, y1, x2, y2).

    Returns:
        str: The detected color of the car.
    """
    # Extract the region of interest (ROI) defined by the bounding box
    x1, y1, x2, y2 = bbox
    roi = frame[int(y1):int(y2), int(x1):int(x2)]

    # Convert the ROI to the HSV color space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define color ranges for common car colors
    color_ranges = {
        'red': ([0, 100, 100], [10, 255, 255]),
        'yellow': ([20, 100, 100], [30, 255, 255]),
        'blue': ([110, 100, 100], [130, 255, 255]),
        'green': ([40, 100, 100], [80, 255, 255]),
        'white': ([0, 0, 200], [180, 80, 255]),
        'black': ([0, 0, 0], [180, 255, 30]),
        'silver': ([0, 0, 150], [180, 30, 200]),
        'gray': ([0, 0, 80], [180, 30, 150]),
        'orange': ([10, 100, 100], [20, 255, 255]),
        'purple': ([130, 100, 100], [140, 255, 255]),
        'pink': ([150, 100, 100], [170, 255, 255]),
        'brown': ([10, 30, 30], [20, 255, 200]),
        'beige': ([20, 20, 20], [30, 255, 255]),
        'cream': ([0, 0, 200], [180, 80, 255]),
        'gold': ([20, 100, 100], [30, 255, 255]),
        'bronze': ([20, 100, 100], [30, 255, 255]),
        'maroon': ([0, 100, 100], [10, 255, 255]),
        'turquoise': ([80, 100, 100], [100, 255, 255]),
        'cyan': ([90, 100, 100], [100, 255, 255]),
        'teal': ([80, 100, 100], [90, 255, 255]),
        'navy': ([100, 100, 100], [110, 255, 255]),
        'indigo': ([110, 100, 100], [120, 255, 255]),
        'lavender': ([120, 100, 100], [130, 255, 255]),
        'mauve': ([140, 100, 100], [150, 255, 255]),
        'olive': ([50, 100, 100], [60, 255, 255]),
        'lime': ([70, 100, 100], [80, 255, 255]),
        'mint': ([70, 100, 100], [80, 255, 255]),
        'coral': ([0, 100, 100], [10, 255, 255]),
        'peach': ([0, 100, 100], [10, 255, 255])
        # Other color ranges...
    }

    # Initialize variables to store the number of non-zero pixels for each color
    color_pixels = {}

    # Iterate over color ranges and count non-zero pixels for each
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        color_pixels[color] = cv2.countNonZero(mask)

    # Determine the color with the maximum number of pixels
    detected_color = max(color_pixels, key=color_pixels.get)
    return detected_color

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                      'license_plate_bbox', 'license_plate_bbox_score',
                                                      'license_number', 'license_number_score', 'car_color', 'object_type'))

        for frame_nmr, frame_data in results.items():
            for car_id, car_data in frame_data.items():
                car_bbox = car_data['car']['bbox']
                car_color = car_data['car']['color'] if 'color' in car_data['car'] else 'Unknown'
                license_plate_data = car_data['license_plate']
                license_plate_bbox = license_plate_data['bbox']
                license_plate_bbox_score = license_plate_data['bbox_score']
                license_plate_text = license_plate_data['text']
                license_plate_text_score = license_plate_data['text_score']

                # Determine object type based on the presence of license plate text
                object_type = 'Car' if license_plate_text else 'Motorcycle'

                f.write('{},{},{},{},{},{},{},{},{}\n'.format(frame_nmr, car_id,
                                                              '[{} {} {} {}]'.format(car_bbox[0], car_bbox[1],
                                                                                      car_bbox[2], car_bbox[3]),
                                                              '[{} {} {} {}]'.format(license_plate_bbox[0],
                                                                                      license_plate_bbox[1],
                                                                                      license_plate_bbox[2],
                                                                                      license_plate_bbox[3]),
                                                              license_plate_bbox_score,
                                                              license_plate_text,
                                                              license_plate_text_score,
                                                              car_color,
                                                              object_type))




def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1



