from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import base64
from io import BytesIO
import json

app = Flask(__name__)

def process_image(image_path):
    # Load and resize image
    image = cv2.imread(image_path)
    if image is None:
        return None, "Error: Image not found."

    image = cv2.resize(image, (800, 600))
    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Yellow range in HSV
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([40, 255, 255])

    # Mask yellow areas
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Morphological clean
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter large contours
    min_area = 500
    large_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if len(large_contours) < 2:
        return None, "Not enough bars detected to identify two rows."

    # Sort contours top to bottom
    large_contours.sort(key=lambda c: cv2.boundingRect(c)[1])

    # Calculate vertical centers
    centers_y = [cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] // 2 for c in large_contours]

    # Identify two row centers
    row_centers = [min(centers_y), max(centers_y)]
    row_threshold = 100

    rows = [[], []]

    # Assign bars to rows
    for idx, c in enumerate(large_contours, 1):
        x, y, w, h = cv2.boundingRect(c)
        center_y = y + h // 2
        dist0 = abs(center_y - row_centers[0])
        dist1 = abs(center_y - row_centers[1])
        if dist0 < dist1 and dist0 < row_threshold:
            rows[0].append((x, y, w, h, idx))
        elif dist1 < row_threshold:
            rows[1].append((x, y, w, h, idx))

    # Sort bars left to right in each row
    rows[0].sort(key=lambda b: b[0])
    rows[1].sort(key=lambda b: b[0])

    # Draw rectangles and labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r_idx, row in enumerate(rows):
        for l_idx, (x, y, w, h, orig_idx) in enumerate(row, 1):
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, f"{r_idx+1}-{l_idx}", (x, y-10), font, 0.6, (0, 255, 0), 2)

    # Calculate counts
    upper_count = len(rows[0])
    lower_count = len(rows[1])

    # Glucose concentration
    glucose = "No Glucose detected or below detection limit"
    if upper_count == 1:
        glucose = "100 ppm Glucose"
    elif upper_count == 2:
        glucose = "125 ppm Glucose"
    elif upper_count == 3:
        glucose = "150 ppm Glucose"
    elif upper_count == 4:
        glucose = "175 ppm Glucose"
    elif upper_count == 5:
        glucose = "200 ppm Glucose"
    elif upper_count > 5:
        glucose = "Above 200 ppm Glucose"

    # Sucrose concentration
    sucrose = "No Sucrose detected or below detection limit"
    if lower_count == 1:
        sucrose = "100 ppm Sucrose"
    elif lower_count == 2:
        sucrose = "125 ppm Sucrose"
    elif lower_count == 3:
        sucrose = "150 ppm Sucrose"
    elif lower_count == 4:
        sucrose = "175 ppm Sucrose"
    elif lower_count == 5:
        sucrose = "200 ppm Sucrose"
    elif lower_count > 5:
        sucrose = "Above 200 ppm Sucrose"

    # Encode annotated image to base64
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "upper_count": upper_count,
        "lower_count": lower_count,
        "glucose": glucose,
        "sucrose": sucrose,
        "annotated_image": image_base64
    }, None

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    # Save image temporarily
    os.makedirs('uploads', exist_ok=True)
    image_path = os.path.join('uploads', file.filename)
    file.save(image_path)

    # Process image
    result, error = process_image(image_path)

    # Clean up
    os.remove(image_path)

    if error:
        return jsonify({'error': error}), 500
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)