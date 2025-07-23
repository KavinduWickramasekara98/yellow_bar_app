import logging
import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def home():
    logger.info("Health check endpoint accessed")
    return jsonify({'status': 'Yellow Bar App is running', 'version': '1.0'})

def process_image(image_path):
    try:
        logger.info(f"Processing image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            logger.error("Image not found or invalid")
            return None, "Error: Image not found."

        image = cv2.resize(image, (800, 600))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        kernel = np.ones((5, 5), np.uint8)
        clean_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 500
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        if len(large_contours) < 2:
            logger.warning("Not enough bars detected")
            return None, "Not enough bars detected to identify two rows."

        large_contours.sort(key=lambda c: cv2.boundingRect(c)[1])
        centers_y = [cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] // 2 for c in large_contours]
        row_centers = [min(centers_y), max(centers_y)]
        row_threshold = 100
        rows = [[], []]

        for idx, c in enumerate(large_contours, 1):
            x, y, w, h = cv2.boundingRect(c)
            center_y = y + h // 2
            dist0 = abs(center_y - row_centers[0])
            dist1 = abs(center_y - row_centers[1])
            if dist0 < dist1 and dist0 < row_threshold:
                rows[0].append((x, y, w, h, idx))
            elif dist1 < row_threshold:
                rows[1].append((x, y, w, h, idx))

        rows[0].sort(key=lambda b: b[0])
        rows[1].sort(key=lambda b: b[0])
        font = cv2.FONT_HERSHEY_SIMPLEX
        for r_idx, row in enumerate(rows):
            for l_idx, (x, y, w, h, orig_idx) in enumerate(row, 1):
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, f"{r_idx+1}-{l_idx}", (x, y-10), font, 0.6, (0, 255, 0), 2)

        upper_count = len(rows[0])
        lower_count = len(rows[1])
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

        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        logger.info("Image processed successfully")
        return {
            "upper_count": upper_count,
            "lower_count": lower_count,
            "glucose": glucose,
            "sucrose": sucrose,
            "annotated_image": image_base64
        }, None
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None, f"Error: {str(e)}"

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            logger.warning("No image provided in request")
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            logger.warning("No image selected")
            return jsonify({'error': 'No image selected'}), 400

        temp_dir = '/tmp/uploads'
        os.makedirs(temp_dir, exist_ok=True)
        image_path = os.path.join(temp_dir, file.filename)
        file.save(image_path)
        logger.info(f"Image saved to: {image_path}")

        result, error = process_image(image_path)
        try:
            os.remove(image_path)
            logger.info(f"Image file {image_path} deleted")
        except:
            logger.warning(f"Failed to delete image file {image_path}")

        if error:
            return jsonify({'error': error}), 500
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in upload_image: {str(e)}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    try:
        logger.info("Starting Flask app")
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"Failed to start Flask app: {str(e)}")
        raise