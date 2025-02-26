import os
import cv2
import numpy as np
import pytesseract
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Directories for uploaded files and annotated images
UPLOAD_FOLDER = 'uploads'
ANNOTATED_FOLDER = 'annotated'
for folder in [UPLOAD_FOLDER, ANNOTATED_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANNOTATED_FOLDER'] = ANNOTATED_FOLDER

# Set Tesseract path if needed (adjust for your server)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

#############################
# Circuit Grading Functions
#############################

def load_model(model_path):
    """Load trained YOLOv8 model."""
    print("Loading YOLOv8 model...")
    model = YOLO(model_path)
    print("Model loaded successfully.")
    return model

def detect_components(model, image_path):
    """Detect circuit components in the image."""
    print("Running component detection...")
    results = model.predict(source=image_path, imgsz=800)
    detections = []
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            label = result.names[int(cls)]
            detections.append((label, (x1, y1, x2, y2)))
    print(f"Detected {len(detections)} components.")
    return detections, results

def save_annotated_image(image_path, detections, save_path):
    """
    Read the image, adjust brightness/contrast, upscale it for better visibility,
    and then draw dark red bounding boxes and text on the upscaled image.
    Bounding box coordinates are scaled accordingly.
    """
    # Read the original image
    img = cv2.imread(image_path)
    
    # Adjust brightness/contrast
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
    
    # Upscale the image by a fixed scale factor (e.g., 1.5x) for clarity
    scale_factor = 1.5
    height, width = img.shape[:2]
    img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_CUBIC)
    
    # Scale the detection coordinates accordingly
    scaled_detections = []
    for label, (x1, y1, x2, y2) in detections:
        scaled_coords = (int(x1 * scale_factor), int(y1 * scale_factor),
                         int(x2 * scale_factor), int(y2 * scale_factor))
        scaled_detections.append((label, scaled_coords))
    
    # Convert image to RGB for consistent drawing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Set dark red color for annotations (BGR: (0, 0, 139) or RGB: (139, 0, 0))
    annotation_color = (139, 0, 0)
    
    # Draw bounding boxes and text with thicker lines and larger font
    for label, (x1, y1, x2, y2) in scaled_detections:
        cv2.rectangle(img, (x1, y1), (x2, y2), annotation_color, thickness=3)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, annotation_color, thickness=2)
    
    # Convert back to BGR before saving
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img)
    return save_path

def check_drawing_clarity(image_path):
    """Analyze drawing clarity using edge detection."""
    image = cv2.imread(image_path, 0)
    edges = cv2.Canny(image, 50, 150)
    noise = np.sum(edges) / edges.size
    score = max(5, min(100, int(100 * (1 - noise / 255))))
    return score

def extract_text(image_path):
    """Extract text labels using OCR."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    extracted_text = set(text.split())
    return extracted_text

def check_labeling_accuracy(detections, extracted_text):
    """Grade based on detection of junctions and text."""
    detected_labels = {label for label, _ in detections}
    return 100 if 'junction' in detected_labels and 'text' in detected_labels else 10

def check_voltage_accuracy(extracted_text, detections):
    """Grade voltage values accuracy."""
    expected_values = {"9V", "5V", "12V", "3.3V", "-5V"}
    detected_values = {word for word in extracted_text if "V" in word}
    voltage_detected = any(label == 'voltage-battery' for label, _ in detections)
    if not detected_values and not voltage_detected:
        return "Skipped"
    score = (len(detected_values & expected_values) / len(expected_values)) * 100
    return max(10, min(100, score))

def grade_circuit(image_path, model_path):
    """Evaluate a circuit drawing and return a grading report and annotated image filename."""
    print("Grading circuit...")
    model = load_model(model_path)
    detections, results = detect_components(model, image_path)
    annotated_filename = "annotated_" + os.path.basename(image_path)
    annotated_path = os.path.join(app.config['ANNOTATED_FOLDER'], annotated_filename)
    save_annotated_image(image_path, detections, annotated_path)
    extracted_text = extract_text(image_path)
    drawing_clarity_score = check_drawing_clarity(image_path)
    labeling_accuracy_score = check_labeling_accuracy(detections, extracted_text)
    voltage_accuracy_score = check_voltage_accuracy(extracted_text, detections)
    if isinstance(voltage_accuracy_score, int):
        total_score = (drawing_clarity_score + labeling_accuracy_score + voltage_accuracy_score) / 3
    else:
        total_score = (drawing_clarity_score + labeling_accuracy_score + 100) / 3

    report = {
        "Overall Score": round(total_score),
        "Drawing Clarity": drawing_clarity_score,
        "Labeling Accuracy": labeling_accuracy_score,
        "Voltage Accuracy": voltage_accuracy_score,
        "Analysis": {
            "Drawing Clarity": f"{drawing_clarity_score}% - Based on image noise/edge clarity.",
            "Labeling Accuracy": f"{labeling_accuracy_score}% - Detection of text/junctions.",
            "Voltage Accuracy": f"{voltage_accuracy_score if isinstance(voltage_accuracy_score, int) else 'Skipped'}% - Voltage check."
        }
    }
    print("Grading complete.")
    return report, annotated_filename

#########################
# FLASK ROUTES
#########################

@app.route('/')
def home():
    return "Hello from Flask!"

@app.route('/upload', methods=['POST'])
def upload_and_grade():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Update the model path to match where your model file is located on the server
    model_path = "/var/www/project-root/backend/model/best_yolov8_augmented.pt"
    report, annotated_filename = grade_circuit(filepath, model_path)
    annotated_url = f"/annotated/{annotated_filename}"

    return jsonify({
        "message": "Image uploaded and graded successfully.",
        "report": report,
        "annotated_image": annotated_url
    }), 200

@app.route('/annotated/<filename>')
def serve_annotated_image(filename):
    return send_from_directory(app.config['ANNOTATED_FOLDER'], filename)

if __name__ == '__main__':
    # For development; in production, run via Gunicorn + systemd
    app.run(debug=True, host='0.0.0.0', port=5000)

