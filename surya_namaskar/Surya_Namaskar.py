import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Load model
model_path = "surya_mobilenet_model.h5"  # Update path if needed
model = load_model(model_path)

# Get class labels (in the same order as training)
class_names = ['Adho Mukha Svanasana', 'Ashtanga Namaskara', 'Ashwa Sanchalanasana', 'Bhujangasana', 'Dandasana', 'Hasta Uttanasana', 'Padahastasana', 'Pranamasana'] # Replace with your actual class names

# Webcam
cap = cv2.VideoCapture(1)
IMG_SIZE = 224

print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and resize
    frame_flipped = cv2.flip(frame, 1)
    input_frame = cv2.resize(frame_flipped, (IMG_SIZE, IMG_SIZE))
    input_array = img_to_array(input_frame) / 255.0
    input_array = np.expand_dims(input_array, axis=0)

    # Predict
    pred = model.predict(input_array)
    pred_index = np.argmax(pred)
    pred_label = class_names[pred_index]
    confidence = np.max(pred)

    # Overlay prediction on the frame
    cv2.putText(frame_flipped, f"{pred_label} ({confidence*100:.1f}%)", 
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)

    cv2.imshow("Surya Namaskar Pose Detection", frame_flipped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
