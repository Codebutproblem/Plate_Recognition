from ultralytics import YOLO
import cv2
import os
from read_plate import read_license_plate
from image_processing import process_image

# load models
license_plate_detector = YOLO('./train/weights/best.pt')

# load file
file_name = "plate8.jpg"

if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
    img = cv2.imread(os.path.join("input", file_name))
    license_plates = license_plate_detector(img)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        license_plate_crop = img[int(y1):int(y2), int(x1):int(x2), :]

        processed_license_plate = process_image(license_plate_crop)
        cv2.imshow("License Plate", processed_license_plate)
        cv2.waitKey(0)

        license_plate_text = read_license_plate(processed_license_plate)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color = (0,0,225), thickness = 2)
        cv2.putText(img, license_plate_text, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 2)
    
    cv2.imshow("License Plate Detection", img)
    cv2.waitKey(0)
    cv2.imwrite(os.path.join("output", file_name), img)