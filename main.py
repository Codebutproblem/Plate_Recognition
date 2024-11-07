from ultralytics import YOLO
import cv2
import easyocr
import os

license_plate_chars = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 
    'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z',
    '-', '.', ' '
]

def contains_digit(s):
    return any(char.isdigit() for char in s)

def format_license_plate(license_plate_text):
    return ''.join(char for char in license_plate_text if char in license_plate_chars)

def read_license_plate(license_plate_crop):

    detections =  reader.readtext(license_plate_crop)
    if len(detections) == 0:
        return "Unknown", 0
    
    final_text = ""
    total_score = 0
    count = 0
    for detection in detections:
        bbox, text, score = detection
        if contains_digit(str(text)) and float(score) > 0.5 and count < 2:
            final_text += text.upper() + " "
            total_score += float(score)
            count += 1
    if final_text == "":
        return format_license_plate(detections[0][1]), detections[0][2]
    
    return format_license_plate(final_text[:-1]), total_score / max(count, 1)
    


# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# load models
license_plate_detector = YOLO('./train/weights/best.pt')

# load file
file_name = "plate5.jpg"

if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
    img = cv2.imread(os.path.join("input", file_name))
    license_plates = license_plate_detector(img)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        license_plate_crop = img[int(y1):int(y2), int(x1):int(x2), :]
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 128, 255, cv2.THRESH_BINARY_INV)
        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
        final_score = int(license_plate_text_score / 2 * 100)
        cv2.imshow("License Plate", license_plate_crop_gray)
        cv2.imshow("License Plate Thresh", license_plate_crop_thresh)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color = (0,0,225), thickness = 2)
        cv2.putText(img, license_plate_text, (int(x1), int(y1-35)), cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 2)
        cv2.putText(img, "Score: " + str(final_score) + "%", (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)
    
    cv2.waitKey(0)
    cv2.imwrite(os.path.join("output", file_name), img)
elif file_name.lower().endswith((".mp4", ".avi", ".mkv", ".gif")):
    cap = cv2.VideoCapture(os.path.join("input",file_name))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(3))
    height = int(cap.get(4))
    out = cv2.VideoWriter(os.path.join("output",file_name), fourcc, fps, (width, height))
    
    frame_nmr = -1
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        # detect license plates
        if ret:
            license_plates = license_plate_detector(frame)[0]
            detections = []
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                detections.append((x1, y1, x2, y2, score, class_id))
                
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                final_score = int((license_plate_text_score + score) / 2 * 100)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color = (0,0,225), thickness = 2)
                cv2.putText(frame, license_plate_text, (int(x1), int(y1-35)), cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 2)
                cv2.putText(frame, "Score: " + str(final_score) + "%", (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)
            out.write(frame)
    out.release()
    cap.release()

