import cv2
def process_image(license_plate_crop):
    scale_factor = 1.5
    license_plate_crop = cv2.resize(license_plate_crop, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(license_plate_crop_gray, (5, 5), 0)
    license_plate_unsharp = cv2.addWeighted(license_plate_crop_gray, 1.5, blurred, -0.5, 0)

    license_plate_thresh = cv2.adaptiveThreshold(license_plate_unsharp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    license_plate_clahe = clahe.apply(license_plate_thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    license_plate_cleaned = cv2.morphologyEx(license_plate_clahe, cv2.MORPH_CLOSE, kernel)

    return license_plate_cleaned 