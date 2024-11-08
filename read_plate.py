import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

license_plate_chars = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 
    'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z',
    ' ', '-'
]

def format_license_plate(license_plate_text):
    s = ''.join(char for char in license_plate_text if char in license_plate_chars)
    print(s)
    if len(s) < 7 or len(s) > 10 or not s[0].isdigit() or not s[1].isdigit():
        return "Unknown"
    return s

def read_license_plate(license_plate_crop):

    detections =  reader.readtext(license_plate_crop)
    if len(detections) == 0:
        return "Unknown"
    elif len(detections) == 1:
        return format_license_plate(detections[0][1].upper())
    else:
        return format_license_plate((detections[0][1] + " " +detections[1][1]).upper())
    