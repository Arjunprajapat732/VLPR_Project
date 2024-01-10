import cv2
from PIL import Image
import pytesseract
import os

# Read input image
img = cv2.imread("number.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Path to Haar cascade XML file
cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
cascade = cv2.CascadeClassifier(cascade_path)

plates = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
print('Number of detected license plates:', len(plates))

# Ensure the 'path_to_images/' directory exists
output_directory = 'path_to_images/'
os.makedirs(output_directory, exist_ok=True)

# Loop over all plates
for idx, (x, y, w, h) in enumerate(plates):
    # Draw bounding rectangle around the license number plate
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Extract the region of interest (ROI), i.e., the number plate
    plate_roi = gray[y:y+h, x:x+w]

    # Save number plate detected with a unique filename
    filename = f'Numberplate_{idx + 1}.jpg'
    cv2.imwrite(os.path.join(output_directory, filename), plate_roi)

    cv2.imshow(f'Number Plate {idx + 1}', plate_roi)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# text extraction
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

# Loop over all saved number plates
for idx in range(len(plates)):
    # Construct the filename dynamically
    filename = f'Numberplate_{idx + 1}.jpg'
    
    # Full path to the image file
    image_path = os.path.join(output_directory, filename)

    extracted_text = extract_text_from_image(image_path)

    print(f"Extracted Text from {filename}:", extracted_text)
