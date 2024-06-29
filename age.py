import cv2
import numpy as np

# Load pre-trained models
age_net = cv2.dnn.readNetFromCaffe(
    'deploy_age.prototxt', 
    'age_net.caffemodel'
)
gender_net = cv2.dnn.readNetFromCaffe(
    'deploy_gender.prototxt', 
    'gender_net.caffemodel'
)

# List of age ranges and genders
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Load a pre-trained face detector model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def predict_age_and_gender(image_path):
    # Read the input image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_img = image[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (104.0, 177.0, 123.0), swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]
           # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_LIST[age_preds[0].argmax()]

        # Display the results
        label = f'{gender}, {age}'
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the output
    cv2.imshow('Age and Gender Prediction', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
predict_age_and_gender('path/to/your/image.jpg')
