import cv2
import math

# Load the pre-trained models
age_model = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
gender_model = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize the frame to 227x227 (the input size for AgeGenderNet)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Predict age and gender
    gender_model.setInput(blob)
    gender_preds = gender_model.forward()
    gender = "Male" if gender_preds[0][0] > gender_preds[0][1] else "Female"

    age_model.setInput(blob)
    age_preds = age_model.forward()
    age = int(age_preds[0][0])

    # Display age and gender
    cv2.putText(frame, "Age: " + str(age), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Gender: " + gender, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Age and Gender Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
