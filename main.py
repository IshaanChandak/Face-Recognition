import cv2
import numpy as np
import pip
import dlib
#
# import sys
# print(sys.path)
#
# #
# img = cv2.imread('img.png',0)
# print(img)
#
# cv2.imshow('image',img)
# cv2.waitKey(5000)
# cv2.destroyAllWindows()


# Load the pre-trained face detector and facial expression classifier
face_detector = dlib.get_frontal_face_detector()
expression_classifier = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the Haar cascade for face detection (alternative to dlib face detector)
# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'model.caffemodel')

# Load the pre-trained facial expression model
expression_model = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt.txt",
    "emotion_net.caffemodel"
)

# Define a dictionary mapping expression labels to human-readable names
expression_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}


# Function to detect and analyze facial expressions
def analyze_expression(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector(gray)

    # Iterate over detected faces
    for face in faces:
        # Get the facial landmarks for expression analysis
        landmarks = expression_classifier(gray, face)

        # Extract the region of interest (ROI) for the face
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        roi = gray[y:y + h, x:x + w]

        # Resize the ROI to match the input size of the expression model
        roi = cv2.resize(roi, (48, 48))

        # Normalize the ROI pixel values
        roi = roi.astype("float") / 255.0

        # Preprocess the ROI for the expression model
        roi = cv2.dnn.blobFromImage(roi, 1.0, (48, 48), (0, 0, 0), swapRB=True, crop=False)

        # Pass the ROI through the expression model to predict the expression
        expression_model.setInput(roi)
        predictions = expression_model.forward()

        # Get the index of the predicted expression
        expression_index = np.argmax(predictions)

        # Get the label and probability of the predicted expression
        expression_label = expression_labels[expression_index]
        expression_prob = predictions[0][expression_index]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the predicted expression label and probability
        text = f"{expression_label}: {expression_prob:.2f}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame


# Open a video capture object
video_capture = cv2.VideoCapture(0)

# Process video frames until interrupted
while True:
    # Read a frame from the video capture
    ret, frame = video_capture.read()

    # Call the function to analyze facial expressions
    frame = analyze_expression(frame)

    # Display the resulting frame
    cv2.imshow("Facial Expression Analysis", frame)

    # Check for key press and break the loop if 'q' is pressed

