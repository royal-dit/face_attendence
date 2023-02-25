i=0

import cv2
import os
import time
# Define the folder to save the images
main_fold = 'FaceData'
save_folder = input("Enter the name")
new_fold = os.path.join(main_fold,save_folder)
facec = cv2.CascadeClassifier('object detection\obj_detection_adv\haarcascade_frontalface_alt.xml')

# Create the folder if it doesn't exist
if not os.path.exists(new_fold):
    os.makedirs(new_fold)
cap = cv2.VideoCapture(0)

# Continuously capture frames from the camera stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()  
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_coordinates = facec.detectMultiScale(gray, 1.3, 4)
        for (x, y, w, h) in face_coordinates:
            faces = frame[y:y+h+5, x:x+w+5, :]
            resized_faces = cv2.resize(faces, (150, 150))
            if i % 2 == 0 and i<=200:
                    timestamp = int(round(time.time() * 1000))
                    filename = os.path.join(new_fold, f"image_{timestamp}.png")
                    # Save the current frame as an image file
                    cv2.imwrite(filename, resized_faces)
                       # Print a message to indicate that the image was saved
                    print(f"Image saved to {filename}")
        
                    
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        i += 1
      
        cv2.imshow('frames', frame)
        
        if cv2.waitKey(1) & 0XFF==ord('q'):
            break

# Release the video capture device and close the window
cap.release()
cv2.destroyAllWindows()
print(resized_faces.shape)
