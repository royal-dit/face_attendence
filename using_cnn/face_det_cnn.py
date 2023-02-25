#using CNN
import cv2

import datetime
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator






facec = cv2.CascadeClassifier('object detection\obj_detection_adv\haarcascade_frontalface_alt.xml')

cam = cv2.VideoCapture(0)

train_dir = 'FaceData'
train_datagen = ImageDataGenerator(rescale=1./255)
train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                 shuffle = True,
                                                                 image_size = (150,150),
                                                                 label_mode = "categorical",
                                                                
                                                                 )

print(len(train_data))

class_names = train_data.class_names
print(class_names)
#cnn model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation="relu",input_shape = (150,150,3)),
  
    tf.keras.layers.MaxPool2D(pool_size = 2),
    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(128,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(len(class_names),activation='softmax'),   
])
model.summary()
model.compile(loss = 'categorical_crossentropy',
              optimizer = tf.keras.optimizers.Adam(),
              metrics = ['accuracy']
              )
model.fit(train_data,epochs = 10)




# Face Recognition using KNN
while True:
    ret, fr = cam.read()
    if ret == True:
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        face_coordinates = facec.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in face_coordinates:
            fc = fr[y:y + h, x:x + w, :]
            datet = str(datetime.datetime.now())
            r = cv2.resize(fc, (150, 150))
            r = r/255
            r = tf.expand_dims(r,axis=0)
            text = model.predict(r)
            text = tf.argmax(text,axis=1)
            text_classname = class_names[int(tf.round(text))]
            cv2.putText(fr, text_classname, (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(fr, datet, (x-10, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x + w, y + w), (0, 0, 255), 2)

        cv2.imshow('livetime face recognition', fr)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    else:
        print("error")
        break
cam.release()
cv2.destroyAllWindows()
