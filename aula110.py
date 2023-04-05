# importe a biblioteca opencv
import cv2
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model('keras_model.h5')
  
# defina um objeto de captura de vídeo
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture o vídeo quadro a quadro
    ret, frame = vid.read()
    img1=cv2.resize(frame,(0, 0), fx = 0.1,fy = 0.1)

    img=cv2.resize(frame,(224,224))

    timg=np.array(img,dtype=np.float32)
    timg=np.expand_dims(timg,axis=0)

    normimg=img/255.0
    prediction=model.predict(normimg)
    print("previsao",prediction)
    # Exiba o quadro resultante
    cv2.imshow('quadro', frame)
      
    # Saia da tela com a barra de espaço
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# Após o loop, libere o objeto capturado
vid.release()

# Destrua todas as janelas
cv2.destroyAllWindows()