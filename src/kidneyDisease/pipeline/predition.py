import tensorflow as tf
import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image 
import os 


class PredictionPipline:
    def __init__(self,filename) -> None:
     self.filenamae=filename
    


    def prediction(self):
       model = load_model(os.path.join("artifacts","training","model.h5"))


       Image_Data = self.filenamae
       test_image = image.load_img(Image_Data,targetsize = (244,244))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image,axis=0)
       result = np.argmax(model.predict(test_image),axis=1)
       print(result)



       if result[0]== 1:
          prediction = "Tumor affected CT-SCAN"
          return prediction
       else:
          prediction= "Normal "
          return










    
    

    


    
    
    
     