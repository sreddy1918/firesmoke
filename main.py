# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:27:00 2024

@author: sredd
"""
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = FastAPI()

# Load the pre-trained model
model = tf.keras.models.load_model("fire_smoke_model.h5")

classes = ['default', 'fire', 'smoke']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Preprocess image
    image = image.resize((250, 250))  # Adjust according to your model's input shape
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make prediction
    prediction = model.predict(image)
    predicted_class = classes[np.argmax(prediction)]

    return {"prediction": predicted_class}
    
    

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:7000
if __name__ == '__main__':
     uvicorn.run(app, host='127.0.0.1', port=7000)
    
#uvicorn main:app --reload