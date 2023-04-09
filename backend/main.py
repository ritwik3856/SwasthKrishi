from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, json
import numpy as np
from io import BytesIO
from PIL import Image  
import tensorflow as tf

app=FastAPI()


origins = [
    "https://localhost",
    "http://localhost:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Model=tf.keras.models.load_model("../saved_models/5")

CLASS_NAMES=['Pepper__bell___Bacterial_spot',
 'Pepper__bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']      


@app.get("/")  
async def ping():
    return "Hello"

def read_file_as_image(data)->np.ndarray:  
    image=np.array(Image.open(BytesIO(data)))  
    return image

#main entry point
@app.post("/predict")
async def predict( 
    file: UploadFile = File(...)   
):
    
    image=read_file_as_image(await file.read())   
    image_batch=np.expand_dims(image, 0)  
    prediction=Model.predict(image_batch)
    predicted_class=CLASS_NAMES[np.argmax(prediction[0])]  

    disdata=json.load(open("C:\\Users\\KIIT\\OneDrive\\Desktop\\Hackfest\\Disease-detection\\training\\data1.json"))

    prediction = str(predicted_class)
    
    caused = disdata[prediction][0]["What caused it?"]
    chem_cont = disdata[prediction][0]["Chemical Control"]
    prev_meas = disdata[prediction][0]["Preventive Measures"]
    med = disdata[prediction][0]["Medicines"]
    
    return {
        'class':predicted_class,
        'caused':caused,
        'chem_cont':chem_cont,
        'prev_meas':prev_meas,
        'med':med,
    }


if __name__=="__main__":
    uvicorn.run(app, host='localhost', port=8000)