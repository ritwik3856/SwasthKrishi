from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, json
import numpy as np
from io import BytesIO
from PIL import Image  #used to read images
import tensorflow as tf

app=FastAPI()

#add and handling chors
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

Model=tf.keras.models.load_model("../saved_models/3")
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

#to check wheather our server is running or not
@app.get("/")  
async def ping():
    return "Hello"

def read_file_as_image(data)->np.ndarray:  #takes bytes as an input and return np.ndarray as output
    image=np.array(Image.open(BytesIO(data)))  #Image.open() converts bytes data into pillow image, np.array() converts that pillow image to numpy
    return image

#main entry point
@app.post("/predict")
async def predict(  #here function name can be different it is not necessary that it should be same
    file: UploadFile = File(...)   #after colon specifies the datatype(UploadFile) of variable(here file) 
):
    #converting image into numpy or tensor so that we can do our prediction
    image=read_file_as_image(await file.read())    #file.read() will convert data into bytes
    image_batch=np.expand_dims(image, 0)  #expanding dimension from 1D to 2D so that it can take the multiple images as input
    prediction=Model.predict(image_batch)
    predicted_class=CLASS_NAMES[np.argmax(prediction[0])]  #prediction[0] gives the 1st image of the batch
    confidence=np.max(prediction[0])

    disdata=json.load(open("C:\\Users\\KIIT\\OneDrive\\Desktop\\College\\Sem-6\\Minor Project\\CODE1\\CODE1\\potato_disease\\training\\data1.json"))

    prediction = str(predicted_class)
    # Filter the data based on the prediction
    # filtered_data = [d for d in data if d[0] == "Potato___Early_blight"]
    dissol1 = disdata[prediction][0]['solution1']
    dissol2 = disdata[prediction][0]['solution2']
    dissol3 = disdata[prediction][0]['solution3']
    # print(filtered_data)
    return {
        'class':predicted_class,
        'confidence':float(confidence),
        'dis1':dissol1,
        'dis2':dissol2,
        'dis3':dissol3,
    }
    
    pass

if __name__=="__main__":
    uvicorn.run(app, host='localhost', port=8000)