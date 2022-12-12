import os
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import pandas as pd

# pre-processing images and extracting feature vectors
def pre_process_img(model, image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_tensor = preprocess_input(x)
    vector = model.predict(img_tensor)[0]
    vector = vector / np.linalg.norm(vector)
    return vector


def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.get_layer("fc1").output)
    return extract_model


data_folder = "./static/img/train_set"
model = get_extract_model()

vectors = []
paths = []


# Parsing through images and extracting feature vectors
for folder_name in os.listdir(data_folder):
    full_path = os.path.join(data_folder, folder_name)
    for im_path in os.listdir(full_path):
        full_im_path = os.path.join(data_folder, folder_name, im_path)
        im_vector = pre_process_img(model, full_im_path)
        vectors.append(im_vector)
        paths.append(full_im_path)

# Dataframe to store image vectors and paths of images
df = pd.DataFrame(np.array(vectors))
df['Path'] = pd.Series(paths, index=df.index)

df.to_csv("./static/im_features.csv", index=False)
