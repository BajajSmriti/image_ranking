from urllib import request
from PIL import Image
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import os


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


def calculate_distances(image_test, clusters, centroids):
    model = get_extract_model()
    test_vector = pre_process_img(model, image_test)
    df_vectors = clusters
    distance = np.linalg.norm(np.array(centroids[centroids.columns[0:4096]]) - test_vector, axis=1)

    # cluster index that represents the min distance with the test vector
    min_cluster = list(distance).index(np.min(distance))

    df_vectors = df_vectors[df_vectors["cluster"] == min_cluster]

    distance = np.linalg.norm(np.array(df_vectors[df_vectors.columns[0:4096]]) - test_vector, axis=1)
    df_vectors['distance'] = pd.Series(distance, index=df_vectors.index)
    df_vectors['rank'] = df_vectors['distance'].rank(ascending=1)
    df_vectors = df_vectors.set_index('rank')
    df_vectors = df_vectors.sort_index()

    result = df_vectors[0:100]
    cat = result['Path'].iloc[0].split("\\")[1]
    content_compare = []
    for content in result['Path']:
        if content.split("\\")[1] == cat:
            content_compare.append(True)
        else:
            content_compare.append(False)
    correct_result = content_compare.count(True)
    precision = correct_result / len(content_compare)
    return precision


def test_ranking(data_folder, clusters, centroids, model):
    ps = []
    # Save query image
    for folder_name in os.listdir(data_folder):
        full_path = os.path.join(data_folder, folder_name)
        for im_path in os.listdir(full_path):
            full_im_path = os.path.join(data_folder, folder_name, im_path)
            ps.append(calculate_distances(full_im_path, clusters, centroids))
        print(folder_name, " precision: " + str(sum(ps)/len(ps)))


if __name__ == "__main__":
    data_folder = "./static/img/test_set"
    clusters = pd.read_csv('./static/clusters.csv')
    centroids = pd.read_csv('./static/centroids.csv')
    model = get_extract_model()
    test_ranking(data_folder, clusters, centroids, model)
