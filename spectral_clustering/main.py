from urllib import request
from PIL import Image
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from flask import Flask, request, render_template


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

    # df_vect = df_vectors[df_vectors["cluster"] == min_cluster]

    # distance = np.linalg.norm(np.array(df_vect[df_vect.columns[0:4096]]) - search_vector, axis=1)
    # df_vect['distance'] = pd.Series(distance, index=df_vect.index)
    # df_vect['rank'] = df_vect['distance'].rank(ascending=1)
    # df_vect = df_vect.set_index('rank')
    # df_vect = df_vect.sort_index()

    result = df_vectors[0:100]
    cat = result['Path'].iloc[0].split("\\")[1]
    content_compare = []
    for content in result['Path']:
        if content.split("\\")[1] == cat:
            content_compare.append(True)
        else:
            content_compare.append(False)
    result['Content_compare'] = pd.Series(content_compare, index=result.index)
    correct_result = content_compare.count(True)
    precision = correct_result / len(content_compare)
    print('Precision:', precision)
    return result, precision


clusters = pd.read_csv('./static/clusters.csv')
centroids = pd.read_csv('./static/centroids.csv')

# build web Flask
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        # Save query image
        img = Image.open(file)  # PIL image
        uploaded_img_path = "static/uploaded/" + file.filename
        img.save(uploaded_img_path)
        result, ps = calculate_distances(uploaded_img_path, clusters, centroids)
        rs = result[['Path', 'Content_compare']]
        rs = rs.to_records(index=False)
        rs = list(rs)
        precision = "Precision: " + str(ps)
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=rs,
                               precision=precision)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=9080, debug=True)
