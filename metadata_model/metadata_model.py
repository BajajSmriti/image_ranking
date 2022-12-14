# -*- coding: utf-8 -*-
"""texual.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ze2uX363ZhsfrBklzvIgLPFAp78DRZ6Q

Spectral Clustering
"""

import sys
import sklearn
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

from google.colab import drive
drive.mount('/content/drive/')

#train_dir = "/content/drive/MyDrive/USML_Project/photos"

train_dir = "/content/drive/MyDrive/USML_Project/Tim Pics, Metadata, and Code/Original"

from keras.utils import image_dataset_from_directory
train_ds = image_dataset_from_directory(
  train_dir,
  shuffle=False,
  image_size=(128, 128))

class_names = train_ds.class_names
img_name = train_ds.file_paths

img_name[100]

class_names[1]

x_train = []
y_train = []

for image_batch, labels_batch in train_ds:
    for i in image_batch:
        x_train.append(i)
    for i in labels_batch:
        y_train.append(i)

x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape)
print(y_train.shape)

X = x_train.reshape(len(x_train),-1)
Y = y_train
X = X.astype(float) / 255.

print(X.shape)
print(Y.shape)

num_clusters = 10
num_samples = 160 #2000
#data = X[np.random.choice(X.shape[0], num_samples), :]
data = X

num_clusters = 10

digit = X[0,:]
digit = np.reshape(digit, (128,128,3)) #(28,28))

plt.imshow(digit, cmap='Greys')

plt.show()

import numpy.linalg as nla

# Gaussian kernel with variance sigma^2.
sigma = 100.0
def kernel(s_i, s_j):
    return np.exp(-nla.norm(s_i - s_j)**2.0 / (2.0*sigma**2.0))

# The kernel matrix of the data.
kernel_matrix = np.zeros((num_samples, num_samples))
for i in range(num_samples):
    for j in range(i, num_samples):
        kernel_matrix[i,j] = kernel(data[i,:], data[j,:])
        kernel_matrix[j,i] = kernel_matrix[i,j]
        
#plt.matshow(kernel_matrix)

degrees = np.sum(kernel_matrix, axis=0)
D = np.diag(degrees)
K = kernel_matrix

L = D - K
L
#plt.matshow(L)

import numpy.linalg as nla

(eigvals, eigvecs) = nla.eigh(L)

print('Smallest eigenvalues = {}'.format(eigvals[0:2*num_clusters]))

plt.plot(eigvals[1:], 'ro')
plt.xlabel('Eigenvalue index')
plt.ylabel('Eigenvalue')

plt.show()

from sklearn.cluster import KMeans, SpectralClustering

sigma = 500.0

# spectral clustering
sc = SpectralClustering(n_clusters=10, gamma=1.0/sigma**2.0, affinity='rbf', n_init=100, random_state=0, assign_labels='kmeans').fit(data)

skl_sc_clusters_info = []
for ind_cluster in range(10):
    skl_sc_clusters_info.append([])

for ind_point in range(num_samples):
    ind_cluster = sc.labels_[ind_point]
    skl_sc_clusters_info[ind_cluster].append(ind_point)

sc.labels_, skl_sc_clusters_info[0][0]

sc.labels_[0]

"""Texual imputs"""

img_name[100]

ids = []
tags = []
for i in range(len(img_name)):
  count = 0
  start = 0
  while count < 7:
    if img_name[i][start] == "/":
      count +=1
    start +=1
    tag = ""
  while count < 8:
    if img_name[i][start] == "/":
      count +=1
      start +=1
    else:
      tag = tag + img_name[i][start]
      start +=1

  tags.append(tag.lower())

  id = ""
  while count < 9:
    if img_name[i][start] == "_":
      count +=1
      start +=1
    else:
      id = id + img_name[i][start]
      start +=1
  
  ids.append(id)

import pandas as pd

data1 = pd.read_csv("/content/drive/MyDrive/USML_Project/Tim Pics, Metadata, and Code/tim_pics_metadata.csv")
data2 = pd.read_csv("/content/drive/MyDrive/USML_Project/Tim Pics, Metadata, and Code/tim_pics_metadata2.csv")

metadata = pd.concat([data1, data2])

id_index = {}

for i in range(len(metadata)):
  temp = str(int(metadata.iloc[i]['id']))
  if temp not in id_index:
    id_index[temp] = i

for i in range(len(ids)):
  if ids[i] not in id_index:
    print(i)

#id_index[ids[8]] = 5083
#id_index[ids[9]] = 5470
#id_index[ids[319]] = 10373 
#id_index[ids[320]] = 10374
#id_index[ids[534]] = 7995
#id_index[ids[561]] = 7958

id_index[ids[8]] = 5472
id_index[ids[9]] = 5470
id_index[ids[260]] = 5901
id_index[ids[334]] = 6155
id_index[ids[335]] = 5660
id_index[ids[1519]] = 10373
id_index[ids[1520]] = 10374
id_index[ids[1671]] = 10309
id_index[ids[1830]] = 10748
id_index[ids[1831]] = 10749
id_index[ids[1850]] = 11999
id_index[ids[1882]] = 11587
id_index[ids[2534]] = 7995
id_index[ids[2561]] = 7958
id_index[ids[2775]] = 7230
id_index[ids[2907]] = 6836
id_index[ids[2947]] = 6937

for i in range(len(ids)):
  if ids[i] not in id_index:
    print(i)

def texual(tag):
  tag = tag.lower()
  match = [0,0,0,0,0,0,0,0,0,0]
  total = [0,0,0,0,0,0,0,0,0,0]
  for i in range(len(sc.labels_)):
    total[sc.labels_[i]] += 1
    if tags[i] == tag:
      match[sc.labels_[i]] += 1
  perc = []
  for i in range(10):
    perc.append(match[i]/total[i])
  max = 0
  clust = 0
  for i in range(10):
    if match[i] > max:
      max = match[i]
      clust = i
  views = []
  imgs = []
  for i in range(len(sc.labels_)):
    if sc.labels_[i] == clust and tags[i] == tag:
      imgs.append(i)
      views.append(metadata.iloc[id_index[ids[i]]]['count_views'])
  

  most = [x for _, x in sorted(zip(views, imgs), reverse=True)]


  #most = most[:10]


  return most

def pics_with_unique_authors(tag):
  most = texual(tag)
  pics = []
  auth = {}
  i=0
  while len(pics)<10:
    if metadata.iloc[id_index[ids[most[i]]]]['owner'] not in auth:
      pics.append(most[i])
      auth[metadata.iloc[id_index[ids[most[i]]]]['owner']] = 1
    i +=1
  #for i in range(10):
    #print(metadata.iloc[id_index[ids[most[i]]]]['owner'])
  return pics

def get_top_images(tag):

  most = pics_with_unique_authors(tag)

  plt.figure(figsize=(10 * 5, 15))

  for idx in range(10):
    plt.subplot(2, 10/2, idx+1)
    digit = X[most[idx],:]
    digit = np.reshape(digit, (128,128,3)) #(28,28))
    imgplot = plt.imshow(digit)
    # plt.show()
    plt.axis('off')
  plt.suptitle("Top " + tag.title() + " Images", fontsize=80)

get_top_images("canyon")

get_top_images("castle")

get_top_images("coast")

get_top_images("fish")

get_top_images("flower")

get_top_images("horses")

get_top_images("plane")

get_top_images("skyline")

get_top_images("sunset")

get_top_images("yosemite")

"""Above shows the best images for each topic by first finding the cluster that has the highest percentage of images with the requested tag, and then returns the images with the tags and the most views. This is similar to how flickrs querying system actually works, except instead of specral clustering they use a complex neural network to cluster images. It is very difficult to quantify the performance of such a task as it is subjective what the "best" images in a certain tag are, but the results do seem to have filtered out noise images and only returned quality images. Whether they are the "most" quality images in their respective tags is subjective"""

import random
def texual_test(tag):
  random.seed(1)
  tag = tag.lower()
  temp_tag = []
  match = [0,0,0,0,0,0,0,0,0,0]
  total = [0,0,0,0,0,0,0,0,0,0]
  for i in range(len(sc.labels_)):
    choices = [0,0,0,0,0,0,0,0,0,1]
    x = random.choice(choices)
    total[sc.labels_[i]] += 1
    if tags[i] == tag:
      match[sc.labels_[i]] += 1
      temp_tag.append(tags[i])
    elif x == 1:
      temp_tag.append(tag)
      match[sc.labels_[i]] += 1
    else:
      temp_tag.append(tags[i])
  perc = []
  for i in range(10):
    perc.append(match[i]/total[i])
  max = 0
  clust = 0
  for i in range(10):
    if match[i] > max:
      max = match[i]
      clust = i
    
  views = []
  imgs = []
  for i in range(len(sc.labels_)):
    if sc.labels_[i] == clust: 
      if temp_tag[i] == tag:
        imgs.append(i)
        views.append(metadata.iloc[id_index[ids[i]]]['count_views'])

  most = [x for _, x in sorted(zip(views, imgs), reverse=True)]


  #most = most[:10]


  return most

def pics_with_unique_authors2(tag):
  most = texual_test(tag)
  pics = []
  auth = {}
  i=0
  while len(pics)<10:
    if metadata.iloc[id_index[ids[most[i]]]]['owner'] not in auth:
      pics.append(most[i])
      auth[metadata.iloc[id_index[ids[most[i]]]]['owner']] = 1
    i +=1
  #for i in range(10):
    #print(metadata.iloc[id_index[ids[most[i]]]]['owner'])
  return pics

def get_top_images2(tag):

  most = texual_test(tag)

  plt.figure(figsize=(10 * 5, 15))

  for idx in range(10):
    plt.subplot(2, 10/2, idx+1)
    digit = X[most[idx],:]
    digit = np.reshape(digit, (128,128,3)) #(28,28))
    imgplot = plt.imshow(digit)
    # plt.show()
    plt.axis('off')
  plt.suptitle("Top " + tag.title() + " Images w/ Noise", fontsize=80)

  count = 0
  for i in range(10):
    if tags[most[i]] != tag:
      count+=1
  print((10-count)/10)

get_top_images2("canyon")

get_top_images2("castle")

get_top_images2("coast")

get_top_images2("fish")

get_top_images2("flower")

get_top_images2("horses")

get_top_images2("plane")

get_top_images2("skyline")

get_top_images2("sunset")

get_top_images2("yosemite")

def avg_views(tag):
  count = 0
  sum = 0
  for i in range(len(tags)):
    if tags[i]==tag:
      sum += metadata.iloc[id_index[ids[i]]]['count_views']
      count +=1
  print("Average Views for", tag +":", (sum/count))

x = ["canyon", "castle", "coast", "fish", "flower", "horses", "plane", "skyline", "sunset", "yosemite"]

for i in x:
  avg_views(i)

import numpy as np
def perc_views(tag, perc):
  views = []
  for i in range(len(tags)):
    if tags[i]==tag:
      views.append(metadata.iloc[id_index[ids[i]]]['count_views'])
  print(str(perc)+"th percentile Views for", tag +":", np.percentile(views, perc))

x = ["canyon", "castle", "coast", "fish", "flower", "horses", "plane", "skyline", "sunset", "yosemite"]

for i in x:
  perc_views(i, 90)

avg_view = {}
avg_view["canyon"] = 131
avg_view["castle"] = 662
avg_view["coast"] = 923
avg_view["fish"] = 767
avg_view["flower"] = 507
avg_view["horses"] = 331
avg_view["plane"] = 713
avg_view["skyline"] = 784
avg_view["sunset"] = 982
avg_view["yosemite"] = 620

perc_view = {}
perc_view["canyon"] = 200
perc_view["castle"] = 1391
perc_view["coast"] = 2387
perc_view["fish"] = 1731
perc_view["flower"] = 1298
perc_view["horses"] = 517
perc_view["plane"] = 1254
perc_view["skyline"] = 1711
perc_view["sunset"] = 2037
perc_view["yosemite"] = 847

import random
def texual_test2(tag):
  random.seed(1)
  tag = tag.lower()
  temp_tag = []
  match = [0,0,0,0,0,0,0,0,0,0]
  total = [0,0,0,0,0,0,0,0,0,0]
  for i in range(len(sc.labels_)):
    choices = [0,0,0,0,0,0,0,0,0,1]
    x = random.choice(choices)
    total[sc.labels_[i]] += 1
    if tags[i] == tag:
      match[sc.labels_[i]] += 1
      temp_tag.append(tags[i])
    elif x == 1:
      temp_tag.append(tag)
      match[sc.labels_[i]] += 1
    else:
      temp_tag.append(tags[i])
  perc = []
  for i in range(10):
    perc.append(match[i]/total[i])
  max = 0
  clust = 0
  for i in range(10):
    if match[i] > max:
      max = match[i]
      clust = i
    
  views = []
  imgs = []
  for i in range(len(sc.labels_)):
    if sc.labels_[i] == clust: 
      if temp_tag[i] == tag:
        imgs.append(i)
        if tags[i] == tag:
          views.append(metadata.iloc[id_index[ids[i]]]['count_views'])
        else:
          views.append(perc_view[tag])

  most = [x for _, x in sorted(zip(views, imgs), reverse=True)]


  #most = most[:10]


  return most
def pics_with_unique_authors3(tag):
  most = texual_test2(tag)
  pics = []
  auth = {}
  i=0
  while len(pics)<10:
    if metadata.iloc[id_index[ids[most[i]]]]['owner'] not in auth:
      pics.append(most[i])
      auth[metadata.iloc[id_index[ids[most[i]]]]['owner']] = 1
    i +=1
  #for i in range(10):
    #print(metadata.iloc[id_index[ids[most[i]]]]['owner'])
  return pics

def get_top_images3(tag):

  most = pics_with_unique_authors3(tag)

  plt.figure(figsize=(10 * 5, 15))

  for idx in range(10):
    plt.subplot(2, 10/2, idx+1)
    digit = X[most[idx],:]
    digit = np.reshape(digit, (128,128,3)) #(28,28))
    imgplot = plt.imshow(digit)
    # plt.show()
    plt.axis('off')
  plt.suptitle("Top " + tag.title() + " Images w/ Noise of adj. views", fontsize=80)

  count = 0
  for i in range(10):
    if tags[most[i]] != tag:
      count+=1
  print((10-count)/10)

get_top_images3("canyon")

get_top_images3("castle")

get_top_images3("coast")

get_top_images3("fish")

get_top_images3("flower")

get_top_images3("horses")

get_top_images3("plane")

get_top_images3("skyline")

get_top_images3("sunset")

get_top_images3("yosemite")
