import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

n_colours = 32 #Number of colours the final image will end with

image = Image.open("image3.jpg") #Load an image into Python

imgarr = np.array(image)/255 #Convert the image into numpy array of RGB values
                          #Divide by 255 for the image to be presented properly by imshow

red = []
green = []
blue = []

i = 0
j = 0

while i < len(imgarr):
    while j < len(imgarr[i]):
        red.append(imgarr[i][j][0])
        green.append(imgarr[i][j][1])
        blue.append(imgarr[i][j][2])
        j += 10
    j = 0
    i += 10
        

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(red)):
    ax.scatter(red[i] * 255, green[i] * 255, blue[i] * 255, '.', c=(red[i], green[i], blue[i]))
ax.set_xlim([0,255])
ax.set_ylim([0,255])
ax.set_zlim([0,255])
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
plt.show()

height, width, dimensions = tuple(imgarr.shape) #Finds the height, width and dimensions
                                             #Dimensions is 3, RGB

#Reshape the data to 2 dimensions for the KMeans
imgarr = np.reshape(imgarr, (height*width, dimensions))#Reshape from [[1,2,3][1,2,3]] to [1 2 3, 1 2 3]
                                                 #Single dimension of all the RGB colours in one line

#Since the dataset is so large (h*w ~10,000,000) we shuffle the dataset and choose the first 1000 to analyse
arr_sample = shuffle(imgarr)[:1000]

#Fit the data to the model
model = KMeans(n_clusters=n_colours, max_iter=500).fit(arr_sample) #Create a K-Means clustering model for the 3 dimensional (RGB) data
                                                     #Fits n_colours centers, so final image has this many colors

print("Cluster Centers")
print(model.cluster_centers_)

cluster_red = []
cluster_blue = []
cluster_green = []

for i in range(len(model.cluster_centers_)):
    cluster_red.append(model.cluster_centers_[i][0])
    cluster_green.append(model.cluster_centers_[i][1])
    cluster_blue.append(model.cluster_centers_[i][2])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(cluster_red)):
    ax.scatter(cluster_red[i] * 255, cluster_green[i] * 255, cluster_blue[i] * 255, marker=(5,1), edgecolors='r',
               c=(cluster_red[i], cluster_green[i], cluster_blue[i]), s=200, hatch='|')
for i in range(len(red)):
    ax.scatter(red[i] * 255, green[i] * 255, blue[i] * 255, '.', c=(red[i], green[i], blue[i]), alpha=0.1)
ax.set_xlim([0,255])
ax.set_ylim([0,255])
ax.set_zlim([0,255])
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
plt.show()