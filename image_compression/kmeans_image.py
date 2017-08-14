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

print(model.cluster_centers_)

#Replace each point with it's nearest cluster center
predictions = model.predict(imgarr) #For each pixel in the original image, label it with [0], [1] ... [n_colours]
                                    #indicating that the closest color cluster to the pixel is cluster_centers_[label[x]]

new_image = np.zeros((height*width, dimensions)) #Create a new image with the same dimensions as the old
for i in range(len(new_image)):
    new_image[i] = model.cluster_centers_[predictions[i]] #Replace the pixels in the new image with the compressed old

#Reshape the image
new_image = np.reshape(new_image, (height, width, dimensions)) #Reshape to get into the old dimensions

#Recreate the new image using matplotlib
plt.imshow(new_image)
plt.show()

im = Image.fromarray((new_image * 255).astype(np.uint8))
im.save("image3_compressed.jpeg")
