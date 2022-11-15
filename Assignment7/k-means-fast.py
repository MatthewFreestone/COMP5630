#!/usr/bin/python
from PIL import Image
import numpy as np


# =========
# converged
# =========
#
# Will determine if the centroids have converged or not.
# Essentially, if the current centroids and the old centroids
# are virtually the same, then there is convergence.
#
# Absolute convergence may not be reached, due to oscillating
# centroids. So a given range has been implemented to observe
# if the comparisons are within a certain ballpark
#


def converged(centroids, old_centroids):
	if len(old_centroids) == 0:
		return False


	if len(centroids) <= 5:
		a = 1
	elif len(centroids) <= 10:
		a = 2
	else:
		a = 4
	res: np.ndarray = (centroids - a) <= old_centroids <= (centroids + a)
	return res.all()

#end converged


# ======
# getMin
# ======
#
# Method used to find the closest centroid to the given pixel.
#
def getMin(pixel: np.ndarray, centroids: np.ndarray):
	dists = np.sqrt(((pixel - centroids) ** 2).sum(axis=1))
	return np.argmin(dists)



# ============
# assignPixels
# ============
def assignPixels(centroids):
	clusters = {}
	img_width, img_height, _ = px.shape

	for x in range(img_width):
		for y in range(img_height):
			minIndex = getMin(px[x, y], centroids)
			if minIndex in clusters:
				clusters[minIndex].append(px[x, y])
			else:
				clusters[minIndex] = [px[x, y]]

	for i in clusters.keys():
		clusters[i] = np.array(clusters[i])
	return clusters

#end assignPixels



# ===============
# adjustCentroids
# ===============

def adjustCentroids(clusters):
	new_centroids = np.zeros((len(clusters), 3), dtype=np.uint8)
	i = 0
	for _, cluster in clusters.items():
		new_centroids[i] = cluster.mean(axis=0)
		i += 1
	print(new_centroids)
	return new_centroids

#end adjustCentroids


# ===========
# initializeKmeans
# ===========
#
# Used to initialize the k-means clustering
#
def initializeKmeans(someK):
	img_width, img_height, _ = px.shape
	centroids = np.zeros((someK,3), dtype=np.uint8)
	for k in range(0, someK):
		i,j = np.random.randint(0, img_width), np.random.randint(0, img_height)
		centroids[k] = px[i, j]


	print("Centroids Initialized")
	print("===========================================")

	return centroids
#end initializeKmeans

# ===========
# iterateKmeans
# ===========
#
# Used to iterate the k-means clustering
#
def iterateKmeans(centroids):
	print("Starting Assignments")
	print("===========================================")
	MAX_ITERATIONS = 20
	for i in range(MAX_ITERATIONS):
		print("Iteration: ", i)
		old_centroids = centroids

		clusters = assignPixels(centroids)
		centroids = adjustCentroids(clusters)
		print(type(centroids))
		print(type(old_centroids))
		if converged(centroids, old_centroids):
			break

	print("===========================================")
	print("Convergence Reached!")
	return centroids
#end iterateKmeans


# ==========
# drawWindow
# ==========
#
# Once the k-means clustering is finished, this method
# generates the segmented image and opens it.
#
def drawWindow(result):
	img = Image.new('RGB', (img_width, img_height), "white")
	p = img.load()

	for x in range(img.size[0]):
		for y in range(img.size[1]):
			RGB_value = result[getMin(px[x, y], result)]
			p[x, y] = tuple(RGB_value)

	img.show()

#end drawWindow



num_input = str(input("Enter image number: "))
k_input = int(input("Enter K value: "))

img = "img/test" + num_input.zfill(2) + ".jpg"
im = Image.open(img)
img_width, img_height = im.size
# px = im.load()
px = np.array(im)
initial_centroid=initializeKmeans(k_input)
result = iterateKmeans(initial_centroid)
# drawWindow(result)




