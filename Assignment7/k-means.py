#!/usr/bin/python
from PIL import Image, ImageStat
import numpy


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

	for i in range(0, len(centroids)):
		cent = centroids[i]
		old_cent = old_centroids[i]

		if ((int(old_cent[0]) - a) <= cent[0] <= (int(old_cent[0]) + a)) and ((int(old_cent[1]) - a) <= cent[1] <= (int(old_cent[1]) + a)) and ((int(old_cent[2]) - a) <= cent[2] <= (int(old_cent[2]) + a)):
			continue
		else:
			return False

	return True

#end converged


# ======
# getMin
# ======
#
# Method used to find the closest centroid to the given pixel.
#
def getMin(pixel, centroids):
	minDist = 9999
	minIndex = 0

	for i in range(0, len(centroids)):
		d = numpy.sqrt(int((centroids[i][0] - pixel[0]))**2 + int((centroids[i][1] - pixel[1]))**2 + int((centroids[i][2] - pixel[2]))**2)
		if d < minDist:
			minDist = d
			minIndex = i

	return minIndex
#end getMin




# ============
# assignPixels
# ============
def assignPixels(centroids):
	clusters = {}
	
	for x in range(im.size[0]):
		for y in range(im.size[1]):
			minIndex = getMin(px[x, y], centroids)
			if minIndex in clusters:
				clusters[minIndex].append(px[x, y])
			else:
				clusters[minIndex] = [px[x, y]]
	return clusters

#end assignPixels



# ===============
# adjustCentroids
# ===============

def adjustCentroids(clusters):
	new_centroids = []
	for i in clusters.keys():
		if i not in clusters:
			continue
		cluster = clusters[i]
		r = 0
		g = 0
		b = 0
		for j in range(len(cluster)):
			r += cluster[j][0]
			g += cluster[j][1]
			b += cluster[j][2]
		new_centroids.append([r//len(cluster), g//len(cluster), b//len(cluster)])

        ## Write your code here
	return new_centroids

#end adjustCentroids


# ===========
# initializeKmeans
# ===========
#
# Used to initialize the k-means clustering
#
def initializeKmeans(someK):
	centroids = []
	img_width, img_height = im.size
	for _ in range(0, someK):
		i,j = numpy.random.randint(0, img_width), numpy.random.randint(0, img_height)
		centroids.append(px[i, j])	

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
	old_centroids = []
	print("Starting Assignments")
	print("===========================================")
	MAX_ITERATIONS = 20
	iteration = 0
	while not converged(centroids, old_centroids) and iteration < MAX_ITERATIONS:
		old_centroids = centroids
		clusters = assignPixels(centroids)
		centroids = adjustCentroids(clusters)
		iteration += 1
	
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
px = im.load()
initial_centroid=initializeKmeans(k_input)
result = iterateKmeans(initial_centroid)
drawWindow(result)




