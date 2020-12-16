import concurrent.futures
import time
import math
import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import argparse
import sys

# img : RBG Color
def toGrayScale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Calculating Cluster Centers
def cluster_center(m, N, U, j):
    '''
    :param m : Fuzzifier
    :param N : Image array
    :param U : memberShip array
    :param j : Target Cluster Index
    '''
    num = 0
    den = 0
    for i, Uij in enumerate(U[j]):
        num += N[i] * Uij
        den += Uij
    return num/den

# Calculating Membership degree for pixel
def membership_degree(m, C, N, i, j):
    '''
    :param m : Fuzzifier
    :param C : Cluster Center Vector
    :param N : Image Array
    :param i : Point i in Image Array
    :param j : Index of Targeted Cluster
    '''
    fraction = 0
    for Ck in C:
        num = euclidean(N[i], C[j])
        den = euclidean(N[i], Ck)
        if(num == 0 or den == 0):
            fraction += 0
        else:
            fraction +=  math.pow(num/den, 2/(m-1))
    if fraction == 0:
        return 0
    return 1/fraction

# Calculates membership for chunk
def distributed_membership(img_flattend, C, fuzzifier, j):
    '''
    :params img_flattend: 1D data points
    :params C: Cluster Center Vectors
    :params fuzzifier: fuzzifier
    :params j: 1D data points
    '''
    # Hashmap for improved performance
    U = np.empty((img_flattend.shape[0]))
    color_membership_hashmap = dict()
    for i in range(img_flattend.shape[0]):
        value = color_membership_hashmap.get(f'{img_flattend[i]}')
        if(value):
            U[i] = value
        else:
            U[i] = membership_degree(fuzzifier, C, img_flattend, i, j)
            color_membership_hashmap[f'{img_flattend[i]}'] = U[i]
    return U

# Returns Distance between two clusters
def helper_get_distance_clusters(c1, c2):
    D = np.array([])
    for p1 in c1:
        for p2 in c2:
            dist = euclidean(p1,p2)
            D = np.append(D, dist)
    return np.mean(D)

# Automatic Cluster Center Vectors Extraction
def get_clusters_auto(img, peeks_minimum_height, peeks_minimum_width, centers_minimum_distance_treshold):

    print("Creating Channel Histograms...")

    img_flat = img.reshape((-1, 3))

    # Extracting Color Channels
    img_r = img[:,:,0].flatten()
    img_g = img[:,:,1].flatten()
    img_b = img[:,:,2].flatten()

    # Getting uniq values and their count
    uniq_r, counts_r = np.unique(img_r, return_counts=True)
    uniq_g, counts_g = np.unique(img_g, return_counts=True)
    uniq_b, counts_b = np.unique(img_b, return_counts=True)

    # Creating a map of (value, count)
    u_r = dict(zip(uniq_r, counts_r))
    u_g = dict(zip(uniq_g, counts_g))
    u_b = dict(zip(uniq_b, counts_b))

    # Initializing channel Histograms with 0
    hist_r = np.zeros((256))
    hist_g = np.zeros((256))
    hist_b = np.zeros((256))

    # Filling Histograms
    for (i, j, k) in zip(u_r, u_g, u_b):
        hist_r[i] = u_r[i]
        hist_g[j] = u_g[j]
        hist_b[k] = u_b[k]

    # Identifying Peeks
    print("Extracting Local Peeks...")
    peaks_r, _ = find_peaks(hist_r, height=peeks_minimum_height, width=peeks_minimum_width)
    peaks_g, _ = find_peaks(hist_g, height=peeks_minimum_height, width=peeks_minimum_width)
    peaks_b, _ = find_peaks(hist_b, height=peeks_minimum_height, width=peeks_minimum_width)

    # Forming All Possible Cluster Centers
    print("Forming Possible Cluster Centers...")
    cluster_centers = []
    for r in peaks_r:
        for g in peaks_g:
            for b in peaks_b:
                cluster_centers.append([r, g, b])
    cluster_centers = np.array(cluster_centers)

    # Eliminating non optimal Clusters
    print("Eliminating non optimal Clusters Centers...")
    optimal = False
    while(not optimal):
        scores = np.zeros(len(cluster_centers))
        for point in img_flat:
            distances = []
            for center in cluster_centers:
                distance = euclidean(center, point)
                distances.append(distance)
            scores[np.argmin(distances)] += 1

        
        coords = np.argwhere(scores < 0.008*img_flat.shape[0])
        if len(coords) == 0:
            optimal = True
            break
        for coord in coords:
            cluster_centers = np.delete(cluster_centers, coord, axis=0)

    # Merging Close centers
    print("Merging Close Cluster Centers...")
    optimal = False
    while(not optimal):
        nb_centers = cluster_centers.shape[0]
        if(nb_centers <= 2):
            optimal=True
            break
        for i_c1 in range(nb_centers-1):
            distances = []
            for i_c2 in range(i_c1+1, nb_centers):
                distance = int(euclidean(cluster_centers[i_c1], cluster_centers[i_c2]))
                distances.append(distance)

            arg_min = np.argmin(distances)
            min_dist = min(distances)

            if min_dist <= centers_minimum_distance_treshold:
                mean = np.mean(np.array([cluster_centers[i_c1], cluster_centers[arg_min+1]]), axis=0).astype(int)
                cluster_centers = np.delete(cluster_centers, i_c1, axis=0)
                cluster_centers = np.delete(cluster_centers, arg_min, axis=0)
                cluster_centers = np.append(cluster_centers, [mean], axis=0)
                break

            if i_c1 == nb_centers-2:
                optimal = True
    return cluster_centers

# Algorithm Variable Declaration
parser = argparse.ArgumentParser(description='HTFCM Params')
parser.add_argument('image_dir', type=str, nargs='?', help='image path')
parser.add_argument('-output_prefix', type=str, nargs='?', default='', help='result images prefix')
parser.add_argument('-output_dir', type=str, nargs='?', default='.', help='result images output folder eg: ./res')
parser.add_argument('-fuzzifier', type=int, nargs='?', default=2, help='Fuzzifier')
parser.add_argument('-termination_criteria', type=int, nargs='?', default=0.1, help='Termination Criteria')
parser.add_argument('-scale_percent', type=int, nargs='?', default=10, help='Scale image to percentage of original image')
parser.add_argument('-peeks_minimum_height', type=int, nargs='?', default=20, help='minimum histogram peek heights')
parser.add_argument('-peeks_minimum_width', type=int, nargs='?', default=5, help='minimum histogram peek widths')
parser.add_argument('-centers_minimum_distance_treshold', type=int, nargs='?', default=30, help='minimum distance between centers')
parser.add_argument('-save', type=bool, nargs='?', default=False, help='Store result images')
parser.add_argument('-plot', type=bool, nargs='?', default=False, help='Plot result cluster points')
args = parser.parse_args()

if(len(sys.argv) < 2):
    print("Not Enough Arguments")
    exit(-1)

#########################################################

# Algorithm Start
start_time = time.time()

# Reading image
img = cv2.imread(args.image_dir) 
original_dim =  (img.shape[0], img.shape[1])

# Resizing image (performance)
width = int(img.shape[1] * args.scale_percent / 100)
height = int(img.shape[0] * args.scale_percent / 100)
dim = (width, height)
img = cv2.resize(img, dim)  

# Log Parameters
print("********")
print(f'fuzzifier: {args.fuzzifier}')
print(f'Termination Criteria: {args.termination_criteria}')
print(f'Image Scaling Percentage: {args.scale_percent}%')
print(f'Original image Dimensions: {original_dim[0]}x{original_dim[1]}')
print(f'Resized image Dimensions: {img.shape[0]}x{img.shape[1]}')
print("********")
print("Initialization...")

# Flattening image matrix (2D -> 1D)
img_flattend = np.reshape(img, (img.shape[0]*img.shape[1], 3)) 

# Unique image colors
uniq_colors = np.unique(img_flattend, axis=1)

# C cluster center vector
print("Center Vector Initialization...")
C = get_clusters_auto(img, args.peeks_minimum_height, args.peeks_minimum_width, args.centers_minimum_distance_treshold)
k_clusters = len(C)
print(f'# of Clusters: {k_clusters}')
Co = np.copy(C)

# U membership array initialization
print("Membership matrix initialization...")
U = np.empty((k_clusters, img_flattend.shape[0]))

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    for j in range(U.shape[0]):
        future = executor.submit(distributed_membership, img_flattend, C, args.fuzzifier, j)
        return_value = future.result()
        U[j,:] = return_value
        

print("Minimization")
iterations = 0
while(True):

    Uk = np.copy(U)

    print('Updating Center Vectors...')
    # Center Vector Calculation
    for j in range(len(C)):
        C[j] = cluster_center(args.fuzzifier, img_flattend, U, j)

    print('Updating Membership matrix...')
    # Membership Matrix Update
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for j in range(U.shape[0]):
            future = executor.submit(distributed_membership, img_flattend, C, args.fuzzifier, j)
            return_value = future.result()
            U[j,:] = return_value

    # Verifying Termination Criteria
    distance = np.linalg.norm(np.subtract(U, Uk))
    print(f'||Uk+1 - Uk|| : {distance:.2f}')
    iterations += 1
    if distance < args.termination_criteria:
        break

# Showing centroids
print(f'Centroids : {C}')
print(f'Distance to final : {helper_get_distance_clusters(C, Co)}')
print(f'Iterations : {iterations}')

# Store image
if args.save:
    # Showing Results as images
    for i in range(k_clusters):
        # Adding Cluster Centroid Color
        cluster = np.full_like(img_flattend, C[i])
        # Reshaping into image form
        cluster = np.reshape(cluster, img.shape)
        # Adding Alpha Layer
        cluster = cv2.cvtColor(cluster, cv2.COLOR_RGB2RGBA)
        # Adding Alpha layer Values
        alpha_layer = U[i]*255
        alpha_layer = alpha_layer.astype(int)
        alpha_layer = np.reshape(alpha_layer, (cluster.shape[0], cluster.shape[1]))
        cluster[:, :, 3] = alpha_layer
        # Resizing to original dimensions
        cluster = cv2.resize(cluster, original_dim)
        #cv2.imshow(f'Cluster {i}', cluster)
        cv2.imwrite(f'{args.output_dir}/{args.output_prefix}_cluster{i}.png', cluster)

    img_upscaled = cv2.resize(img, original_dim)
    #cv2.imshow("Original", img)
    cv2.imwrite(f'{args.output_dir}/{args.output_prefix}_original.png', img_upscaled)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

# Plot data
if args.plot:
    # Showing Results as Scatter 
    for cluster_index in range(k_clusters):
        # Separating RGB Image Values + Adding Cluster Centroid
        r = np.append(C[cluster_index][0], img_flattend[:, 0])
        g = np.append(C[cluster_index][1], img_flattend[:, 1])
        b = np.append(C[cluster_index][2], img_flattend[:, 2])
        # Alpha Values
        color_values = np.append(1, U[cluster_index])
        # Cluster Centroid Color [0, 1]
        cluster_color = C[cluster_index]/255
        # Initializating Color Vector With Cluster centroid color (alpha = 1)
        colors = np.full((r.shape[0], 4), np.append(cluster_color, 1))
        # Adding Alpha Values
        colors[:, 3] = color_values
        # Adding centroid Color (Red)
        colors[0] = [1, 0, 0, 1]
        # Ploting points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(r, g, b, c = colors)
        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.set_zlabel('B')
        plt.show()



print("--- %s seconds ---" % (time.time() - start_time))