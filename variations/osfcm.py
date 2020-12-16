import cv2
import numpy as np
from scipy.spatial.distance import euclidean
import math
import matplotlib.pyplot as plt
import concurrent.futures
import time
import argparse
import sys

# img : RBG Color
def toGrayScale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Calculates Center Vectors
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

# Calculates Membership for pixel
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

# Calculates Membership for image chunk
def distributed_membership(img_flattend, C, fuzzifier, j):
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

# Returns euclidean distance between 2d arrays
def helper_get_distance_clusters(c1, c2):
    D = np.array([])
    for p1 in c1:
        for p2 in c2:
            dist = euclidean(p1,p2)
            D = np.append(D, dist)
    return np.mean(D)

# Center Vector Initialization 
def ordering_split(points, k_clusters):
    # Computing m
    m = []
    for xkj in points:
        mk = 1/3 * np.sum(xkj)
        m.append(mk)

    # Applying ordering function
    m.sort()

    # Uniform splitting
    l = []
    for i in range(k_clusters+1):
        li = i * math.floor(len(points)/k_clusters)
        l.append(li)

    # Building the subsets
    V = []
    for i in range(1, k_clusters+1):
        Si = [li for li in range(l[i-1]+1, l[i]) ]
        Ci = sorted(Si, reverse=True)
        Vi = 0
        for j in Ci:
            Vi += points[j]*1/len(Ci)
        V.append(Vi.astype(int))

    return np.array(V)


# Algorithm Variable Declaration
parser = argparse.ArgumentParser(description='OSFCM Params')
parser.add_argument('image_dir', type=str, nargs='?', help='image path')
parser.add_argument('k_clusters', type=int, nargs='?', help='number of clusters')
parser.add_argument('-output_dir', type=str, nargs='?', default='.', help='result images output folder eg: ./res')
parser.add_argument('-output_prefix', type=str, nargs='?', default='', help='result images prefix')
parser.add_argument('-fuzzifier', type=int, nargs='?', default=2, help='Fuzzifier')
parser.add_argument('-termination_criteria', type=int, nargs='?', default=0.1, help='Termination Criteria')
parser.add_argument('-scale_percent', type=int, nargs='?', default=10, help='Scale image to percentage of original image')
parser.add_argument('-save', type=bool, nargs='?', default=False, help='Store result images')
parser.add_argument('-plot', type=bool, nargs='?', default=False, help='Plot result cluster points')
args = parser.parse_args()

if(len(sys.argv) < 2):
    print("Not Enough Arguments")
    exit(-1)

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
print(f'# of Clusters: {args.k_clusters}')
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
C = ordering_split(uniq_colors, args.k_clusters)
Co = np.copy(C)

# U membership array initialization
print("Membership matrix initialization...")
U = np.empty((args.k_clusters, img_flattend.shape[0]))

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

if args.save:
    # Showing Results as images
    for i in range(args.k_clusters):
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

if args.plot:
    # Showing Results as Scatter 
    for cluster_index in range(args.k_clusters):
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