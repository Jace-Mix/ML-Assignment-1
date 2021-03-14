import random
import time
import pandas as pd
import numpy as np

# Target function for change
def distance(instance1, instance2, metric):
    if instance1 == None or instance2 == None:
        return float("inf")
    
    return_distance = 0
    A = np.array(instance1[1:])
    B = np.array(instance2[1:])
    
    if metric == 'Euclidean':
        return np.sqrt(np.sum(np.square(A - B)))
    elif metric == 'Manhattan':
        return np.sum(np.absolute(A - B))
    elif metric == 'Cosine':
        return 1 - np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
    elif metric == 'Jaccard':
        return 1 - (np.sum(np.minimum(A, B)) / np.sum(np.maximum(A, B)))

    return return_distance

def meanInstance(name, instanceList):
    numInstances = len(instanceList)
    if (numInstances == 0):
        return
    numAttributes = len(instanceList[0])
    means = [name] + [0] * (numAttributes-1)
    for instance in instanceList:
        for i in range(1, numAttributes):
            means[i] += instance[i]
    for i in range(1, numAttributes):
        means[i] /= float(numInstances)
    return tuple(means)

def assign(instance, centroids, metric):
    minDistance = distance(instance, centroids[0], metric)
    minDistanceIndex = 0
    for i in range(1, len(centroids)):
        d = distance(instance, centroids[i], metric)
        if (d < minDistance):
            minDistance = d
            minDistanceIndex = i
    return minDistanceIndex

def createEmptyListOfLists(numSubLists):
    myList = []
    for i in range(numSubLists):
        myList.append([])
    return myList

def assignAll(instances, centroids, metric):
    clusters = createEmptyListOfLists(len(centroids))
    for instance in instances:
        clusterIndex = assign(instance, centroids, metric)
        clusters[clusterIndex].append(instance)
    return clusters

def computeCentroids(clusters):
    centroids = []
    for i in range(len(clusters)):
        name = "centroid" + str(i)
        centroid = meanInstance(name, clusters[i])
        centroids.append(centroid)
    return centroids

def kmeans(instances, k, initCentroids=None, metric='Euclidean', task=1):
    result = {}
    if (initCentroids == None or len(initCentroids) < k):
        random.seed(time.time())
        centroids = random.sample(list(instances), k)
    else:
        centroids = initCentroids
    prevCentroids = []
    iteration = 0
    prev_sse = float('inf')

    while (centroids != prevCentroids or iteration > 100):
        iteration += 1
        clusters = assignAll(instances, centroids, metric)
        prevCentroids = centroids
        centroids = computeCentroids(clusters)
        withinss = computeWithinss(clusters, centroids, metric)
        
        if withinss > prev_sse:
            print('Increase in SSE detected. Terminating KMeans...')
            break
        prev_sse = withinss
        
        if (iteration == 1 and task == 1):
            print('Centroids after 1 iteration:',centroids)
    
    if task == 2:    
        print('Iterations taken:', iteration)
    result["clusters"] = clusters
    result["centroids"] = centroids
    result["withinss"] = withinss
    return result

def computeWithinss(clusters, centroids, metric):
    result = 0
    for i in range(len(centroids)):
        centroid = centroids[i]
        cluster = clusters[i]
        for instance in cluster:
            result += np.square(distance(centroid, instance, metric))
    return result

def repeatedKMeans(instances, k, n, metric='Euclidean', task=2):
    bestClustering = {}
    bestClustering['withinss'] = float('inf')
    for i in range(n):
        trialClustering = kmeans(instances, k, metric=metric, task=task)
        if trialClustering['withinss'] < bestClustering['withinss']:
            bestClustering = trialClustering
    return bestClustering

########################################
#              Task 1                  #
########################################
print()
print('################# TASK 1 #################')

instances = [
    ['X1', 3, 5],
    ['X2', 3, 4],
    ['X3', 2, 8],
    ['X4', 2, 3],
    ['X5', 6, 2],
    ['X6', 6, 4],
    ['X7', 7, 3],
    ['X8', 7, 4],
    ['X9', 8, 5],
    ['X10', 7, 6]
]

centroids1 = [['centroid0', 4,6], ['centroid1', 5,4]]
centroids2 = [['centroid0', 3,3], ['centroid1', 8,3]]
centroids3 = [['centroid0', 3,2], ['centroid1', 4,8]]
part1 = kmeans(instances, 2, initCentroids=centroids1, metric='Manhattan')
part2 = kmeans(instances, 2, initCentroids=centroids1, metric='Euclidean')
part3 = kmeans(instances, 2, initCentroids=centroids2, metric='Manhattan')
part4 = kmeans(instances, 2, initCentroids=centroids3, metric='Euclidean')
print()
print('Part 1 Centroids', part1["centroids"])
print('Part 1 Clusters', part1["clusters"])
print()
print('Part 2 Centroids', part2["centroids"])
print('Part 2 Clusters', part2["clusters"])
print()
print('Part 3 Centroids', part3["centroids"])
print('Part 3 Clusters', part3["clusters"])
print()
print('Part 4 Centroids', part4["centroids"])
print('Part 4 Clusters', part4["clusters"])
print()


########################################
#              Task 2                  #
########################################
print('################# TASK 2 #################')

data = pd.read_csv('iris.data', sep=',', header=None)

# Rearrange the target classification column to be the first column
data = data.reindex(columns=[4, 0, 1, 2, 3])
data = data.values.tolist()

print('Processing Euclidean KMeans...')
euclidean = repeatedKMeans(data, 3, 5)
print('Processing Cosine KMeans...')
cosine = repeatedKMeans(data, 3, 5, metric='Cosine')
print('Processing Jaccard KMeans...')
jacc = repeatedKMeans(data, 3, 5, metric='Jaccard')
print()
print('Euclidean SSE:', euclidean['withinss'])
print('Cosine SSE:', cosine['withinss'])
print('Jaccard SSE:', jacc['withinss'])
print()

Euclidean_count = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
Cosine_count = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
Jacc_count = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

for j in range(3):
    for i in range(len(euclidean['clusters'][j])):
        if euclidean['clusters'][j][i][0] == 'Iris-setosa':
            Euclidean_count[j][0] += 1
        elif euclidean['clusters'][j][i][0] == 'Iris-versicolor':
            Euclidean_count[j][1] += 1
        elif euclidean['clusters'][j][i][0] == 'Iris-virginica':
            Euclidean_count[j][2] += 1
            
    for i in range(len(cosine['clusters'][j])):
        if cosine['clusters'][j][i][0] == 'Iris-setosa':
            Cosine_count[j][0] += 1
        elif cosine['clusters'][j][i][0] == 'Iris-versicolor':
            Cosine_count[j][1] += 1
        elif cosine['clusters'][j][i][0] == 'Iris-virginica':
            Cosine_count[j][2] += 1
            
    for i in range(len(jacc['clusters'][j])):
        if jacc['clusters'][j][i][0] == 'Iris-setosa':
            Jacc_count[j][0] += 1
        elif jacc['clusters'][j][i][0] == 'Iris-versicolor':
            Jacc_count[j][1] += 1
        elif jacc['clusters'][j][i][0] == 'Iris-virginica':
            Jacc_count[j][2] += 1


euclidean_accuracy = cosine_accuracy = jacc_accuracy = 0.0
for i in range(3):
    euclidean_accuracy += max(Euclidean_count[i])
    cosine_accuracy += max(Cosine_count[i])
    jacc_accuracy += max(Jacc_count[i])
    
print('Euclidean accuracy: {}%'.format((euclidean_accuracy/150)*100))
print('Cosine accuracy: {}%'.format((cosine_accuracy/150)*100))
print('Jaccard accuracy: {}%'.format((jacc_accuracy/150)*100))
print()


########################################
#              Task 3                  #
########################################
print('################# TASK 3 #################')

instances1 = [
    ['Red', 4.7, 3.2],
    ['Red', 4.9, 3.1],
    ['Red', 5.0, 3.0],
    ['Red', 4.6, 2.9]
]
instances2 = [
    ['Blue', 5.9, 3.2],
    ['Blue', 6.7, 3.1],
    ['Blue', 6.0, 3.0],
    ['Blue', 6.2, 2.8]
]

max_distance = -1
max_distance_points = [[], []]
min_distance = float('inf')
min_distance_points = [[], []]
path_count = 0
path_distance = 0

for i in range(len(instances1)):
    for j in range(len(instances2)):
        d = distance(instances1[i], instances2[j], metric='Euclidean')
        if d > max_distance:
            max_distance = d
            max_distance_points[0] = instances1[i]
            max_distance_points[1] = instances2[j]
        elif d < min_distance:
            min_distance = d
            min_distance_points[0] = instances1[i]
            min_distance_points[1] = instances2[j]
        path_count += 1
        path_distance += d

print('Max distance: {} between points {} and {}'.format(max_distance, max_distance_points[0], max_distance_points[1]))
print('Min distance: {} between points {} and {}'.format(min_distance, min_distance_points[0], min_distance_points[1]))
print('Average distance:', path_distance/path_count)