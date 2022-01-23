'''
calculate the distance between each pair of points
then take the max of these distances
'''

def maximalDiameter(points):
    '''
    Inputs: set of points in the figure listed as an array
    Returns the maximum distance between the pairs of points
    '''

    #set up an array that will hold distances between pairs of points
    distances = np.zeros[[1,1]]

    for i in range(points.shape):
        for j in range(i+1, points.shape-1):
            distances = np.append(distances, abs(points[i] - points[j]))

    max = np.max(distances)

    return max
