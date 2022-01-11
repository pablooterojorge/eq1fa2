def scale2one(array_to_one):
    array_to_one = np.zeros(len(array_to_one))
    for i in range(len(array_to_one)):
        array_to_one[i] = (array_to_one[i] - np.amin(array_to_one))/(np.amax(array_to_one)-np.amin(array_to_one))
    return array_to_one
