def igreen(pics):
    xaxis = np.zeros(len(pics))
    yaxis = np.zeros(len(pics))
    for i in range(len(pics)):
        xaxis[i] = float(pics[i][-7:-4])
        yaxis[i] = np.mean(cv2.imread(pics[i])[:,:,1])
    plt.plot(xaxis, yaxis, '*r')
    plt.xlabel(r'$indexpic$')
    plt.ylabel(r'$igreen$')
    return xaxis, yaxis 
