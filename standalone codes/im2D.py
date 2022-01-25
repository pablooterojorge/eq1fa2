def im2D(img, p = 10.0):
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    retval,imgthresh = cv2.threshold(gray, p, 255, cv2.THRESH_BINARY)
    
    
    #image edging
    edges = cv2.Canny(imgthresh,254,255)
    edgesblur  = cv2.GaussianBlur(edges,(5,5),0)
    
    # plt.imshow(edgesblur)
    # plt.imshow(edges)
    
    #apply hough transform
    lines = cv2.HoughLines(edgesblur,1,np.pi/60,500)
    
    #avoid NONE array when no lines are detected
    if lines is None :
        lines = np.ones((2,2))
    
    #reshape array for calculation
    lines = np.reshape(lines,(len(lines),2))
    
    # print(lines)
    #select appropiate lines for fiber edge
    lines_selected = np.zeros((2,2))
    lines_selected[0] = lines.max(0)
    lines_selected[1] = lines.min(0)

    
    coord = np.zeros((2,4))
    

    for count, polars in enumerate(lines_selected):

        m = 1 / np.tan(polars[1]) ## slope of line
        x0, y0 = polars[0] * np.cos(polars[1]), polars[0] * np.sin(polars[1]) ## point in line
        
        x1, y1 = np.shape(img)[1], y0 + m * np.shape(img)[1]
        x2, y2 = 0, -m * polars[0] * np.cos(polars[1]) + polars[0] * np.sin(polars[1])

# represent lines overimposed on original img.
# cv2.line(img, (int(x1),int(y1)),(int(x2),int(y2)),(255,255,255),5)

        coord[count] = [int(x1), int(y1), int(x2), int(y2)]

    return coord
