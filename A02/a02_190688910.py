import numpy as np
import cv2
import math
import random

DIV = 255

#---------------------------------------------------------------------------------------------------------------------------------------------------
#
# :::: Task 1 ::::
#
#---------------------------------------------------------------------------------------------------------------------------------------------------

#Implement from scratch the convolution operations for:

camera_man = cv2.imread("A02/cameraman.tif", -1)
assert camera_man is not None, "Could not find the cameraman image."
cv2.imshow("Original Cameraman", camera_man)
#print(camera_man.shape)

####################################################################################################################################################
# a. An averaging smoothing filter (filter size: 3*3)                                                                                              #
####################################################################################################################################################

def average_smoothing_filter(matrix):
    
    resultAverage = []
    #print(matrix.shape)
    m, n = matrix.shape

    for i in range(len(matrix)):
        row = matrix[i]
        newRow = []
        if i < 2 or i >= m-2:
            resultAverage.append(row/DIV)
        else:
            for j in range(0, len(row)):
                
                if j < 2 or j >= n-2:
                    newRow.append(matrix[i, j]/DIV)
                else:
                    #"simply the average of the pixels contained in a neighborhood"
                    # so I want to recreate this:
                    # f(x-1, y-1) f(x-1, y) f(x-1, y+1)
                    # f(x, y-1)   f(x,y)    f(x, y+1)
                    # f(x+1,y-1)  f(x+1,y)  f(x+1,y+1)
                    average = (( matrix[i-1, j-1]/9 + matrix[i-1, j]/9 + matrix[i-1, j+1]/9 + 
                                 matrix[i, j-1]/9   + matrix[i, j]/9   + matrix[i, j+1]/9   + 
                                 matrix[i+1, j-1]/9 + matrix[i+1, j]/9 + matrix[i+1, j+1]/9 ) ) /DIV
                    
                    #print(average)

                    newRow.append(average)

            resultAverage.append(newRow)

    #print(resultAverage)

    return np.array(resultAverage)

avg_smooth_cameraman = average_smoothing_filter(np.array(camera_man))
#print(avg_smooth_cameraman.shape)
cv2.imshow("T1a Smoothed Cameraman", avg_smooth_cameraman)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite("t1a.tif", avg_smooth_cameraman)



####################################################################################################################################################
# b. A Gaussian smoothing filter (filter size: 7*7, sigma = 1, mean = 0)                                                                           #
####################################################################################################################################################
camera_man = cv2.imread("A02/cameraman.tif", -1)
assert camera_man is not None, "Could not find the cameraman image."

def create_gaussian_value(m, n):
    gValue = np.exp(-(m**2 + n**2) / (2 * 1**2)) #where sigma = 1
    return gValue

def create_gaussian_filter(m, n):
    gaussianFilter = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            gaussianFilter[i, j] = create_gaussian_value(i-3, j-3)

    #normalize the filter to sum to 1
    gaussianFilter *= 1/np.sum(gaussianFilter)
    
    return gaussianFilter

def gaussian_smoothing(matrix, gaussian_filter):
    
    resultAverage = []
    #print(matrix.shape)
    m, n = matrix.shape

    for i in range(len(matrix)):
        row = matrix[i]
        newRow = []
        if i < 7 or i >= m-7:
            resultAverage.append(row/DIV)
        else:
            for j in range(0, len(row)):
                
                if j < 7 or j >= n-7:
                    newRow.append(matrix[i, j]/DIV)
                
                else:
                    # apply the filter to each 7*7 region to find the new pixel value
                    value = [[ matrix[i-3, j-3] , matrix[i-3, j-2] , matrix[i-3, j-1] , matrix[i-3, j] , matrix[i-3, j+1] , matrix[i-3, j+2] , matrix[i-3, j+3] ],
                            [  matrix[i-2, j-3] , matrix[i-2, j-2] , matrix[i-2, j-1] , matrix[i-2, j] , matrix[i-2, j+1] , matrix[i-2, j+2] , matrix[i-2, j+3] ],
                            [  matrix[i-1, j-3] , matrix[i-1, j-2] , matrix[i-1, j-1] , matrix[i-1, j] , matrix[i-1, j+1] , matrix[i-1, j+2] , matrix[i-1, j+3] ],
                            [  matrix[i, j-3]   , matrix[i, j-2]   , matrix[i, j-1]   , matrix[i, j]   , matrix[i, j+1]   , matrix[i, j+2]   , matrix[i, j+3]   ],
                            [  matrix[i+1, j-3] , matrix[i+1, j-2] , matrix[i-3, j-1] , matrix[i-3, j] , matrix[i-3, j+1] , matrix[i-3, j+2] , matrix[i-3, j+3] ],
                            [  matrix[i+2, j-3] , matrix[i+2, j-2] , matrix[i-3, j-1] , matrix[i-3, j] , matrix[i-3, j+1] , matrix[i-3, j+2] , matrix[i-3, j+3] ],
                            [  matrix[i+3, j-3] , matrix[i+3, j-2] , matrix[i-3, j-1] , matrix[i-3, j] , matrix[i-3, j+1] , matrix[i-3, j+2] , matrix[i-3, j+3] ]]
                    
                    #print(gaussian_filter)
                    value *= gaussian_filter
                    newRow.append(np.sum(value)/DIV)
            
            resultAverage.append(newRow)

    #print(resultAverage)
    return np.array(resultAverage)

gaussian_filter = create_gaussian_filter(7, 7) # 7*7
gaussian_smooth_cameraman = gaussian_smoothing(np.array(camera_man), gaussian_filter)
#print(gaussian_smooth_cameraman.shape)
cv2.imshow("T1b Guassian Cameraman", gaussian_smooth_cameraman)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite("t1b.tif", gaussian_smooth_cameraman)

####################################################################################################################################################
# c.The sobel sharpening filter (filter size: 3*3, both horizontal and vertical)                                                                   #
####################################################################################################################################################

camera_man = cv2.imread("A02/cameraman.tif", -1)
assert camera_man is not None, "Could not find the cameraman image."

def sobel_sharpening(matrix):

    resultSobel = []
    #print(matrix.shape)
    m, n = matrix.shape

    for i in range(len(matrix)):
        row = matrix[i]
        newRow = []
        if i < 2 or i >= m-2:
            resultSobel.append(row/DIV)
        else:
            for j in range(0, len(row)):
                
                if j < 2 or j >= n-2:
                    newRow.append(matrix[i, j]/DIV)
                else:
                    horizontalValue = (( matrix[i-1, j-1]*-1  + matrix[i-1, j]*0 + matrix[i-1, j+1]*1 + 
                                          matrix[i, j-1]*-2    + matrix[i, j]*0   + matrix[i, j+1]*2   + 
                                          matrix[i+1, j-1]*-1  + matrix[i+1, j]*0 + matrix[i+1, j+1]*1 ) ) /DIV
                    
                    verticalValue = (( matrix[i-1, j-1]*-1  + matrix[i-1, j]*0 + matrix[i-1, j+1]*1 + 
                                          matrix[i, j-1]*-2    + matrix[i, j]*0   + matrix[i, j+1]*2   + 
                                          matrix[i+1, j-1]*-1  + matrix[i+1, j]*0 + matrix[i+1, j+1]*1 ) ) /DIV
                    #print(average)

                    # magnitude of a gradient = sqrt(Gx**2, Gy**2), this combines the vertical and horizontal and removes any negatives
                    sobelCombined = math.sqrt(horizontalValue**2 + verticalValue**2) 

                    newRow.append(sobelCombined)
            

            resultSobel.append(newRow)

    #print(resultAverage)

    return np.array(resultSobel)



sobel_cameraman = sobel_sharpening(np.array(camera_man))
cv2.imshow("T1c Sobel Cameraman", sobel_cameraman)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("t1c.tif", sobel_cameraman)


#---------------------------------------------------------------------------------------------------------------------------------------------------
#
# :::: Task 2 ::::
#
#---------------------------------------------------------------------------------------------------------------------------------------------------

####################################################################################################################################################
# a. An averaging smoothing filter (filter size: 3*3)                                                                                              #
####################################################################################################################################################

# https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
camera_man = cv2.imread("A02/cameraman.tif", -1)
assert camera_man is not None, "Could not find the cameraman image."

avg_blur_cameraman = cv2.blur(camera_man, ksize=(3, 3))

cv2.imshow("T2a Average BuiltIn Cameraman", avg_blur_cameraman)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite("t2a.tif", avg_blur_cameraman)


####################################################################################################################################################
# b. A Gaussian smoothing filter (filter size: 7*7, sigma = 1, mean = 0)                                                                           #
####################################################################################################################################################

camera_man = cv2.imread("A02/cameraman.tif", -1)
assert camera_man is not None, "Could not find the cameraman image."

guassian_blur_cameraman = cv2.GaussianBlur(camera_man, ksize=(7, 7), sigmaX=1)

cv2.imshow("T2b Gaussian BuiltIn Cameraman", guassian_blur_cameraman)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite("t2b.tif", guassian_blur_cameraman)

####################################################################################################################################################
# c.The sobel sharpening filter (filter size: 3*3, both horizontal and vertical)                                                                   #
####################################################################################################################################################

#https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d
#https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga3460e9c9f37b563ab9dd550c4d8c4e7d

camera_man = cv2.imread("A02/cameraman.tif", -1)
assert camera_man is not None, "Could not find the cameraman image."

horizontal = sobel_filter_cameraman = cv2.Sobel(camera_man, ddepth=cv2.CV_64F, dx=1, dy=0, dst=None, ksize=3)
vertical = sobel_filter_cameraman = cv2.Sobel(camera_man, ddepth=cv2.CV_64F, dx=0, dy=1, dst=None, ksize=3)
#removes the negatives, essentially the same as sqrt(Gx**2, Gy**2) where |Gx| + |Gy|
combineX = cv2.convertScaleAbs(src=horizontal)
combineY = cv2.convertScaleAbs(src=vertical)
sobel_filter_cameraman = cv2.addWeighted(src1=combineX, alpha=0.5, src2=combineY, beta=0.5, gamma=0)

cv2.imshow("T2c Sobel BuiltIn Cameraman", sobel_filter_cameraman)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("t2c.tif", sobel_filter_cameraman)


#---------------------------------------------------------------------------------------------------------------------------------------------------
#
# :::: Task 3 ::::
#
#----------------------------------------------------------------------------------------------------------------------------------------------------

####################################################################################################################################################
# a. Marr-Hildreth Edge Detector                                                                                                                   #
####################################################################################################################################################

camera_man = cv2.imread("A02/cameraman.tif", -1)
assert camera_man is not None, "Could not find the cameraman image."

# Algorithm:
#"1. Smooth image by Gaussian filter"
gaussian_filter = cv2.GaussianBlur(camera_man, ksize=(7, 7), sigmaX=1)

#"2. Apply Laplacian to smoothed image"
laplacian_of_gaussian = cv2.Laplacian(gaussian_filter, cv2.CV_64F)

#"3. Find zero crossing"
# Normalize into a 255 range
get_absolute_laplacian = cv2.convertScaleAbs(src=laplacian_of_gaussian)
min, max, _, _ = cv2.minMaxLoc(get_absolute_laplacian)
zero_crossing = ((255 * (get_absolute_laplacian - min)) / (max - min)) / DIV

# Apply the threshold to get the final result, the more you increase the threshold, the more exaggerated the edges
marr_Hildreth_edge_detection = cv2.threshold(np.uint8(zero_crossing * DIV), 40, 255, type=cv2.THRESH_BINARY)[1]

#cv2.imshow("Zero_crossings BuiltIn Cameraman", zero_crossing)
kernel_to_dilate = [[1,1,1],[1,1,1],[1,1,1]]
kernel_to_dilate = np.array(kernel_to_dilate, np.uint8)
dilated_edges_of_Hildreth = cv2.dilate(marr_Hildreth_edge_detection, kernel_to_dilate, iterations=1)

cv2.imshow("T3a Marr-Hildreth BuiltIn Cameraman", marr_Hildreth_edge_detection)
cv2.imshow("T3a+ Marr-Hildreth Dilated BuiltIn Cameraman", dilated_edges_of_Hildreth)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite("t3a.tif", marr_Hildreth_edge_detection)

####################################################################################################################################################
# b. Canny Edge Dectector                                                                                                                          #
####################################################################################################################################################

camera_man = cv2.imread("A02/cameraman.tif", -1)
assert camera_man is not None, "Could not find the cameraman image."

#https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
#threshold 1 and threshold 2 decides which of the edges are "really" edges (openCV)

canny_edge_detection = cv2.Canny(camera_man, threshold1=120, threshold2=180)

kernel_to_dilate = [[1,1,1],[1,1,1],[1,1,1]]
kernel_to_dilate = np.array(kernel_to_dilate, np.uint8)
dilated_edges_of_Canny = cv2.dilate(canny_edge_detection, kernel_to_dilate, iterations=1)

cv2.imshow("T3b Canny BuiltIn Cameraman", canny_edge_detection)
cv2.imshow("T3b+ Canny Dilated BuiltIn Cameraman", dilated_edges_of_Canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("t3b.tif", canny_edge_detection)



#----------------------------------------------------------------------------------------------------------------------------------------------------
#
# :::: Task 4 ::::
#
#----------------------------------------------------------------------------------------------------------------------------------------------------


# Implement the algorithm to group adjacent pixels in an edge map, using 4 neighbourhood

def group_adjacent(edge_map):
    label_edge_map = []
    edge_map = edge_map.tolist()
    #convert every value in the array to a tuple
    for list in edge_map:
        newRow = []
        for value in list:
            # 1 label is for the unlabelled, 0 is for black (edges)
            if value == 255:
                newRow.append((value, 0))
            else:
                newRow.append((value, 1))

        label_edge_map.append(newRow)

    #print(label_edge_map[255])

    currentLabel = 2
    queue = []

    #1, -1 to avoid collisions (outside the array) right now
    for x in range(1, len(label_edge_map)-1):
        for y in range(1, len(label_edge_map)-1):
            if x == 1 and y == 1:
                queue.append((x, y, currentLabel))
                currentLabel += 1
            
            #if it equals 1 then its unlabeled and we can do the things
            elif label_edge_map[x][y][1] == 1:
                queue.append((x, y, currentLabel))
                currentLabel += 1

            while(queue):
                #pop out an element from the queue and look at its neighbors. 
                i, j, currentLabel = queue.pop()

                above = None
                below = None
                left = None
                right = None
                
                #get above, right, left, below
                if i - 1 > 0 and j - 1 > 0 and i < len(label_edge_map)-1 and j < len(label_edge_map)-1:
                    above = label_edge_map[i - 1][j]
                    left = label_edge_map[i][j-1]
                    below = label_edge_map[i+1][j]
                    right = label_edge_map[i][j+1]    
                elif i - 1 <= 0 and j < len(label_edge_map)-1:
                    left = label_edge_map[i][j-1]
                    below = label_edge_map[i+1][j]
                    right = label_edge_map[i][j+1] 
                elif j - 1 <= 0 and i < len(label_edge_map)-1:
                    above = label_edge_map[i - 1][j]
                    below = label_edge_map[i+1][j]
                    right = label_edge_map[i][j+1] 
                elif i - 1 <= 0:
                    left = label_edge_map[i][j-1]
                    below = label_edge_map[i+1][j]
                elif j - 1 <= 0:
                    above = label_edge_map[i - 1][j]
                    right = label_edge_map[i][j+1] 


                #If a neighbor is a foreground pixel and not already labeled give it the curlab label and add it to the queue
                #repeat 3 until there are no more elements in queue
                #note: update the edge_map_labeled
                if above != None:
                    value, label = above
                    if label == 1:
                        label_edge_map[i - 1][j] = (value, currentLabel)
                        queue.append((i-1, j, currentLabel))

                if below != None:
                    value, label = below
                    if label == 1:
                        label_edge_map[i+1][j] = (value, currentLabel)
                        queue.append((i+1, j, currentLabel))

                if right != None:
                    #print(right)
                    value, label = right
                    if label == 1:
                        label_edge_map[i][j+1] = (value, currentLabel)
                        queue.append((i, j+1, currentLabel))

                if left != None:
                    value, label = left
                    if label == 1:
                        label_edge_map[i][j-1] = (value, currentLabel)
                        queue.append((i, j-1, currentLabel))

                #(2) if the pixel is a 0 then give it a new label and add it to the queue.
                #if it already has a label then igore and more on
                #if its a 255 value then ignore and move on

                #4 go to (2) for the next pixel in the image and update labelCount


                currentLabel+=1
    
    #now we have a list of the image containing the value at the pixels and the group flag
    #go through all of the labels from 2 to currentLabel and set a different colour to each label
    label_to_greyscale = {}
    final_label_edge_map = label_edge_map.copy
    for labelNum in range(1, currentLabel):
        label_to_greyscale[labelNum] = random.randint(0, 255)

    #print(label_to_greyscale)
    
    final_label_edge_map = []
    for list in label_edge_map:
        newRow = []
        for tuple in list:
            _, label = tuple
            #print(label)
            if label != 0:
                newRow.append(label_to_greyscale[label])
            else:
                newRow.append(255)
        final_label_edge_map.append(newRow)
    final_label_edge_map = np.array(final_label_edge_map)


    return np.uint8(final_label_edge_map)


canny_grouped = group_adjacent(canny_edge_detection)
canny_dilated_grouped = group_adjacent(dilated_edges_of_Canny)
marr_Hildreth_grouped = group_adjacent(marr_Hildreth_edge_detection)
marr_Hildreth_Dilated_grouped = group_adjacent(dilated_edges_of_Hildreth)


cv2.imshow("T4b Canny Grouped Cameraman", canny_grouped)
cv2.imshow("T4b+ Canny Dilated Grouped Cameraman", canny_dilated_grouped)
cv2.imshow("T4a Marr-Hildreth Grouped Cameraman", marr_Hildreth_grouped)
cv2.imshow("T4a+ Marr-Hildreth Grouped Cameraman", marr_Hildreth_Dilated_grouped)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("t4b.tif", marr_Hildreth_grouped)
cv2.imwrite("t4a.tif", canny_grouped)


