import numpy as np
import matplotlib.pyplot as plt
import cv2
DIV = 512

#------------------------------------------------------------------------------------------
#
# :::: Image Interpolation ::::
#
#------------------------------------------------------------------------------------------

def resizeQuarter(image):
    
    #reduce the number of rows by half
    rescaled = shrinkWidthByHalf(image, DIV)

    #reduce, re-use, recycle
    rescaled = transpose(rescaled)
    
    #reduce the number of cols by half
    rescaled = shrinkWidthByHalf(rescaled, 2)

    #return to normal
    rescaled = transpose(rescaled)

    return rescaled

def transpose(matrix):
    result = []
    for i in range(len(matrix[0])):
        row = []
        for item in matrix:
            row.append(item[i])
        
        result.append(row)
    
    return np.array(result)

def shrinkWidthByHalf(matrix, div):
    resultShrink = []

    for i in range(len(matrix)):
        row = matrix[i]
        newRow = []
        for j in range(0, len(row) - 1, 2):
            #for every pair of values, add them together and divide by image total size to get the greyscale output
            #from 0 to 1. otherwise, no image will appear but white under testing as the image is greyscale by default(?).
            newValue = (row[j].item() + row[j+1].item())/div
            newRow.append(newValue)
        
        resultShrink.append(newRow)

    return resultShrink


#
# :::: 1. Rescale the image to 1/4 its size ::::
#
camera_man = cv2.imread("A01/cameraman.tif", -1)
assert camera_man is not None, "Could not find the image you tried to load 'camera_man'."

'''print(camera_man.shape)
cv2.imshow("camera_man", camera_man)
cv2.waitKey(0)
cv2.destroyAllWindows()'''


# (Height/ Width/ Depth)
# print(camera_man.shape)

cameraman_rescaled = resizeQuarter(camera_man)
'''print(cameraman_rescaled.shape)
print(cameraman_rescaled)
cv2.imshow("newIMg", cameraman_rescaled)
cv2.waitKey(0)
cv2.destroyAllWindows()'''


#
# :::: 2. Rescale the image to original size (4x) using interpolation ::::
#

# ::: Nearest Neighbor :::

def nearest_neighbor(matrix):
    resultNearest = []

    for i in range(len(matrix)):
        row = matrix[i]
        newRow = []
        for j in range(0, len(row)):
            newRow.append(row[j].item())
            newRow.append(row[j].item())
        resultNearest.append(newRow)
        resultNearest.append(newRow)

    return resultNearest


cameraman_nearest = nearest_neighbor(cameraman_rescaled)
cameraman_nearest = np.array(cameraman_nearest)
'''print(type(cameraman_nearest))
print(cameraman_nearest.shape)
cv2.imshow("cameraman_nearest", cameraman_nearest)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
cv2.imwrite("cameraman_nearest.tif", cameraman_nearest)


# ::: Bilinear interpolation :::
def bilinear_interpolation(matrix):

    resultBilinearWidth = []
    for i in range(len(matrix)):
        row = matrix[i]
        newRow = []
        for j in range(0, len(row) - 1, 2):
            newValue = (row[j].item() + row[j+1].item())/2
            newRow.append(row[j].item())
            newRow.append(newValue)
            newRow.append(newValue)
            newRow.append(row[j+1].item())
            
        
        resultBilinearWidth.append(newRow)

    resultBilinear = []
    for i in range(0, len(resultBilinearWidth), 2):
        above = resultBilinearWidth[i]
        below = resultBilinearWidth[i+1]
        newRow = []
        for j in range(0, len(above)):
            newValue = (above[j] + below[j])/2
            newRow.append(newValue)
        
        resultBilinear.append(above)
        resultBilinear.append(newRow)
        resultBilinear.append(newRow)
        resultBilinear.append(below)
        

    resultBilinear = np.array(resultBilinear)
    #print(resultBilinear.shape)
    return resultBilinear


cameraman_bilinear = bilinear_interpolation(cameraman_rescaled)
"""print(type(cameraman_bilinear))
print(cameraman_bilinear.shape)
cv2.imshow("cameraman_bilinear", cameraman_bilinear)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

cv2.imwrite("cameraman_bilinear.tif", cameraman_bilinear)

# ::: Bicubic interpolation :::

def bicubic_interpolation(matrix):
    


    return


cameraman_bicubic = bicubic_interpolation(cameraman_rescaled)



#------------------------------------------------------------------------------------------
#
# :::: Point Operations ::::
#
#------------------------------------------------------------------------------------------

#
# :::: 1. Find the negative of the image and store the output  ::::
#

def negative(matrix):

    resultNegative = []

    for i in range(len(matrix)):
        row = matrix[i]
        newRow = []
        for j in range(len(row)):
            newValue = float(256-1-row[j].item())/DIV
            newRow.append(newValue)
        resultNegative.append(newRow)


    return np.array(resultNegative)

cameraman_negative = negative(camera_man)
'''print(cameraman_negative)
print(cameraman_negative.shape)

cv2.imshow("cameraman_negative", cameraman_negative)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("cameraman_negative.tif", cameraman_negative)'''


#
# :::: 2. Apply Power-law transformation on the image  ::::
#

def powerlaw_transformation(matrix):

    resultPowerLaw = []

    for i in range(len(matrix)):
        row = matrix[i]
        newRow = []
        for j in range(len(row)):
            newValue = float((row[j].item())/DIV)**0.58
            newRow.append(newValue)

        resultPowerLaw.append(newRow)

    return np.array(resultPowerLaw)

cameraman_power = powerlaw_transformation(camera_man)
'''print(cameraman_power)
print(cameraman_power.shape)

cv2.imshow("cameraman_power", cameraman_power)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("cameraman_power.tif", cameraman_power)'''

#
# :::: 3. Apply contrast stretching on the image and store the output  ::::
#

def contrast_stretching(matrix):
    
    resultContrastStretch = []

    for i in range(len(matrix)):
        row = matrix[i]
        newRow = []
        for j in range(len(row)):
            newValue = (row[j].item()-np.min(matrix))/(np.max(matrix)-np.min(matrix))
            newRow.append(newValue)

        resultContrastStretch.append(newRow)
    
    return np.array(resultContrastStretch)

cameraman_contrast = powerlaw_transformation(camera_man)
'''print(cameraman_contrast)
print(cameraman_contrast.shape)

cv2.imshow("camera_man",  camera_man)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("cameraman_contrast", cameraman_contrast)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("cameraman_contrast.tif", cameraman_contrast)'''



#------------------------------------------------------------------------------------------
#
# :::: Histogram Processing ::::
#
#------------------------------------------------------------------------------------------

#
# :::: 1. Apply histogram equalization on the "Einstein" image and store the output  ::::
#


def histogram_equalization(matrix):

    # Create a histogram from the image, put every value in bins ranging 1-256
    imageHistogram = np.zeros(shape=(256, 1))
    imageShape = matrix.shape

    #print(range(imageShape[0]))
    #print(range(imageShape[1]))
    for i in range(imageShape[0]):
        for j in range(imageShape[1]):
            currGreyValue = matrix[i, j]
            #print(currGreyValue)
            imageHistogram[currGreyValue, 0] = imageHistogram[currGreyValue, 0] + 1

    #plt.plot(imageHistogram)
    #plt.show()

    # Create the array for the equalized histogram and grab totals of gray level bin counts from original
    pixelCountInBin = imageHistogram.reshape(1, 256)
    equalizedHistogram = np.array([])
    equalizedHistogram = np.append(equalizedHistogram, pixelCountInBin[0, 0]) 
    #print(pixelCountInBin)
    #print(equalizedHistogram)

    for i in range(255):
        runningTotal = pixelCountInBin[0, i + 1] + equalizedHistogram[i]
        equalizedHistogram = np.append(equalizedHistogram, runningTotal)
    
    #print(imageShape)
    M = imageShape[0]
    N = imageShape[1]

    # Normalize all the values (P(r_k) = bin values normalized) P(r_k) = n_k / M*N
    # Multiply resulting CDF by (L-1) (all values should already be running total so PDF = CDF off the bat)
    # Round all of the values to the nearest int
    equalizedHistogram = np.round((equalizedHistogram / (M * N)) * (256 - 1))
    #print(equalizedHistogram)
    plt.plot(equalizedHistogram)
    plt.show()

    # Apply the equalized histogram onto the original image
    for i in range(M):
        for j in range(N):
            currGreyValue = matrix[i, j]
            matrix[i, j] = equalizedHistogram[currGreyValue]
    
    return matrix


einstein = cv2.imread("A01/einstein.tif", -1)
assert einstein is not None, "Could not find the image you tried to load 'einstein'."

'''cv2.imshow("einstein", einstein)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

#Einstein_equalized = histogram_equalization(einstein)
'''cv2.imshow("Einstein_equalized", Einstein_equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("Einstein_equalized.tif", Einstein_equalized)'''

#
# :::: 2. Apply histogram specification on "chest_x-ray1" iomage so it matches the histogram for "chest_x-ray2" and store the output ::::
#


def histogram_specification(matrix1, matrix2):

    resultHistogramSpec = []
    
    #Get the equalized Original and Target image
    #originalEqualized = histogram_equalization(matrix1)
    #targetEqualized = histogram_equalization(matrix2)

    # Create a histogram from the original image, put every value in bins ranging 1-256
    imageHistogram = np.zeros(shape=(256, 1))
    image1Shape = matrix1.shape

    for i in range(image1Shape[0]):
        for j in range(image1Shape[1]):
            currGreyValue = matrix1[i, j]
            imageHistogram[currGreyValue, 0] = imageHistogram[currGreyValue, 0] + 1

    #plt.plot(imageHistogram)
    #plt.show()

    pixelCountInBin = imageHistogram.reshape(1, 256)
    equalizedImage1 = np.array([])
    equalizedImage1 = np.append(equalizedImage1, pixelCountInBin[0, 0]) 

    for i in range(255):
        runningTotal = pixelCountInBin[0, i + 1] + equalizedImage1[i]
        equalizedImage1 = np.append(equalizedImage1, runningTotal)
    
    M = image1Shape[0]
    N = image1Shape[1]

    equalizedImage1 = np.round((equalizedImage1 / (M * N)) * (256 - 1))


    # Create a histogram from the target image, put every value in bins ranging 1-256
    imageHistogram = np.zeros(shape=(256, 1))
    image2Shape = matrix2.shape

    for i in range(image2Shape[0]):
        for j in range(image2Shape[1]):
            currGreyValue = matrix2[i, j]
            imageHistogram[currGreyValue, 0] = imageHistogram[currGreyValue, 0] + 1

    #plt.plot(imageHistogram)
    #plt.show()

    pixelCountInBin = imageHistogram.reshape(1, 256)
    equalizedImage2 = np.array([])
    equalizedImage2 = np.append(equalizedImage2, pixelCountInBin[0, 0]) 


    for i in range(255):
        runningTotal = pixelCountInBin[0, i + 1] + equalizedImage2[i]
        equalizedImage2 = np.append(equalizedImage2, runningTotal)
    
    M = image2Shape[0]
    N = image2Shape[1]

    equalizedImage2 = np.round((equalizedImage2 / (M * N)) * (256 - 1))

    #plt.plot(equalizedImage1)
    #plt.show()
    #plt.plot(equalizedImage2)
    #plt.show()
    
    #print(image1Shape[0])
    #print(image1Shape[1])
    #print(matrix1.shape)
    #Now map the values of the original to target image (must use the size of image 1)
    for i in range(image1Shape[0]):
        for j in range(image1Shape[1]):
           """ currGreyValue = matrix1[i, j]
            originalEqualizedValue = equalizedImage1[currGreyValue]
            matrix1[i, j] = equalizedImage2[int(originalEqualizedValue)]"""
           currGreyValue = matrix1[i, j]
           matrix1[i, j] = equalizedImage1[currGreyValue]
    
    for i in range(image1Shape[0]):
        for j in range(image1Shape[1]):
           """ currGreyValue = matrix1[i, j]
            originalEqualizedValue = equalizedImage1[currGreyValue]
            matrix1[i, j] = equalizedImage2[int(originalEqualizedValue)]"""
           currGreyValue = matrix1[i, j]
           matrix1[i, j] = equalizedImage2[currGreyValue]

    plt.plot(equalizedImage2)
    plt.show()
 
    return matrix1


chest_xray1 = cv2.imread("A01/chest_x-ray1.jpeg", -1)
chest_xray2 = cv2.imread("A01/chest_x-ray2.jpeg", -1)
assert chest_xray1 is not None, "Could not find the image you tried to load 'chest_xray1'."
assert chest_xray2 is not None, "Could not find the image you tried to load 'chest_xray2'."

cv2.imshow("chest_xray1", chest_xray1)
cv2.imshow("chest_xray2", chest_xray2)

chest_xray3 = histogram_specification(chest_xray1, chest_xray2)

cv2.imshow("chest_xray3", chest_xray3)
cv2.waitKey(0)
cv2.destroyAllWindows()