import numpy as np
import cv2

eyeball1 = cv2.imread("A03/1.bmp", -1)
eyeball2 = cv2.imread("A03/2.bmp", -1)
eyeball3 = cv2.imread("A03/3.bmp", -1)
eyeball4 = cv2.imread("A03/4.bmp", -1)
eyeball5 = cv2.imread("A03/5.bmp", -1)
assert eyeball1 is not None, "Could not find the eyeball1 image."
assert eyeball2 is not None, "Could not find the eyeball2 image."
assert eyeball3 is not None, "Could not find the eyeball3 image."
assert eyeball4 is not None, "Could not find the eyeball4 image."
assert eyeball5 is not None, "Could not find the eyeball5 image."

#
#   Circle Detection and Results Compilation
#

def find_my_eyes(eyeball_image):

    #gaussian blurring and regular blur seemed to end up in no iris detection at all, medianblur takes the median in a set of pixels
    eyeball_image_preprocessed = cv2.medianBlur(eyeball_image, 5) #in the opencv documentation it says its good for salt and pepper noise

    #find the pupil right away, no other preprocessing needs to be done on the image for this to be successful
    pupil_circles = cv2.HoughCircles(image=eyeball_image_preprocessed, method=cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=170, param2=30, minRadius=0, maxRadius=0)

    # Preprocessing for iris detection:
    kernel = np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
    eyeball_image_preprocessed = cv2.erode(eyeball_image_preprocessed, kernel, iterations = 5)
    eyeball_image_preprocessed = cv2.dilate(eyeball_image_preprocessed, kernel, iterations = 3)
    
    #alpha is the scaling factor, beta is the delta -> setting these will change my image contrast 
    eyeball_image_preprocessed = cv2.convertScaleAbs(eyeball_image_preprocessed, alpha=1.8, beta=-70)

    #demonstrate the canny detection occuring in the hough circles
    canny_edge_detection = cv2.Canny(eyeball_image_preprocessed, threshold1=45, threshold2=55) #upper threshold is that of the hough circles param1 (represents the same map)
    iris_circles = cv2.HoughCircles(image=eyeball_image_preprocessed, method=cv2.HOUGH_GRADIENT, dp=1, minDist=400, param1=55, param2=30, minRadius=40, maxRadius=0)
    
    pupil_circles = np.round(pupil_circles)
    pupil_circles = np.uint8(pupil_circles)
    pupil_circles = pupil_circles[0] # for some reason the houghcircles array is nested.
    iris_circles = np.round(iris_circles)
    iris_circles = np.uint8(iris_circles)
    iris_circles = iris_circles[0]

    #print("circle: ", circles)
    for i in pupil_circles:
        cv2.circle(eyeball_image, (i[0], i[1]), i[2], (0,0,0), 2)
        #cv2.circle(eyeball_image, (i[0], i[1]), 2, (0,0,0), 2) #center dot

    #print("iris_circle: ", iris_circles)
    for i in iris_circles:
        cv2.circle(eyeball_image, (i[0], i[1]), i[2], (0,0,0), 2)
        #cv2.circle(eyeball_image, (i[0], i[1]), 2, (0,0,0), 2) #center dot

    return eyeball_image, canny_edge_detection


# Call and get the output:
eyeball1_circles, eyeball1_canny = find_my_eyes(eyeball1)
eyeball2_circles, eyeball2_canny = find_my_eyes(eyeball2)
eyeball3_circles, eyeball3_canny = find_my_eyes(eyeball3)
eyeball4_circles, eyeball4_canny = find_my_eyes(eyeball4)
eyeball5_circles, eyeball5_canny = find_my_eyes(eyeball5)

cv2.imshow("eyeball1", eyeball1_circles)
cv2.imshow("eyeball2", eyeball2_circles)
cv2.imshow("eyeball3", eyeball3_circles)
cv2.imshow("eyeball4", eyeball4_circles)
cv2.imshow("eyeball5", eyeball5_circles)
cv2.imshow("eyeball1_canny", eyeball1_canny)
cv2.imshow("eyeball2_canny", eyeball2_canny)
cv2.imshow("eyeball3_canny", eyeball3_canny)
cv2.imshow("eyeball4_canny", eyeball4_canny)
cv2.imshow("eyeball5_canny", eyeball5_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("1_segmented.bmp", eyeball1_circles)
cv2.imwrite("2_segmented.bmp", eyeball2_circles)
cv2.imwrite("3_segmented.bmp", eyeball3_circles)
cv2.imwrite("4_segmented.bmp", eyeball4_circles)
cv2.imwrite("5_segmented.bmp", eyeball5_circles)
cv2.imwrite("1_edge.bmp", eyeball1_canny)
cv2.imwrite("2_edge.bmp", eyeball2_canny)
cv2.imwrite("3_edge.bmp", eyeball3_canny)
cv2.imwrite("4_edge.bmp", eyeball4_canny)
cv2.imwrite("5_edge.bmp", eyeball5_canny)
