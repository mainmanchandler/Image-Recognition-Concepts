import numpy as np
import cv2

#----------------------------------------------------------------------------------------------------------------------------------------------------------
# Task 01 - Get Key Features
#----------------------------------------------------------------------------------------------------------------------------------------------------------

#https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#gac1fc3598018010880e370e2f709b4345


t01_01 = cv2.imread("A04/T01_01.png", -1)
t01_02 = cv2.imread("A04/T01_02.png", -1)
t01_03 = cv2.imread("A04/T01_03.png", -1)

# convert to grayscale for simplicity
t01_01_gray = cv2.cvtColor(t01_01, cv2.COLOR_BGR2GRAY)
t01_02_gray = cv2.cvtColor(t01_02, cv2.COLOR_BGR2GRAY)
t01_03_gray = cv2.cvtColor(t01_03, cv2.COLOR_BGR2GRAY)

def find_features(gray_scale_image):

    image_corners = cv2.cornerHarris(src=np.float32(gray_scale_image), blockSize=3, ksize=5, k=0.04)
    #print(image_corners)
    kernel_to_dilate = [[1,1,1],[1,1,1],[1,1,1]]
    kernel_to_dilate = np.array(kernel_to_dilate, np.uint8)
    dilated_image_corners = cv2.dilate(image_corners, kernel_to_dilate, iterations=2)

    #only get the most important features, higher the threshold the less points
    feature_threshold = 0.01 * dilated_image_corners.max()
    gray_scale_image[feature_threshold <= dilated_image_corners] = 0

    return gray_scale_image

t01_01_features = find_features(t01_01_gray)
t01_02_features = find_features(t01_02_gray)
t01_03_features = find_features(t01_03_gray)
cv2.imshow("t01_01_features", t01_01_features)
cv2.imshow("t01_02_features", t01_02_features)
cv2.imshow("t01_03_features", t01_03_features)
cv2.waitKey(0)
cv2.destroyAllWindows()


#----------------------------------------------------------------------------------------------------------------------------------------------------------
# Task 02 - Simple Image Manipulation
#----------------------------------------------------------------------------------------------------------------------------------------------------------

#https://docs.opencv.org/3.4/df/da0/group__photo__clone.html


t02_01 = cv2.imread("A04/T02_01.jpg")
t02_02 = cv2.imread("A04/T02_02.jpg")

def insert_image(background, photo_to_insert):

    photo_to_insert_grayscale = cv2.cvtColor(photo_to_insert, cv2.COLOR_BGR2GRAY)

    #print(background.shape)
    #print(photo_to_insert.shape)

    grayscale_mask = cv2.threshold(photo_to_insert_grayscale, 245, 255, cv2.THRESH_BINARY)[1]

    # invert the mask so the white is the object we want to dilate (dilate increase white region) and black is the background
    grayscale_mask = cv2.bitwise_not(grayscale_mask)
    #cv2.imshow("mask", grayscale_mask)
    
    # dilate the white region 
    kernel_to_dilate = [[1,1,1],[1,1,1],[1,1,1]]
    kernel_to_dilate = np.array(kernel_to_dilate, np.uint8)
    dilated_mask = cv2.dilate(grayscale_mask, kernel_to_dilate, iterations=4)
    #cv2.imshow("dilated_mask", dilated_mask)

    # the -20, +90 change the point to place right from the center of the image
    y = (background.shape[0] // 2) - 20 
    x = (background.shape[1] // 2) + 90
    point_to_place = (x, y)

    #blended_normal = cv2.convertScaleAbs(background, alpha=0.7, beta=1)
    blended_normal = cv2.seamlessClone(src=photo_to_insert, dst=background, mask=dilated_mask, p=point_to_place, flags=cv2.NORMAL_CLONE)
    #blended_mixed = cv2.seamlessClone(src=photo_to_insert, dst=background, mask=dilated_mask, p=point_to_place, flags=cv2.MIXED_CLONE)
    #blended_mixed = cv2.convertScaleAbs(blended_mixed, alpha=1.5, beta=5)
    
    return blended_normal

t02_combined = insert_image(t02_01, t02_02)
cv2.imshow("t02_combined", t02_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()



#----------------------------------------------------------------------------------------------------------------------------------------------------------
# Task 03 - Stitching
#----------------------------------------------------------------------------------------------------------------------------------------------------------

# https://docs.opencv.org/4.x/d2/d8d/classcv_1_1Stitcher.html

t03_01 = cv2.imread("A04/T03_01.png")
t03_02 = cv2.imread("A04/T03_02.png")

def combine_images(first_half, second_half):
    stitcher_object = cv2.Stitcher.create()
    _, stitched_photos = stitcher_object.stitch((first_half, second_half)) 
    return stitched_photos

t03_stitched = combine_images(t03_01, t03_02)

cv2.imshow('First Half', t03_01)
cv2.imshow('Second Half', t03_02)
cv2.imshow('Stitched Image', t03_stitched)
cv2.waitKey(0)
cv2.destroyAllWindows()


#write all to folder
#cv2.imwrite("t01_01_features.png", t01_01_features)
#cv2.imwrite("t01_02_features.png", t01_02_features)
#cv2.imwrite("t01_03_features.png", t01_03_features)
#cv2.imwrite("t02_combined.png", t02_combined)
#cv2.imwrite("t03_stitched.png", t03_stitched)