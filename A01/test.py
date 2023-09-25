import numpy as np
import matplotlib, cv2


#open the image
camera_man = cv2.imread("cameraman.tif", -1)
assert camera_man is not None, "Could not find the image you tried to load 'camera_man'."
'''cv2.imshow("camera_man", camera_man)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

#convert the image to an np array (?) and resize the image smaller 
test = np.array(camera_man)
assert test is not None, "Could not find the image you tried to load 'camera_man'."
# print(test.shape) # (Height/ Width/ Depth)


def resizeQuarter(img):
    result = shrinkWidthByHalf(img, 512);
    result = transpose(result);
    result = np.array(result);
    result = shrinkWidthByHalf(result, 2);
    result = transpose(result);
    return np.array(result);

def transpose(matrix):
    result = []
    for i in range(len(matrix[0])):
        row = [];
        for item in matrix:
            row.append(item[i]);
        
        result.append(row);
    
    return result;

def shrinkWidthByHalf(matrix, div):
    resultShrink = []
    for i in range(len(matrix)):
        row = matrix[i]
        newRow = []
        for j in range(0, len(row) - 1, 2):
            a = (row[j].item() + row[j+1].item())/div;
            newRow.append(a);
        
        resultShrink.append(newRow);

    return resultShrink;

print(test.shape)

newImg = resizeQuarter(test)
print(newImg.shape)
print(newImg)

cv2.imshow("newIMg", newImg)
cv2.waitKey(0)
cv2.destroyAllWindows()


