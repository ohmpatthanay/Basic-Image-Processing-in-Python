import cv2
import numpy as np
img = cv2.imread('m.jpg',1)
# originimg = cv2.imread('D:/New folder/m.jpg',1)

img = cv2.resize(img, (1100, 900))
# originimg = cv2.resize(originimg, (1100, 900))

grayscale = img[0:120,0:1100]
redgrayscale = img[120:330,0:550]
bluegreen = img[120:330,550:1100]
negative = img[330:450,0:1100]
brighter = img[450:900,0:320]
lowercontrast = img[450:900,798:1100]
gaussian = img[450:600,320:798]
sobeledgeH = img[600:750,320:798]
saltnpepper = img[750:900,320:798]


#Category 1=============================================

#Grayscale
gray = cv2.cvtColor(grayscale,cv2.COLOR_BGR2GRAY)
for i in np.arange(0,3):
    grayscale[:,:,i] = gray
img[0:120,0:1100] = grayscale

#Red channel in grayscale
for i in np.arange(0,3):
    redgrayscale[:,:,i] = redgrayscale[:,:,2]
img[120:330,0:550] = redgrayscale

#BlueGreen
bluegreen[:,:,2] = 0
img[120:330,550:1100] = bluegreen

#Category 1==============================================



#Category 2==============================================

#Negative
negative = 255-negative
img[330:450,0:1100] = negative

#Brighter
weight = 100
bright = np.ones(brighter.shape,dtype='uint8')

for i in np.arange(0,3):
    bright[:,:,i] = cv2.add(brighter[:,:,i],weight)
img[450:900,0:320] = bright

#Lowercontrast
weight = 2
lower = np.ones(lowercontrast.shape,dtype='uint8')

for i in np.arange(0,3):
    lower[:,:,i] = cv2.divide(lowercontrast[:,:,i],weight)
img[450:900,798:1100] = lower

#Category 2==============================================



#Category 3==============================================

#Guassianblur
gauss = cv2.GaussianBlur(gaussian,(9,9),0)
img[450:600, 320:798] = gauss

#Sobel edge horizontal
sobelH = cv2.Sobel(sobeledgeH,-1,1,0,ksize=3)
img[600:750, 320:798] = sobelH

#Salt-and-pepper noise
def saltpepper_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

saltnpepper = saltpepper_noise(saltnpepper,.03)
medianFiltered = cv2.medianBlur(saltnpepper,9)

img[750:900,320:798] = saltnpepper

#Category 3==============================================




# cv2.imshow('Original image',originimg)
cv2.imshow('Transformed image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

