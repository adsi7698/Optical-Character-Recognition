from PIL import Image
import cv2
from decimal import getcontext, Decimal
import pickle
import network
import numpy as np

def convert_image():

    image = Image.open('z.jpg').convert('L')
    new_image = image.resize((28, 28), Image.ANTIALIAS)
    quality_val = 100
    new_image.save('img_28.jpg', quality=quality_val)

    gray_img = cv2.imread('img_28.jpg', cv2.IMREAD_GRAYSCALE)

    
    """Removing noise from images"""
    for i in range(28):
	    for j in range(28):
		    if gray_img[i, j] < 70:
			    gray_img[i, j] = 0
		    elif gray_img[i, j] >= 70 and gray_img[i, j] <= 100:
			    gray_img[i, j] = 85
		    elif gray_img[i, j] >= 101 and gray_img[i, j] <= 130:
			    gray_img[i, j] = 115
		    elif gray_img[i, j] >= 131 and gray_img[i, j] <= 160:
			    gray_img[i, j] = 145
		    else:
			    gray_img[i, j] = 255

    input_data = np.ndarray(shape=(784,1))
    getcontext().prec = 1

    for i in range(28):
	    for j in range(28):
		    input_data[i*28+j] = (round(float(255.0-gray_img[i, j])/(255.0), 1))

    return input_data
