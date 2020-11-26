import cv2
import numpy as np
#obtenemos la imagen
img = cv2.resize(cv2.imread('./imagenes/IMG_20201125_102237.jpg'),(600,600), interpolation=cv2.INTER_CUBIC)
#img = cv2.imread('./imagenes/IMG_20201125_102237.jpg')
#kernel2 = np.ones((1,1), np.uint8)
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], np.float32)
kernel2 = np.ones((1,1), np.uint8)

img_sharpen = cv2.filter2D(img, -1, kernel)
#cambiamos el canal de color de BRG A HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_sharpen_hsv = cv2.cvtColor(img_sharpen, cv2.COLOR_BGR2HSV)
#Establecemos el rango mínimo y máximo de (Blue, Green, Red):
""" verde_bajos = np.array([ 28, 35, 20])
verde_altos = np.array([ 82, 131, 134]) """
verde_bajos = np.array([28, 60, 20])
verde_altos = np.array([100, 254, 254])

#Discriminando el fondo
mask = cv2.inRange(img_sharpen_hsv, verde_bajos, verde_altos)

#Quitando el ruido despues de haber borrado el fondo
mask = cv2.bitwise_not(mask) 
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)

mask = cv2.medianBlur(mask,11)
inv_mask = cv2.bitwise_not(mask)
cv2.imshow("mask", inv_mask)
cv2.imshow("img_sharpen_hsv", img)

k = cv2.waitKey(0) & 0xFF
if k == 27:
    cv2.destroyAllWindows()