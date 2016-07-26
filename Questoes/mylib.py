import matplotlib
import numpy
import scipy

from PIL import Image

#Questao 02 a)
def imread(filename):
    image = Image.open(filename)
    return numpy.asarray(image)

#Questao 02 a)
#Digitar os comandos na seguinte ordem:
# from PIL import Image
# img = Image.open('test.png')
# img.show()

#Questao 02 b)
#Digitar os comandos na seguinte ordem:
# from PIL import Image
# img = Image.open('test.png').convert('L')
# img.show()

#Questao 02 c)
# from PIL import Image
# img = Image.open('test.png')
# img.resize((50,50)).show()

#Questao 03
# image.layers para imagens
def nchannels(image):
    return 1 if len(image.shape) == 2 else 3

#Questao 04
def size(image):
    vector = range(2)
    vector[0] = image.shape[0] #Altura
    vector[1] = image.shape[1] #Largura
    return vector

#Questao 05
def rgb2gray(image):
    newImage = image.copy()
    vector = size(newImage)
    for x in range(0, vector[0]):
        for y in range(0, vector[1]):
            greyColor = calcavarage(newImage[x][y][0], newImage[x][y][1], newImage[x][y][2])
            newImage[x][y][0] = greyColor
            newImage[x][y][1] = greyColor
            newImage[x][y][2] = greyColor
    return newImage

def calcavarage(pixelRed, pixelGreen, pixelBlue):
    red     = pixelRed * 0.299
    green   = pixelGreen * 0.587
    blue    = pixelBlue * 0.114
    return (red + green + blue) / 3

#testes
imageRGB = imread('zenfoneGO.jpg')
imageGrey = imread('imagemCinza.jpg')
# print(nchannels(imageRGB))
# print(nchannels(imageGrey))
#
# print(size(imageRGB))
# print(size(imageGrey))
# print(type(size(imageGrey)[0]))
# rgb2gray(imageRGB)