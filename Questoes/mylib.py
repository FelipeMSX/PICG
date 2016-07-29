import matplotlib
import numpy
import scipy
import matplotlib.pyplot as plt

from PIL import Image

MAX = 255
MIN = 0


# Questao 02 OK!
def imread(filename):
    image = Image.open(filename)
    return numpy.asarray(image)


# Questao 02 a)
# Digitar os comandos na seguinte ordem:
# from PIL import Image
# img = Image.open('test.png')
# img.show()

# Questao 02 b)
# Digitar os comandos na seguinte ordem:
# from PIL import Image
# img = Image.open('test.png').convert('L')
# img.show()

# Questao 02 c)
# from PIL import Image
# img = Image.open('test.png')
# img.resize((50,50)).show()

# Questao 03 OK!!!
# image.layers para imagens
def nchannels(image):
    return 1 if len(image.shape) == 2 else 3


# precisa ser melhorada passivel de bugs
def isgray(image):
    return nchannels(image) == 1


# precisa ser melhorada passivel de bugs
def isrgb(image):
    return nchannels(image) == 3


# Questao 04 OK!!!
def size(image):
    vector = [0] * 2
    vector[0] = image.shape[0]  # Altura
    vector[1] = image.shape[1]  # Largura
    return vector


# Questao 05 OK!!!
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
    red = pixelRed * 0.299
    green = pixelGreen * 0.587
    blue = pixelBlue * 0.114
    return (red + green + blue) / 3


# Questao 06 OK!!
def imreadgray(filename):
    img = imread(filename)
    return rgb2gray(img) if isrgb(img) else img


# Questo 07
# --------------------- Incompelta, nao consigo entender o que o professor quer.
def imshow(image):
    Image.fromarray(image).show()
    return


# Questao 08 OK!!!
def thresh(image, limit):
    newImage = image.copy()
    vector = size(newImage)
    if isgray(image):
        for x in range(0, vector[0]):
            for y in range(0, vector[1]):
                newImage[x][y] = MAX if (newImage[x][y] >= limit) else MIN
    else:
        for x in range(0, vector[0]):
            for y in range(0, vector[1]):
                newImage[x][y][0] = MAX if (newImage[x][y][0] >= limit) else MIN
                newImage[x][y][1] = MAX if (newImage[x][y][1] >= limit) else MIN
                newImage[x][y][2] = MAX if (newImage[x][y][2] >= limit) else MIN
    return newImage


# Questao 09 OK!!!
def negative(image):
    newImage = image.copy()
    vector = size(newImage)
    if isgray(image):
        for x in range(0, vector[0]):
            for y in range(0, vector[1]):
                newImage[x][y] = MAX - newImage[x][y]
    else:
        for x in range(0, vector[0]):
            for y in range(0, vector[1]):
                newImage[x][y][0] = MAX - (newImage[x][y][0])
                newImage[x][y][1] = MAX - (newImage[x][y][1])
                newImage[x][y][2] = MAX - (newImage[x][y][2])
    return newImage


# Questao 10 OK depende da questao 07 q estou com duvida.
def contrast(r, m, f):
    return r(f - m) + m


# Questao 11 OK!!!
def hist(image):
    imgSize = size(image)

    if isgray(image):
        count = [0] * 256
        for x in range(0, imgSize[0]):
            for y in range(0, imgSize[1]):
                count[image[x][y]] += 1
        return count
    else:
        count = [[0] * 256, [0] * 256, [0] * 256]
        for x in range(0, imgSize[0]):
            for y in range(0, imgSize[1]):
                count[0][image[x][y][0]] += 1
                count[1][image[x][y][1]] += 1
                count[2][image[x][y][2]] += 1
        return count


# Questao 12 OK!!!
def showhist(hist):
    N = 256
    ind = numpy.arange(N)  # the x locations for the groups
    width = 1  # the width of the bars
    fig, ax = plt.subplots()
    if len(hist) == 256: #indica que 'e' cinza.
        greyrect = ax.bar(ind, hist, width, color='w')

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Quantidade')
        ax.set_title('Pixels por Intensidade em escala de cinza')
    else:

        redrect = ax.bar(ind, hist[0], width, color='r')
        greenrect = ax.bar(ind, hist[1], width, color='g')
        bluerect = ax.bar(ind, hist[2], width, color='b')

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Quantidade')
        ax.set_title('Pixels por Intensidade')
        ax.legend((redrect[0], greenrect[0], bluerect[0]), ('RED', 'GREEN', 'BLUE'))

    plt.show()
    return


# testes----------------------------------------------------
imageRGB = imread('zenfoneGO.jpg')
imageGrey = imread('zenfoneGOGrey.jpg')
energiaRG = imread('Energia.jpg')
energiaGrey = imread('EnergiaCinza.jpg')
# print(nchannels(imageRGB))
# print(nchannels(imageGrey))
#
# print(size(imageRGB))
# print(size(imageGrey))
# print(type(size(imageGrey)[0]))
# rgb2gray(imageRGB)

# Questao 08
# Image.fromarray(thresh(imageRGB,150)).show()

# Questao 09
# Image.fromarray(negative(imageGrey)).show()

# Questao 10
# v = hist(energiaRG)
# v = hist(energiaGrey)
# v = hist(energiaGrey)

# Questao 11
# showhist(hist(energiaGrey))
# showhist(hist(imageRGB))
