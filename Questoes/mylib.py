import matplotlib
import numpy
import scipy
import matplotlib.pyplot as plt
import math
from types import *

from PIL import Image

MAXCAPACITY = 256
MINCAPACITY = 0
MININTENSITY = 0
MAXINTENSITY = 255

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


# Se a imagem que foi usada para usar o hist for gray retorna true.
def ishistgray(hist):
    return len(hist) == 256


# Se a intensidade do pixel for maior ou menor que o permitido o ajusto.
def truncate(value):
    v = int(value)
    if   v > MAXINTENSITY:
        return MAXINTENSITY
    elif v < MININTENSITY:
        return MININTENSITY

    return v


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
                newImage[x][y] = MAXINTENSITY if (newImage[x][y] >= limit) else MININTENSITY
    else:
        for x in range(0, vector[0]):
            for y in range(0, vector[1]):
                newImage[x][y][0] = MAXINTENSITY if (newImage[x][y][0] >= limit) else MININTENSITY
                newImage[x][y][1] = MAXINTENSITY if (newImage[x][y][1] >= limit) else MININTENSITY
                newImage[x][y][2] = MAXINTENSITY if (newImage[x][y][2] >= limit) else MININTENSITY
    return newImage


# Questao 09 OK!!!
def negative(image):
    newImage = image.copy()
    vector = size(newImage)
    if isgray(image):
        for x in range(0, vector[0]):
            for y in range(0, vector[1]):
                newImage[x][y] = MAXINTENSITY - newImage[x][y]
    else:
        for x in range(0, vector[0]):
            for y in range(0, vector[1]):
                newImage[x][y][0] = MAXINTENSITY - (newImage[x][y][0])
                newImage[x][y][1] = MAXINTENSITY - (newImage[x][y][1])
                newImage[x][y][2] = MAXINTENSITY - (newImage[x][y][2])
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
# Questao 13 OK!!!
def showhist(hist, bin = 1):

    length      = int(calchistlength(hist, bin))
    xvalues     = numpy.arange(length)*bin  # as posicoes do que serao exibidas no eixo X

    if bin != 1:
        for x in range(0, length-1):
            xvalues[x] = xvalues[x+1]-1

        xvalues[length-1] = 255

    width   = 0.2  # the width of the bars
    fig, ax = plt.subplots()
    if ishistgray(hist):
        groupvector = grouphist(hist, bin, length)
        greyrect = ax.bar(xvalues, groupvector, width, color='w')

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Quantidade')
        ax.set_title('Pixels por Intensidade em escala de cinza')
    else:
        groupvectorred      = grouphist(hist[0], bin, length)
        groupvectorgreen    = grouphist(hist[1], bin, length)
        groupvectorblue     = grouphist(hist[2], bin, length)
        redrect             = ax.bar(xvalues, groupvectorred, width, color='r')
        greenrect           = ax.bar(xvalues+width, groupvectorgreen, width, color='g')
        bluerect            = ax.bar(xvalues+width*2, groupvectorblue, width, color='b')

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Quantidade')
        ax.set_title('Pixels por Intensidade')
        ax.legend((redrect[0], greenrect[0], bluerect[0]), ('RED', 'GREEN', 'BLUE'))

    # Configuracoes visuais:
    if bin >= 5:
        ax.set_xticks(xvalues)

    if bin > 1:
        ax.set_xlabel('Pixels agrupados por ' + str(bin))
    plt.show()
    return


# Agrupa os valores do vetor de acordo com o valor bin.
def grouphist(hist, bin, length):
    groupvector = [0] * length
    count       = 0
    acum        = 0
    position    = 0
    for x in range(0, 256):
        if count <= bin:
            acum += hist[x]

        count += 1
        if count > (bin - 1):
            groupvector[position] = acum
            acum                  = 0
            count                 = 0
            position += 1

    if count != 0:
        groupvector[position] = acum

    return groupvector


# Calcula o tamanho de posicoes necessarias com base no valor de agrupamento 'bin'.
def calchistlength(hist, bin):

    if ishistgray(hist):
        length = (len(hist) / bin)
        if len(hist) % bin != 0:
            length += 1
    else:
        length = (len(hist[0]) / bin)
        if len(hist[0]) % bin != 0:
            length += 1

    return length

# Questao 14 OK!!! necessario refatoracao
def histeq(image):
    imgequalized    = image.copy()
    imgsize         = size(image)
    qtdpixels       = imgsize[0]*imgsize[1]
    histogram       = hist(image)
    histqualized    = pmf(histogram, qtdpixels)
    cdf(histqualized)

    if ishistgray(histqualized):
        for x in range(0, 256):
            histqualized[x] *= MAXINTENSITY
    else:
        for x in range(0, 256):
            histqualized[0][x] *= MAXINTENSITY
            histqualized[1][x] *= MAXINTENSITY
            histqualized[2][x] *= MAXINTENSITY


    # Aplica o resultado da equalizacao na imagem
    if ishistgray(histqualized):
        for x in range(0, imgsize[0]):
            for y in range(0, imgsize[1]):
                imgequalized[x][y] = int(histqualized[imgequalized[x][y]])

    else:
        for x in range(0, imgsize[0]):
            for y in range(0, imgsize[1]):
                imgequalized[x][y][0] = int(histqualized[0][imgequalized[x][y][0]])
                imgequalized[x][y][1] = int(histqualized[1][imgequalized[x][y][1]])
                imgequalized[x][y][2] = int(histqualized[2][imgequalized[x][y][2]])
    return imgequalized


# Divide o total de pixels pela quantidade em cada posicao do histograma.
def pmf(hist, qtdpixels):

    if ishistgray(hist):
        histequ = [float(i) for i in hist]
        for x in range(0, 256):
            histequ[x] /= qtdpixels
        return histequ
    else:
        histequ = [0] * 3
        histequ[0] = [float(i) for i in hist[0]]
        histequ[1] = [float(i) for i in hist[1]]
        histequ[2] = [float(i) for i in hist[2]]
        for x in range(0, 256):
            histequ[0][x] /= qtdpixels
            histequ[1][x] /= qtdpixels
            histequ[2][x] /= qtdpixels
        return histequ


# Acumula os valores do histograma, a ultima posicao do vetor deve conter o valor 1.
def cdf(hist):
    if ishistgray(hist):
        for x in range(1, 256):
            hist[x] = hist[x] + hist[x - 1]
    else:
        for x in range(1, 256):
            hist[0][x] = hist[0][x] + hist[0][x - 1]
            hist[1][x] = hist[1][x] + hist[1][x - 1]
            hist[2][x] = hist[2][x] + hist[2][x - 1]

    return hist


# Questao 15
def convolve(image, mask):
    if type(mask) is numpy.ndarray:
        mask = mask.tolist() #converte para lista casa nao seja.
    imagelength = size(image)
    imageconvolved = image.copy()

    for i in range(0, imagelength[0]):
        for j in range(0, imagelength[1]):
            if isgray(image):
                result = calcconvolve(i, j, image, mask)
                result = truncate(result)
                imageconvolved[i][j] = result
            else:
                resultRGB = calcconvolve(i, j, image, mask)
                imageconvolved[i][j][0] = truncate(resultRGB[0])
                imageconvolved[i][j][1] = truncate(resultRGB[1])
                imageconvolved[i][j][2] = truncate(resultRGB[2])

    return imageconvolved


# Define o centro da mascara.
def center(mask):
    length = masklen(mask)

    return int(length[0]/2), int(length[1]/2)


# retorna uma tupla com a quantidade de linhas e colunas da mascara.
def masklen(mask):
    return len(mask), len(mask[0])


def calcconvolve(positionx, positiony, image, mask):
    centerposition = center(mask)
    masklength = masklen(mask)
    maskposition = relativeposition(positionx, positiony, centerposition)
    result = 0.0
    resultr = 0.0
    resultg = 0.0
    resultb = 0.0
    if isgray(image):
        for i in range(0, masklength[0]):
            for j in range(0, masklength[1]):
                x = maskposition[0]+i
                y = maskposition[1]+j
                result += calcpixel(x, y, mask[i][j], image)
        return result
    else:
        for i in range(0, masklength[0]):
            for j in range(0, masklength[1]):
                x = maskposition[0] + i
                y = maskposition[1] + j
                valuergb = calcpixel(x, y, mask[i][j], image)
                resultr += valuergb[0]
                resultg += valuergb[1]
                resultb += valuergb[2]
        return resultr, resultg, resultb


# Coloca na posicao inicial da mascara de acordo com a imagem.
def relativeposition(positionx, positiony, centerposition):
    positionx -= centerposition[0]
    positiony -= centerposition[1]

    return positionx, positiony


# Verifica se a coordenada da mascara esta dentro da imagem, se nao estiver aproxima para uma coordenada.
# Mesma ideia do algoritmo de replicar os pixels mais proximos.
# Retorna a valida do pixel dentro da imagem
def outofbounds(positionx, positiony, image,):
    imagelength = size(image)
    if positionx < 0:
        positionx = 0
    elif positionx > (imagelength[0]-1):
        positionx = imagelength[0]-1

    if positiony < 0:
        positiony = 0
    elif positiony > (imagelength[1]-1):
        positiony = imagelength[1]-1

    return positionx, positiony


# Calcula o valor do pixel da imagem multiplicado pelo peso.
def calcpixel(positionx, positiony, weight, image):
    # Verifica se alguam coordenada esta fora da imagem e ajusta para o correto.
    position = outofbounds(positionx, positiony, image)
    if isgray(image):
        return (image[position[0]][position[1]]) * weight
    else:
        #RGB
        r = image[position[0]][position[1]][0]
        g = image[position[0]][position[1]][1]
        b = image[position[0]][position[1]][2]

        return r * weight, g * weight, b * weight


# 16 - Crie uma funcao chamada maskBlur que retorna a mascara 1/16 * [[1, 2, 1], [2, 4, 2], [1,2,1]].-----------OK
def maskBlur():
    return 0.0625*numpy.asarray([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])


# Questao 17
def blur(image):
    return convolve(image, maskBlur())

# 18 - Crie uma funcao chamada seSquare3, que retorna o elemento estruturante binario [[1,1, 1], [1, 1, 1], [1, 1, 1]].-----OK
def seSquare3():
    return numpy.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]])


# 19 - Crie uma funcao chamada seCross3, que retorna o elemento estruturante binario [[0, 1,0], [1, 1, 1], [0, 1, 0]].-------OK
def seCross3():
    return numpy.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).tolist()

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
# showhist(hist(imageRGB),25)

# Questao 13
#showhist(hist(energiaGrey),50)
#showhist(hist(imageRGB),1)

# Questao 14
#Image.fromarray(imageRGB).show()
#Image.fromarray(histeq(imageGrey)).show()

#Questao 15
# mask1x1 = [[0.0]]
# mask1x7 = [[0.0]*7]
# mask1x7[0][1] = 0.3
# mask1x7[0][2] = 0.6
# mask1x7[0][3] = 0.6
# mask1x7[0][4] = 0.3
# mask3x3 = [[0.0]*3, [0.0]*3, [0.0]*3]
# mask3x3[0][1] = float(1.0/25.0)
# mask3x3[1][0] = float(1.0/25.0)
# mask3x3[1][1] = float(1.0/25.0)
# mask3x3[1][2] = float(1.0/25.0)
# mask3x3[2][1] = float(1.0/25.0)
#filtrodemedia = [[0.04, 0.04, 0.04, 0.04, 0.04], [0.04, 0.04, 0.04, 0.04, 0.04], [0.04, 0.04, 0.04, 0.04, 0.04],
# [0.04, 0.04, 0.04, 0.04, 0.04], [0.04, 0.04, 0.04, 0.04, 0.04]]
# mask1x3 = [[0.0]*3]
# mask5x5 = [[0.0]*5, [0.0]*5, [0.0]*5, [0.0]*5, [0.0]*5]
# mask3x7 = [[0.0]*7, [0.0]*7, [0.0]*7]
# mask7x3 = [[0.0]*3, [0.0]*3, [0.0]*3, [0.0]*3, [0.0]*3, [0.0]*3, [0.0]*3]
# print(center(mask3x7))
#img = convolve(imageRGB, filtrodemedia)
#Image.fromarray(img).save('convolvetest.jpg')
#print type(maskBlur()) is numpy.ndarray
# Questao 16
#s = maskBlur()

# Questao 17
#Image.fromarray(blur(imageRGB)).show()
