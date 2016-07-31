import matplotlib
import numpy
import scipy
import matplotlib.pyplot as plt
import math

from PIL import Image

MAXCAPACITY = 256
MAXINTENSITY = 255
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
                newImage[x][y] = MAXINTENSITY if (newImage[x][y] >= limit) else MIN
    else:
        for x in range(0, vector[0]):
            for y in range(0, vector[1]):
                newImage[x][y][0] = MAXINTENSITY if (newImage[x][y][0] >= limit) else MIN
                newImage[x][y][1] = MAXINTENSITY if (newImage[x][y][1] >= limit) else MIN
                newImage[x][y][2] = MAXINTENSITY if (newImage[x][y][2] >= limit) else MIN
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
    # Questao 12--------------------------------------------------
    # N = 256
    # ind = numpy.arange(N)  # the x locations for the groups
    # width = 1  # the width of the bars
    # fig, ax = plt.subplots()
    # if len(hist) == 256: #indica que 'e' cinza.
    #     greyrect = ax.bar(ind, hist, width, color='w')
    #
    #     # add some text for labels, title and axes ticks
    #     ax.set_ylabel('Quantidade')
    #     ax.set_title('Pixels por Intensidade em escala de cinza')
    # else:
    #
    #     redrect = ax.bar(ind, hist[0], width, color='r')
    #     greenrect = ax.bar(ind, hist[1], width, color='g')
    #     bluerect = ax.bar(ind, hist[2], width, color='b')
    #
    #     # add some text for labels, title and axes ticks
    #     ax.set_ylabel('Quantidade')
    #     ax.set_title('Pixels por Intensidade')
    #     ax.legend((redrect[0], greenrect[0], bluerect[0]), ('RED', 'GREEN', 'BLUE'))
    #
    # plt.show()
    #
    # Questao 12--------------------------------------------------

    lenght      = calchistlenght(hist, bin)
    xvalues     = numpy.arange(lenght)*bin  # as posicoes do que serao exibidas no eixo X

    if bin != 1:
        for x in range(0, lenght-1):
            xvalues[x] = xvalues[x+1]-1

        xvalues[lenght-1] = 255

    width   = 1  # the width of the bars
    fig, ax = plt.subplots()
    if len(hist) == 256: #indica que o hist foi feito com base em uma base de cinza
        groupvector = grouphist(hist, bin, lenght)
        greyrect = ax.bar(xvalues, groupvector, width, color='w')

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Quantidade')
        ax.set_title('Pixels por Intensidade em escala de cinza')
    else:
        groupvectorred      = grouphist(hist[0], bin, lenght)
        groupvectorgreen    = grouphist(hist[1], bin, lenght)
        groupvectorblue     = grouphist(hist[2], bin, lenght)
        redrect             = ax.bar(xvalues, groupvectorred, width, color='r')
        greenrect           = ax.bar(xvalues, groupvectorgreen, width, color='g')
        bluerect            = ax.bar(xvalues, groupvectorblue, width, color='b')

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Quantidade')
        ax.set_title('Pixels por Intensidade')

        ax.legend((redrect[0], greenrect[0], bluerect[0]), ('RED', 'GREEN', 'BLUE'))

    # Configuracoes visuais:
    if bin >= 8:
        ax.set_xticks(xvalues)

    if bin > 1:
        ax.set_xlabel('Pixels agrupados por '+ str(bin))
    plt.show()
    return


# Agrupa os valores do vetor de acordo com o valor bin.
def grouphist(hist, bin, lenght):
    groupvector = [0] * lenght
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
def calchistlenght(hist,bin):

    if len(hist) == 256: # indica que o hist foi feito com base em uma base de cinza
        lenght = (len(hist) / bin)
        if len(hist) % bin != 0:
            lenght += 1
    else:
        lenght = (len(hist[0]) / bin)
        if len(hist[0]) % bin != 0:
            lenght += 1

    return lenght

# Questao 14 OK!!! necessario refatoracao
def histeq(image):
    imgequalized    = image.copy()
    imgsize         = size(image)
    qtdpixels       = imgsize[0]*imgsize[1]
    histogram       = hist(image)
    histqualized    = pmf(histogram, qtdpixels)
    cdf(histqualized)

    if len(histqualized) == 256:
        for x in range(0, 256):
            histqualized[x] *= MAXINTENSITY
    else:
        for x in range(0, 256):
            histqualized[0][x] *= MAXINTENSITY
            histqualized[1][x] *= MAXINTENSITY
            histqualized[2][x] *= MAXINTENSITY


    # Aplica o resultado da equalizacao na imagem
    if len(histogram) == 256:
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

    if len(hist) == 256:
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


# Acumula os valores do histograma, a ultima posicao do vetor deve ser 1.
def cdf(hist):
    if len(hist) == 256:
        for x in range(1, 256):
            hist[x] = hist[x] + hist[x - 1]
    else:
        for x in range(1, 256):
            hist[0][x] = hist[0][x] + hist[0][x - 1]
            hist[1][x] = hist[1][x] + hist[1][x - 1]
            hist[2][x] = hist[2][x] + hist[2][x - 1]

    return hist
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
# showhist(hist(energiaGrey))
# showhist(hist(imageRGB),25)

# Questao 14

Image.fromarray(imageRGB).show()
Image.fromarray(histeq(imageRGB)).show()
