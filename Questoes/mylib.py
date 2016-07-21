import matplotlib
import numpy
import scipy

from PIL import Image

#Questão 02
def imread(filename):
    image = Image.open(filename)
    return numpy.asarray(image)

#Questão 03
def nchannels(image):
    return 1


#testes Questão 02

image = imread("C:\\Users\\aluno\\Desktop\\zenfoneGO.jpg")
image2 = imread("C:\\Users\\aluno\\Desktop\\imagem cinza.png")
print(type(image))

print(image.size)
#image.show()
