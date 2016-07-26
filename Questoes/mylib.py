import matplotlib
import numpy
import scipy

from PIL import Image

#Questao 02 a)
def imread(filename):
    image = Image.open(filename)
    return numpy.asarray(image)

#Questao 02 b)
#Digitar os comandos na seguinte ordem:
# from PIL import Image
# img = Image.open('test.png')
# img.show()

#Questao 02 c)
# from PIL import Image
# img = Image.open('test.png')
# img.resize((50,50)).show()

#Questao 03
def nchannels(image):
    return 1

#testes Questao 02
img = Image.open("C:\\Users\\felipemsx\\Desktop\\zenfoneGO.jpg")
img.resize((50,50)).show()

