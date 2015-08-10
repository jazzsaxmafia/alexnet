from layers import *
import theano
import theano.tensor as T
import cv2
import skimage.io as io

x = T.tensor4()

conv_pool = ConvPoolLayer(x)
ff = theano.function(inputs=[x],
                     outputs=conv_pool.output,
                     allow_input_downcast=True)

image = io.imread('dog.jpg')
image = cv2.resize(image, (227,227))
image = np.swapaxes(image, 2, 0)
image = image[None, :,:,:]

result = np.asarray(ff(image))
