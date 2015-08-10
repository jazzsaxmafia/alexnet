
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv
from theano.sandbox.cuda.dnn import dnn_pool

class AlexNet():
    x = T.ftensor4('x')
    input_shuffled = x.dimshuffle(3, 0, 1, 2)

    conv_1 = dnn_conv(
            img=x,
            kerns=

