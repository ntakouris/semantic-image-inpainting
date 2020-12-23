import tensorflow as tf
import numpy as np
import external.poissonblending as blending
from scipy.signal import convolve2d

class ModelInpaint():
    def __init__(self, modelfilename, config,
                 model_name='dcgan',
                 gen_input='z:0', gen_output='Tanh:0', gen_loss='logistic_loss_2:0', # gen_loss='Mean_2:0',
                 disc_input='real_images:0', disc_output='Sigmoid:0',
                 z_dim=100, batch_size=64, awesome_gan=False):
        """
        Model for Semantic image inpainting.
        Loads frozen weights of a GAN and create the graph according to the
        loss function as described in paper

        Arguments:
            modelfilename - tensorflow .pb file with weights to be loaded
            config - training parameters: lambda_p, nIter
            gen_input - node name for generator input
            gen_output - node name for generator output
            disc_input - node name for discriminator input
            disc_output - node name for discriminator output
            z_dim - latent space dimension of GAN
            batch_size - training batch size
        """

        self.config = config

        self.batch_size = batch_size
        self.z_dim = z_dim

        self.awesome_gan = awesome_gan
        if self.awesome_gan:
            # model_name = 'awesome_gan'
            self.z_dim = 128
            gen_input = 'z-noise:0'
            gen_output = 'generator/Tanh:0'
            # gen_loss = 'loss/g_loss:0' / string for summary
            gen_loss = 'logistic_loss_2:0'
            disc_input = 'x-images:0'
            disc_output = 'discriminator/Sigmoid:0'

        self.graph, self.graph_def = ModelInpaint.loadpb(modelfilename,
                                                         model_name)

        self.gi = self.graph.get_tensor_by_name(model_name+'/'+gen_input)
        self.go = self.graph.get_tensor_by_name(model_name+'/'+gen_output)
        self.gl = self.graph.get_tensor_by_name(model_name+'/'+gen_loss)
        self.di = self.graph.get_tensor_by_name(model_name+'/'+disc_input)
        self.do = self.graph.get_tensor_by_name(model_name+'/'+disc_output)

        # for name, value in zip(['gi','go','gl','do','do'], [self.gi, self.go, self.gl, self.di, self.do]):
        #     print(f'{name} --> {value}')

        self.image_shape = self.go.shape[1:].as_list()
        print(f'self.image_shape = {self.image_shape}')

        if hasattr(config, 'lambda_p'):
            self.l = config.lambda_p

        self.sess = tf.Session(graph=self.graph)

        self.init_z()

    def init_z(self):
        """Initializes latent variable z"""
        self.z = np.random.randn(self.batch_size, self.z_dim)

    def discriminate(self, inp):
        return self.sess.run(self.do, feed_dict={self.di: inp})

    def sample(self, z=None):
        """GAN sampler. Useful for checking if the GAN was loaded correctly"""
        if z is None:
            z = self.z
        sample_out = self.sess.run(self.go, feed_dict={self.gi: z})
        return sample_out

    def preprocess(self, images, imask, useWeightedMask = True, nsize=7):
        """Default preprocessing pipeline
        Prepare the data to be fed to the network. Weighted mask is computed
        and images and masks are duplicated to fill the batch.

        Arguments:
            image - input image
            mask - input mask

        Returns:
            None
        """
        images = ModelInpaint.imtransform(images)
        if useWeightedMask:
            mask = ModelInpaint.createWeightedMask(imask, nsize)
        else:
            mask = imask
        mask = ModelInpaint.create3ChannelMask(mask)
        
        bin_mask = ModelInpaint.binarizeMask(imask, dtype='uint8')
        self.bin_mask = ModelInpaint.create3ChannelMask(bin_mask)

        self.masks_data = np.repeat(mask[np.newaxis, :, :, :],
                                    self.batch_size,
                                    axis=0)

        #Generate multiple candidates for completion if single image is given
        if len(images.shape) is 3:
            self.images_data = np.repeat(images[np.newaxis, :, :, :],
                                         self.batch_size,
                                         axis=0)
        elif len(images.shape) is 4:
            #Ensure batch is filled
            num_images = images.shape[0]
            self.images_data = np.repeat(images[np.newaxis, 0, :, :, :],
                                         self.batch_size,
                                         axis=0)
            ncpy = min(num_images, self.batch_size)
            self.images_data[:ncpy, :, :, :] = images[:ncpy, :, :, :].copy()

    def postprocess(self, g_out, blend = True):
        """Default post processing pipeline
        Applies poisson blending using binary mask. (default)

        Arguments:
            g_out - generator output
            blend - Use poisson blending (True) or alpha blending (False)
        """
        images_out = ModelInpaint.iminvtransform(g_out)
        images_in = ModelInpaint.iminvtransform(self.images_data)

        if blend:
            for i in range(len(g_out)):
                images_out[i] = ModelInpaint.poissonblending(
                    images_in[i], images_out[i], self.bin_mask
                )
        else:
            images_out = np.multiply(images_out, 1-self.masks_data) \
                         + np.multiply(images_in, self.masks_data)

        return images_out

    def build_inpaint_graph(self, log_generator_loss=False):
        """Builds the context and prior loss objective"""
        with self.graph.as_default():
            self.masks = tf.placeholder(tf.float32,
                                        [None] + self.image_shape,
                                        name='mask')
            self.images = tf.placeholder(tf.float32,
                                         [None] + self.image_shape,
                                         name='images')
            
            self.context_loss = tf.reduce_sum(
                    tf.contrib.layers.flatten(
                        tf.abs(tf.multiply(self.masks, self.go) -
                               tf.multiply(self.masks, self.images))), 1
                ) # (batch_size, )

            if not log_generator_loss:
                self.perceptual_loss = self.gl
            
            if log_generator_loss:
                self.perceptual_loss = tf.math.log(self.gl)
            
            self.perceptual_loss = tf.squeeze(self.perceptual_loss) # (batch_size, )

            self.inpaint_loss = self.context_loss + self.l * self.perceptual_loss

            self.inpaint_grad = tf.gradients(tf.reduce_sum(self.inpaint_loss), self.gi)

    def inpaint(self, image, mask, blend=True, proposed_loss=False):
        """Perform inpainting with the given image and mask with the standard
        pipeline as described in paper. To skip steps or try other pre/post
        processing, the methods can be called seperately.

        Arguments:
            image - input 3 channel image
            mask - input binary mask, single channel. Nonzeros values are 
                   treated as 1
            blend - Flag to apply Poisson blending on output, Default = True

        Returns:
            post processed image (merged/blneded), raw generator output
        """
        log_generator_loss = not proposed_loss
        self.build_inpaint_graph(log_generator_loss=log_generator_loss)
        self.preprocess(image, mask)

        imout, losses, generator_losses = self.backprop_to_input()
        
        return self.postprocess(imout, blend), imout, losses, generator_losses

    def backprop_to_input(self, verbose=True):
        """Main worker function. To be called after all initilization is done.
        Performs backpropagation to input using (accelerated) gradient descent
        to obtain latent space representation of target image

        Returns:
            generator output image
        """
        losses = np.zeros([self.config.nIter, self.images_data.shape[0]])
        generator_losses = np.zeros([self.config.nIter, self.images_data.shape[0]])

        v = 0
        for i in range(self.config.nIter):
            out_vars = [self.inpaint_loss, self.inpaint_grad, self.go, self.gl]

            in_dict = {self.masks: self.masks_data,
                       self.gi: self.z,
                       self.images: self.images_data}

            loss, grad, imout, gl = self.sess.run(out_vars, feed_dict=in_dict)
            losses[i, :] = loss
            generator_losses[i, :] = gl.flatten()

            v_prev = np.copy(v)
            v = self.config.momentum*v - self.config.lr*grad[0]
            self.z += (-self.config.momentum * v_prev +
                       (1 + self.config.momentum) * v)
            self.z = np.clip(self.z, -1, 1)

            if verbose:
                print('Iteration {}: {}'.format(i, np.mean(loss)))

        return imout, losses, generator_losses

    @staticmethod
    def loadpb(filename, model_name='dcgan'):
        """Loads pretrained graph from ProtoBuf file

        Arguments:
            filename - path to ProtoBuf graph definition
            model_name - prefix to assign to loaded graph node names

        Returns:
            graph, graph_def - as per Tensorflow definitions
        """
        with tf.gfile.GFile(filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def,
                                input_map=None,
                                return_elements=None,
                                op_dict=None,
                                producer_op_list=None,
                                name=model_name)

        return graph, graph_def

    @staticmethod
    def imtransform(img):
        """Helper: Rescale pixel value ranges to -1 and 1"""
        return np.array(img) / 127.5-1

    @staticmethod
    def iminvtransform(img):
        """Helper: Rescale pixel value ranges to 0 and 1"""
        return (np.array(img) + 1.0) / 2.0

    @staticmethod
    def poissonblending(img1, img2, mask):
        """Helper: interface to external poisson blending"""
        return blending.blend(img1, img2, 1 - mask)

    @staticmethod
    def createWeightedMask(mask, nsize=7):
        """Takes binary weighted mask to create weighted mask as described in 
        paper.

        Arguments:
            mask - binary mask input. numpy float32 array
            nsize - pixel neighbourhood size. default = 7
        """
        ker = np.ones((nsize,nsize), dtype=np.float32)
        ker = ker/np.sum(ker)
        wmask = mask * convolve2d(mask, ker, mode='same', boundary='symm')
        return wmask

    @staticmethod
    def binarizeMask(mask, dtype=np.float32):
        """Helper function, ensures mask is 0/1 or 0/255 and single channel
        If dtype specified as float32 (default), output mask will be 0, 1
        if required dtype is uint8, output mask will be 0, 255
        """
        assert(np.dtype(dtype) == np.float32 or np.dtype(dtype) == np.uint8)
        bmask = np.array(mask, dtype=np.float32)
        bmask[bmask>0] = 1.0
        bmask[bmask<=0] = 0
        if dtype == np.uint8:
            bmask = np.array(bmask*255, dtype=np.uint8)
        return bmask
    
    @staticmethod
    def create3ChannelMask(mask):
        """Helper function, repeats single channel mask to 3 channels"""
        assert(len(mask.shape)==2)
        return np.repeat(mask[:,:,np.newaxis], 3, axis=2)
