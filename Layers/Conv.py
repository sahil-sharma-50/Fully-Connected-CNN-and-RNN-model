import numpy as np
from scipy import signal


class Conv:
    def __init__(self, stride_shape=np.random.uniform(0, 1, 1)[0], convolution_shape=np.random.uniform(0, 1, 2),
                 num_kernels=np.random.uniform(0, 1, 1)[0]):
        self.trainable = True
        self.input_tensor = None
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.bias = np.random.rand(num_kernels)
        self.weights = np.random.rand(num_kernels, *convolution_shape)
        self._optimizer = None
        self._bias_optimizer = None
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        # For 2D Images

        '''int(np.ceil(input_tensor.shape[2] / self.stride_shape[0])),
            int(np.ceil(input_tensor.shape[3] / self.stride_shape[1]))
            This ensures that the output dimensions are adjusted properly to handle 
            stride values that do not evenly divide the input dimensions.'''
        if len(self.convolution_shape) == 3:
            result = np.zeros((
                input_tensor.shape[0],
                self.num_kernels,
                int(np.ceil(input_tensor.shape[2] / self.stride_shape[0])),
                int(np.ceil(input_tensor.shape[3] / self.stride_shape[1]))
            ))
        # For 1D Images
        elif len(self.convolution_shape) == 2:
            result = np.zeros((
                input_tensor.shape[0],
                self.num_kernels,
                int(np.ceil(input_tensor.shape[2] / self.stride_shape[0]))
            ))

        # Loop over batches
        for batch in range(input_tensor.shape[0]):
            # Loop over different kernels
            for kernel in range(self.weights.shape[0]):
                # List to save outputs of convolution for each channel
                conv_outputs = []
                # Loop over each channel of the kernel and input
                for channel in range(self.weights.shape[1]):
                    # Correlation between input and weights -> output same shape
                    ''' mede = 'same' -> Same dimension 
                    method = 'direct' -> straightforward mathematical calculations without any optimization 
                                                                                techniques or frequency-domain transformations'''
                    conv_outputs.append(
                        signal.correlate(
                            input_tensor[batch, channel],
                            self.weights[kernel, channel],
                            mode='same', method='direct')
                    )

                # Stacking the output of the correlation of each channel
                conv_result = np.stack(conv_outputs, axis=0)

                # Sum it over channels to get a 2D one-channel image
                conv_result = conv_result.sum(axis=0)  # axis = 0 -> column

                # Stride(Step size) Implementation: Down-sampling
                # Down-sampling : reduce the spatial dimensions of the feature maps
                if len(self.convolution_shape) == 3:
                    conv_result = conv_result[::self.stride_shape[0], ::self.stride_shape[1]]  # Height and width
                elif len(self.convolution_shape) == 2:
                    conv_result = conv_result[::self.stride_shape[0]]  # First Dimension

                # Element-wise addition of bias for every kernel
                result[batch, kernel] = conv_result + self.bias[kernel]

        return result

    def backward(self, error_tensor):
        # helps to store gradient with respect to the input
        gradient_input = np.zeros_like(self.input_tensor)
        new_weights = np.copy(self.weights)
        '''Beginning of the gradient weight calculation'''
        if len(self.convolution_shape) == 3:
            temp_gradient_weights = np.zeros((error_tensor.shape[0], *self.weights.shape))
            # Padding of input width and height
            conv_plane_out = []

            # Iterate over batches in input tensor and output channels
            for batch_idx in range(self.input_tensor.shape[0]):
                ch_conv_out = []
                # Loop over different kernels (Output channels)
                for out_ch in range(self.input_tensor.shape[1]):
                    # Pad the input tensor to match the kernel size and stride.
                    ch_conv_out.append(np.pad(self.input_tensor[batch_idx, out_ch],
                                              ((self.convolution_shape[1] // 2, self.convolution_shape[1] // 2),
                                               (self.convolution_shape[2] // 2, self.convolution_shape[2] // 2)), mode='constant'))

                    if self.convolution_shape[2] % 2 == 0:  # Checks if padding is needed in the width dimension.
                        ch_conv_out[out_ch] = ch_conv_out[out_ch][:, :-1]  # removes one column from the right side of the ch_conv_out

                    if self.convolution_shape[1] % 2 == 0:  # Checks if padding is needed in the height dimension.
                        ch_conv_out[out_ch] = ch_conv_out[out_ch][:-1, :]  # removes one row from the bottom of the ch_conv_out

                conv_plane = np.stack(ch_conv_out, axis=0)  # This array represents the complete padded input tensor for a single batch.
                conv_plane_out.append(conv_plane)  # This list will contain the padded input tensors for all batches.

            padded_input = np.stack(conv_plane_out, axis=0)  # stacks the padded input tensors along the batch dimension.
            # Loop over batches
            for batch_idx in range(error_tensor.shape[0]):
                for out_ch in range(error_tensor.shape[1]):
                    # Stride implementation : Up-sampling the error tensor to match the input tensor shape
                    temp = signal.resample(error_tensor[batch_idx, out_ch], error_tensor[batch_idx, out_ch].shape[0] * self.stride_shape[0],
                                           axis=0)
                    temp = signal.resample(temp, error_tensor[batch_idx, out_ch].shape[1] * self.stride_shape[1], axis=1)
                    # Slice it to match the correct shape if the last step of up-sampling was not full
                    temp = temp[:self.input_tensor.shape[2], :self.input_tensor.shape[3]]
                    # We need zero-interpolation, so we put zero for interpolated values
                    if self.stride_shape[1] > 1:
                        for i, row in enumerate(temp):
                            for ii, element in enumerate(row):
                                if ii % self.stride_shape[1] != 0:
                                    row[ii] = 0

                    if self.stride_shape[0] > 1:
                        for i, row in enumerate(temp):
                            for ii, element in enumerate(row):
                                if i % self.stride_shape[0] != 0:
                                    row[ii] = 0
                    # Loop over input channels
                    for in_ch in range(self.input_tensor.shape[1]):
                        # Compute the correlation between the padded input and the up_sampled error tensor.
                        # ‘Valid’/no padding:
                        # : The output is smaller than the input
                        '''mode = 'valid' -> computing the correlation only in the overlapping regions and excluding the padded regions.'''
                        temp_gradient_weights[batch_idx, out_ch, in_ch] = signal.correlate(padded_input[batch_idx, in_ch], temp,
                                                                                           mode='valid')
            # Sum the gradient over the batches to get the final gradient_weights
            self.gradient_weights = temp_gradient_weights.sum(axis=0)
            '''End of the gradient weight calculation'''

        '''Beginning of the gradient input calculation'''
        # Rearranging the weights if the convolution shape has three dimensions.
        if len(self.convolution_shape) == 3:
            new_weights = np.transpose(new_weights, (1, 0, 2, 3))
        elif len(self.convolution_shape) == 2:
            new_weights = np.transpose(new_weights, (1, 0, 2))

        # [CONVOLUTION operation] for the input gradient there's flipping, so we use the convolution.
        # Loop over batches
        for batch_idx in range(error_tensor.shape[0]):
            # Loop over different kernels
            for out_ch in range(new_weights.shape[0]):
                # List to save the outputs of convolution for each channel
                ch_conv_out = []
                # Loop over each channel of the kernel and input
                for in_ch in range(new_weights.shape[1]):

                    ''' Stride implementation : Up-sampling the error tensor to match the input tensor shape.'''
                    # Up-samplying for 2D images
                    if len(self.convolution_shape) == 3:
                        temp = signal.resample(error_tensor[batch_idx, in_ch],
                                               error_tensor[batch_idx, in_ch].shape[0] * self.stride_shape[0], axis=0)
                        temp = signal.resample(temp, error_tensor[batch_idx, in_ch].shape[1] * self.stride_shape[1], axis=1)
                        # Slice to match the correct shape if the last step of up-sampling was not full
                        temp = temp[:self.input_tensor.shape[2], :self.input_tensor.shape[3]]
                        # For zero-interpolation, we put zero for interpolated values
                        if self.stride_shape[1] > 1:
                            for i, row in enumerate(temp):
                                for ii, element in enumerate(row):
                                    if ii % self.stride_shape[1] != 0:
                                        row[ii] = 0

                        if self.stride_shape[0] > 1:
                            for i, row in enumerate(temp):
                                for ii, element in enumerate(row):
                                    if i % self.stride_shape[0] != 0:
                                        row[ii] = 0

                    # Up-sampling for 1D images
                    elif len(self.convolution_shape) == 2:
                        # Up-sampling the error tensor to match the input tensor shape.
                        temp = signal.resample(error_tensor[batch_idx, in_ch],
                                               error_tensor[batch_idx, in_ch].shape[0] * self.stride_shape[0], axis=0)
                        temp = temp[:self.input_tensor.shape[2]]
                        # For zero-interpolation, put zero for interpolated values
                        if self.stride_shape[0] > 1:
                            for i, element in enumerate(temp):
                                if i % self.stride_shape[0] != 0:
                                    temp[i] = 0
                    # ‘Same’ padding (usually zero padding):
                    # : Input and output have the same size
                    ch_conv_out.append(signal.convolve(temp, new_weights[out_ch, in_ch], mode='same', method='direct'))

                temp2 = np.stack(ch_conv_out, axis=0)
                # Sum the results over the input channels to get a 2D one-channel image for each output channel.
                temp2 = temp2.sum(axis=0)
                gradient_input[batch_idx, out_ch] = temp2
                '''Ending of the gradient input calculation'''

        '''Gradient bias calculation'''
        if len(self.convolution_shape) == 3:
            # sum the error tensor over batch, height, and width dimensions to get the gradient for the bias
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
        elif len(self.convolution_shape) == 2:
            # sum the error tensor over the batch and height dimensions to get the gradient for the bias.
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2))

        '''Optimizer and Bias calculation'''
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        if self.bias_optimizer:
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        return gradient_input

    ''' Initialize Weights and Bias: '''

    def initialize(self, weights_initializer, bias_initializer):
        if len(self.convolution_shape) == 3:
            ''' self.weights = weights_initializer.initialize(weights_shape, weights_fan_in, weights_fan_out)
                self.bias = bias_initializer.initialize(bias_shape, bias_fan_in, bias_fan_out)
                flatten() -> Converts 2D array to 1D, containing only last row values'''
            self.weights = weights_initializer.initialize((self.num_kernels, self.convolution_shape[0], self.convolution_shape[1],
                                                           self.convolution_shape[2]),
                                                          self.convolution_shape[0] * self.convolution_shape[1] * self.convolution_shape[2],
                                                          self.num_kernels * self.convolution_shape[1] * self.convolution_shape[2])
            self.bias = bias_initializer.initialize(self.num_kernels, 1, self.num_kernels).flatten()

        elif len(self.convolution_shape) == 2:
            self.weights = weights_initializer.initialize((self.num_kernels, self.convolution_shape[0], self.convolution_shape[1]),
                                                          self.convolution_shape[0] * self.convolution_shape[1],
                                                          self.num_kernels * self.convolution_shape[1])
            self.bias = bias_initializer.initialize((1, self.num_kernels), 1, self.num_kernels).flatten()

    # Gradient Weights Property:
    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @gradient_weights.deleter
    def gradient_weights(self):
        del self._gradient_weights

    # Gradient Bias Property:
    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    @gradient_bias.deleter
    def gradient_bias(self):
        del self._gradient_bias

    # Optimizer Property:
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @optimizer.deleter
    def optimizer(self):
        del self._optimizer

    # Bias Optimizer Property:
    @property
    def bias_optimizer(self):
        return self._bias_optimizer

    @bias_optimizer.setter
    def bias_optimizer(self, value):
        self._bias_optimizer = value

    @bias_optimizer.deleter
    def bias_optimizer(self):
        del self._bias_optimizer
