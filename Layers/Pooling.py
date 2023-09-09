from Layers.Base import BaseLayer
import numpy as np

"""It reduce the dimensionality of the input by sliding the filter
 and calculating the maximum or average of the input."""


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_tensor = None
        self.batch_size = None
        self.num_channels = None
        self.input_shape = None
        self.pool_positions = None
        self.cache = {}

    def forward(self, input_tensor):
        self.input_tensor = np.array(input_tensor, copy=True)  # Later use for backward pass
        self.batch_size, self.num_channels, height, width = input_tensor.shape

        '''Formula used = ((input_tensor - pooling_shape) // stride_shape) + 1'''
        output_height = (height - self.pooling_shape[0]) // self.stride_shape[0] + 1
        output_width = (width - self.pooling_shape[1]) // self.stride_shape[1] + 1

        output = np.zeros((self.batch_size, self.num_channels, output_height, output_width))

        '''Iteration over the height dimension of the output tensor'''
        for i in range(output_height):
            '''Iteration over the width dimension of the output tensor'''
            for j in range(output_width):
                '''This line selects a specific region from the input tensor based on the current position (i, j) and the pooling shape.'''
                input_tensor_slice = input_tensor[:, :,
                                     i * self.stride_shape[0]:i * self.stride_shape[0] + self.pooling_shape[0],
                                     j * self.stride_shape[1]:j * self.stride_shape[1] + self.pooling_shape[1]]

                '''The maximum value along the spatial dimensions (height and width) is computed for each channel,
                and the result is assigned to the corresponding position in the output tensor.'''
                output[:, :, i, j] = np.max(input_tensor_slice, axis=(2, 3))

                '''The indices of the maximum values along the spatial dimensions (height and width) are computed for each channel.
                These indices are used to create a mask of the same shape as the input_tensor_slice,'''
                mask = np.zeros_like(input_tensor_slice)
                n_indices, c_indices, h_indices, w_indices = np.indices(input_tensor_slice.shape)
                mask[n_indices, c_indices, h_indices, w_indices] = (input_tensor_slice == output[:, :, i, j][:, :, np.newaxis, np.newaxis])

                '''The mask is stored in the self.cache dictionary with the position (i, j) as the key.'''
                self.cache[(i, j)] = mask

        return output

    def backward(self, error_tensor):
        output = np.zeros_like(self.input_tensor)
        _, _, output_height, output_width = error_tensor.shape

        '''i iterate over the height dimension'''
        for i in range(output_height):
            '''j iterate over the width dimension'''
            for j in range(output_width):
                '''This line performs the accumulation of gradients. 
                It selects a specific region in the output tensor corresponding to the 
                pooling region for the current position (i, j). It then adds the element-wise product of the corresponding region in the 
                error_tensor and the stored value in self.cache[(i, j)].'''
                output[:, :,
                i * self.stride_shape[0]:i * self.stride_shape[0] + self.pooling_shape[0],
                j * self.stride_shape[1]:j * self.stride_shape[1] + self.pooling_shape[1]] += \
                    error_tensor[:, :,
                    i:i + 1,
                    j:j + 1] * self.cache[(i, j)]

        return output
