""" Flatten : flattens the input tensor into a one-dimensional array."""
# We flatten to pass it to fully connected layer


class Flatten:
    def __init__(self):
        self.shape = 0
        self.trainable = False

    def forward(self, input_tensor):
        self.shape = input_tensor.shape  # Eg = (14,5,3) 14 sb arrays, 5 rows of each sub array, and 3 columns for each sub array row
        # Reshaping array into one dimensional array
        batch_size = self.shape[0]
        return input_tensor.reshape(batch_size, -1)  # New Size (14,15) 14 rows and 15 columns: Make sub arrays in one row

    def backward(self, error_tensor):
        # reshaped into the original shape of the input tensor.
        return error_tensor.reshape(self.shape)

