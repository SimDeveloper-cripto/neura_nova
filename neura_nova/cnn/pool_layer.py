import numpy as np


class MaxPoolLayer:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride      = stride
        self.input       = None
        self.argmax      = None

    def forward(self, input_data):
        self.input = input_data
        batch_size, input_channels, H, W = input_data.shape
        out_h = (H - self.kernel_size) // self.stride + 1
        out_w = (W - self.kernel_size) // self.stride + 1

        input_reshaped = input_data.reshape(batch_size, input_channels, out_h, self.stride, out_w, self.stride)
        input_reshaped = input_reshaped.transpose(0, 1, 2, 4, 3, 5)
        output = input_reshaped.max(axis=(-1, -2))

        self.argmax = np.argmax(input_reshaped.reshape(batch_size, input_channels, out_h, out_w, -1), axis=-1)
        return output

    def backward(self, grad_output):
        # grad_output.shape[0]: batch_size
        # grad_output.shape[1]: input_channels
        # grad_output.shape[2]: H
        # grad_output.shape[3]: W

        batch_size, input_channels, H, W = self.input.shape
        out_h = (H - self.kernel_size) // self.stride + 1
        out_w = (W - self.kernel_size) // self.stride + 1

        dinput = np.zeros(self.input.shape, dtype=grad_output.dtype)

        input_reshaped  = self.input.reshape(batch_size, input_channels, out_h, self.stride, out_w, self.stride)
        input_reshaped  = input_reshaped.transpose(0, 1, 2, 4, 3, 5).reshape(batch_size, input_channels, out_h, out_w, -1)
        dinput_reshaped = np.zeros_like(input_reshaped)

        for b in range(batch_size):
            for c in range(input_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        idx = self.argmax[b, c, i, j]
                        dinput_reshaped[b, c, i, j, idx] = grad_output[b, c, i, j]

        dinput_reshaped = dinput_reshaped.reshape(batch_size, input_channels, out_h, out_w, self.kernel_size, self.kernel_size)
        dinput_reshaped = dinput_reshaped.transpose(0, 1, 2, 4, 3, 5)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end   = h_start + self.kernel_size
                w_start = j * self.stride
                w_end   = w_start + self.kernel_size
                dinput[:, :, h_start:h_end, w_start:w_end] += dinput_reshaped[:, :, i, :, j, :]

        return dinput
