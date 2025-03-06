import numpy as np

def glorot_uniform_init_dense(input_dim, output_dim):
    fan_in  = input_dim
    fan_out = output_dim
    limit   = np.sqrt(6.0 / (fan_in + fan_out))

    return np.random.uniform(low=-limit, high=limit, size=(output_dim, input_dim)).astype(np.float32)


def glorot_uniform_init_conv(num_filters, input_channels, kernel_size):
    fan_in  = input_channels * kernel_size * kernel_size
    fan_out = num_filters    * kernel_size * kernel_size
    limit   = np.sqrt(6.0 / (fan_in + fan_out))

    return np.random.uniform(
        low=-limit, high=limit,
        size=(num_filters, input_channels, kernel_size, kernel_size)
    ).astype(np.float32)
