import numpy as np

def glorot_uniform_init_dense(input_dim, output_dim):
    """
    Range [-limit, limit] in which limit = sqrt(6 / (fan_in + fan_out))

    Parameters:
    - input_dim  : number of input neurons  (fan_in)
    - output_dim : number of output neurons (fan_out)

    Returns:
    - A numpy array with shape (output_dim, input_dim)
    """
    fan_in  = input_dim
    fan_out = output_dim
    limit   = np.sqrt(6.0 / (fan_in + fan_out))

    # Uniformly in [-limit, +limit]
    return np.random.uniform(low=-limit, high=limit, size=(output_dim, input_dim)).astype(np.float32)


def glorot_uniform_init_conv(num_filters, input_channels, kernel_size):
    """
    - fan_in  = (input_channels  * kernel_size * kernel_size)
    - fan_out = (num_filters     * kernel_size * kernel_size)

    Returns:
    - A numpy array with shape (num_filters, input_channels, kernel_size, kernel_size)
    """
    fan_in  = input_channels * kernel_size * kernel_size
    fan_out = num_filters    * kernel_size * kernel_size
    limit   = np.sqrt(6.0 / (fan_in + fan_out))

    return np.random.uniform(
        low=-limit, high=limit,
        size=(num_filters, input_channels, kernel_size, kernel_size)
    ).astype(np.float32)
