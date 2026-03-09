"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filename, "rb") as f:
        magic_bytes = f.read(4)
        magic = struct.unpack('>I', magic_bytes)[0]
        if magic != 2051: # 0x00000803 is 2051 in decimal
            raise ValueError(f"Invalid magic number: expected 2051, got {magic}")
        '''
        num_image_byte = f.read(4)
        num_image = struct.unpack('>I', num_image_byte)[0]

        row_byte = f.read(4)
        row = struct.unpack('>I', row_byte)[0]
        col_byte = f.read(4)
        col = struct.unpack('>I', col_byte)[0]
        '''
        num_images, rows, cols = struct.unpack(">III", f.read(12))
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(num_images, rows * cols)
        X = data.astype(np.float32) / 255.0



    with gzip.open(label_filename, "rb") as f:
        magic = struct.unpack('>I', f.read(4))[0]
        if magic != 2049: # 0x00000801 is 2049 in decimal
            raise ValueError(f"Invalid magic number: expected 2049, got {magic}")
        num_images = struct.unpack(">I", f.read(4))[0]
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8)

    return (X, data)
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    Z_exp = ndl.exp(Z)
    log_sum_exp = ndl.log(Z_exp.sum(axes=(1,)))

    correct_class_score = (Z * y_one_hot)

    correct_class_score = correct_class_score.sum(axes=(1,))

    loss = log_sum_exp - correct_class_score
    
    result = loss.sum() / Z.shape[0]
    return result


    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION

    m = X.shape[0] # # of example
    k = W2.shape[1] # # of classes
    n = X.shape[1] # input_dimesion
    d = W1.shape[1] # hidden dimension

    y_one_hot = np.zeros((m, k))
    y_one_hot[np.arange(m), y] = 1

    # mini batch start
    for i in range(0, m, batch):

        # end idx limits to avoid index out of bounds
        end_idx = min(i + batch, m)
        X_batch = ndl.Tensor(X[i : end_idx])
        # y_batch is one hot vector
        y_batch = ndl.Tensor(y_one_hot[i : end_idx])

        # B is mini batch size
        B = X_batch.shape[0]

        Z1_temp = X_batch.matmul(W1) # can I use matmul 대신에 @?
        Z1 = ndl.relu(Z1_temp)
        Z2 = Z1.matmul(W2)

        loss = softmax_loss(Z2, y_batch)
        loss.backward()



        # Update weights and detach them from the computation graph.
        # We explicitly cast back to np.float32, as multiplying by `lr` (a python float) 
        # implicitly converts the array type to np.float64, which causes needle ops to fail!
        W1 = ndl.Tensor(np.float32(W1.numpy() - lr * W1.grad.numpy()))
        W2 = ndl.Tensor(np.float32(W2.numpy() - lr * W2.grad.numpy()))
    
    return (W1, W2)
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
