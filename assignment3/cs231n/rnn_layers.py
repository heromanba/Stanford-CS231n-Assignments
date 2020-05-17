from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    # u.shape: (N, H)
    u = prev_h.dot(Wh) + x.dot(Wx) + b
    next_h = np.tanh(u)
    
    cache = {
        'u': u,
        'prev_h': prev_h,
        'Wx': Wx,
        'Wh': Wh,
        'b': b,
        'x': x,
        'next_h': next_h
    }
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    # u.shape: (N, H)
    u = cache['u']
    # prev_h.shape: (N, H)
    prev_h = cache['prev_h']
    # Wx.shape: (D, H)
    Wx = cache['Wx']
    # Wh.shape: (H, H)
    Wh = cache['Wh']
    # b.shape: (H, )
    b = cache['b']
    # x.shape: (N, D)
    x = cache['x']
    # next_h.shape: (N, H)
    next_h = cache['next_h']
    
    # tanh(u) = 1 - 2 / (e^(2u) + 1)
    # dtanh(u)/du = 4 * e^(2u) / (e^(2u) + 1)^2)
    # dtanh.shape: (N, H)
    dtanh = dnext_h * 4 * np.exp(2 * u) / np.power((np.exp(2 * u) + 1), 2)
    
    # dx.shape: (N, D)
    dx = dtanh.dot(Wx.T)
    
    # dprev_h.shape: (N, H)
    dprev_h = dtanh.dot(Wh.T)
    
    dWx = x.T.dot(dtanh)
    
    # dWh.shape: (H, H)
    dWh = prev_h.T.dot(dtanh)
    
    # db.shape: (H, )
    db = np.sum(dtanh, axis=0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    prev_h = h0
    
    N, T, D = x.shape
    _, H = h0.shape
    
    # This RNN is a 'many-to-many' sequence to squence model.
    h = np.zeros((N, T, H))
    
    # Store the cache of each time step.
    step_cache_list = []
    
    x = x.transpose(1, 0, 2)
    for t_i, x_i in enumerate(x):
        next_h, step_cache = rnn_step_forward(x_i, prev_h, Wx, Wh, b)
        h[:, t_i, :] = next_h
        prev_h = next_h
        
        step_cache_list.append(step_cache)
    cache = {
        'N, T, D, H': (N, T, D, H),
        'step_cache_list': step_cache_list
    }
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    N, T, D, H = cache['N, T, D, H']
    step_cache_list = cache['step_cache_list']
    
    dx = np.zeros((N, T, D))

    # Gradients w.r.t Wx over all time step T.
    dWx_T = np.zeros((T, D, H))
    
    # Gradients w.r.t Wh over all time step T.
    dWh_T = np.zeros((T, H, H))

    # Gradients w.r.t b over all time step T.
    db_T = np.zeros((T, H))
    
    # Backprop from time step T to 1.
    for t_i in reversed(range(T)):
        step_cache = step_cache_list[t_i]
        if t_i == T - 1:
            # For the last time step, no upstream gradients.
            dh_i = dh[:, T - 1, :]
        else:
            # For others, add upstream gradients.
            dh_i += dh[:, t_i, :]
        dx_i, dh_i, dWx_i, dWh_i, db_i = rnn_step_backward(dh_i, step_cache)

        dx[:, t_i, :] = dx_i
        dWx_T[t_i] = dWx_i
        dWh_T[t_i] = dWh_i
        db_T[t_i] = db_i
    
    dh0 = dh_i

    # Sum over time steps.
    dWx = np.sum(dWx_T, axis=0)
    dWh = np.sum(dWh_T, axis=0)
    db = np.sum(db_T, axis=0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    N, T = x.shape
    V, D = W.shape
    
    x_T = np.zeros((N, T, V))
    # TODO: do this in one line with numpy.
    for i_sample in range(N):
        for i_t_step in range(T):
            x_T[i_sample, i_t_step, x[i_sample, i_t_step]] = 1
    out = np.matmul(x_T, W[np.newaxis, :, :])
    cache = x_T
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that Words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    """
    Numpy Doc:
    numpy.ufunc.at(a, indices, b=None)
        Performs unbuffered in place operation on operand 'a' for elements specified
        by 'indices'. For addition ufunc, this method is equivalent to a[indices] += b,
        except that results are accumulated for elements that are indexed more than once.
        For example, a[[0,0]] += 1 will only increment the first element once because of 
        buffering, wheras add.at(a, [0,0], 1) will increment the first element twice.
    """
    # TODO: use np.add.at.
    # x_T.shape: (N, T, V)
    x_T = cache
    dW = np.sum(np.matmul(x_T.transpose(0, 2, 1), dout), axis=0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    N, H = prev_h.shape

    # A.shape: (N, 4H)
    A = x.dot(Wx) + prev_h.dot(Wh) + b
    
    a_i = A[:, :H]
    a_f = A[:, H:2*H]
    a_o = A[:, 2*H:3*H]
    a_g = A[:, 3*H:]

    i = sigmoid(a_i)
    f = sigmoid(a_f)
    o = sigmoid(a_o)
    g = np.tanh(a_g)

    next_c = f * prev_c + i * g
    
    next_c_tanh = np.tanh(next_c)

    next_h = o * next_c_tanh

    cache = {
        'next_c': next_c,
        'next_c_tanh': next_c_tanh,
        'o': o,
        'f': f,
        'prev_c': prev_c,
        'i': i,
        'g': g,
        'x': x,
        'Wx': Wx,
        'prev_h': prev_h,
        'Wh': Wh
    }
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    # next_c, next_c_tanh, o, f, prev_c, i, g shape: (N, H)
    next_c = cache['next_c']
    next_c_tanh = cache['next_c_tanh']
    o = cache['o']
    f = cache['f']
    prev_c = cache['prev_c']
    i = cache['i']
    g = cache['g']
    
    # x.shape: (N, D)
    x = cache['x']
    # Wx.shape: (D, 4H)
    Wx = cache['Wx']
    # prev_h.shape: (N, H)
    prev_h = cache['prev_h']
    # Wh.shape: (H, 4H)
    Wh = cache['Wh']
    
    N, H = next_c.shape

    # y = tanh(x) --> dy/dx = (1 + y) * (1 - y)
    # y = sigmoid(x) --> dy/dx = y * (1 - y)

    # do.shape: (N, H)
    do = dnext_h * next_c_tanh
    
    # dnext_c_tanh.shape: (N, H)
    dnext_c_tanh = dnext_h * o

    # dnext_c_0.shape: (N, H)
    dnext_c_0 = dnext_c_tanh * (1 + next_c_tanh) * (1 - next_c_tanh)
    dnext_c += dnext_c_0

    # df, dprev_c, df, dg shape: (N, H)
    df = dnext_c * prev_c
    dprev_c = dnext_c * f
    di = dnext_c * g
    dg = dnext_c * i

    # da_o, da_i, da_f, da_g shape: (N, H)
    da_o = do * o * (1 - o)
    da_i = di * i * (1 - i)
    da_f = df * f * (1 - f)
    da_g = dg * (1 + g) * (1 - g)

    # Map back to A.
    dA = np.zeros((N, 4 * H))
    dA[:, :H] = da_i
    dA[:, H: 2*H] = da_f
    dA[:, 2*H: 3*H] = da_o
    dA[:, 3*H:] = da_g

    dx = dA.dot(Wx.T)
    dWx = x.T.dot(dA)
    dprev_h = dA.dot(Wh.T)
    dWh = prev_h.T.dot(dA)
    db = np.sum(dA, axis=0)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    prev_h = h0

    # Initialize cell state to 0.
    prev_c = np.zeros_like(h0)

    N, T, D = x.shape
    _, H = h0.shape
    
    # This RNN is a 'many-to-many' sequence to squence model.
    h = np.zeros((N, T, H))
    
    # Store the cache of each time step.
    step_cache_list = []
    
    x = x.transpose(1, 0, 2)
    for t_i, x_i in enumerate(x):
        next_h, next_c, step_cache = lstm_step_forward(x_i, prev_h, prev_c, Wx, Wh, b)
        h[:, t_i, :] = next_h

        prev_h = next_h
        prev_c = next_c
        
        step_cache_list.append(step_cache)
    cache = {
        'N, T, D, H': (N, T, D, H),
        'step_cache_list': step_cache_list
    }
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    N, T, D, H = cache['N, T, D, H']
    step_cache_list = cache['step_cache_list']
    
    dx = np.zeros((N, T, D))

    # Gradients w.r.t Wx over all time step T.
    dWx_T = np.zeros((T, D, 4 * H))
    
    # Gradients w.r.t Wh over all time step T.
    dWh_T = np.zeros((T, H, 4 * H))

    # Gradients w.r.t b over all time step T.
    db_T = np.zeros((T, 4 * H))
    
    # Backprop from time step T to 1.
    for t_i in reversed(range(T)):
        step_cache = step_cache_list[t_i]
        if t_i == T - 1:
            # For the last time step, no upstream gradients.
            dh_i = dh[:, T - 1, :]
            # Note: for the last time step, the value of cell
            #       has no direct contribution for loss.
            dc_i = np.zeros_like(dh_i)
        else:
            # For others, add upstream gradients.
            dh_i += dh[:, t_i, :]
        dx_i, dh_i, dc_i, dWx_i, dWh_i, db_i = lstm_step_backward(dh_i, dc_i, step_cache)

        dx[:, t_i, :] = dx_i
        dWx_T[t_i] = dWx_i
        dWh_T[t_i] = dWh_i
        db_T[t_i] = db_i
    
    dh0 = dh_i

    # Sum over time steps.
    dWx = np.sum(dWx_T, axis=0)
    dWh = np.sum(dWh_T, axis=0)
    db = np.sum(db_T, axis=0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
