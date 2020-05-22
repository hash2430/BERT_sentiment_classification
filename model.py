from abc import ABC
import math
import random
from random import randint

import torch
import torch.nn as nn 

### You can import any Python standard libraries or pyTorch sub directories here

### END YOUR LIBRARIES

from bpe import BytePairEncoding

def dot_scaled_attention(
    query: torch.Tensor, 
    key: torch.Tensor,
    value: torch.Tensor,
    padding_mask: torch.Tensor
):
    """ Dot scaled attention
    Implement dot-product scaled attention which takes query, key, value and gives attention scores.
    Like assignment 2, <PAD> should not be attended when calculating attention distribution.

    Hint: If you get still stuck on the test cases, remind the structure of the Transformer decoder.
    In the Transformer decoder, value and key from the encoder have same shape but query does not.

    Arguments:
    query -- Query tensor
                in shape (sequence_length, batch_size, d_k)
    key -- Key tensor 
                in shape (sequence_length, batch_size, d_k)
    value -- Value tensor
                in shape (sequence_length, batch_size, d_k)
    padding_mask -- Padding mask tensor in torch.bool type
                in shape (sequence_length, batch_size)
                True for <PAD>, False for non-<PAD>

    Returns:
    attention -- Attention result tensor
                in shape (sequence_length, batch_size, d_k)
    """

    # Because we will use only Transformer's encoder, all of input tensors have same shape
    assert query.shape == key.shape == value.shape
    assert padding_mask.shape == query.shape[:2]
    query_shape = query.shape
    _, _, d_k = query_shape

    # All vlues in last dimension (d_k dimension) are zeros for <PAD> location
    assert (padding_mask == (query == 0.).all(-1)).all()
    assert (padding_mask == (key == 0.).all(-1)).all()
    assert (padding_mask == (value == 0.).all(-1)).all()

    ### YOUR CODE HERE (~4 lines)
    attention: torch.Tensor = None
    q = query.transpose(0,1)
    k = key.transpose(0,1).transpose(1,2)
    denominator = torch.FloatTensor([d_k**0.5])
    dev = torch.div(q.bmm(k), denominator)
    mask_expand = padding_mask.T.unsqueeze(1).expand(query_shape[1], query_shape[0], -1)
    dev[mask_expand] = float('-inf')
    attention_weights = nn.functional.softmax(dev, dim=-1)
    v = value.transpose(0, 1)
    attention = attention_weights.bmm(v).transpose(0,1)
    ### END YOUR CODE

    # Don't forget setting attention result of <PAD> to zeros.
    # This will be useful for debuging
    attention[padding_mask, :] = 0. 

    assert attention.shape == query_shape
    return attention


class MultiHeadAttention(nn.Module):
    def __init__(self,
        hidden_dim: int=256, 
        n_head: int=8
    ):
        """ Multi-head attention initializer
        Use below attributes to implement the forward function

        Attributes:
        n_head -- the number of heads
        d_k -- Hidden dimension of the dot scaled attention
        V_linear -- Linear function to project hidden_dim of value to d_k
        K_linear -- Linear function to project hidden_dim of key to d_k
        Q_linear -- Linear function to project hidden_dim of query to d_k
        O_linear -- Linear function to project collections of d_k to hidden_dim
        """
        super().__init__()
        assert hidden_dim % n_head == 0
        self.n_head = n_head
        self.d_k = hidden_dim // n_head

        self.V_linear = nn.Linear(hidden_dim, self.n_head * self.d_k, bias=False)
        self.K_linear = nn.Linear(hidden_dim, self.n_head * self.d_k, bias=False)
        self.Q_linear = nn.Linear(hidden_dim, self.n_head * self.d_k, bias=False)
        self.O_linear = nn.Linear(self.n_head * self.d_k, hidden_dim, bias=False)

    def forward(self,
        value: torch.Tensor,
        key: torch.Tensor,
        query: torch.Tensor,
        padding_mask: torch.Tensor
    ):
        """ Multi-head attention forward function
        Implement multi-head attention which takes value, key, query, and gives attention score.
        Use dot-scaled attention you have implemented above.

        Note: If you adjust the dimension of batch_size dynamically,
              you can implement this function without any iteration.

        Parameters:
        value -- Value tensor
                    in shape (sequence_length, batch_size, hidden_dim)
        key -- Key tensor
                    in shape (sequence_length, batch_size, hidden_dim)
        query -- Query tensor
                    in shape (sequence_length, batch_size, hidden_dim)
        padding_mask -- Padding mask tensor in torch.bool type
                    in shape (sequence_length, batch_size)
                    True for <PAD>, False for non-<PAD>

        Returns:
        attention -- Attention result tensor
                    in shape (sequence_length, batch_size, hidden_dim)
        """
        assert value.shape == key.shape == query.shape
        assert padding_mask.shape == query.shape[:2]
        input_shape = value.shape
        seq_length, batch_size, hidden_dim = input_shape

        ### YOUR CODE HERE (~6 lines)
        attention: torch.Tensor = None
        # query = query.transpose(0,1)
        # value = value.transpose(0,1)
        # key = key.transpose(0,1)

        multi_query = self.Q_linear(query).view(seq_length, batch_size * self.n_head, -1)
        multi_key = self.K_linear(key).view(seq_length, batch_size * self.n_head, -1)
        multi_value = self.V_linear(value).view(seq_length, batch_size * self.n_head, -1)
        extended_mask = padding_mask.unsqueeze(2).expand(seq_length, batch_size, self.n_head).reshape(seq_length, -1)
        attention = dot_scaled_attention(multi_query, multi_key, multi_value, extended_mask)
        attention = attention.reshape(seq_length, batch_size, hidden_dim)
        attention = self.O_linear(attention)
        ### END YOUR CODE
        assert attention.shape == input_shape
        return attention


class TransformerEncoderBlock(nn.Module):
    def __init__(self,
        hidden_dim: int=256,
        dropout: float=.1,
        n_head: int=8,
        feed_forward_dim: int=512
    ):
        """ Transformer Encoder Block initializer
        Use below attributes to implement the forward function

        Attributes:
        attention -- Multi-head attention layer
        output -- Output layer
        dropout1, dropout2 -- Dropout layers
        norm1, norm2 -- Layer normalization layers
        """
        super().__init__()

        # Attention Layer
        self.attention = MultiHeadAttention(hidden_dim, n_head)

        # Output Layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, feed_forward_dim, bias=True),
            nn.GELU(),
            nn.Linear(feed_forward_dim, hidden_dim, bias=True)
        )

        # Dropout Layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Layer Normalization Layers
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self,
        x: torch.Tensor,
        padding_mask: torch.Tensor
    ):
        """  Transformer Encoder Block forward function
        Implement transformer encoder block with the given attributes.
        We will stack this module to constuct a BERT model.

        Note: Dropout should be applied before adding residual connections

        Paramters:
        x -- Input tensor
                in shape (sequence_length, batch_size, hidden_dim)
        padding_mask -- Padding mask tensor in torch.bool type
                in shape (sequence_length, batch_size)
                True for <PAD>, False for non-<PAD>

        Returns:
        output -- output tensor
                in shape (sequence_length, batch_size, hidden_dim)
        """
        input_shape = x.shape

        # All vlues in last dimension (hidden dimension) are zeros for <PAD> location
        assert (padding_mask == (x == 0.).all(-1)).all()

        ### YOUR CODE HERE (~5 lines)
        output: torch.Tensor = None
        residu = x
        x1 = self.attention(x, x, x, padding_mask)
        x1 = self.dropout1(x1)
        x1 = x1 + residu
        x1 = self.norm1(x1)

        residu = x1
        x2 = self.output(x1)
        x2 = self.dropout2(x2)
        x2 = x2 + residu
        x2 = self.norm2(x2)
        output = x2
        ### END YOUR CODE

        # Don't forget setting output result of <PAD> to zeros.
        # This will be useful for debuging
        output[padding_mask] = 0. 

        assert output.shape == input_shape
        return output

#######################################################
# Helper functions below. DO NOT MODIFY!              #
# Read helper classes to implement trainers properly! #
#######################################################

class PositionalEncoding(nn.Module):
    """ Positional encoder from pyTorch tutorial
    This class injects token position information to the tensor
    Link: https://pytorch.org/tutorials/beginner/transformer_tutorial
    """
    def __init__(self, hidden_dim, max_len=1000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe[:, None, :]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), ...]

class SegmentationEmbeddings(nn.Module):
    """ Segmentaion embedding layer
    This class injects segmentation information to the tensor.
    """
    def __init__(self, hidden_dim, max_seg_id=3):
        super().__init__()
        self.embedding = nn.Embedding(max_seg_id, hidden_dim)

    def forward(self, x, tokens):
        seg_ids = torch.cumsum(tokens == BytePairEncoding.SEP_token_idx, dim=0) \
                        - (tokens == BytePairEncoding.SEP_token_idx).to(torch.long)
        return x + self.embedding(seg_ids)

class BaseModel(nn.Module):
    """ BERT base model
    MLM & NSP pretraining model and IMDB classification model share the structure of this class.
    """
    def __init__(
        self, token_num: int,
        hidden_dim: int=256, num_layers: int=4,
        dropout: float=0.1, max_len: int=1000,
        **kwargs
    ):
        super().__init__()
        self.embedding = nn.Embedding(token_num, hidden_dim, padding_idx=BytePairEncoding.PAD_token_idx)
        self.position_encoder = PositionalEncoding(hidden_dim, max_len)
        self.segmentation_embedding = SegmentationEmbeddings(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        encoders = [TransformerEncoderBlock(hidden_dim=hidden_dim, dropout=dropout, **kwargs) \
                                                                                for _ in range(num_layers)]
        self.encoders = nn.ModuleList(encoders)

    def forward(self, x):
        padding_mask = x == BytePairEncoding.PAD_token_idx
        out = self.embedding(x)
        out = self.position_encoder(out)
        out = self.segmentation_embedding(out, x)
        out = self.dropout(out)

        out[padding_mask] = 0.
        for encoder in self.encoders:
            out = encoder(out, padding_mask=padding_mask)

        return out

class MLMandNSPmodel(BaseModel):
    """ MLM & NSP model
    Pretraining model for MLM & NSP
    """
    def __init__(self, token_num: int, hidden_dim=256, **kwargs):
        super().__init__(token_num, hidden_dim=hidden_dim, **kwargs)
        self.MLM_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, token_num)
        )
        self.NSP_output = nn.Linear(hidden_dim, 2) # Binary classes, 0 for False and 1 for True.

    def forward(self, x):
        out = super().forward(x)
        return self.MLM_output(out), self.NSP_output(out[0, ...])

class IMDBmodel(BaseModel):
    """ IMDB classification model
    IMDB review classification model which generates binary classes.
    """
    def __init__(self, token_num: int, hidden_dim=256, **kwargs):
        super().__init__(token_num, hidden_dim=hidden_dim, **kwargs)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 2) # Binary classes, 0 for False and 1 for True
        ) 

    def forward(self, x):
        out = super().forward(x)
        return self.output(out[0, ...])

#############################################
# Testing functions below.                  #
#############################################

def dot_scaled_attention_test():
    print("======Dot Scaled Attention Test Case======")
    sequence_length = 10
    batch_size = 8
    d_k = 3

    padding_mask = []
    for _ in range(0, batch_size):
        padding_length = randint(0, sequence_length // 2)
        padding_mask.append([False] * (sequence_length - padding_length) + [True] * padding_length)
    padding_mask = torch.Tensor(padding_mask).to(torch.bool).T
    x = torch.normal(0, 1, [sequence_length, batch_size, d_k], requires_grad=True)
    x.data[padding_mask, :] = 0.

    attention = dot_scaled_attention(query=x, key=x, value=x, padding_mask=padding_mask)

    # the first test
    expected_attn = torch.Tensor([[ 0.12579274, -0.68755102, -0.23434006],
                                  [ 0.03455823, -0.43622059, -0.38654214],
                                  [ 1.03157961, -1.71973670, -0.77598560]])
    assert attention[:3, :3, 0].allclose(expected_attn, atol=1e-7), \
        "Your attention does not match the expected result"
    print("The first test passed!")

    # the second test
    (attention ** 2).sum().backward()
    expected_grad = torch.Tensor([[ 0.16342157, -0.95254862, -0.46277216],
                                  [ 0.06600342, -1.18473446, -1.33262730],
                                  [ 3.98706675, -7.13838005, -2.49680638]])
    assert x.grad[:3, :3, 0].allclose(expected_grad, atol=1e-7), \
        "Your gradient does not match the expected result"
    print("The second test passed!")

    print("All 2 tests passed!")

def multi_head_attention_test():
    print("======Multi-Head Attention Test Case======")
    sequence_length = 10
    batch_size = 8
    hidden_dim = 6
    n_head = 3

    padding_mask = []
    for _ in range(0, batch_size):
        padding_length = randint(0, sequence_length // 2)
        padding_mask.append([False] * (sequence_length - padding_length) + [True] * padding_length)
    padding_mask = torch.Tensor(padding_mask).to(torch.bool).T
    x = torch.normal(0, 1, [sequence_length, batch_size, hidden_dim], requires_grad=True)
    x.data[padding_mask, :] = 0.

    layer = MultiHeadAttention(hidden_dim=hidden_dim, n_head=n_head)
    attention = layer(value=x, key=x, query=x, padding_mask=padding_mask)

    # the first test
    expected_attn = torch.Tensor([[ 0.05339770, -0.01818473, -0.08113014],
                                  [ 0.02698230,  0.00074039,  0.02418777],
                                  [-0.00273581, -0.08840958, -0.04401136]])
    assert attention[:3, :3, 0].allclose(expected_attn, atol=1e-7), \
        "Your attention does not match the expected result"
    print("The first test passed!")

    # the second test
    (attention ** 2).sum().backward()
    expected_grad = torch.Tensor([[ 0.00302105, -0.02852733,  0.01455163],
                                  [-0.03849730, -0.03096544, -0.01094508],
                                  [-0.08066120, -0.01674306,  0.04139223]])
    assert x.grad[:3, :3, 0].allclose(expected_grad, atol=1e-7), \
        "Your gradient does not match the expected result"
    print("The second test passed!")

    print("All 2 tests passed!")

def transformer_encoder_block_test():
    print("======Transformer Encoder Block Test Case======")
    sequence_length = 10
    batch_size = 8
    hidden_dim = 6
    n_head = 3
    feed_forward_dim = 12

    padding_mask = []
    for _ in range(0, batch_size):
        padding_length = randint(0, sequence_length // 2)
        padding_mask.append([False] * (sequence_length - padding_length) + [True] * padding_length)
    padding_mask = torch.Tensor(padding_mask).to(torch.bool).T
    x = torch.normal(0, 1, [sequence_length, batch_size, hidden_dim], requires_grad=True)
    x.data[padding_mask, :] = 0.

    layer = TransformerEncoderBlock(hidden_dim=hidden_dim, n_head=n_head, feed_forward_dim=feed_forward_dim)
    encoded = layer(x, padding_mask=padding_mask)

    # the test case
    expected_value = torch.Tensor([[ 0.17558186,  1.45592594,  0.78872836],
                                   [-0.69810724, -0.71614712,  0.25665924],
                                   [-2.15096855,  1.82032120,  1.11762428]])
    assert encoded[:3, :3, 0].allclose(expected_value, atol=1e-7), \
        "Your encoded value does not match the expected result"
    print("The test case passed!")

if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    random.seed(1234)
    torch.manual_seed(1234)

    dot_scaled_attention_test()
    multi_head_attention_test()
    transformer_encoder_block_test()
