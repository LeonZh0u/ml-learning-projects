import math
import torch
from torch import nn
from torch.nn import functional as F

from typing import Optional

class SinusoidEncoding(nn.Module):
    """
    Implementation of Positional Encoding.
    Refer to the paper "Attention is All You Need" for more details: https://arxiv.org/abs/1706.03762.
    """
    def __init__(self, hidden_dim, max_len=5000):
        """
        Inputs:
            hidden_dim: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence.
        """
        super().__init__()

        # create the matrix of representing the positional encoding for max_len inputs
        # then fill in the positional encoding matrix w/ desired values.
        # the positional encoding matrix is static, so we will use "register buffer" method.
        embed = torch.zeros(max_len, hidden_dim)
        pos = torch.arange(0, max_len)
        i = torch.arange(0, hidden_dim, 2)
        terms = pos * torch.exp(-i/hidden_dim * log(10000))
        embed[:, ::2] = torch.sin(terms)
        embed[:, 1::2] = torch.cos(terms)
        self.pos_embed = embed
        

    def forward(self, x):
        """
        Adds positional embeddings to token embeddings.

        :param x: token embeddings. Shape: [batch_size, seq_length, emb_dim]
        :return: token_embeddings + positional embeddings
        """
        # return the result of adding positional embeddings to x.
        batch_size = x.shape[1]
        return x + self.pos_embed[:, :batch_size]


class MultiHeadSelfAttention(nn.Module):
    """
    This is the implementation of the multi-head self-attention mechanism, 
        as we are only implementing the encoder transformer. 
    We will (hopefully) implement other multi-head attention mechanisms (cross-attention and masked self-attention)) 
        in the decoder transformer next time.
    """
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()

        assert hidden_dim % num_heads == 0
        
        # register the important hyperparameters about this multi-head attention layer.
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dv = hidden_dim // num_heads
        
        # register the linear layers for the query, key, and value projections, without bias;
        # here we only use one matrix to implement the three projections, 
        # which is the same trick as the probabilistic encoder in VAE, remmeber? 
        self.QKV = nn.Linear(hidden_dim, 3*self.hidden_dim)
       

        # register the output linear layer.
        self.fc1 = nn.Linear(3*self.hidden_dim, hidden_dim) 

        # reset the parameters of the linear layers.
        self._reset_parameters()

    def _reset_parameters(self):
        """ Weight initialization taken from the UvA DL1 PyTorch Transformer tutorial. """
        pass

    def forward(
        self,
        x: torch.Tensor,
        src_padding_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        Perform multi-head attention using one projection matrix. Self attention is performed when encoder_hidden_states
        is None, in which case input x represents encoder token embeddings. Otherwise, cross-attention is performed.
        In that case, input x represents the decoder hidden states.

        N = batch size
        S = source sequence length
        T = target sequence length
        E = embedding dimensionality

        :param x: Either encoder or decoder hidden states. Shape: (N, S or T, E)
        :param src_padding_mask: Used for encoder self-attention and cross-attention to handle pad tokens.
        Masks all incoming "connections" or "logits" from any token position to any pad token in a sequence.
        Shape: (T, T).
        :return: Contextualized token embeddings. Shape depends on attention type. (N, S, E) for encoder self-attention
        and decoder cross-attention. (N, T, E) for decoder self-attention.
        """
        batch_size, sequence_length, hidden_dim = x.size()

        # self-attention: project input token embeddings into q, k, v.
        # first use the qkv_proj and then chunck the result to q, k, and v.
        pass

        # swap dimensions to (batch_size, n_heads, seq_len, qkv_dim), 
        # which is required for the matrix multiplication below
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Compute (contextualized) value vector for each "head"
        # the scaled_dot_product() will be responsible for the attention mechanism.
        values, attn = self.scaled_dot_product(q, k, v, src_padding_mask)

        # Concatenate contextualized value vectors from all heads
        values = values.reshape(batch_size, sequence_length, hidden_dim)

        # linearly transform the concatenation of all heads' value vectors (8*64=512) to the original hidden dim (512)
        # with the output linear layer.
        pass

        # return the hidden states
        pass

    def scaled_dot_product(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        For cross-attention, the sequence length of q and (k,v) may differ as q is projected from decoder hidden states
        and kv from encoder hidden states.

        N = batch size
        S = source sequence length
        T = target sequence length
        E = embedding dimensionality
        H = number of heads

        :param q: Tensor stacking query vectors for all tokens and all heads. Shape: (N, H, S or T, E/H)
        :param k: Tensor stacking key vectors for all tokens and all heads. Shape: (N, H, S or T, E/H)
        :param v: Tensor stacking value vectors for all tokens and all heads. Shape: (N, H, S or T, E/H)
        :param padding_mask: Used for encoder self-attention to handle pad tokens.
        Masks all incoming "connections" or "logits" from any token position to any pad token in a sequence.
        Shape: (N, S)

        :return: values (N, H, S or T, E/H), attention scores (N, H, S or T, S or T)
        """

        # Compute attention logits: 
        # dot product between each query and key vector, through one matrix multiplication.
        # shape: (batch_size, num_heads, seq_length, seq_length)
        pass

        # Scale logits by constant to create less spiky softmax distribution, 
        # a trick described in the paper.
        pass

        # attention mask (for padding tokens)
        # here we implemented the mask_logits() method for you, 
        # but feel free to implement it in your own way.
        attn_logits = self.mask_logits(attn_logits, padding_mask)

        # apply softmax function to the logits to 
        # attention probability distribution (one distribution per non-masked token index)
        pass 

        # weighted sum of value vectors for each input token using attention scores -> new contextualized representation
        # shape: (batch_size, num_heads, sequence_length, qkv_dim)
        pass

        # return the new contextualized representation and the attention scores.
        pass

    @staticmethod
    def mask_logits(
        logits: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        """
        Reshape masks to fit the shape of the logits and set all indices with "False" to -inf

        N = batch size
        S = source sequence length
        T = target sequence length
        E = embedding dimensionality
        H = number of heads

        :param logits: Tensor containing attention logits. Shape: (N, H, S or T, S or T)
        :param mask: Used for encoder self-attention to handle pad tokens.
        Masks all incoming "connections" or "logits" from any token position to any pad token in a sequence.
        Shape: (N, S)

        :return: masked_logits (N, H, S or T, S or T)
        """
        if mask is None: return logits
        masked_logits = logits.masked_fill(mask[:, None, None, :] == 0, float("-inf"))
        return masked_logits
    

class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, ff_dim: int, num_heads: int, dropout_p: float):
        super().__init__()
        # create the multi-head self-attention layer
        pass

        # create the feed-forward layer
        pass

        # create the dropout layers
        pass

        # create the layer_norm layers
        pass

    def forward(self, x: torch.FloatTensor, src_padding_mask: torch.BoolTensor = None):
        """
        Performs one encoder *block* forward pass given the previous block's output and an optional attention mask.

        N = batch size
        S = source sequence length
        E = embedding dimensionality

        :param x: Tensor containing the output of the previous encoder block. Shape: (N, S, E)
        :param src_padding_mask: An attention mask to ignore pad-tokens in the source input. Shape (N, S)
        :return: Updated intermediate encoder (contextualized) token embeddings. Shape: (N, S, E)
        """

        # 1. multi-head self-attention layer
        pass
        # 2. dropout1
        pass
        # 3. layer normalization1
        pass
        # 4. feed-forward layer
        pass
        # 5. dropout2
        pass
        # 6. skip-connection and layer normalization2
        pass
        # return the output 
        pass


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embedding: torch.nn.Embedding,
        hidden_dim: int,
        ff_dim: int,
        num_heads: int,
        num_layers: int,
        dropout_p: float,
    ):
        super().__init__()
        self.embed = embedding
        self.hidden_dim = hidden_dim
        self.positional_encoding = SinusoidEncoding(hidden_dim, max_len=5000)
        self.dropout = nn.Dropout(p=dropout_p)
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(hidden_dim, ff_dim, num_heads, dropout_p)
                for _ in range(num_layers)
            ]
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, input_ids: torch.Tensor, src_padding_mask: torch.BoolTensor = None
    ):
        """
        Performs one encoder forward pass given input token ids and an optional attention mask.

        N = batch size
        S = source sequence length
        E = embedding dimensionality

        :param input_ids: Tensor containing input token ids. Shape: (N, S)
        :param src_padding_mask: An attention mask to ignore pad-tokens in the source input. Shape (N, S)
        :return: The encoder's final (contextualized) token embeddings. Shape: (N, S, E)
        """

        x = self.embed(input_ids)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block.forward(x, src_padding_mask=src_padding_mask)
        return x


class TransformerEncoderClassifier(nn.Module):
    """
    Here we use our implemented TransformerEncoder to build a simple classifier.
    """
    def __init__(self, embedding, num_classes, hidden_dim=128, ff_dim=128, num_heads=8, num_layers=2, dropout_p=0.1):
        super(TransformerEncoderClassifier, self).__init__()
        # creat self.encoder as the TransformerEncoder
        pass
        
        # create the output linear layer as the final multi-class classifier
        pass
    
    def forward(self, x, mask):
        # feed forward the input x to the encoder
        pass

        # here we only use the CLS token to make the prediction,
        # merely following the BERT model.
        pass

        # feed forward of x to the final linear classifier.
        pass

        # return the logits
        pass