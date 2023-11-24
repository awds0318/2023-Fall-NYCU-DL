import torch
import torch.nn as nn

# TODO:改成這個
# https://zhuanlan.zhihu.com/p/411311520


class MutiAttention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self, input_dim, dim_k, dim_v, nums_head):
        super(MutiAttention, self).__init__()
        assert dim_k % nums_head == 0
        assert dim_v % nums_head == 0
        self.q = nn.Linear(input_dim, dim_k, bias=False)
        self.k = nn.Linear(input_dim, dim_k, bias=False)
        self.v = nn.Linear(input_dim, dim_v, bias=False)

        self.nums_head = nums_head
        self.dim_k = dim_k
        self.dim_v = dim_v
        self._norm_fact = 1 / (dim_k ** (1 / 2))

    def forward(self, x):
        Q = self.q(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.nums_head)
        K = self.k(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.nums_head)
        V = self.v(x).reshape(-1, x.shape[0], x.shape[1], self.dim_v // self.nums_head)
        print(x.shape)
        print(Q.size())

        atten = nn.Softmax(dim=-1)(torch.matmul(Q, K.permute(0, 1, 3, 2)))

        output = torch.matmul(atten, V).reshape(x.shape[0], x.shape[1], -1)

        return nn.LayerNorm(x.size()[1:]).cuda()(output + x)


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super(FeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=False),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model, bias=False),
            nn.Dropout(dropout),
        )
        self.d_model = d_model

    def forward(self, x):
        output = self.fc(x)
        return nn.LayerNorm(self.d_model).cuda()(output + x)  # ADD And Layer Norm


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, nhead, dropout):
        """
        d_model: token embedding's dimension
        dim_feedforaward: feedforward hidden layer's dimension = d_model * 4
        """
        self.dim_k = d_model % nhead
        self.dim_v = d_model % nhead
        super(TransformerEncoderLayer, self).__init__()
        self.muti_attention = MutiAttention(d_model, self.dim_k, self.dim_v, nhead)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)

    def forward(self, x):
        output = self.muti_attention.forward(x)
        output = self.feed_forward.forward(output)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer1 = TransformerEncoderLayer()
        self.encoder_layer2 = TransformerEncoderLayer()
        self.encoder_layer3 = TransformerEncoderLayer()
        self.encoder_layer4 = TransformerEncoderLayer()

    def forward(self, x):
        return


class Classifier(nn.Module):
    def __init__(self, d_model=80, n_spks=600, dropout=0.1):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        """
        #   TODO:
        #   Build vanilla transformer encoder block from scratch !!!
        #   You can't call the transformer encoder block function from PyTorch library !!!
        #   The number of encoder layers should be less than 4 !!!
        #   The parameter size of transformer encoder block should be less than 500k !!!
        """
        # self.encoder_layer = TransformerEncoderLayer(d_model, dim_feedforward, nhead)  # your own transformer encoder layer
        # self.encoder = TransformerEncoder(self.encoder_layer, num_layers)  # your own transformer encoder
        # Project the the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        out = self.prenet(mels)
        out = out.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder_layer(out)
        out = out.transpose(0, 1)
        # mean pooling
        stats = out.mean(dim=1)
        out = self.pred_layer(stats)
        return out


encoderblock = Classifier()
param_enc = sum(p.numel() for p in encoderblock.encoder.parameters())
print(f"The parameter size of encoder block is {param_enc/1000}k")
