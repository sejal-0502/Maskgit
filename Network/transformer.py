import torch
from torch import nn


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        """ PreNorm module to apply layer normalization before a given function
            :param:
                dim  -> int: Dimension of the input
                fn   -> nn.Module: The function to apply after layer normalization
            """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """ Forward pass through the PreNorm module
            :param:
                x        -> torch.Tensor: Input tensor
                **kwargs -> _ : Additional keyword arguments for the function
            :return
                torch.Tensor: Output of the function applied after layer normalization
        """
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        """ Initialize the Multi-Layer Perceptron (MLP).
            :param:
                dim        -> int : Dimension of the input
                dim        -> int : Dimension of the hidden layer
                dim        -> float : Dropout rate
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """ Forward pass through the MLP module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                torch.Tensor: Output of the function applied after layer
        """
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        """ Initialize the Attention module.
            :param:
                embed_dim     -> int : Dimension of the embedding
                num_heads     -> int : Number of heads
                dropout       -> float : Dropout rate
        """
        super(Attention, self).__init__()
        self.dim = embed_dim
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True, bias=True)

    def forward(self, x):
        """ Forward pass through the Attention module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                attention_value  -> torch.Tensor: Output the value of the attention
                attention_weight -> torch.Tensor: Output the weight of the attention
        """
        attention_value, attention_weight = self.mha(x, x, x)
        return attention_value, attention_weight


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        """ Initialize the Attention module.
            :param:
                dim       -> int : number of hidden dimension of attention
                depth     -> int : number of layer for the transformer
                heads     -> int : Number of heads
                mlp_dim   -> int : number of hidden dimension for mlp
                dropout   -> float : Dropout rate
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        """ Forward pass through the Attention module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                x -> torch.Tensor: Output of the Transformer
                l_attn -> list(torch.Tensor): list of the attention
        """
        l_attn = []
        for idx, (attn, ff) in enumerate(self.layers):
            attention_value, attention_weight = attn(x)
            x = attention_value + x
            x = ff(x) + x
            l_attn.append(attention_weight)
        return x, l_attn

class MaskTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=16, hidden_dim=768, codebook_size=16384, depth=16, heads=8, mlp_dim=3072, dropout=0.1, nclass=1000):
        """ Initialize the Transformer model.
            :param:
                img_size       -> int:     Input image size (default: 256)
                hidden_dim     -> int:     Hidden dimension for the transformer (default: 768)
                codebook_size  -> int:     Size of the codebook (default: 1024)
                depth          -> int:     Depth of the transformer (default: 24)
                heads          -> int:     Number of attention heads (default: 8)
                mlp_dim        -> int:     MLP dimension (default: 3072)
                dropout        -> float:   Dropout rate (default: 0.1)
                nclass         -> int:     Number of classes (default: 1000)
        """

        super().__init__()
        self.nclass = nclass
        self.patch_size = patch_size
        self.num_tokens = img_size // self.patch_size
        self.codebook_size = codebook_size
        self.tok_emb = nn.Embedding(codebook_size+1, hidden_dim)  
        self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, (self.num_tokens*self.num_tokens), hidden_dim)), 0., 0.02)

        # First layer before the Transformer block
        self.first_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
        )

        self.transformer = TransformerEncoder(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

        # Last layer after the Transformer block
        self.last_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
        )

        # Bias for the last linear output
        self.bias = nn.Parameter(torch.zeros((self.num_tokens*self.num_tokens), codebook_size+1))

    def forward(self, img_token, return_attn=False):
        """ Forward.
            :param:
                img_token      -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
                return_attn    -> Bool: return the attn for visualization
            :return:
                logit:         -> torch.FloatTensor: bsize x path_size*path_size * 1024, the predicted logit
                attn:          -> list(torch.FloatTensor): list of attention for visualization
        """
        b, w, h = img_token.size()
        input = img_token.view(b, -1)

        tok_embeddings = self.tok_emb(input)

        # Position embedding
        pos_embeddings = self.pos_emb
        x = tok_embeddings + pos_embeddings

        # transformer forward pass
        x = self.first_layer(x)
        x, attn = self.transformer(x)
        x = self.last_layer(x)

        logit = torch.matmul(x, self.tok_emb.weight.T) + self.bias   # Shared layer with the embedding

        if return_attn:  # return list of attention
            return logit[:, :self.num_tokens * self.num_tokens, :self.codebook_size + 1], attn

        return logit[:, :self.num_tokens*self.num_tokens, :self.codebook_size+1]


class MaskTransformerNextFrame(MaskTransformer):
    def __init__(self, img_size=256, patch_size=16, hidden_dim=768, codebook_size=16384, depth=24, heads=8, mlp_dim=3072, dropout=0.1, num_frames=4):
        """ Initialize the Transformer model.
            :param:
                img_size       -> int:     Input image size (default: 256)
                hidden_dim     -> int:     Hidden dimension for the transformer (default: 768)
                codebook_size  -> int:     Size of the codebook (default: 1024)
                depth          -> int:     Depth of the transformer (default: 24)
                heads          -> int:     Number of attention heads (default: 8)
                mlp_dim        -> int:     MLP dimension (default: 3072)
                num_frames     -> int:     Number of frames (context + 1) (default: 4)
                dropout        -> float:   Dropout rate (default: 0.1)
        """

        super().__init__(img_size, patch_size, hidden_dim, codebook_size, depth, heads, mlp_dim, dropout)
        self.num_frames = num_frames
        self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, (self.num_frames*self.num_tokens*self.num_tokens), hidden_dim)), 0., 0.02)
        self.bias = nn.Parameter(torch.zeros((self.num_frames*self.num_tokens*self.num_tokens), codebook_size+1))

    def forward(self, img_token, drop_frame, drop_token, return_attn=False):
        """ Forward.
            :param:
                img_token      -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
                return_attn    -> Bool: return the attn for visualization
            :return:
                logit:         -> torch.FloatTensor: bsize x path_size*path_size * 1024, the predicted logit
                attn:          -> list(torch.FloatTensor): list of attention for visualization
        """
        b, f, w, h = img_token.size()
        input = img_token.view(b, -1)
        tok_embeddings = self.tok_emb(input)

        # Position embedding
        pos_embeddings = self.pos_emb
        #print (f'pos_emb: {pos_embeddings.shape}', f'tok_emb: {tok_embeddings.shape}')
        x = tok_embeddings + pos_embeddings

        # transformer forward pass
        x = self.first_layer(x)
        x, attn = self.transformer(x)
        x = self.last_layer(x)

        logit = torch.matmul(x, self.tok_emb.weight.T) + self.bias   # Shared layer with the embedding

        if return_attn:  # return list of attention
            return logit[:, :self.num_tokens * self.num_tokens, :self.codebook_size + 1], attn
        
        return logit[:, :, :self.codebook_size+1]


class MaskTransformerNextFrameSep(MaskTransformerNextFrame):
    def __init__(self, img_size=256, patch_size=16, hidden_dim=768, codebook_size=16384, depth=24, heads=8, mlp_dim=3072, dropout=0.1, num_frames=5):
        super().__init__(img_size, patch_size, hidden_dim, codebook_size, depth, heads, mlp_dim, dropout, num_frames)
        #self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, (self.num_tokens*self.num_tokens), hidden_dim)), 0., 0.02)
        #self.bias = nn.Parameter(torch.zeros((self.num_tokens*self.num_tokens), codebook_size+1))
        self.tok_emb = nn.Embedding(codebook_size+1, hidden_dim//2)
        self.tok_emb2 = nn.Embedding(codebook_size+1, hidden_dim//2)

        self.last_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim//2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim//2, eps=1e-12),
        )

        self.last_layer2 = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim//2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim//2, eps=1e-12),
        )

        #self.bias2 = nn.Parameter(torch.zeros((self.num_tokens*self.num_tokens), codebook_size//2+1))
        self.bias2 = nn.Parameter(torch.zeros((self.num_frames*self.num_tokens*self.num_tokens), codebook_size+1))

    def forward(self, img_tokens, return_attn=False):
        if isinstance(img_tokens, tuple):
            img_token_rec = img_tokens[0]
            img_token_sem = img_tokens[1]
        
        b, f, w, h = img_token_rec.size()
        input_rec = img_token_rec.view(b, -1)
        input_sem = img_token_sem.view(b, -1)

        tok_embeddings_rec = self.tok_emb(input_rec)
        tok_embeddings_sem = self.tok_emb2(input_sem)

        # concatenate the two embeddings
        tok_embeddings = torch.cat((tok_embeddings_rec, tok_embeddings_sem), dim=-1)

        # Position embedding
        pos_embeddings = self.pos_emb
        x = tok_embeddings + pos_embeddings

        # transformer forward pass
        x = self.first_layer(x)
        x, attn = self.transformer(x)
        x_rec = self.last_layer(x)
        x_sem = self.last_layer2(x)

        logit_rec = torch.matmul(x_rec, self.tok_emb.weight.T) + self.bias   # Shared layer with the embedding
        logit_sem = torch.matmul(x_sem, self.tok_emb2.weight.T) + self.bias2   # Shared layer with the embedding

        return logit_rec[:, :, :self.codebook_size+1], logit_sem[:, :, :self.codebook_size+1]


class MaskTransformerNextFrameSep3x(MaskTransformerNextFrame):
    def __init__(self, img_size=256, patch_size=16, hidden_dim=768, codebook_size=16384, depth=24, heads=8, mlp_dim=3072, dropout=0.1, num_frames=5):
        super().__init__(img_size, patch_size, hidden_dim, codebook_size, depth, heads, mlp_dim, dropout, num_frames)
        #self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, (self.num_tokens*self.num_tokens), hidden_dim)), 0., 0.02)
        #self.bias = nn.Parameter(torch.zeros((self.num_tokens*self.num_tokens), codebook_size+1))
        self.tok_emb = nn.Embedding(codebook_size+1, hidden_dim//3)
        self.tok_emb2 = nn.Embedding(codebook_size+1, hidden_dim//3)
        self.tok_emb3 = nn.Embedding(codebook_size+1, hidden_dim//3)

        self.last_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim//3),
            nn.GELU(),
            nn.LayerNorm(hidden_dim//3, eps=1e-12),
        )

        self.last_layer2 = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim//3),
            nn.GELU(),
            nn.LayerNorm(hidden_dim//3, eps=1e-12),
        )

        self.last_layer3 = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim//3),
            nn.GELU(),
            nn.LayerNorm(hidden_dim//3, eps=1e-12),
        )

        #self.bias2 = nn.Parameter(torch.zeros((self.num_tokens*self.num_tokens), codebook_size//2+1))
        self.bias2 = nn.Parameter(torch.zeros((self.num_frames*self.num_tokens*self.num_tokens), codebook_size+1))
        self.bias3 = nn.Parameter(torch.zeros((self.num_frames*self.num_tokens*self.num_tokens), codebook_size+1))

    def forward(self, img_tokens, return_attn=False):
        if isinstance(img_tokens, tuple):
            img_token_rec = img_tokens[0]
            img_token_sem = img_tokens[1]
            img_token_depth = img_tokens[2]
        
        b, f, w, h = img_token_rec.size()
        input_rec = img_token_rec.view(b, -1)
        input_sem = img_token_sem.view(b, -1)
        input_depth = img_token_depth.view(b, -1)

        tok_embeddings_rec = self.tok_emb(input_rec)
        tok_embeddings_sem = self.tok_emb2(input_sem)
        tok_embeddings_depth = self.tok_emb3(input_depth)

        # concatenate the two embeddings
        tok_embeddings = torch.cat((tok_embeddings_rec, tok_embeddings_sem, tok_embeddings_depth), dim=-1)

        # Position embedding
        pos_embeddings = self.pos_emb
        x = tok_embeddings + pos_embeddings

        # transformer forward pass
        x = self.first_layer(x)
        x, attn = self.transformer(x)
        x_rec = self.last_layer(x)
        x_sem = self.last_layer2(x)
        x_depth = self.last_layer3(x)

        logit_rec = torch.matmul(x_rec, self.tok_emb.weight.T) + self.bias   # Shared layer with the embedding
        logit_sem = torch.matmul(x_sem, self.tok_emb2.weight.T) + self.bias2   # Shared layer with the embedding
        logit_depth = torch.matmul(x_depth, self.tok_emb3.weight.T) + self.bias3   # Shared layer with the embedding

        return logit_rec[:, :, :self.codebook_size+1], logit_sem[:, :, :self.codebook_size+1], logit_depth[:, :, :self.codebook_size+1]
