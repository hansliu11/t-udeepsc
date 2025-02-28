import math
import numpy as np
from timm.models.registry import register_model
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

import torch
import torch.nn as nn
from channel import *
from functools import partial
import torch.nn.functional as F


import timm
net = timm.create_model("vit_base_patch16_384", pretrained=True)




def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }
# nohup   > log_files/demo_t_14dB.log 2>&1 &
def noise_gen(is_train):
    min_snr, max_snr = -6, 18
    diff_snr = max_snr - min_snr
    
    min_var, max_var = 10**(-min_snr / 20), 10**(-max_snr / 20)
    diff_var = max_var - min_var
    if is_train:
        # b = torch.bernoulli(1 / 5.0 * torch.ones(1))
        # if b > 0.5:
        #     channel_snr = torch.FloatTensor([20])
        # else:               
        #     channel_snr = torch.rand(1) * diff_snr + min_snr
        # noise_var = 10**(-channel_snr / 20)
        
        # noise_var = torch.rand(1) * diff_var+min_var  
        # channel_snr = 10 * torch.log10((1 / noise_var)**2)
        
        # channel_snr = torch.rand(1) * diff_snr + min_snr
        # noise_var = 10**(-channel_snr/20)
        
        ## paper only train on 12 and -2
        channel_snr = torch.FloatTensor([12])
        noise_var = torch.FloatTensor([1]) * 10**(-channel_snr/20)  
    else:
        channel_snr = torch.FloatTensor([12])
        noise_var = torch.FloatTensor([1]) * 10**(-channel_snr/20)  
    return channel_snr, noise_var 



class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
class BertEmbed(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, vocab_size=30522, embed_dim=512, max_position_embedd=512, config=None):
        super().__init__()

        # Embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_position_embedd, embed_dim)
        self.token_type_embeddings = nn.Embedding(2, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text):
        """
            Args:
                inputs: a tensor with shape (batch_size, seq_len)

            Returns:
                a tensor with shape (batch_size, embed_dim, seq_len)

        """
        # Create position IDs
        position_ids = torch.arange(text.size(1), dtype=torch.long, device=text.device)
        position_ids = position_ids.unsqueeze(0).expand_as(text)

         # Get embeddings
        word_embeds = self.word_embeddings(text)
        position_embeds = self.position_embeddings(position_ids)
        token_type_ids = torch.zeros_like(text)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        # Combine embeddings
        embeds = word_embeds + position_embeds + token_type_embeds
        embeds = self.layer_norm(embeds)
        embeds = self.dropout(embeds)
        
        return embeds
    
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model)) #math.log(math.exp(1)) = 1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #[1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
        
        #self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)
        
        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)
        # print(key.shape)
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)

        x, self.attn = self.attention(query, key, value, mask=mask)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.num_heads * self.d_k)
             
        x = self.dense(x)
        x = self.dropout(x)
        
        return x
    
    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        #print(mask.shape)
        if mask is not None:
            scores += (mask * -1e9)
            # attention weights
        p_attn = F.softmax(scores, dim = -1)
        return torch.matmul(p_attn, value), p_attn

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x) 
        return x

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.src_mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout = 0.1)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        
        #self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        "Follow Figure 1 (right) for connections."
        attn_output = self.self_mha(x, x, x, look_ahead_mask)
        x = self.layernorm1(x + attn_output)
     
        src_output = self.src_mha(x, memory, memory, trg_padding_mask) # q, k, v
        x = self.layernorm2(x + src_output)
        
        fnn_output = self.ffn(x)
        x = self.layernorm3(x + fnn_output)
        return x


class Decoder(nn.Module):
    def __init__(self, depth=4, embed_dim=128, num_heads=4, dff=128, drop_rate=0.1):
        super(Decoder, self).__init__()
    
        self.d_model = embed_dim
        self.pos_encoding = PositionalEncoding(embed_dim, drop_rate, 50)
        self.dec_layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, dff, drop_rate) 
                                            for _ in range(depth)])
        
    def forward(self, x, memory, look_ahead_mask=None, trg_padding_mask=None):
        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, look_ahead_mask, trg_padding_mask)  
        return x


class ChannelDecoder(nn.Module):
    def __init__(self, 
                 num_symbols: int,
                 output_dim: int, 
                 ):
        super(ChannelDecoder, self).__init__()
        
        self.num_symbols = num_symbols
        self.output_dim = output_dim

        self.linear1 = nn.Linear(num_symbols, output_dim)
        self.linear2 = nn.Linear(output_dim, 256)
        self.linear3 = nn.Linear(256, output_dim)
        # self.linear4 = nn.Linear(size1, d_model)
        
        self.LN = nn.LayerNorm(output_dim, eps=1e-6)
        
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = F.relu(x1)
        x3 = self.linear2(x2)
        x4 = F.relu(x3)
        x5 = self.linear3(x4)
        
        # output = self.layernorm(x1 + x5)
        output = self.LN(x5)

        return output


class ViTEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed_imgr = PatchEmbed(
            img_size=32, patch_size=4, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_imgc = PatchEmbed(
            img_size=224, patch_size=32, in_chans=in_chans, embed_dim=embed_dim)
        self.linear_embed_vqa = nn.Linear(2048, self.embed_dim)
        self.linear_embed_msa = nn.Linear(35, self.embed_dim)
        num_patches_imgr = self.patch_embed_imgr.num_patches
        num_patches_imgc = self.patch_embed_imgc.num_patches
        # TODO: Add the cls token
        self.cls_token = {}
        self.cls_token['imgr'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token['imgc'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token['vqa'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token['msa'] = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.task_embedd = {}
        self.task_embedd['imgr'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.task_embedd['imgc'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.task_embedd['vqa'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.task_embedd['msa'] = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if use_learnable_pos_emb:
            self.pos_embed_imgc = nn.Parameter(torch.zeros(1, num_patches_imgc + 1, embed_dim))
            self.pos_embed_imgr = nn.Parameter(torch.zeros(1, num_patches_imgr + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed_imgc = get_sinusoid_encoding_table(num_patches_imgc + 1, embed_dim)
            self.pos_embed_imgr = get_sinusoid_encoding_table(num_patches_imgr + 1, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        for key in self.cls_token.keys():
            trunc_normal_(self.cls_token[key], std=.02)
            trunc_normal_(self.task_embedd[key], std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x, ta_perform):
        '''
            input: (batch, channel, img_w, img_h)
        '''
        if ta_perform.startswith('vqa'):
            x = self.linear_embed_vqa(x)
            batch_size = x.shape[0]
            cls_tokens = self.cls_token[ta_perform].expand(batch_size, -1, -1).to(x.device) 
            task_embedd = self.task_embedd[ta_perform].expand(batch_size, -1, -1).to(x.device)
            x = torch.cat((cls_tokens, x, task_embedd), dim=1)
        elif ta_perform.startswith('msa'):
            x = self.linear_embed_msa(x)
            batch_size = x.shape[0]
            cls_tokens = self.cls_token[ta_perform].expand(batch_size, -1, -1).to(x.device)
            task_embedd = self.task_embedd[ta_perform].expand(batch_size, -1, -1).to(x.device)
            x = torch.cat((cls_tokens, x, task_embedd), dim=1)
        elif ta_perform.startswith('img'):
            x = self.patch_embed_imgr(x) if ta_perform.startswith('imgr') else self.patch_embed_imgc(x)
            batch_size = x.shape[0]
            cls_tokens = self.cls_token[ta_perform].expand(batch_size, -1, -1).to(x.device) 
            task_embedd = self.task_embedd[ta_perform].expand(batch_size, -1, -1).to(x.device) 
            x = torch.cat((cls_tokens, x), dim=1)
            if ta_perform.startswith('imgr'):
                x = x + self.pos_embed_imgr.type_as(x).to(x.device).clone().detach()
            elif ta_perform.startswith('imgc'):
                x = x + self.pos_embed_imgc.type_as(x).to(x.device).clone().detach()
            x = torch.cat((x, task_embedd), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x, ta_perform):
        x = self.forward_features(x, ta_perform)
        return x


    

class SPTEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.linear_embed = nn.Linear(74, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.task_embedd = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.task_embedd, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, ta_perform):
        if ta_perform.startswith('msa'):
            x = self.linear_embed(x)
            batch_size = x.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(x.device) 
            task_embedd = self.task_embedd.expand(batch_size, -1, -1).to(x.device) 
            x = torch.cat((cls_tokens, x, task_embedd), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x, ta_perform=None):
        x = self.forward_features(x, ta_perform)
        return x



# class BertTextEncoder(nn.Module):
#     def __init__(self,  vocab_size=30522, embed_dim=512, num_hidden_layers=4, num_heads=8, intermediate_size=2048, max_position_embedd=512, config=None):
#         super().__init__()
#         self.num_features = self.embed_dim = embed_dim
        
#         if config is None:
#             config = BertConfig(
#                 hidden_size=embed_dim,
#                 num_hidden_layers=4,
#                 num_attention_heads=8,
#                 intermediate_size=2048,
#                 hidden_act="gelu",
#                 hidden_dropout_prob=0.1,
#                 attention_probs_dropout_prob=0.1,
#                 max_position_embeddings=512,
#                 type_vocab_size=2,
#                 initializer_range=0.02,
#                 layer_norm_eps=1e-12,
#                 pad_token_id=0,
#                 vocab_size=30522,
#             )
        
#         # self.bert_ckpt = f"/prajjwal1/bert-{mode}"
#         # self.bert = BertModel.from_pretrained(self.bert_ckpt)
        
#         # Embeddings
#         self.embeddings = BertEmbed(vocab_size, embed_dim, max_position_embedd, config)
        
#         self.cls_token = nn.ParameterDict({
#             'vqa': nn.Parameter(torch.zeros(1, 1, embed_dim)),
#             'msa': nn.Parameter(torch.zeros(1, 1, embed_dim)),
#             'textr': nn.Parameter(torch.zeros(1, 1, embed_dim)),
#             'textc': nn.Parameter(torch.zeros(1, 1, embed_dim)),
#         })
        
#         # Task embeddings
#         self.task_embedd = nn.ParameterDict({
#             'vqa': nn.Parameter(torch.zeros(1, 1, embed_dim)),
#             'msa': nn.Parameter(torch.zeros(1, 1, embed_dim)),
#             'textr': nn.Parameter(torch.zeros(1, 1, embed_dim)),
#             'textc': nn.Parameter(torch.zeros(1, 1, embed_dim)),
#         })
        
#         # Bert layers
#         self.encoder = BertEncoder(config)
    
#         self.pooler = BertPooler(config)
        
#         for key in self.cls_token.keys():
#             trunc_normal_(self.cls_token[key], std=.02)
#             trunc_normal_(self.task_embedd[key], std=.02)
        
#         self.apply(self._init_weights)
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def get_num_layers(self):
#         return len(self.blocks)
              
#     def forward_features(self, text, ta_perform=None):
#         """
#             Args:
#                 inputs: a tensor with shape (batch_size, seq_len)
            
#             Returns:
#                 a tensor with shape (batch_size, seq_len + 2, embed_dim)

#         """
#         # Text has been tokenized first
        
#         # Create position IDs
#         # position_ids = torch.arange(text.size(1), dtype=torch.long, device=text.device)
#         # position_ids = position_ids.unsqueeze(0).expand_as(text)
        
#         # if ta_perform.startswith('vqa'):
            
#         # elif ta_perform.startswith('msa'):
            
#         # elif ta_perform.startswith('text'):
                
#         # else:
#         #     raise ValueError(f"Task {ta_perform} not supported")
        
#         # # Get embeddings
#         # word_embeds = self.word_embedding(text)
#         # position_embeds = self.position_embedding(position_ids)
#         # token_type_ids = torch.zeros_like(text)
#         # token_type_embeds = self.token_type_embedding(token_type_ids)
        
#         # # Combine embeddings
#         # embeds = word_embeds + position_embeds + token_type_embeds
#         # embeds = self.layer_norm(embeds)
#         # embeds = self.dropout(embeds)
        
#         embeds = self.embeddings(text)
        
#         batch_size = embeds.shape[0]
#         # cls_tokens = self.cls_token[ta_perform].expand(batch_size, -1, -1).to(embeds.device) 
#         task_embedd = self.task_embedd[ta_perform].expand(batch_size, -1, -1).to(embeds.device)
#         embeds = torch.cat((embeds, task_embedd), dim=1)
        
#         encoder_outputs = self.encoder(embeds)
#         sequence_output = encoder_outputs[0]
#         pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

#         return (sequence_output, pooled_output) + encoder_outputs[1:]
        
#     def forward(self, text, ta_perform):
#         return self.forward_features(text, ta_perform)


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents, SNRdB, bit_per_index):
        latents = latents # [B x L x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BL x D]
        device = latents.device
        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BL x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BL, 1]
        shape = encoding_inds.shape
        Rx_signal = transmit(encoding_inds, SNRdB, bit_per_index)
        encoding_inds = torch.from_numpy(Rx_signal).to(device).reshape(shape)
        
        # Convert to one-hot encodings   
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BL x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BL, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x L x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        #print((quantized_latents - latents))
        return quantized_latents.contiguous(), vq_loss  # [B x L x D]
    

class Channels():    
    def AWGN_Var(self, Tx_sig, n_std):
        device = Tx_sig.device
        noise = torch.normal(0, n_std/math.sqrt(2), size=Tx_sig.shape).to(device)
        Rx_sig = Tx_sig + noise
        return Rx_sig

    def AWGN(self, Tx_sig: torch.Tensor, SNRdb: float, ouput_power=False):
        '''
            make AWGN noise for a signal (applying the signal on it)
            the noise for a signal will be n ~ CN(0, sigma^2 * I),
            where CN is circularly symmetric complex Gaussian distribution

            Args:
                Tx_sig: complex tensor with any shape
                SNRdb: signal-to-noise ratio in decibel
                        can be float or a real tensor (same shape and device as signal)
            Return:
                the signal with complex noise applying onto it
        '''
        device = Tx_sig.device
        signal_power = torch.mean(Tx_sig.abs()**2, dim=-1, keepdim=True)
        snr_linear = 10 ** (SNRdb / 10.0)
        sigma2 = signal_power / snr_linear # calculate noise power based on signal power and SNR
        if ouput_power:
            print ("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))     
        # Generate complex noise with given variance
        # noise = torch.sqrt(sigma2 / 2) * (torch.randn_like(Tx_sig.real) + 1j*torch.randn_like(Tx_sig.imag)).to(device)
        noise_real = torch.randn_like(Tx_sig.real) * torch.sqrt(sigma2 / 2)
        noise_imag = torch.randn_like(Tx_sig.imag) * torch.sqrt(sigma2 / 2)
        noise = torch.complex(noise_real, noise_imag).to(device)
        return noise.add_(Tx_sig)

    def AWGNMultiChannel(self, Tx_sig: torch.Tensor, SNRdb: float, user_dim_index: int):
        """
            AWGN channel for multiple users
            Superimpose signal before adding noise
        """
        dim = tuple(Tx_sig.size()[user_dim_index + 1:-1])
        batch_size = Tx_sig.size()[0]
        symbol_dim = Tx_sig.size()[-1]
        
        # make superimposed signal
        Tx_sig = torch.sum(Tx_sig, dim=user_dim_index) 
        Tx_sig = Tx_sig.view(batch_size, 1, *dim, symbol_dim)

        return self.AWGN(Tx_sig, SNRdb)
 
    def Rayleigh(self, Tx_sig, SNRdb):
        # slow fading -> all symbol use same channel gain
        device = Tx_sig.device
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, SNRdb)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        return Rx_sig

    def Rician(self, Tx_sig, n_std, K=1):
        device = Tx_sig.device
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_std)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        return Rx_sig

class SIC(nn.Module):
    def __init__(self, 
                 img_embed_dim: int, 
                 text_embed_dim: int, 
                 speech_embed_dim: int, 
                 num_antennas :int):
        super(SIC, self).__init__()
        """
            Signal detection by channel reconstruction successive interference cancellation
            paper ref.
                - https://ieeexplore.ieee.org/abstract/document/10233614?casa_token=qufCIaPszeYAAAAA:ujSza85Bd7w0j07U8hyQbtGNGHd0mqYw3A7-xwDGcllLhEDxdRiPnmIuwkUhVDI-7gjrkRZFqw
        """
        
        self.img_channel_decoder  =  nn.Linear(num_antennas, img_embed_dim)
        self.text_channel_decoder  = nn.Linear(num_antennas, text_embed_dim)
        self.spe_channel_decoder  =  nn.Linear(num_antennas, speech_embed_dim)
        
        self.channel_decoders = nn.ModuleList([self.text_channel_decoder, self.img_channel_decoder, self.spe_channel_decoder])
        
    def forward(self, signal: torch.Tensor, user_dim_index: int, power_constraints: list[float], channel_encoders: list[nn.Module], sig_power:list, channel_type: Literal['AWGN', 'Fading'], h=None):
        """            
            Args
                signal: real tensor in (batch_size, 1, *dim, symbol_dim)
                power_constraint: the power constraint for users, length n_user (The order is [text, img, speech])
                h: channel gain (Rayleigh or Rician)
            Return
                decode signal of shape (batch_size, n_user, *dim, symbol_dim)
        """
        device = signal.device
        batch_size = signal.size()[0]
        num_elements = signal.size()[-1]
        dim = tuple(signal.size()[user_dim_index + 1:])

        num_users = len(power_constraints)
        K = num_elements // 2 # number of complex symbols
        
        if(num_users == 1):
            return signal

        if channel_type == "AWGN":
            # Sort users by transmit power (descending order)
            user_indices = np.argsort(power_constraints)[::-1]
        else: # Rayleigh or Rician
            if h is None:
                raise ValueError("Channel gains (h) must be provided for Rayleigh channel.")
            # Compute effective received power |h_i|^2 * P_i
            effective_power = torch.abs(h)**2 * power_constraints
            user_indices = np.argsort(effective_power)[::-1]


        decoded_signals = torch.zeros((batch_size, num_users, *dim)).to(device)
        remaining_signal  = signal.clone().detach().to(device)

        for i in user_indices:
            """
                Decoding steps:
                    1. Use channel decoder to decode stronger signal
                    
                    2. Channel encoding (same as transmitter) the signal to simulate the signal state of the transmitter
                    
                    3. Subtract encoded estimated signal from Y
                    
                    4. Repeat until all signal been detected
            """
            
            if channel_type == "AWGN":
                estimated = remaining_signal[:, 0, :]
                estimated = self.channel_decoders[i](estimated)
                estimated = channel_encoders[i](estimated)
                # print(estimated.shape)
                
                decoded_signals[:,i,:] = estimated
                power_constr = torch.full((batch_size, 1), power_constraints[i]).to(device)
                sig = estimated.clone().detach().to(device)
                sig = torch.flatten(sig, start_dim=1)

                estimated_pow = torch.sqrt((power_constr * K) / torch.sum(sig ** 2, dim=1, keepdim=True))
                estimated_pow = estimated_pow.view(batch_size, *[1] * (remaining_signal.ndim - 1))
                
                remaining_signal = remaining_signal - (estimated_pow * estimated)
            else: # Rayleigh or Rician
                if h is None:
                    raise ValueError("Channel gains (h) must be provided for Rayleigh channel.")
                # Estimate the i-th user's signal based on remaining signal
                estimated_signal = remaining_signal / (h[i] * sig_power[i])  
                current_signal = torch.abs(estimated_signal)
                decoded_signals[:,i,:] = current_signal
                
                # subtract decoded signal from received signal
                remaining_signal -= h[i] * current_signal

        return decoded_signals