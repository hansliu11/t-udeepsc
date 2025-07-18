import math
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import *

from channel import *
from model_util import *
from functools import partial
from trans_decoder import Decoder
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from transformers import BertModel
from model_util import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from base_args import IMGC_NUMCLASS,TEXTC_NUMCLASS,IMGR_LENGTH,TEXTR_NUMCLASS,VQA_NUMCLASS,MSA_NUMCLASS, AVE_NUMCLASS


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

__all__ = [
    'UDeepSC_model']


class UDeepSC_M1(nn.Module):
    def __init__(self,mode='tiny',
                 img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0, 
                 img_embed_dim=384, text_embed_dim=384, speech_embed_dim=128, img_encoder_depth=4, 
                 text_encoder_depth=4, speech_encoder_depth=4, encoder_num_heads=12, decoder_num_classes=768, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, init_values=0.,use_learnable_pos_emb=False,num_classes=0, 
                 ):

        super().__init__()
        self.img_encoder = ViTEncoder(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, 
                                num_classes=encoder_num_classes, embed_dim=img_embed_dim,depth=img_encoder_depth,
                                num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        
        # bert_ckpt = f"/Data1/zhangguangyi/SemanRes2/JSACCode/UDeepSC_Base/pretrained_models/bert-{mode}"
        bert_ckpt = f"/prajjwal1/bert-{mode}"
        self.text_encoder = BertModel.from_pretrained(bert_ckpt) 
        
        self.spe_encoder = SPTEncoder(in_chans=encoder_in_chans,num_classes=encoder_num_classes, embed_dim=speech_embed_dim,
                                depth=speech_encoder_depth,num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        
        if mode=='tiny':
            text_embed_dim = 128
        elif mode=='small':
            text_embed_dim = 512
        else:
            text_embed_dim = 512
        
        self.num_symbols_img = 16
        self.num_symbols_text = 6
        self.num_symbols_spe = 16

        self.text_encoder_to_channel = nn.Linear(text_embed_dim, self.num_symbols_text)
        self.img_encoder_to_channel = nn.Linear(img_embed_dim, self.num_symbols_img)
        self.spe_encoder_to_channel = nn.Linear(speech_embed_dim, self.num_symbols_spe)

        self.text_channel_to_decoder = nn.Linear(self.num_symbols_text, decoder_embed_dim)
        self.img_channel_to_decoder = nn.Linear(self.num_symbols_img, decoder_embed_dim)
        self.spe_channel_to_decoder = nn.Linear(self.num_symbols_spe, decoder_embed_dim)


        self.task_dict = nn.ModuleDict()
        self.task_dict['imgc'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['imgr'] = nn.Embedding(64, decoder_embed_dim)
        self.task_dict['textc'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['vqa'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['msa']  = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['textr'] = nn.Embedding(66, decoder_embed_dim)


        self.head = nn.ModuleDict()
        self.head['imgc'] = nn.Linear(decoder_embed_dim, IMGC_NUMCLASS)
        self.head['textc'] = nn.Linear(decoder_embed_dim, TEXTC_NUMCLASS)
        self.head['textr'] = nn.Linear(decoder_embed_dim, TEXTR_NUMCLASS)
        self.head['vqa'] = nn.Linear(decoder_embed_dim, VQA_NUMCLASS)
        self.head['imgr'] = nn.Linear(decoder_embed_dim, IMGR_LENGTH)
        self.head['msa'] = nn.Linear(decoder_embed_dim, MSA_NUMCLASS)


        self.decoder = Decoder(depth=decoder_depth,embed_dim=decoder_embed_dim, 
                                num_heads=decoder_num_heads, dff=mlp_ratio*decoder_embed_dim, 
                                drop_rate=drop_rate)
        self.channel = Channels()
        self.sigmoid_layer = nn.Sigmoid()

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
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, text=None, img=None, speech=None, ta_perform=None, test_snr=torch.FloatTensor([-2])):
        if self.training:
            noise_snr, noise_std = noise_gen(self.training)
            noise_std,noise_snr = noise_std.cuda(), noise_snr.cpu().item()
        else:
            noise_std = torch.FloatTensor([1]) * 10**(-test_snr/20) 
        if text is not None:
            x_text = self.text_encoder(ta_perform, text, return_dict=False)[0]
            x_text = self.text_encoder_to_channel(x_text)

            if ta_perform.startswith('textc'):
                x_text = x_text[:,0,:].unsqueeze(1)
            elif ta_perform.startswith('textr'):
                x_text = x_text[:,1:-1,:]  
            elif ta_perform.startswith('vqa'):
                x_text = x_text[:,0:2,:]
            elif ta_perform.startswith('msa'):
                x_text = x_text[:,0].unsqueeze(1)

            x_text = power_norm_batchwise(x_text)
            x_text = self.channel.AWGN(x_text, noise_std.item())
            x_text = self.text_channel_to_decoder(x_text)
        if img is not None:
            x_img = self.img_encoder(img, ta_perform)
            x_img = self.img_encoder_to_channel(x_img)
            if ta_perform.startswith('imgc'):
                x_img = x_img[:,0,:].unsqueeze(1)
            elif ta_perform.startswith('imgr'):
                x_img = x_img[:,1:-1,:]
            elif ta_perform.startswith('vqa'):
                x_img = x_img[:,0:3,:]
            elif ta_perform.startswith('msa'):
                x_img = x_img[:,0,:].unsqueeze(1)
            x_img = power_norm_batchwise(x_img)
            x_img = self.channel.AWGN(x_img, noise_std.item())
            x_img = self.img_channel_to_decoder(x_img)
            
        
        if speech is not None:
            x_spe = self.spe_encoder(speech, ta_perform)
            x_spe = self.spe_encoder_to_channel(x_spe)
            x_spe = x_spe[:,0,:].unsqueeze(1)
            x_spe = power_norm_batchwise(x_spe)
            x_spe = self.channel.AWGN(x_spe, noise_std.item())
            x_spe = self.spe_channel_to_decoder(x_spe)
        
        if ta_perform.startswith('img'):
            x = x_img
        elif ta_perform.startswith('text'):
            x = x_text
        elif ta_perform.startswith('vqa'):
            x = torch.cat([x_img,x_text], dim=1)
        elif ta_perform.startswith('msa'):
            x = torch.cat([x_img,x_text,x_spe], dim=1)

        batch_size = x.shape[0]
        if ta_perform.endswith('r'):
            x = self.decoder(x, x, None, None, None) 
            x = self.head[ta_perform](x)
            return x
        else:
            query_embed = self.task_dict[ta_perform].weight.unsqueeze(0).repeat(batch_size, 1, 1)
            x = self.decoder(query_embed, x, None, None, None) 
            if ta_perform.startswith('textr'): 
                x = self.head[ta_perform](x)
            else:
                x = self.head[ta_perform](x.mean(1))
            if ta_perform.startswith('vqa'):
                x = self.sigmoid_layer(x)
            return x


class UDeepSC_M2(nn.Module):
    """
        Similar to UDeepSC_M1, but 
            1. include more details (e.g. symbol number) for each task
            2. extract transmit part as a function
    """
    def __init__(self,mode='tiny',
                 img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0, 
                 img_embed_dim=384, text_embed_dim=384, speech_embed_dim=128, img_encoder_depth=4, 
                 text_encoder_depth=4, speech_encoder_depth=4, encoder_num_heads=12, decoder_num_classes=768, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, init_values=0.,use_learnable_pos_emb=False,num_classes=0, 
                 ):

        super().__init__()
        
        if mode=='tiny':
            text_embed_dim = 128
        elif mode=='small':
            text_embed_dim = 512
        else:
            text_embed_dim = 512
            
        self.img_encoder = ViTEncoder(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, 
                                num_classes=encoder_num_classes, embed_dim=img_embed_dim,depth=img_encoder_depth,
                                num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        
        print(f"Load pre-train weight from bert-{mode}")
        
        bert_ckpt = f"prajjwal1/bert-{mode}"
        self.text_encoder = BertModel.from_pretrained(bert_ckpt)
        # text_encoder_pretrained = BertModel.from_pretrained(bert_ckpt)

        # Get state_dict of pretrained bert
        # pretrained_state_dict = text_encoder_pretrained.state_dict()
        
        # self.text_encoder = BertTextEncoder(embed_dim=text_embed_dim)
        # Load pretrained state_dict
        # self.text_encoder.load_state_dict(pretrained_state_dict, strict=False)
        
        self.spe_encoder = SPTEncoder(in_chans=encoder_in_chans,num_classes=encoder_num_classes, embed_dim=speech_embed_dim,
                                depth=speech_encoder_depth,num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        
        self.num_symbols_imgc = 16
        self.num_symbols_imgr = 16
        self.num_symbols_textc = 4
        self.num_symbols_textr = 24
        self.num_symbols_vqa_img = 16
        self.num_symbols_vqa_text = 6
        self.num_symbols_msa_img = 16
        self.num_symbols_msa_text = 6
        self.num_symbols_msa_spe = 16

        self.num_symbols_vqa_max = max(self.num_symbols_vqa_img, self.num_symbols_vqa_text)
        self.num_symbols_msa_max = max(self.num_symbols_msa_img, self.num_symbols_msa_text, self.num_symbols_msa_spe)

        self.textc_encoder_to_channel =     nn.Linear(text_embed_dim, self.num_symbols_textc)
        self.imgc_encoder_to_channel =      nn.Linear(img_embed_dim, self.num_symbols_imgc)
        self.textr_encoder_to_channel =     nn.Linear(text_embed_dim, self.num_symbols_textr)
        self.imgr_encoder_to_channel =      nn.Linear(img_embed_dim, self.num_symbols_imgr)
        self.vqa_img_encoder_to_channel =   nn.Linear(img_embed_dim, self.num_symbols_vqa_img)
        self.vqa_text_encoder_to_channel =  nn.Linear(text_embed_dim, self.num_symbols_vqa_text)
        self.msa_img_encoder_to_channel =   nn.Linear(img_embed_dim, self.num_symbols_msa_img)
        self.msa_text_encoder_to_channel =  nn.Linear(text_embed_dim, self.num_symbols_msa_text)
        self.msa_spe_encoder_to_channel =   nn.Linear(speech_embed_dim, self.num_symbols_msa_spe)
        
        
        self.textc_channel_to_decoder  =    nn.Linear(self.num_symbols_textc, decoder_embed_dim)
        self.imgc_channel_to_decoder  =     nn.Linear(self.num_symbols_imgc, decoder_embed_dim)
        self.textr_channel_to_decoder  =    nn.Linear(self.num_symbols_textr, decoder_embed_dim)
        self.imgr_channel_to_decoder =      nn.Linear(self.num_symbols_imgr, decoder_embed_dim)
        self.vqa_img_channel_to_decoder  =  nn.Linear(self.num_symbols_vqa_img, decoder_embed_dim)
        self.vqa_text_channel_to_decoder  = nn.Linear(self.num_symbols_vqa_max, decoder_embed_dim)
        self.msa_img_channel_to_decoder  =  nn.Linear(self.num_symbols_msa_img, decoder_embed_dim)
        self.msa_text_channel_to_decoder  = nn.Linear(self.num_symbols_msa_max, decoder_embed_dim)
        self.msa_spe_channel_to_decoder  =  nn.Linear(self.num_symbols_msa_spe, decoder_embed_dim)
        

        self.task_dict = nn.ModuleDict()
        self.task_dict['imgc'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['imgr'] = nn.Embedding(64, decoder_embed_dim)
        self.task_dict['textc'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['vqa'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['msa']  = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['textr'] = nn.Embedding(66, decoder_embed_dim)


        self.head = nn.ModuleDict()
        self.head['imgc'] = nn.Linear(decoder_embed_dim, IMGC_NUMCLASS)
        self.head['textc'] = nn.Linear(decoder_embed_dim, TEXTC_NUMCLASS)
        self.head['textr'] = nn.Linear(decoder_embed_dim, TEXTR_NUMCLASS) ## shape( , , 34000)
        self.head['vqa'] = nn.Linear(decoder_embed_dim, VQA_NUMCLASS)
        self.head['imgr'] = nn.Linear(decoder_embed_dim, IMGR_LENGTH) ## shape (, , 48)
        self.head['msa'] = nn.Linear(decoder_embed_dim, MSA_NUMCLASS)


        self.decoder = Decoder(depth=decoder_depth, embed_dim=decoder_embed_dim, 
                                num_heads=decoder_num_heads, dff=mlp_ratio*decoder_embed_dim, 
                                drop_rate=drop_rate)
        self.channel = Channels()
        self.sigmoid_layer = nn.Sigmoid()

        # self.LN = nn.LayerNorm(text_embed_dim)
        
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
    
    def transmit(self, x, SNRdb, channel_to_decoder, power_constraint: float):
        # x = encoder_to_channel(input_signal)
        x, _ = power_norm_batchwise(x, power_constraint)
        
        x = tensor_real2complex(x, 'concat')
        # x = self.channel.AWGN_Var(x, noise_std)
        x = self.channel.AWGN(x, SNRdb.item())
        x = tensor_complex2real(x, 'concat')
        
        x = channel_to_decoder(x)
        return x
    
    def pad_to_length(self, x, target_length):
        current_length = x.shape[-1]  
        padding_size = target_length - current_length
        
        if padding_size > 0:
            return F.pad(x, (0, padding_size), "constant", 0)
        else:
            return x 

    def get_signals(self, text=None, img=None, speech=None, ta_perform=None, SNRdb:torch.FloatTensor=torch.FloatTensor([12])):
        """
            just get the signal, for testing purposes
            almost the same as forward()
            
            Args:
                inputs: 
                    text, image, speech data for task
                    ta_perform: target task need to perform
                    SNRdb: SNR of the channel for test
            Returns:
                A dict of at most 3 modalities signal list, each list include:
                0: signal before channel
                1: signal after tranmitting through channel
        """
        signals = {}
        noise_std = torch.FloatTensor([1]) * 10**(-SNRdb/20) 
        noise_snr = SNRdb
        
        if text is not None:
            x_text = self.text_encoder(text, ta_perform)[0]
            if ta_perform.startswith('textc'):
                x_text = x_text[:,0,:].unsqueeze(1)
                encoder_to_channel = self.textc_encoder_to_channel
                channel_to_decoder = self.textc_channel_to_decoder
            elif ta_perform.startswith('textr'):
                x_text = x_text[:,1:-1,:]  
                encoder_to_channel = self.textr_encoder_to_channel
                channel_to_decoder = self.textr_channel_to_decoder
            elif ta_perform.startswith('vqa'):
                x_text = x_text[:,0:2,:]
                encoder_to_channel = self.vqa_text_encoder_to_channel
                channel_to_decoder = self.vqa_text_channel_to_decoder
            elif ta_perform.startswith('msa'):
                x_text = x_text[:,-2:-1,:]
                encoder_to_channel = self.msa_text_encoder_to_channel
                channel_to_decoder = self.msa_text_channel_to_decoder
            
            # x_text = self.transmit(x_text_before_ch, noise_snr,encoder_to_channel, channel_to_decoder)
            
            x_text = encoder_to_channel(x_text)
            x_text_before_ch = power_norm_batchwise(x_text)
            x_text = tensor_real2complex(x_text_before_ch, 'concat')
            x_text = self.channel.AWGN(x_text, SNRdb.item())
            x_text = tensor_complex2real(x_text, 'concat')
            
            signals['text'] = [x_text_before_ch, x_text]
            
        if img is not None:
            x_img = self.img_encoder(img, ta_perform)
            if ta_perform.startswith('imgc'):
                x_img = x_img[:,0,:].unsqueeze(1)
                encoder_to_channel = self.imgc_encoder_to_channel
                channel_to_decoder = self.imgc_channel_to_decoder
                
            elif ta_perform.startswith('imgr'):
                x_img = x_img[:,1:-1,:]
                encoder_to_channel = self.imgr_encoder_to_channel
                channel_to_decoder = self.imgr_channel_to_decoder
                
            elif ta_perform.startswith('vqa'):
                x_img = x_img[:,0:3,:]
                encoder_to_channel = self.vqa_img_encoder_to_channel
                channel_to_decoder = self.vqa_img_channel_to_decoder
                
            elif ta_perform.startswith('msa'):
                x_img = x_img[:,0,:].unsqueeze(1)
                encoder_to_channel = self.msa_img_encoder_to_channel
                channel_to_decoder = self.msa_img_channel_to_decoder
    
            # x_img = self.transmit(x_img_before_ch, noise_snr,encoder_to_channel, channel_to_decoder)
            
            x_img = encoder_to_channel(x_img)
            x_img_before_ch = power_norm_batchwise(x_img)
            x_img = tensor_real2complex(x_img_before_ch, 'concat')
            x_img = self.channel.AWGN(x_img, SNRdb.item())
            x_img = tensor_complex2real(x_img, 'concat')
            
            signals['img'] = [x_img_before_ch, x_img]
        
        if speech is not None:
            x_spe = self.spe_encoder(speech, ta_perform)
            x_spe = x_spe[:,0,:].unsqueeze(1)
           
            # x_spe = self.transmit(x_spe_before_ch, noise_snr, self.msa_spe_encoder_to_channel, self.msa_spe_channel_to_decoder)
            
            x_spe = self.msa_spe_encoder_to_channel(x_spe)
            x_spe_before_ch = power_norm_batchwise(x_spe)
            x_spe = tensor_real2complex(x_spe_before_ch, 'concat')
            x_spe = self.channel.AWGN(x_spe, SNRdb.item())
            x_spe = tensor_complex2real(x_spe, 'concat')
            
            signals['spe'] = [x_spe_before_ch, x_spe]
            
        return signals

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, text=None, img=None, speech=None, ta_perform=None, power_constraint:list[float]=[1, 1, 1], test_snr:torch.FloatTensor=torch.FloatTensor([12])):
        """
            Input:
                text: text data (tokenized first) for task need text
                img: image data for task need image
                speech: audio data for task need speech
                ta_perform: The task to be eval (execute)
                power_constraint: The power constraint of each user [text, img, speech]
            Output:
                Task result executed by decoder of receiver
                Different shape for different tasks and data type
                Ex:
                    Textr: (batch_size, seq length, vocab size) for vacab size = 34000
        """
        
        if self.training:
            noise_snr, noise_std = noise_gen(self.training)
            noise_std = noise_std.cuda()
        else:
            noise_std = torch.FloatTensor([1]) * 10**(-test_snr/20) 
            noise_snr = test_snr
            # print(f"SNR: {noise_snr}")
        if text is not None:
            x_text = self.text_encoder(ta_perform, text, return_dict=False)[0]
            power = power_constraint[0]
            # x_text = self.text_encoder(text, ta_perform)[0]
            # x_text = self.LN(x_text)
            if ta_perform.startswith('textc'):
                x_text = x_text[:,0,:].unsqueeze(1)
                x_text = self.textc_encoder_to_channel(x_text)
                x_text = self.transmit(x_text, noise_snr, self.textc_channel_to_decoder, power)
            elif ta_perform.startswith('textr'):
                x_text = x_text[:,1:-1,:]  
                x_text = self.textr_encoder_to_channel(x_text)
                x_text = self.transmit(x_text, noise_snr, self.textr_channel_to_decoder, power)
            elif ta_perform.startswith('vqa'):
                x_text = x_text[:,0:2,:]
                x_text = self.vqa_text_encoder_to_channel(x_text)
                # padding to longest symbol numbers
                x_text = self.pad_to_length(x_text, self.num_symbols_vqa_max)
                x_text = self.transmit(x_text, noise_snr, self.vqa_text_channel_to_decoder, power)
            elif ta_perform.startswith('msa'):
                x_text = x_text[:,-2:-1,:]
                x_text = self.msa_text_encoder_to_channel(x_text)
                # padding to longest symbol numbers
                x_text = self.pad_to_length(x_text, self.num_symbols_msa_max)
                x_text = self.transmit(x_text, noise_snr, self.msa_text_channel_to_decoder, power)

            
        if img is not None:
            x_img = self.img_encoder(img, ta_perform)
            if ta_perform.startswith('imgc'):
                power = power_constraint[0]
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img = self.imgc_encoder_to_channel(x_img)
                x_img = self.transmit(x_img, noise_snr, self.imgc_channel_to_decoder, power)
                
            elif ta_perform.startswith('imgr'):
                power = power_constraint[0]
                x_img = x_img[:,1:-1,:]
                x_img = self.imgr_encoder_to_channel(x_img)
                x_img = self.transmit(x_img, noise_snr, self.imgr_channel_to_decoder, power)
                
            elif ta_perform.startswith('vqa'):
                power = power_constraint[1]
                x_img = x_img[:,0:3,:]
                x_img = self.vqa_img_encoder_to_channel(x_img)
                x_img = self.transmit(x_img, noise_snr, self.vqa_img_channel_to_decoder, power)
                
            elif ta_perform.startswith('msa'):
                power = power_constraint[1]
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img = self.msa_img_encoder_to_channel(x_img)
                x_img = self.transmit(x_img, noise_snr, self.msa_img_channel_to_decoder, power)

        if speech is not None:
            power = power_constraint[2]
            x_spe = self.spe_encoder(speech, ta_perform)
            x_spe = x_spe[:,0,:].unsqueeze(1)
            x_spe = self.msa_spe_encoder_to_channel(x_spe)
            x_spe = self.transmit(x_spe, noise_snr, self.msa_spe_channel_to_decoder, power)
        
        if ta_perform.startswith('img'):
            x = x_img
        elif ta_perform.startswith('text'):
            x = x_text
        elif ta_perform.startswith('vqa'):
            x = torch.cat([x_img,x_text], dim=1)
        elif ta_perform.startswith('msa'):
            # print(x_img.shape, x_text.shape, x_spe.shape) # (batch_size, 1, 128)
            x = torch.cat([x_img,x_text,x_spe], dim=1)
            # print(x.shape) # (batch_size, 3, 128)

        batch_size = x.shape[0]
        if ta_perform.endswith('r'):
            x = self.decoder(x, x, None, None, None) 
            x = self.head[ta_perform](x)
            return x
        else:
            query_embed = self.task_dict[ta_perform].weight.unsqueeze(0).repeat(batch_size, 1, 1)
            x = self.decoder(query_embed, x, None, None, None) 
            if ta_perform.startswith('textr'): 
                x = self.head[ta_perform](x)
            else:
                x = self.head[ta_perform](x.mean(1))
            if ta_perform.startswith('vqa'):
                x = self.sigmoid_layer(x)
            return x

class UDeepSC_M3(nn.Module):
    """
        UDeepSC with channel decoder reconstruct signal first version
    """

    def __init__(self,mode='tiny',
                 img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0, 
                 img_embed_dim=384, text_embed_dim=384, speech_embed_dim=128, img_encoder_depth=4, 
                 text_encoder_depth=4, speech_encoder_depth=4, encoder_num_heads=12, decoder_num_classes=768, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, init_values=0.,use_learnable_pos_emb=False,num_classes=0, 
                 ):

        super(UDeepSC_M3, self).__init__()

        # self.decoder_embed_dim = decoder_embed_dim
        
        if mode=='tiny':
            text_embed_dim = 128
        elif mode=='small':
            text_embed_dim = 512
        else:
            text_embed_dim = 512
        
        self.img_encoder = ViTEncoder(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, 
                                num_classes=encoder_num_classes, embed_dim=img_embed_dim,depth=img_encoder_depth,
                                num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        
        bert_ckpt = f"prajjwal1/bert-{mode}"
        self.text_encoder = BertModel.from_pretrained(bert_ckpt)
        
        self.spe_encoder = SPTEncoder(in_chans=encoder_in_chans,num_classes=encoder_num_classes, embed_dim=speech_embed_dim,
                                depth=speech_encoder_depth,num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        
        self.img2_encoder = ViTEncoder(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, 
                                num_classes=encoder_num_classes, embed_dim=img_embed_dim,depth=img_encoder_depth,
                                num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        
        self.num_symbols_imgc = 16
        self.num_symbols_imgr = 16
        self.num_symbols_textc = 4
        self.num_symbols_textr = 24
        # self.num_symbols_vqa_img = 16
        # self.num_symbols_vqa_text = 16
        # self.num_symbols_msa_img = 16
        # self.num_symbols_msa_text = 16
        # self.num_symbols_msa_spe = 16
        self.num_symbols_vqa = 16
        self.num_symbols_msa = 16
        self.num_symbols_ave = 16
 
        # channel encoder
        self.textc_encoder_to_channel =     nn.Linear(text_embed_dim, self.num_symbols_textc)
        self.imgc_encoder_to_channel =      nn.Linear(img_embed_dim, self.num_symbols_imgc)
        self.textr_encoder_to_channel =     nn.Linear(text_embed_dim, self.num_symbols_textr)
        self.imgr_encoder_to_channel =      nn.Linear(img_embed_dim, self.num_symbols_imgr)
        self.vqa_img_encoder_to_channel =   nn.Linear(img_embed_dim, self.num_symbols_vqa)
        self.vqa_text_encoder_to_channel =  nn.Linear(text_embed_dim, self.num_symbols_vqa)
        self.msa_img_encoder_to_channel =   nn.Linear(img_embed_dim, self.num_symbols_msa)
        self.msa_text_encoder_to_channel =  nn.Linear(text_embed_dim, self.num_symbols_msa)
        self.msa_spe_encoder_to_channel =   nn.Linear(speech_embed_dim, self.num_symbols_msa)
        self.ave_img_encoder_to_channel =   nn.Linear(img_embed_dim, self.num_symbols_ave)
        self.ave_spe_encoder_to_channel =   nn.Linear(speech_embed_dim, self.num_symbols_ave)
        self.ave_img2_encoder_to_channel =   nn.Linear(img_embed_dim, self.num_symbols_ave)
        
        # channel decoder
        self.textc_channel_decoder = ChannelDecoder(self.num_symbols_textc, text_embed_dim)
        self.imgc_channel_decoder  = ChannelDecoder(self.num_symbols_imgc, img_embed_dim)
        self.vqa_img_channel_decoder  = ChannelDecoder(self.num_symbols_vqa, img_embed_dim)
        self.vqa_text_channel_decoder = ChannelDecoder(self.num_symbols_vqa, text_embed_dim)
        self.msa_img_channel_decoder  = ChannelDecoder(self.num_symbols_msa, img_embed_dim)
        self.msa_text_channel_decoder = ChannelDecoder(self.num_symbols_msa, text_embed_dim)
        self.msa_spe_channel_decoder  = ChannelDecoder(self.num_symbols_msa, speech_embed_dim)
        self.ave_img_channel_decoder  = ChannelDecoder(self.num_symbols_ave, img_embed_dim)
        self.ave_spe_channel_decoder  = ChannelDecoder(self.num_symbols_ave, speech_embed_dim)
        self.ave_img2_channel_decoder  = ChannelDecoder(self.num_symbols_ave, img_embed_dim)

        self.textc_channel_to_decoder  =    nn.Linear(text_embed_dim, decoder_embed_dim)
        self.imgc_channel_to_decoder  =     nn.Linear(img_embed_dim, decoder_embed_dim)
        self.vqa_img_channel_to_decoder  =  nn.Linear(img_embed_dim, decoder_embed_dim)
        self.vqa_text_channel_to_decoder  = nn.Linear(text_embed_dim, decoder_embed_dim)
        self.msa_img_channel_to_decoder  =  nn.Linear(img_embed_dim, decoder_embed_dim)
        self.msa_text_channel_to_decoder  = nn.Linear(text_embed_dim, decoder_embed_dim)
        self.msa_spe_channel_to_decoder  =  nn.Linear(speech_embed_dim, decoder_embed_dim)
        self.ave_img_channel_to_decoder  =  nn.Linear(img_embed_dim, decoder_embed_dim)
        self.ave_spe_channel_to_decoder = nn.Linear(speech_embed_dim, decoder_embed_dim)
        self.ave_img2_channel_to_decoder  =  nn.Linear(img_embed_dim, decoder_embed_dim)
        

        self.task_dict = nn.ModuleDict()
        self.task_dict['imgc'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['imgr'] = nn.Embedding(64, decoder_embed_dim)
        self.task_dict['textc'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['vqa'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['msa']  = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['textr'] = nn.Embedding(66, decoder_embed_dim)
        self.task_dict['ave'] = nn.Embedding(25, decoder_embed_dim)


        self.head = nn.ModuleDict()
        self.head['imgc'] = nn.Linear(decoder_embed_dim, IMGC_NUMCLASS)
        self.head['textc'] = nn.Linear(decoder_embed_dim, TEXTC_NUMCLASS)
        self.head['textr'] = nn.Linear(decoder_embed_dim, TEXTR_NUMCLASS)
        self.head['vqa'] = nn.Linear(decoder_embed_dim, VQA_NUMCLASS)
        self.head['imgr'] = nn.Linear(decoder_embed_dim, IMGR_LENGTH)
        self.head['msa'] = nn.Linear(decoder_embed_dim, MSA_NUMCLASS)
        self.head['ave'] = nn.Linear(decoder_embed_dim, AVE_NUMCLASS)


        self.decoder = Decoder(depth=decoder_depth, embed_dim=decoder_embed_dim, 
                                num_heads=decoder_num_heads, dff=mlp_ratio*decoder_embed_dim, 
                                drop_rate=drop_rate)
        # self.channel = Channels()
        self.channel = AWGNSingleChannel()
        # self.channel = RayleighFadingSingleChannel()
        self.sigmoid_layer = nn.Sigmoid()

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

    
    def transmit(self, x, SNRdb):
        # x = encoder_to_channel(input_signal)
        
        x = tensor_real2complex(x, 'concat')
        x = self.channel.interfere(x, SNRdb.item()) # AWGN channel
        # x = self.channel.interfere(x, SNRdb.item(), self.channel_gain_var)
        x = tensor_complex2real(x, 'concat')
        
        return x

    def get_features(self, text=None, img=None, speech=None, ta_perform=None, power_constraint:list[float]=[1, 1, 1], test_snr:torch.FloatTensor=torch.FloatTensor([12])):
        """
            Args:
                same as forward
            Output:
                A tuple: (features before transmission, features after channel decoding)
        """
        if self.training:
            noise_snr, noise_std = noise_gen(self.training)
            noise_std = noise_std.cuda()
        else:
            noise_std = torch.FloatTensor([1]) * 10**(-test_snr/20) 
            noise_snr = test_snr
        
        print(f'Noise SNR: {noise_snr}')
        if text is not None:
            ## ##
            x_text = self.text_encoder(ta_perform, text, return_dict=False)[0]
            power = power_constraint[0]
            # x_text = self.LN(x_text)
            if ta_perform.startswith('textc'):
                x_text = x_text[:,0,:].unsqueeze(1)
                x_text_sig = self.textc_encoder_to_channel(x_text)
            elif ta_perform.startswith('vqa'):
                x_text = x_text[:,0:3,:]
                x_text_sig = self.vqa_text_encoder_to_channel(x_text)
            elif ta_perform.startswith('msa'):
                x_text = x_text[:,-2:-1,:]
                x_text_sig = self.msa_text_encoder_to_channel(x_text)
            
            x_text_sig, _ = power_norm_batchwise(x_text_sig, power)
            x_text_sig = self.transmit(x_text_sig, noise_snr)
            x_text_sig = self.msa_text_channel_decoder(x_text_sig)
           
        if img is not None:
            x_img = self.img_encoder(img, ta_perform)
            if ta_perform.startswith('imgc'):
                power = power_constraint[0]
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img_sig = self.imgc_encoder_to_channel(x_img)
                
            elif ta_perform.startswith('vqa'):
                power = power_constraint[1]
                x_img = x_img[:,0:3,:]
                x_img_sig = self.vqa_img_encoder_to_channel(x_img)
                
            elif ta_perform.startswith('msa'):
                power = power_constraint[1]
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img_sig = self.msa_img_encoder_to_channel(x_img)
    
            x_img_sig, _ = power_norm_batchwise(x_img_sig, power)
            x_img_sig = self.transmit(x_img_sig, noise_snr)
            x_img_sig = self.msa_img_channel_decoder(x_img_sig)
        
        if speech is not None:
            power = power_constraint[2]
            x_spe = self.spe_encoder(speech, ta_perform)
            x_spe = x_spe[:,0,:].unsqueeze(1)
            x_spe_sig = self.msa_spe_encoder_to_channel(x_spe)
            x_spe_sig, _ = power_norm_batchwise(x_spe_sig, power)
            x_spe_sig = self.transmit(x_spe_sig, noise_snr)
            x_spe_sig = self.msa_spe_channel_decoder(x_spe_sig)
        
        transmitted = []
        received = []
        if ta_perform.startswith('img'):
            transmitted = [x_img]
            received = [x_img_sig]
        elif ta_perform.startswith('text'):
            transmitted = [x_text]
            received = [x_text_sig]
        elif ta_perform.startswith('vqa'):
            transmitted = [x_img, x_text]
            received = [x_img_sig, x_text_sig]
        elif ta_perform.startswith('msa'):
            transmitted = [x_img, x_text, x_spe]
            received = [x_img_sig, x_text_sig, x_spe_sig]
        
        return transmitted, received
    
    def forward(self, text=None, img=None, speech=None, img2=None, ta_perform=None, power_constraint:list[float]=[1, 1, 1], test_snr:torch.FloatTensor=torch.FloatTensor([12]), channel_gain_var: float | None = None):
        """
            Input:
                text: text data (tokenized first) for task need text
                img: image data for task need image
                speech: audio data for task need speech
                ta_perform: The task to be eval (execute)
                power_constraint: The power constraint of each user [text, img, speech]
            Output:
                Task result executed by decoder of receiver
                Different shape for different tasks and data type
                Ex:
                    Textr: (batch_size, seq length, vocab size) for vacab size = 34000
        """
        if self.training:
            noise_snr, noise_std = noise_gen(self.training)
            noise_std = noise_std.cuda()
        else:
            noise_std = torch.FloatTensor([1]) * 10**(-test_snr/20) 
            noise_snr = test_snr
            # print(f"SNR: {noise_snr}")

        if isinstance(self.channel, RayleighFadingSingleChannel):
            self.channel_gain_var = 1.0 if channel_gain_var is None else channel_gain_var
        
        if text is not None:
            ## ##
            x_text = self.text_encoder(ta_perform, text, return_dict=False)[0]
            power = power_constraint[0]
            # x_text = self.LN(x_text)
            if ta_perform.startswith('textc'):
                x_text = x_text[:,0,:].unsqueeze(1)
                x_text = self.textc_encoder_to_channel(x_text)
            elif ta_perform.startswith('vqa'):
                x_text = x_text[:,0:3,:]
                x_text = self.vqa_text_encoder_to_channel(x_text)
            elif ta_perform.startswith('msa'):
                x_text = x_text[:,-2:-1,:]
                x_text = self.msa_text_encoder_to_channel(x_text)

            x_text, _ = power_norm_batchwise(x_text, power)
            x_text = self.transmit(x_text, noise_snr)

            
        if img is not None:
            AVE_batch_size = img.shape[0]
            if ta_perform.startswith('ave'):
                img = img.view(img.size(0) * img.size(1), -1, 512) # (batch_size * time_steps, 49, 512)

            x_img = self.img_encoder(img, ta_perform)
            if ta_perform.startswith('imgc'):
                power = power_constraint[0]
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img = self.imgc_encoder_to_channel(x_img)
                
            elif ta_perform.startswith('vqa'):
                power = power_constraint[1]
                x_img = x_img[:,0:3,:]
                x_img = self.vqa_img_encoder_to_channel(x_img)
                
            elif ta_perform.startswith('msa'):
                power = power_constraint[1]
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img = self.msa_img_encoder_to_channel(x_img)

            elif ta_perform.startswith('ave'):
                power = power_constraint[0]
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img = self.ave_img_encoder_to_channel(x_img)

            x_img, _ = power_norm_batchwise(x_img, power)
            x_img = self.transmit(x_img, noise_snr)

        if speech is not None:
            if ta_perform.startswith('ave'):
                speech = speech.view(-1, speech.size(-1)) # (batch_size * time_steps, 128)

            x_spe = self.spe_encoder(speech, ta_perform)
            if ta_perform.startswith('msa'):
                power = power_constraint[1]
                x_spe = x_spe[:,0,:].unsqueeze(1)
                x_spe = self.msa_spe_encoder_to_channel(x_spe)
            
            elif ta_perform.startswith('ave'):
                power = power_constraint[1]
                x_spe = x_spe[:,0,:].unsqueeze(1)
                x_spe = self.ave_spe_encoder_to_channel(x_spe)
            
            x_spe, _ = power_norm_batchwise(x_spe, power)
            x_spe = self.transmit(x_spe, noise_snr)

        if img2 is not None:
            if ta_perform.startswith('ave'):
                img2 = img2.view(img2.size(0) * img2.size(1), -1, 512) # (batch_size * time_steps, 49, 512)
                x_img2 = self.img2_encoder(img2, ta_perform)
                
                power = power_constraint[2]
                x_img2 = x_img2[:,0,:].unsqueeze(1)
                x_img2 = self.ave_img2_encoder_to_channel(x_img2)

            x_img2, _ = power_norm_batchwise(x_img2, power)
            x_img2 = self.transmit(x_img2, noise_snr)
        
        if ta_perform.startswith('img'):
            x_img = self.imgc_channel_decoder(x_img)
            x_img = self.imgc_channel_to_decoder(x_img)
            x = x_img
        elif ta_perform.startswith('text'):
            x_text = self.textc_channel_decoder(x_text)
            x_text = self.textc_channel_to_decoder(x_text)
            x = x_text
        elif ta_perform.startswith('vqa'):
            x_text = self.vqa_text_channel_decoder(x_text)
            x_text = self.vqa_text_channel_to_decoder(x_text)
            
            x_img = self.vqa_img_channel_decoder(x_img)
            x_img = self.vqa_img_channel_to_decoder(x_img)

            x = torch.cat([x_img,x_text], dim=1)
        elif ta_perform.startswith('msa'):
            x_text = self.msa_text_channel_decoder(x_text)
            x_text = self.msa_text_channel_to_decoder(x_text)

            x_img = self.msa_img_channel_decoder(x_img)
            x_img = self.msa_img_channel_to_decoder(x_img)

            x_spe = self.msa_spe_channel_decoder(x_spe)
            x_spe = self.msa_spe_channel_to_decoder(x_spe)
            
            x = torch.cat([x_img,x_text,x_spe], dim=1)
            # x = torch.cat([x_text,x_spe], dim=1)
            # print(x.shape) # (batch_size, 3, 128)
        elif ta_perform.startswith('ave'):
            x_img = self.ave_img_channel_decoder(x_img)
            x_img = self.ave_img_channel_to_decoder(x_img)

            x_spe = self.ave_spe_channel_decoder(x_spe)
            x_spe = self.ave_spe_channel_to_decoder(x_spe)

            if img2 is not None:
                x_img2 = self.ave_img2_channel_decoder(x_img2)
                x_img2 = self.ave_img2_channel_to_decoder(x_img2)
            
            x = torch.cat([x_img, x_spe], dim=1)
            # x = torch.cat([x_img, x_spe, x_img2], dim=1)
            # print(x.shape) # (batch_size, 3, 128)

        batch_size = x.shape[0]
        if ta_perform.endswith('r'):
            x = self.decoder(x, x, None, None, None) 
            x = self.head[ta_perform](x)
            return x
        else:
            query_embed = self.task_dict[ta_perform].weight.unsqueeze(0).repeat(batch_size, 1, 1)
            x = self.decoder(query_embed, x, None, None, None) 
            if ta_perform.startswith('textr'): 
                x = self.head[ta_perform](x)
            else:
                x = self.head[ta_perform](x.mean(1))
                if ta_perform.startswith('ave'):
                    x = x.view(AVE_batch_size, -1, x.size(-1))
            if ta_perform.startswith('vqa'):
                x = self.sigmoid_layer(x)
            return x
        
class UDeepSCperfectSIC(nn.Module):
    """
        UDeepSC non-orthogonal assumed have perfect SIC version
    """

    def __init__(self,mode='tiny',
                 img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0, 
                 img_embed_dim=384, text_embed_dim=384, speech_embed_dim=128, img_encoder_depth=4, 
                 text_encoder_depth=4, speech_encoder_depth=4, encoder_num_heads=12, decoder_num_classes=768, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, init_values=0.,use_learnable_pos_emb=False,num_classes=0, 
                 ):

        super(UDeepSCperfectSIC, self).__init__()

        # self.decoder_embed_dim = decoder_embed_dim
        
        if mode=='tiny':
            text_embed_dim = 128
        elif mode=='small':
            text_embed_dim = 512
        else:
            text_embed_dim = 512
        
        self.img_encoder = ViTEncoder(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, 
                                num_classes=encoder_num_classes, embed_dim=img_embed_dim,depth=img_encoder_depth,
                                num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        
        bert_ckpt = f"prajjwal1/bert-{mode}"
        self.text_encoder = BertModel.from_pretrained(bert_ckpt)
        
        self.spe_encoder = SPTEncoder(in_chans=encoder_in_chans,num_classes=encoder_num_classes, embed_dim=speech_embed_dim,
                                depth=speech_encoder_depth,num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        
        self.img2_encoder = ViTEncoder(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, 
                                num_classes=encoder_num_classes, embed_dim=img_embed_dim,depth=img_encoder_depth,
                                num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        
        self.num_symbols_imgc = 16
        self.num_symbols_imgr = 16
        self.num_symbols_textc = 4
        self.num_symbols_textr = 24
        # self.num_symbols_vqa_img = 16
        # self.num_symbols_vqa_text = 16
        # self.num_symbols_msa_img = 16
        # self.num_symbols_msa_text = 16
        # self.num_symbols_msa_spe = 16
        self.num_symbols_vqa = 16
        # using 3 times bandwidth compare to OMA
        self.num_symbols_msa = 48
        self.num_symbols_ave = 16
 
        # channel encoder
        self.textc_encoder_to_channel =     nn.Linear(text_embed_dim, self.num_symbols_textc)
        self.imgc_encoder_to_channel =      nn.Linear(img_embed_dim, self.num_symbols_imgc)
        self.textr_encoder_to_channel =     nn.Linear(text_embed_dim, self.num_symbols_textr)
        self.imgr_encoder_to_channel =      nn.Linear(img_embed_dim, self.num_symbols_imgr)
        self.vqa_img_encoder_to_channel =   nn.Linear(img_embed_dim, self.num_symbols_vqa)
        self.vqa_text_encoder_to_channel =  nn.Linear(text_embed_dim, self.num_symbols_vqa)
        self.msa_img_encoder_to_channel =   nn.Linear(img_embed_dim, self.num_symbols_msa)
        self.msa_text_encoder_to_channel =  nn.Linear(text_embed_dim, self.num_symbols_msa)
        self.msa_spe_encoder_to_channel =   nn.Linear(speech_embed_dim, self.num_symbols_msa)
        self.ave_img_encoder_to_channel =   nn.Linear(img_embed_dim, self.num_symbols_ave)
        self.ave_spe_encoder_to_channel =   nn.Linear(speech_embed_dim, self.num_symbols_ave)
        self.ave_img2_encoder_to_channel =   nn.Linear(img_embed_dim, self.num_symbols_ave)
        
        # channel decoder
        self.textc_channel_decoder = ChannelDecoder(self.num_symbols_textc, text_embed_dim)
        self.imgc_channel_decoder  = ChannelDecoder(self.num_symbols_imgc, img_embed_dim)
        self.vqa_img_channel_decoder  = ChannelDecoder(self.num_symbols_vqa, img_embed_dim)
        self.vqa_text_channel_decoder = ChannelDecoder(self.num_symbols_vqa, text_embed_dim)
        self.msa_img_channel_decoder  = ChannelDecoder(self.num_symbols_msa, img_embed_dim)
        self.msa_text_channel_decoder = ChannelDecoder(self.num_symbols_msa, text_embed_dim)
        self.msa_spe_channel_decoder  = ChannelDecoder(self.num_symbols_msa, speech_embed_dim)
        self.ave_img_channel_decoder  = ChannelDecoder(self.num_symbols_ave, img_embed_dim)
        self.ave_spe_channel_decoder  = ChannelDecoder(self.num_symbols_ave, speech_embed_dim)
        self.ave_img2_channel_decoder  = ChannelDecoder(self.num_symbols_ave, img_embed_dim)

        self.textc_channel_to_decoder  =    nn.Linear(text_embed_dim, decoder_embed_dim)
        self.imgc_channel_to_decoder  =     nn.Linear(img_embed_dim, decoder_embed_dim)
        self.vqa_img_channel_to_decoder  =  nn.Linear(img_embed_dim, decoder_embed_dim)
        self.vqa_text_channel_to_decoder  = nn.Linear(text_embed_dim, decoder_embed_dim)
        self.msa_img_channel_to_decoder  =  nn.Linear(img_embed_dim, decoder_embed_dim)
        self.msa_text_channel_to_decoder  = nn.Linear(text_embed_dim, decoder_embed_dim)
        self.msa_spe_channel_to_decoder  =  nn.Linear(speech_embed_dim, decoder_embed_dim)
        self.ave_img_channel_to_decoder  =  nn.Linear(img_embed_dim, decoder_embed_dim)
        self.ave_spe_channel_to_decoder = nn.Linear(speech_embed_dim, decoder_embed_dim)
        self.ave_img2_channel_to_decoder  =  nn.Linear(img_embed_dim, decoder_embed_dim)
        

        self.task_dict = nn.ModuleDict()
        self.task_dict['imgc'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['imgr'] = nn.Embedding(64, decoder_embed_dim)
        self.task_dict['textc'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['vqa'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['msa']  = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['textr'] = nn.Embedding(66, decoder_embed_dim)
        self.task_dict['ave'] = nn.Embedding(25, decoder_embed_dim)


        self.head = nn.ModuleDict()
        self.head['imgc'] = nn.Linear(decoder_embed_dim, IMGC_NUMCLASS)
        self.head['textc'] = nn.Linear(decoder_embed_dim, TEXTC_NUMCLASS)
        self.head['textr'] = nn.Linear(decoder_embed_dim, TEXTR_NUMCLASS)
        self.head['vqa'] = nn.Linear(decoder_embed_dim, VQA_NUMCLASS)
        self.head['imgr'] = nn.Linear(decoder_embed_dim, IMGR_LENGTH)
        self.head['msa'] = nn.Linear(decoder_embed_dim, MSA_NUMCLASS)
        self.head['ave'] = nn.Linear(decoder_embed_dim, AVE_NUMCLASS)


        self.decoder = Decoder(depth=decoder_depth, embed_dim=decoder_embed_dim, 
                                num_heads=decoder_num_heads, dff=mlp_ratio*decoder_embed_dim, 
                                drop_rate=drop_rate)
        # self.channel = Channels()
        self.channel = AWGNSingleChannel()
        # self.channel = RayleighFadingSingleChannel()
        self.sigmoid_layer = nn.Sigmoid()

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

    
    def transmit(self, x, SNRdb):
        # x = encoder_to_channel(input_signal)
        
        x = tensor_real2complex(x, 'concat')
        x = self.channel.interfere(x, SNRdb.item()) # AWGN channel
        # x = self.channel.interfere(x, SNRdb.item(), self.channel_gain_var)
        x = tensor_complex2real(x, 'concat')
        
        return x
    
    def get_features(self, text=None, img=None, speech=None, ta_perform=None, power_constraint:list[float]=[1, 1, 1], test_snr:torch.FloatTensor=torch.FloatTensor([12])):
        """
            Args:
                same as forward
            Output:
                A tuple: (features before transmission, features after channel decoding)
        """
        if self.training:
            noise_snr, noise_std = noise_gen(self.training)
            noise_std = noise_std.cuda()
        else:
            noise_std = torch.FloatTensor([1]) * 10**(-test_snr/20) 
            noise_snr = test_snr
        
        print(f'Noise SNR: {noise_snr}')
        if text is not None:
            ## ##
            x_text = self.text_encoder(ta_perform, text, return_dict=False)[0]
            power = power_constraint[0]
            # x_text = self.LN(x_text)
            if ta_perform.startswith('textc'):
                x_text = x_text[:,0,:].unsqueeze(1)
                x_text_sig = self.textc_encoder_to_channel(x_text)
            elif ta_perform.startswith('vqa'):
                x_text = x_text[:,0:3,:]
                x_text_sig = self.vqa_text_encoder_to_channel(x_text)
            elif ta_perform.startswith('msa'):
                x_text = x_text[:,-2:-1,:]
                x_text_sig = self.msa_text_encoder_to_channel(x_text)
            
            x_text_sig, _ = power_norm_batchwise(x_text_sig, power)
            x_text_sig = self.transmit(x_text_sig, noise_snr)
            # x_text_sig = power_denorm_batchwise(x_text_sig, power)
            x_text_sig = self.msa_text_channel_decoder(x_text_sig)
           
        if img is not None:
            x_img = self.img_encoder(img, ta_perform)
            if ta_perform.startswith('imgc'):
                power = power_constraint[0]
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img_sig = self.imgc_encoder_to_channel(x_img)
                
            elif ta_perform.startswith('vqa'):
                power = power_constraint[1]
                x_img = x_img[:,0:3,:]
                x_img_sig = self.vqa_img_encoder_to_channel(x_img)
                
            elif ta_perform.startswith('msa'):
                power = power_constraint[1]
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img_sig = self.msa_img_encoder_to_channel(x_img)
    
            x_img_sig, _ = power_norm_batchwise(x_img_sig, power)
            x_img_sig = self.transmit(x_img_sig, noise_snr)
            # x_img_sig = power_denorm_batchwise(x_img_sig, power)
            x_img_sig = self.msa_img_channel_decoder(x_img_sig)
        
        if speech is not None:
            power = power_constraint[2]
            x_spe = self.spe_encoder(speech, ta_perform)
            x_spe = x_spe[:,0,:].unsqueeze(1)
            x_spe_sig = self.msa_spe_encoder_to_channel(x_spe)
            x_spe_sig, _ = power_norm_batchwise(x_spe_sig, power)
            x_spe_sig = self.transmit(x_spe_sig, noise_snr)
            # x_spe_sig = power_denorm_batchwise(x_spe_sig, power)
            x_spe_sig = self.msa_spe_channel_decoder(x_spe_sig)
        
        transmitted = []
        received = []
        if ta_perform.startswith('img'):
            transmitted = [x_img]
            received = [x_img_sig]
        elif ta_perform.startswith('text'):
            transmitted = [x_text]
            received = [x_text_sig]
        elif ta_perform.startswith('vqa'):
            transmitted = [x_img, x_text]
            received = [x_img_sig, x_text_sig]
        elif ta_perform.startswith('msa'):
            # print(x_img.shape, x_text.shape, x_spe.shape) # (batch_size, 1, 384), (batch_size, 1, 512), ( batch_size, 1, 128)
            transmitted = [x_img, x_text, x_spe]
            received = [x_img_sig, x_text_sig, x_spe_sig]
        
        return transmitted, received
    
    def forward(self, text=None, img=None, speech=None, img2=None, ta_perform=None, power_constraint:list[float]=[1, 1, 1], test_snr:torch.FloatTensor=torch.FloatTensor([12]), channel_gain_var: float | None = None):
        """
            Input:
                text: text data (tokenized first) for task need text
                img: image data for task need image
                speech: audio data for task need speech
                ta_perform: The task to be eval (execute)
                power_constraint: The power constraint of each user [text, img, speech]
            Output:
                Task result executed by decoder of receiver
                Different shape for different tasks and data type
                Ex:
                    Textr: (batch_size, seq length, vocab size) for vacab size = 34000
        """
        if self.training:
            noise_snr, noise_std = noise_gen(self.training)
            noise_std = noise_std.cuda()
        else:
            noise_std = torch.FloatTensor([1]) * 10**(-test_snr/20) 
            noise_snr = test_snr
            # print(f"SNR: {noise_snr}")

        if isinstance(self.channel, RayleighFadingSingleChannel):
            self.channel_gain_var = 1.0 if channel_gain_var is None else channel_gain_var
        
        if text is not None:
            ## ##
            x_text = self.text_encoder(ta_perform, text, return_dict=False)[0]
            power = power_constraint[0]
            # x_text = self.LN(x_text)
            if ta_perform.startswith('textc'):
                x_text = x_text[:,0,:].unsqueeze(1)
                x_text = self.textc_encoder_to_channel(x_text)
            elif ta_perform.startswith('vqa'):
                x_text = x_text[:,0:3,:]
                x_text = self.vqa_text_encoder_to_channel(x_text)
            elif ta_perform.startswith('msa'):
                x_text = x_text[:,-2:-1,:]
                x_text = self.msa_text_encoder_to_channel(x_text)

            x_text, _ = power_norm_batchwise(x_text, power)
            x_text = self.transmit(x_text, noise_snr)

            
        if img is not None:
            AVE_batch_size = img.shape[0]
            if ta_perform.startswith('ave'):
                img = img.view(img.size(0) * img.size(1), -1, 512) # (batch_size * time_steps, 49, 512)

            x_img = self.img_encoder(img, ta_perform)
            if ta_perform.startswith('imgc'):
                power = power_constraint[0]
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img = self.imgc_encoder_to_channel(x_img)
                
            elif ta_perform.startswith('vqa'):
                power = power_constraint[1]
                x_img = x_img[:,0:3,:]
                x_img = self.vqa_img_encoder_to_channel(x_img)
                
            elif ta_perform.startswith('msa'):
                power = power_constraint[1]
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img = self.msa_img_encoder_to_channel(x_img)

            elif ta_perform.startswith('ave'):
                power = power_constraint[0]
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img = self.ave_img_encoder_to_channel(x_img)

            x_img, _ = power_norm_batchwise(x_img, power)
            x_img = self.transmit(x_img, noise_snr)

        if speech is not None:
            if ta_perform.startswith('ave'):
                speech = speech.view(-1, speech.size(-1)) # (batch_size * time_steps, 128)

            x_spe = self.spe_encoder(speech, ta_perform)
            if ta_perform.startswith('msa'):
                power = power_constraint[2]
                x_spe = x_spe[:,0,:].unsqueeze(1)
                x_spe = self.msa_spe_encoder_to_channel(x_spe)
            
            elif ta_perform.startswith('ave'):
                power = power_constraint[1]
                x_spe = x_spe[:,0,:].unsqueeze(1)
                x_spe = self.ave_spe_encoder_to_channel(x_spe)
            
            x_spe, _ = power_norm_batchwise(x_spe, power)
            x_spe = self.transmit(x_spe, noise_snr)

        if img2 is not None:
            if ta_perform.startswith('ave'):
                img2 = img2.view(img2.size(0) * img2.size(1), -1, 512) # (batch_size * time_steps, 49, 512)
                x_img2 = self.img2_encoder(img2, ta_perform)
                
                power = power_constraint[2]
                x_img2 = x_img2[:,0,:].unsqueeze(1)
                x_img2 = self.ave_img2_encoder_to_channel(x_img2)

            x_img2, _ = power_norm_batchwise(x_img2, power)
            x_img2 = self.transmit(x_img2, noise_snr)
        
        if ta_perform.startswith('img'):
            x_img = self.imgc_channel_decoder(x_img)
            x_img = self.imgc_channel_to_decoder(x_img)
            x = x_img
        elif ta_perform.startswith('text'):
            x_text = self.textc_channel_decoder(x_text)
            x_text = self.textc_channel_to_decoder(x_text)
            x = x_text
        elif ta_perform.startswith('vqa'):
            x_text = self.vqa_text_channel_decoder(x_text)
            x_text = self.vqa_text_channel_to_decoder(x_text)
            
            x_img = self.vqa_img_channel_decoder(x_img)
            x_img = self.vqa_img_channel_to_decoder(x_img)

            x = torch.cat([x_img,x_text], dim=1)
        elif ta_perform.startswith('msa'):
            x_text = self.msa_text_channel_decoder(x_text)
            x_text = self.msa_text_channel_to_decoder(x_text)

            x_img = self.msa_img_channel_decoder(x_img)
            x_img = self.msa_img_channel_to_decoder(x_img)

            x_spe = self.msa_spe_channel_decoder(x_spe)
            x_spe = self.msa_spe_channel_to_decoder(x_spe)
            
            x = torch.cat([x_img,x_text,x_spe], dim=1)
            # x = torch.cat([x_text,x_spe], dim=1)
            # print(x.shape) # (batch_size, 3, 128)
        elif ta_perform.startswith('ave'):
            x_img = self.ave_img_channel_decoder(x_img)
            x_img = self.ave_img_channel_to_decoder(x_img)

            x_spe = self.ave_spe_channel_decoder(x_spe)
            x_spe = self.ave_spe_channel_to_decoder(x_spe)

            if img2 is not None:
                x_img2 = self.ave_img2_channel_decoder(x_img2)
                x_img2 = self.ave_img2_channel_to_decoder(x_img2)
            
            x = torch.cat([x_img, x_spe], dim=1)
            # x = torch.cat([x_img, x_spe, x_img2], dim=1)
            # print(x.shape) # (batch_size, 3, 128)

        batch_size = x.shape[0]
        if ta_perform.endswith('r'):
            x = self.decoder(x, x, None, None, None) 
            x = self.head[ta_perform](x)
            return x
        else:
            query_embed = self.task_dict[ta_perform].weight.unsqueeze(0).repeat(batch_size, 1, 1)
            x = self.decoder(query_embed, x, None, None, None) 
            if ta_perform.startswith('textr'): 
                x = self.head[ta_perform](x)
            else:
                x = self.head[ta_perform](x.mean(1))
                if ta_perform.startswith('ave'):
                    x = x.view(AVE_batch_size, -1, x.size(-1))
            if ta_perform.startswith('vqa'):
                x = self.sigmoid_layer(x)
            return x

class UDeepSC_M3_withSIC(UDeepSCperfectSIC):
    def __init__(self,
                noise_power_density_dBm,
                reference_distance,
                reference_path_loss,
                path_loss_exponent,
                distance,
                *args, **kwargs):

        super().__init__(*args, **kwargs)
        # self.channel = Channels()
        self.channel = AWGNMultiChannel()
        # self.channel = RayleighFadingMultiChannel(
        #                     noise_power_density_dBm,
        #                     reference_distance,
        #                     reference_path_loss,
        #                     path_loss_exponent,
        #                     distance,
        #                 )

    def SIC(self, signal: torch.Tensor, user_dim_index: int, power_constraints: torch.FloatTensor, 
            channel_encoders: list[nn.Module], channel_decoders: list[nn.Module], 
            channel_type: Literal['AWGN', 'Fading'], h=None):
        """            
            Args
                signal: real tensor in (batch_size, 1, *dim, symbol_dim)
                power_constraint: the power constraint for users, length n_user (The order is [text, img, speech])
                h: channel gain (Rayleigh or Rician)
            Return
                a list of decode signals
        """

        device = signal.device
        batch_size = signal.size()[0]
        dim = tuple(signal.size()[user_dim_index + 1:])

        num_users = len(power_constraints)
        # print(f"{num_users= }")
        
        if(num_users == 1):
            estimated = signal[:, 0, :]
            estimated = channel_decoders[0](estimated)
            return [estimated]
        
        if channel_type == "AWGN":
            # Sort users by transmit power (descending order)
            user_indices = torch.argsort(power_constraints, dim=-1).detach().to('cpu').numpy()
        else: # Rayleigh or Rician
            if h is None:
                raise ValueError("Channel gains (h) must be provided for Rayleigh channel.")
            # Compute effective received power |h_i|^2 * P_i
            # effective_power = torch.abs(h).detach()**2 * power_constraints
            user_indices = torch.argsort(power_constraints, dim=-1).detach().to('cpu').numpy()


        # decoded_signals = torch.zeros((batch_size, num_users, *dim)).to(device)
        decoded_signals = [None] * num_users
        # remaining_signal  = signal.clone().detach().to(device)

        for i in user_indices:
            """
                Decoding steps:
                    1. Use channel decoder to decode stronger signal
                    
                    2. Channel encoding (same as transmitter) the signal to simulate the signal state of the transmitter
                    
                    3. Subtract encoded estimated signal from Y
                    
                    4. Repeat until all signals be detected
            """
            estimated = signal[:, 0, :]

            if channel_type == "Fading":
                estimated = estimated / h[:, i, 0]
                estimated = tensor_complex2real(estimated, 'concat')

            estimated = channel_decoders[i](estimated)
            decoded_signals[i] = estimated
            
            with torch.no_grad():
                estimated = channel_encoders[i](estimated)

            estimated_norm, _ = power_norm_batchwise(estimated, power_constraints[i].item())

            if channel_type == "AWGN":
                signal = signal - estimated_norm
            else: # Fading channel (Rayleigh or Rician)
                # print(f'{estimated_norm.shape=}')
                estimated_norm = tensor_real2complex(estimated_norm, 'concat')
                signal = signal - h[:, i, 0] * estimated_norm

        return decoded_signals
        
    def transmit(self, signal: torch.Tensor, user_dim_index: int, SNRdb: torch.FloatTensor, 
                 power_constraints: list[float], channel_encoders: list[nn.Module], 
                 channel_decoders: list[nn.Module]):
        """
            Args:
                - signal: a real tensor representing the signals from multiple transmitter (user).
                        with shape (batch_size, user_dim, *dim, symbol_dim):
                        - user_dim: for users, indexed by user_dim_index
                        - symbol_dim: for signal symbols
            Return:
                a list of decode signals in order (text, img, (speech))
        """
        # print(f"Transmid Signal Dim = {signal.shape}")
        
        if isinstance(self.channel, AWGNMultiChannel):
            power_constraints = torch.tensor(power_constraints)
            signal = power_normlize_stack(signal, power_constraints)
            signal = tensor_real2complex(signal, 'concat') #(batch_size, 1, 8)
            
            # superimpose and add noise
            signal = self.channel.interfere(signal, SNRdb.item(), user_dim_index)

            signal = tensor_complex2real(signal, 'concat')
            outputs = self.SIC(signal, user_dim_index, power_constraints, channel_encoders, channel_decoders, 'AWGN')

        elif isinstance(self.channel, RayleighFadingMultiChannel):
            total_power = 3
            signal = tensor_real2complex(signal, 'concat')
            power_constraints = self.channel.get_signal_power_constraint(signal, total_power, SNRdb, user_dim_index, self.channel_gain_var) # (n_tx, )
            signal = power_normlize_stack(signal, power_constraints)

            self.channel_gain = self.channel.get_channel_gain().to(signal.device) # (batch_size, n_tx, 1, 1 ,signal length)
            # print(f'{self.channel_gain.shape= }')
            # superimpose and add noise
            signal = self.channel.interfere(
                signal=signal, 
                user_dim_index=user_dim_index, 
                SNRdb=SNRdb, 
                divide_gain=False,
                # channel_gain_var=self.channel_gain_var,
                channel_gain_tensor=self.channel_gain
            )
        
            outputs = self.SIC(signal, user_dim_index, power_constraints, channel_encoders, channel_decoders, 'Fading', self.channel_gain)

            # signal = tensor_complex2real(signal, 'concat')

        return outputs
    
    def get_features(self, text=None, img=None, speech=None, ta_perform=None, power_constraint:list[float]=[1, 1, 1], test_snr:torch.FloatTensor=torch.FloatTensor([12])):
        """
            Args:
                same as forward
            Output:
                A tuple: (features before transmission, features after channel decoding)
        """
        if self.training:
            noise_snr, noise_std = noise_gen(self.training)
            noise_std = noise_std.cuda()
        else:
            noise_std = torch.FloatTensor([1]) * 10**(-test_snr/20) 
            noise_snr = test_snr
            
        if text is not None:
            ## ##
            x_text = self.text_encoder(ta_perform, text, return_dict=False)[0]
            power = power_constraint[0]
            # x_text = self.LN(x_text)
            if ta_perform.startswith('textc'):
                x_text = x_text[:,0,:].unsqueeze(1)
                x_text_sig = self.textc_encoder_to_channel(x_text)
            elif ta_perform.startswith('vqa'):
                x_text = x_text[:,0:3,:]
                x_text_sig = self.vqa_text_encoder_to_channel(x_text)
            elif ta_perform.startswith('msa'):
                x_text = x_text[:,-2:-1,:]
                x_text_sig = self.msa_text_encoder_to_channel(x_text)
            
            x_text_sig, _ = power_norm_batchwise(x_text_sig, power)
            
           
        if img is not None:
            x_img = self.img_encoder(img, ta_perform)
            if ta_perform.startswith('imgc'):
                power = power_constraint[0]
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img_sig = self.imgc_encoder_to_channel(x_img)
                
            elif ta_perform.startswith('vqa'):
                power = power_constraint[1]
                x_img = x_img[:,0:3,:]
                x_img_sig = self.vqa_img_encoder_to_channel(x_img)
                
            elif ta_perform.startswith('msa'):
                power = power_constraint[1]
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img_sig = self.msa_img_encoder_to_channel(x_img)
    
            x_img_sig, _ = power_norm_batchwise(x_img_sig, power)
            
        
        if speech is not None:
            power = power_constraint[2]
            x_spe = self.spe_encoder(speech, ta_perform)
            x_spe = x_spe[:,0,:].unsqueeze(1)
            x_spe_sig = self.msa_spe_encoder_to_channel(x_spe)
            x_spe_sig, _ = power_norm_batchwise(x_spe_sig, power)
            
        if ta_perform.startswith('img'):
            power_constraint = [1]
            x = x_img_sig.unsqueeze(1)
            channel_encoders = [self.imgc_encoder_to_channel]
            channel_decoders = [self.imgc_channel_decoder]
            Rx_sigs = self.transmit(x, 1, noise_snr, power_constraint, channel_encoders, channel_decoders)
        elif ta_perform.startswith('text'):
            power_constraint = [1]
            x = x_text_sig.unsqueeze(1)
            channel_encoders = [self.textc_encoder_to_channel]
            channel_decoders = [self.textc_channel_decoder]
            Rx_sigs = self.transmit(x, 1, noise_snr, power_constraint, channel_encoders, channel_decoders)
        elif ta_perform.startswith('vqa'):
            x = torch.stack((x_img_sig, x_text_sig), dim=1)
            channel_encoders = [self.vqa_text_encoder_to_channel, self.vqa_img_encoder_to_channel]
            channel_decoders = [self.vqa_text_channel_decoder, self.vqa_img_channel_decoder]
            Rx_sigs = self.transmit(x, 1, noise_snr, power_constraint, channel_encoders, channel_decoders)
        elif ta_perform.startswith('msa'):
            x = torch.stack((x_img_sig, x_text_sig, x_spe_sig), dim=1)
            channel_encoders = [self.msa_text_encoder_to_channel, self.msa_img_encoder_to_channel, self.msa_spe_encoder_to_channel]
            channel_decoders = [self.msa_text_channel_decoder, self.msa_img_channel_decoder, self.msa_spe_channel_decoder]
            # x = torch.stack((x_text_sig, x_spe_sig), dim=1)
            # channel_encoders = [self.msa_text_encoder_to_channel, self.msa_spe_encoder_to_channel]
            # channel_decoders = [self.msa_text_channel_decoder, self.msa_spe_channel_decoder]
            Rx_sigs = self.transmit(x, 1, noise_snr, power_constraint, channel_encoders, channel_decoders)
        
        if ta_perform.startswith('img'):
            x_img_sig = Rx_sigs[0]
        elif ta_perform.startswith('text'):
            x_text_sig = Rx_sigs[0]
        elif ta_perform.startswith('vqa'):
            x_text_sig = Rx_sigs[0]
            x_img_sig = Rx_sigs[1]
        elif ta_perform.startswith('msa'):
            x_text_sig = Rx_sigs[0]
            x_img_sig = Rx_sigs[1]
            x_spe_sig = Rx_sigs[2]
            # print(x.shape) # (batch_size, 3, 128)

        transmitted = []
        received = []
        if ta_perform.startswith('img'):
            transmitted = [x_img]
            received = [x_img_sig]
        elif ta_perform.startswith('text'):
            transmitted = [x_text]
            received = [x_text_sig]
        elif ta_perform.startswith('vqa'):
            transmitted = [x_img, x_text]
            received = [x_img_sig, x_text_sig]
        elif ta_perform.startswith('msa'):
            # print(x_img.shape, x_text.shape, x_spe.shape) # (batch_size, 1, 128)
            transmitted = [x_img, x_text, x_spe]
            received = [x_img_sig, x_text_sig, x_spe_sig]
            # transmitted = [x_text, x_spe]
            # received = [x_text_sig, x_spe_sig]
        
        return transmitted, received

    def forward(self, text=None, img=None, speech=None, img2=None, ta_perform=None, power_constraint:list[float]=[1, 1, 1], test_snr:torch.FloatTensor=torch.FloatTensor([12]), channel_gain_var: list[list[float]] | torch.Tensor = None):
        """
            Input:
                text: text data (tokenized first) for task need text
                img: image data for task need image
                speech: audio data for task need speech
                ta_perform: The task to be eval (execute)
                power_constraint: The power constraint of each user [text, img, speech]
            Output:
                Task result executed by decoder of receiver
                Different shape for different tasks and data type
                Ex:
                    Textr: (batch_size, seq length, vocab size) for vacab size = 34000
        """

        n_user = len(power_constraint)
        
        if self.training:
            noise_snr, noise_std = noise_gen(self.training)
            noise_std = noise_std.cuda()
        else:
            noise_std = torch.FloatTensor([1]) * 10**(-test_snr/20) 
            noise_snr = test_snr
        if isinstance(self.channel, RayleighFadingMultiChannel):
            self.channel_gain_var = [[1.0]] * n_user if channel_gain_var is None else channel_gain_var

        if text is not None:
            ## ##
            x_text = self.text_encoder(ta_perform, text, return_dict=False)[0]
            # x_text = self.LN(x_text)
            if ta_perform.startswith('textc'):
                x_text = x_text[:,0,:].unsqueeze(1)
                x_text = self.textc_encoder_to_channel(x_text)
            elif ta_perform.startswith('vqa'):
                x_text = x_text[:,0:3,:]
                x_text = self.vqa_text_encoder_to_channel(x_text)
            elif ta_perform.startswith('msa'):
                x_text = x_text[:,-2:-1,:]
                x_text = self.msa_text_encoder_to_channel(x_text)

            # x_text, _ = power_norm_batchwise(x_text, power)
            
        if img is not None:
            AVE_batch_size = img.shape[0]
            if ta_perform.startswith('ave'):
                img = img.view(img.size(0) * img.size(1), -1, 512) # (batch_size * time_steps, 49, 512)

            x_img = self.img_encoder(img, ta_perform)
            if ta_perform.startswith('imgc'):
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img = self.imgc_encoder_to_channel(x_img)
                
            elif ta_perform.startswith('vqa'):
                x_img = x_img[:,0:3,:]
                x_img = self.vqa_img_encoder_to_channel(x_img)
                
            elif ta_perform.startswith('msa'):
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img = self.msa_img_encoder_to_channel(x_img)

            elif ta_perform.startswith('ave'):
                power = power_constraint[0]
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img = self.ave_img_encoder_to_channel(x_img)

            # x_img, _ = power_norm_batchwise(x_img, power)

        if speech is not None:
            if ta_perform.startswith('ave'):
                speech = speech.view(-1, speech.size(-1)) # (batch_size * time_steps, 128)

            x_spe = self.spe_encoder(speech, ta_perform)
            if ta_perform.startswith('msa'):
                x_spe = x_spe[:,0,:].unsqueeze(1)
                x_spe = self.msa_spe_encoder_to_channel(x_spe)
            
            elif ta_perform.startswith('ave'):
                power = power_constraint[1]
                x_spe = x_spe[:,0,:].unsqueeze(1)
                x_spe = self.ave_spe_encoder_to_channel(x_spe)
            
            # x_spe, _ = power_norm_batchwise(x_spe, power)

        if img2 is not None:
            if ta_perform.startswith('ave'):
                img2 = img2.view(img2.size(0) * img2.size(1), -1, 512) # (batch_size * time_steps, 49, 512)
                x_img2 = self.img2_encoder(img2, ta_perform)
                
                x_img2 = x_img2[:,0,:].unsqueeze(1)
                x_img2 = self.ave_img2_encoder_to_channel(x_img2)

        if ta_perform.startswith('img'):
            power_constraint = [1]
            x = x_img.unsqueeze(1)
            channel_encoders = [self.imgc_encoder_to_channel]
            channel_decoders = [self.imgc_channel_decoder]
            Rx_sigs = self.transmit(x, 1, noise_snr, power_constraint, channel_encoders, channel_decoders)
        elif ta_perform.startswith('text'):
            power_constraint = [1]
            x = x_text.unsqueeze(1)
            channel_encoders = [self.textc_encoder_to_channel]
            channel_decoders = [self.textc_channel_decoder]
            Rx_sigs = self.transmit(x, 1, noise_snr, power_constraint, channel_encoders, channel_decoders)
        elif ta_perform.startswith('vqa'):
            x = torch.stack((x_img, x_text), dim=1)
            channel_encoders = [self.vqa_text_encoder_to_channel, self.vqa_img_encoder_to_channel]
            channel_decoders = [self.vqa_text_channel_decoder, self.vqa_img_channel_decoder]
            Rx_sigs = self.transmit(x, 1, noise_snr, power_constraint, channel_encoders, channel_decoders)
        elif ta_perform.startswith('msa'):
            x = torch.stack((x_text, x_img, x_spe), dim=1)
            channel_encoders = [self.msa_text_encoder_to_channel, self.msa_img_encoder_to_channel, self.msa_spe_encoder_to_channel]
            channel_decoders = [self.msa_text_channel_decoder, self.msa_img_channel_decoder, self.msa_spe_channel_decoder]
            # x = torch.stack((x_img, x_spe), dim=1)
            # x = x_img.unsqueeze(1)
            # channel_encoders = [self.msa_img_encoder_to_channel, self.msa_spe_encoder_to_channel]
            # channel_decoders = [self.msa_img_channel_decoder, self.msa_spe_channel_decoder]
            Rx_sigs = self.transmit(x, 1, noise_snr, power_constraint, channel_encoders, channel_decoders)

        elif ta_perform.startswith('ave'):
            x = torch.stack((x_img, x_spe), dim=1)
            channel_encoders = [self.ave_img_encoder_to_channel, self.ave_spe_encoder_to_channel]
            channel_decoders = [self.ave_img_channel_decoder, self.ave_spe_channel_decoder]
            # x = torch.stack((x_img, x_spe, x_img2), dim=1)
            # channel_encoders = [self.ave_img_encoder_to_channel, self.ave_spe_encoder_to_channel, self.ave_img2_encoder_to_channel]
            # channel_decoders = [self.ave_img_channel_decoder, self.ave_spe_channel_decoder, self.ave_img2_channel_decoder]
            Rx_sigs = self.transmit(x, 1, noise_snr, power_constraint, channel_encoders, channel_decoders)
        
        if ta_perform.startswith('img'):
            x_img = Rx_sigs[0]
            x = self.imgc_channel_to_decoder(x_img)
        elif ta_perform.startswith('text'):
            x_text = Rx_sigs[0]
            x = self.textc_channel_to_decoder(x_text)
        elif ta_perform.startswith('vqa'):
            x_text = Rx_sigs[0]
            x_text = self.vqa_text_channel_to_decoder(x_text)
            x_img = Rx_sigs[1]
            x_img = self.vqa_img_channel_to_decoder(x_img)

            x = torch.cat([x_img,x_text], dim=1)
        elif ta_perform.startswith('msa'):
            x_text = Rx_sigs[0]
            x_text = self.msa_text_channel_to_decoder(x_text)
            x_img = Rx_sigs[1]
            x_img = self.msa_img_channel_to_decoder(x_img)
            x_spe = Rx_sigs[2]
            x_spe = self.msa_spe_channel_to_decoder(x_spe)
            
            # x = torch.cat([x_img, x_spe], dim=1)
            x = torch.cat([x_img, x_text, x_spe], dim=1)
            # print(x.shape) # (batch_size, 3, 128)

        elif ta_perform.startswith('ave'):
            x_img = Rx_sigs[0]
            x_img = self.ave_img_channel_to_decoder(x_img)
            x_spe = Rx_sigs[1]
            x_spe = self.ave_spe_channel_to_decoder(x_spe)

            if img2 is not None:
                x_img2 = Rx_sigs[2]
                x_img2 = self.ave_img2_channel_to_decoder(x_img2)
            
            x = torch.cat([x_img, x_spe], dim=1)
            # x = torch.cat([x_img, x_spe, x_img2], dim=1)
            # print(x.shape) # (batch_size, 3, 128)

        batch_size = x.shape[0]
        if ta_perform.endswith('r'):
            x = self.decoder(x, x, None, None, None) 
            x = self.head[ta_perform](x)
            return x
        else:
            query_embed = self.task_dict[ta_perform].weight.unsqueeze(0).repeat(batch_size, 1, 1)
            x = self.decoder(query_embed, x, None, None, None) 
            if ta_perform.startswith('textr'): 
                x = self.head[ta_perform](x)
            else:
                x = self.head[ta_perform](x.mean(1))
                if ta_perform.startswith('ave'):
                    x = x.view(AVE_batch_size, -1, x.size(-1))
            if ta_perform.startswith('vqa'):
                x = self.sigmoid_layer(x)
            return x

class UDeepSCNOMAnoSIC(nn.Module):
    """
        UDeepSC non-orthogonal version without signal detection
        (signals are superimposed)
    """
    def __init__(self,
                noise_power_density_dBm,
                reference_distance,
                reference_path_loss,
                path_loss_exponent,
                distance,
                 mode='tiny',
                 img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0, 
                 img_embed_dim=384, text_embed_dim=384, speech_embed_dim=128, img_encoder_depth=4, 
                 text_encoder_depth=4, speech_encoder_depth=4, encoder_num_heads=12, decoder_num_classes=768, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, init_values=0.,use_learnable_pos_emb=False,num_classes=0, 
                 ):

        super().__init__()
        
        if mode=='tiny':
            text_embed_dim = 128
        elif mode=='small':
            text_embed_dim = 512
        else:
            text_embed_dim = 512
        
        self.img_encoder = ViTEncoder(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, 
                                num_classes=encoder_num_classes, embed_dim=img_embed_dim,depth=img_encoder_depth,
                                num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        
        bert_ckpt = f"prajjwal1/bert-{mode}"
        self.text_encoder = BertModel.from_pretrained(bert_ckpt)
        
        self.spe_encoder = SPTEncoder(in_chans=encoder_in_chans,num_classes=encoder_num_classes, embed_dim=speech_embed_dim,
                                depth=speech_encoder_depth,num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        
        self.img2_encoder = ViTEncoder(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, 
                                num_classes=encoder_num_classes, embed_dim=img_embed_dim,depth=img_encoder_depth,
                                num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        
        self.num_symbols_imgc = 16
        self.num_symbols_imgr = 16
        self.num_symbols_textc = 4
        self.num_symbols_textr = 24

        self.num_symbols_vqa = 16
        # self.num_symbols_msa = 16
        # using 3 times bandwidth compare to OMA (UDeepSC_M3)
        self.num_symbols_msa = 48 
        self.num_symbols_ave = 16
 
        self.textc_encoder_to_channel =     nn.Linear(text_embed_dim, self.num_symbols_textc)
        self.imgc_encoder_to_channel =      nn.Linear(img_embed_dim, self.num_symbols_imgc)
        self.textr_encoder_to_channel =     nn.Linear(text_embed_dim, self.num_symbols_textr)
        self.imgr_encoder_to_channel =      nn.Linear(img_embed_dim, self.num_symbols_imgr)
        self.vqa_img_encoder_to_channel =   nn.Linear(img_embed_dim, self.num_symbols_vqa)
        self.vqa_text_encoder_to_channel =  nn.Linear(text_embed_dim, self.num_symbols_vqa)
        self.msa_img_encoder_to_channel =   nn.Linear(img_embed_dim, self.num_symbols_msa)
        self.msa_text_encoder_to_channel =  nn.Linear(text_embed_dim, self.num_symbols_msa)
        self.msa_spe_encoder_to_channel =   nn.Linear(speech_embed_dim, self.num_symbols_msa)
        self.ave_img_encoder_to_channel =   nn.Linear(img_embed_dim, self.num_symbols_ave)
        self.ave_spe_encoder_to_channel =   nn.Linear(speech_embed_dim, self.num_symbols_ave)
        self.ave_img2_encoder_to_channel =   nn.Linear(img_embed_dim, self.num_symbols_ave)
        
        
        self.textc_channel_to_decoder  =    nn.Linear(self.num_symbols_textc, decoder_embed_dim)
        self.imgc_channel_to_decoder  =     nn.Linear(self.num_symbols_imgc, decoder_embed_dim)
        self.textr_channel_to_decoder  =    nn.Linear(self.num_symbols_textr, decoder_embed_dim)
        self.imgr_channel_to_decoder =      nn.Linear(self.num_symbols_imgr, decoder_embed_dim)

        self.vqa_channel_to_decoder = nn.Linear(self.num_symbols_vqa, decoder_embed_dim)

        self.msa_channel_to_decoder = nn.Linear(self.num_symbols_msa, decoder_embed_dim)

        self.ave_channel_to_decoder = nn.Linear(self.num_symbols_ave, decoder_embed_dim)
        

        self.task_dict = nn.ModuleDict()
        self.task_dict['imgc'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['imgr'] = nn.Embedding(64, decoder_embed_dim)
        self.task_dict['textc'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['vqa'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['msa']  = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['textr'] = nn.Embedding(66, decoder_embed_dim)
        self.task_dict['ave'] = nn.Embedding(25, decoder_embed_dim)

        self.head = nn.ModuleDict()
        self.head['imgc'] = nn.Linear(decoder_embed_dim, IMGC_NUMCLASS)
        self.head['textc'] = nn.Linear(decoder_embed_dim, TEXTC_NUMCLASS)
        self.head['textr'] = nn.Linear(decoder_embed_dim, TEXTR_NUMCLASS)
        self.head['vqa'] = nn.Linear(decoder_embed_dim, VQA_NUMCLASS)
        self.head['imgr'] = nn.Linear(decoder_embed_dim, IMGR_LENGTH)
        self.head['msa'] = nn.Linear(decoder_embed_dim, MSA_NUMCLASS)
        self.head['ave'] = nn.Linear(decoder_embed_dim, AVE_NUMCLASS)

        self.decoder = Decoder(depth=decoder_depth, embed_dim=decoder_embed_dim, 
                                num_heads=decoder_num_heads, dff=mlp_ratio*decoder_embed_dim, 
                                drop_rate=drop_rate)
        # self.channel = Channels()
        self.channel = AWGNMultiChannel()
        # self.channel = RayleighFadingMultiChannel(
        #                     noise_power_density_dBm,
        #                     reference_distance,
        #                     reference_path_loss,
        #                     path_loss_exponent,
        #                     distance,
        #                 )
        self.sigmoid_layer = nn.Sigmoid()
        
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
    
    def transmit(self, signal: torch.Tensor, user_dim_index: int, SNRdb: torch.FloatTensor, power_constraint: list[float]):
        """
            Args:
                signal: a complex tensor representing the signals from multiple transmitter (user).
                        with shape (batch_size, user_dim, *dim, symbol_dim):
                        - user_dim: for users, indexed by user_dim_index
                        - symbol_dim: for signal symbols
            Return:
                the signal with interference and superposition for the receiver
                will have dimension (batch_size, user_dim, *dim, symbol_dim)
        """
        # print(f"Transmitted Signal Dim = {signal.shape}")

        if isinstance(self.channel, AWGNMultiChannel):
            power_constraint = torch.tensor(power_constraint)
            signal = power_normlize_stack(signal, power_constraint)
            signal = tensor_real2complex(signal, 'concat') #(batch_size, 1, 8)
            
            # superimpose and add noise
            signal = self.channel.interfere(signal, SNRdb.item(), user_dim_index)

        elif isinstance(self.channel, RayleighFadingMultiChannel):
            power_constraint = torch.tensor(power_constraint)
            signal = tensor_real2complex(signal, 'concat')
            # power_constraint = self.channel.get_signal_power_constraint(signal, SNRdb, user_dim_index, self.channel_gain_var) # (n_tx, )

            signal = power_normlize_stack(signal, power_constraint)

            # superimpose and add noise
            signal = self.channel.interfere(
                signal=signal, 
                user_dim_index=user_dim_index, 
                SNRdb=SNRdb, 
                divide_gain = True,
                channel_gain_var=self.channel_gain_var,
                # channel_gain_var=torch.tensor([[2.0], [0.5], [0.05]])
            )
            self.channel_gain = self.channel.get_channel_gain().to(signal.device) # (batch_size, n_tx, 1, 1 ,signal length)
            # print(f'{self.channel_gain=}')
            signal = signal.squeeze(dim=user_dim_index)
            # print(f"{signal.shape=}")

        signal = tensor_complex2real(signal, 'concat')

        return signal

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def get_signals(self, text=None, img=None, speech=None, img2=None, ta_perform=None, power_constraint:list[float]=[1, 1, 1], test_snr:torch.FloatTensor=torch.FloatTensor([12]), channel_gain_var: list[list[float]] | torch.Tensor = None):
        """
            Input:
                text: text data (tokenized first) for task need text
                img: image data for task need image
                speech: audio data for task need speech
                ta_perform: The task to be eval (execute)
                power_constraint: The power constraint of each user [text, img, speech]
            Output:
                Task result executed by decoder of receiver
                Different shape for different tasks and data type
                Ex:
                    Textr: (batch_size, seq length, vocab size) for vacab size = 34000
        """
        n_user = len(power_constraint)

        if isinstance(self.channel, RayleighFadingMultiChannel):
            self.channel_gain_var = [[1.0]] * n_user if channel_gain_var is None else channel_gain_var

        if text is not None:
            x_text = self.text_encoder(ta_perform, text, return_dict=False)[0]
            power = power_constraint[0]
            # x_text = self.LN(x_text)
            if ta_perform.startswith('textc'):
                x_text = x_text[:,0,:].unsqueeze(1)
                x_text = self.textc_encoder_to_channel(x_text)
            elif ta_perform.startswith('vqa'):
                x_text = x_text[:,0:3,:]
                x_text = self.vqa_text_encoder_to_channel(x_text)
            elif ta_perform.startswith('msa'):
                x_text = x_text[:,-2:-1,:]
                x_text = self.msa_text_encoder_to_channel(x_text)

            x_text_norm, _ = power_norm_batchwise(x_text, power)
            
        if img is not None:
            AVE_batch_size = img.shape[0]
            if ta_perform.startswith('ave'):
                img = img.view(img.size(0) * img.size(1), -1, 512) # (batch_size * time_steps, 49, 512)

            x_img = self.img_encoder(img, ta_perform)
            if ta_perform.startswith('imgc'):
                power = power_constraint[0]
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img = self.imgc_encoder_to_channel(x_img)
                
            elif ta_perform.startswith('vqa'):
                power = power_constraint[1]
                x_img = x_img[:,0:3,:]
                x_img = self.vqa_img_encoder_to_channel(x_img)
                
            elif ta_perform.startswith('msa'):
                power = power_constraint[1]
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img = self.msa_img_encoder_to_channel(x_img)

            elif ta_perform.startswith('ave'):
                power = power_constraint[0]
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img = self.ave_img_encoder_to_channel(x_img)

            x_img_norm, _ = power_norm_batchwise(x_img, power)
            
        if speech is not None:
            if ta_perform.startswith('ave'):
                speech = speech.view(-1, speech.size(-1)) # (batch_size * time_steps, 128)

            x_spe = self.spe_encoder(speech, ta_perform)
            if ta_perform.startswith('msa'):
                power = power_constraint[2]
                x_spe = x_spe[:,0,:].unsqueeze(1)
                x_spe = self.msa_spe_encoder_to_channel(x_spe)
            
            elif ta_perform.startswith('ave'):
                power = power_constraint[1]
                x_spe = x_spe[:,0,:].unsqueeze(1)
                x_spe = self.ave_spe_encoder_to_channel(x_spe)
            
            x_spe_norm, _ = power_norm_batchwise(x_spe, power)
        
        if img2 is not None:
            if ta_perform.startswith('ave'):
                img2 = img2.view(img2.size(0) * img2.size(1), -1, 512) # (batch_size * time_steps, 49, 512)
                x_img2 = self.img2_encoder(img2, ta_perform)
                
                power = power_constraint[2]
                x_img2 = x_img2[:,0,:].unsqueeze(1)
                x_img2 = self.ave_img2_encoder_to_channel(x_img2)
            
        if ta_perform.startswith('img'):
            x_img = x_img.unsqueeze(1)

        elif ta_perform.startswith('text'):
            x_text = x_text.unsqueeze(1)

        elif ta_perform.startswith('vqa'):
            x = torch.stack((x_img, x_text), dim=1)

        elif ta_perform.startswith('msa'):
            power = torch.tensor(power_constraint)
            ## stack -> power normalizated
            x = torch.stack((x_img, x_text, x_spe), dim=1)
            # x = power_normlize_superimposed(x, power).clone().detach()
            x, _ = power_norm_batchwise(x, 3)
            # x = x.clone().detach()
            x = torch.sum(x, dim=1).clone().detach()

            ## power normalizated -> stack
            x_norm = torch.stack((x_img_norm, x_text_norm, x_spe_norm), dim=1)
            x_norm = torch.sum(x_norm, dim=1).clone().detach()

        elif ta_perform.startswith('ave'):
            power = 1
            x = torch.stack((x_img, x_spe), dim=1)
        
        return x, x_norm


    def forward(self, text=None, img=None, speech=None, img2=None, ta_perform=None, power_constraint:list[float]=[1, 1, 1], test_snr:torch.FloatTensor=torch.FloatTensor([12]), channel_gain_var: list[list[float]] | torch.Tensor = None):
        """
            Input:
                text: text data (tokenized first) for task need text
                img: image data for task need image
                speech: audio data for task need speech
                ta_perform: The task to be eval (execute)
                power_constraint: The power constraint of each user [text, img, speech]
            Output:
                Task result executed by decoder of receiver
                Different shape for different tasks and data type
                Ex:
                    Textr: (batch_size, seq length, vocab size) for vacab size = 34000
        """
        n_user = len(power_constraint)
        
        if self.training:
            noise_snr, noise_std = noise_gen(self.training)
            noise_std = noise_std.cuda()
        else:
            noise_std = torch.FloatTensor([1]) * 10**(-test_snr/20) 
            noise_snr = test_snr

        if isinstance(self.channel, RayleighFadingMultiChannel):
            self.channel_gain_var = [[1.0]] * n_user if channel_gain_var is None else channel_gain_var

        if text is not None:
            x_text = self.text_encoder(ta_perform, text, return_dict=False)[0]
            # x_text = self.LN(x_text)
            if ta_perform.startswith('textc'):
                x_text = x_text[:,0,:].unsqueeze(1)
                x_text = self.textc_encoder_to_channel(x_text)
            elif ta_perform.startswith('vqa'):
                x_text = x_text[:,0:3,:]
                x_text = self.vqa_text_encoder_to_channel(x_text)
            elif ta_perform.startswith('msa'):
                x_text = x_text[:,-2:-1,:]
                x_text = self.msa_text_encoder_to_channel(x_text)

            # x_text, _ = power_norm_batchwise(x_text, power)
            
        if img is not None:
            AVE_batch_size = img.shape[0]
            if ta_perform.startswith('ave'):
                img = img.view(img.size(0) * img.size(1), -1, 512) # (batch_size * time_steps, 49, 512)

            x_img = self.img_encoder(img, ta_perform)
            if ta_perform.startswith('imgc'):
                power = power_constraint[0]
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img = self.imgc_encoder_to_channel(x_img)
                
            elif ta_perform.startswith('vqa'):
                x_img = x_img[:,0:3,:]
                x_img = self.vqa_img_encoder_to_channel(x_img)
                
            elif ta_perform.startswith('msa'):
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img = self.msa_img_encoder_to_channel(x_img)

            elif ta_perform.startswith('ave'):
                power = power_constraint[0]
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img = self.ave_img_encoder_to_channel(x_img)

            # x_img, _ = power_norm_batchwise(x_img, power)
            
        if speech is not None:
            if ta_perform.startswith('ave'):
                speech = speech.view(-1, speech.size(-1)) # (batch_size * time_steps, 128)

            x_spe = self.spe_encoder(speech, ta_perform)
            if ta_perform.startswith('msa'):
                x_spe = x_spe[:,0,:].unsqueeze(1)
                x_spe = self.msa_spe_encoder_to_channel(x_spe)
            
            elif ta_perform.startswith('ave'):
                power = power_constraint[1]
                x_spe = x_spe[:,0,:].unsqueeze(1)
                x_spe = self.ave_spe_encoder_to_channel(x_spe)
            
            # x_spe, _ = power_norm_batchwise(x_spe, power)
        
        if img2 is not None:
            if ta_perform.startswith('ave'):
                img2 = img2.view(img2.size(0) * img2.size(1), -1, 512) # (batch_size * time_steps, 49, 512)
                x_img2 = self.img2_encoder(img2, ta_perform)
                
                power = power_constraint[2]
                x_img2 = x_img2[:,0,:].unsqueeze(1)
                x_img2 = self.ave_img2_encoder_to_channel(x_img2)
            
        if ta_perform.startswith('img'):
            x_img = x_img.unsqueeze(1)
            x_img = self.transmit(x_img, 1, noise_snr)
        elif ta_perform.startswith('text'):
            x_text = x_text.unsqueeze(1)
            x_text = self.transmit(x_text, 1, noise_snr)
        elif ta_perform.startswith('vqa'):
            x = torch.stack((x_img, x_text), dim=1)
            x = self.transmit(x, 1, noise_snr, 2)
        elif ta_perform.startswith('msa'):
            x = torch.stack((x_text, x_img, x_spe), dim=1)
            # x = torch.stack((x_text, x_spe), dim=1)
            # x = x_img.unsqueeze(1)
            x = self.transmit(x, 1, noise_snr, power_constraint)

        elif ta_perform.startswith('ave'):
            x = torch.stack((x_img, x_spe), dim=1)
            # x = torch.stack((x_img, x_spe, x_img2), dim=1)
            x = self.transmit(x, 1, noise_snr, power_constraint)
        
        
        if ta_perform.startswith('img'):
            x = x_img
        elif ta_perform.startswith('text'):
            x = x_text
        elif ta_perform.startswith('vqa'):
            x = self.vqa_channel_to_decoder(x)
                  
        elif ta_perform.startswith('msa'):
            x = self.msa_channel_to_decoder(x)

        elif ta_perform.startswith('ave'):
            x = self.ave_channel_to_decoder(x)

        batch_size = x.shape[0]
        if ta_perform.endswith('r'):
            x = self.decoder(x, x, None, None, None) 
            x = self.head[ta_perform](x)
            return x
        else:
            query_embed = self.task_dict[ta_perform].weight.unsqueeze(0).repeat(batch_size, 1, 1)
            x = self.decoder(query_embed, x, None, None, None) 
            if ta_perform.startswith('textr'): 
                x = self.head[ta_perform](x)
            else:
                x = self.head[ta_perform](x.mean(1))
                if ta_perform.startswith('ave'):
                    x = x.view(AVE_batch_size, -1, x.size(-1))
            if ta_perform.startswith('vqa'):
                x = self.sigmoid_layer(x)
            return x

@register_model
def UDeepSC_model(pretrained=False, **kwargs):
    model = UDeepSC_M1(
        mode='small',
        img_size=32,
        patch_size=4,
        img_embed_dim=384,
        text_embed_dim=384,
        speech_embed_dim=128,
        img_encoder_depth=6,
        text_encoder_depth=4,
        speech_encoder_depth=4,
        encoder_num_heads=6,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def UDeepSC_new_model(pretrained=False, **kwargs):
    model = UDeepSC_M2(
        mode='small',
        img_size=32,
        patch_size=4,
        img_embed_dim=384,
        text_embed_dim=512,
        speech_embed_dim=128,
        img_encoder_depth=6,
        text_encoder_depth=4,
        speech_encoder_depth=4,
        encoder_num_heads=6,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def UDeepSC_SepCD_model(pretrained=False, **kwargs):
    model = UDeepSC_M3(
        mode='small',
        img_size=32,
        patch_size=4,
        img_embed_dim=384,
        text_embed_dim=512,
        speech_embed_dim=128,
        img_encoder_depth=6,
        text_encoder_depth=4,
        speech_encoder_depth=4,
        encoder_num_heads=6,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def UDeepSC_perfectSIC_model(pretrained=False, **kwargs):
    model = UDeepSCperfectSIC(
        mode='small',
        img_size=32,
        patch_size=4,
        img_embed_dim=384,
        text_embed_dim=512,
        speech_embed_dim=128,
        img_encoder_depth=6,
        text_encoder_depth=4,
        speech_encoder_depth=4,
        encoder_num_heads=6,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def UDeepSC_NOMA_new_model(pretrained=False, **kwargs):
    model = UDeepSC_M3_withSIC(
        noise_power_density_dBm=-90,    # ref. ISSNOMATrainer's note
        reference_distance=1,
        reference_path_loss=pow(10, -30/10),
        path_loss_exponent=4,
        distance=torch.Tensor([83, 33, 133]).reshape(3, 1),
        mode='small',
        img_size=32,
        patch_size=4,
        img_embed_dim=384,
        text_embed_dim=512,
        speech_embed_dim=128,
        img_encoder_depth=6,
        text_encoder_depth=4,
        speech_encoder_depth=4,
        encoder_num_heads=6,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def UDeepSC_NOMANoSIC_model(pretrained=False, **kwargs):
    model = UDeepSCNOMAnoSIC(
        noise_power_density_dBm=-90,    # ref. ISSNOMATrainer's note
        reference_distance=1,
        reference_path_loss=pow(10, -30/10),
        path_loss_exponent=4,
        distance=torch.Tensor([33, 83, 133]).reshape(3, 1),
        mode='small',
        img_size=32,
        patch_size=4,
        img_embed_dim=384,
        text_embed_dim=512,
        speech_embed_dim=128,
        img_encoder_depth=6,
        text_encoder_depth=4,
        speech_encoder_depth=4,
        encoder_num_heads=6,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model