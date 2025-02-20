"""
BERT Encoder
"""

import torch
from torch import nn
import math
import torch.nn.functional as F


class BertEmbedding(nn.Module):

    def __init__(self,config):
        super(BertEmbedding,self).__init__()
        self.token_embedding = nn.Embedding(config.vocab_size,config.hidden_size,padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_position_embeddings,config.hidden_size)
        self.token_type_embedding = nn.Embedding(config.type_vocab_size,config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size,eps=1e-12)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.config = config

    
    def forward(self,input_ids,token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length,dtype=torch.long,device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        token_type_embeddings = self.token_type_embedding(token_type_ids)

        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):

        return x + self.dropout(sublayer(self.norm(x)))    

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    

"""attention """
class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

"""multiHeadAttention"""
class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map

    def forward(self, query, key, value, mask=None,save_attention=False):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        if save_attention:
            self.save_attention_map(attn)

        return self.output_linear(x)
    

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, save_attention=False):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.save_attention = save_attention

    def forward(self,x,y,mask,save_attention=False):
        # x is query, y is key and value
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, y, y, mask=mask,save_attention=save_attention))
        

        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class FusionBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, save_attention=False):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.cross_attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.save_attention = save_attention

    def forward(self,x,y,mask_self,mask_cross,save_attention=False):
        # x is query, y is key and value
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask_self,save_attention=False))
        x = self.input_sublayer(x, lambda _x: self.cross_attention.forward(_x, y, y, mask=mask_cross,save_attention=save_attention))        

        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x) 

class bert_encoder(nn.Module):

    def __init__(self,config):
        
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.n_layers = config.n_layers
        self.attn_heads = config.attn_heads
        self.vocab_size = config.vocab_size
        self.dropout = config.dropout_rate
        self.mode = config.mode

        self.embedding = BertEmbedding(config)

        if self.mode == 'fusion':
            self.transformer_blocks =  nn.ModuleList(
                [FusionBlock(self.hidden_size, self.attn_heads, self.hidden_size * 4, self.dropout) for _ in range(self.n_layers)])

        else:
            self.transformer_blocks = nn.ModuleList(
                [TransformerBlock(self.hidden_size, self.attn_heads, self.hidden_size * 4, self.dropout) for _ in range(self.n_layers)])
            

    
    def forward(self, input_ids=None,selfies_embeds=None,iupac_mol_embeds=None,selfies_mask=None,iupac_mol_mask=None,mode=None,save_attention=False):

        mode = self.mode if mode is None else mode
        if mode == 'iupac' or mode=='selfies':
            x = input_ids

            mask = (x > 0)
            mask_attn = mask.unsqueeze(1).repeat(1, mask.size(1), 1).unsqueeze(1).to(input_ids.device)
            x = self.embedding(x)
            for transformer in self.transformer_blocks:
                x = transformer.forward(x,x,mask_attn)

            return x,mask
        
        if mode == 'fusion':
            # y consit of both iuapc and molecular graph features
            self.transformer_blocks[0].save_attention = True
            mask_selfies = selfies_mask.clone()
            mask_selfies = mask_selfies.unsqueeze(2).to(torch.float32)
            mask_iupac_mol = iupac_mol_mask.unsqueeze(1).to(torch.float32)
            mask_cross = torch.matmul(mask_selfies,mask_iupac_mol).unsqueeze(1).to(selfies_embeds.device)
            mask_self = selfies_mask.unsqueeze(1).repeat(1, selfies_mask.size(1), 1).unsqueeze(1).to(selfies_embeds.device)
            i = 0
            for transformer in self.transformer_blocks:
                selfies_embeds = transformer.forward(selfies_embeds,iupac_mol_embeds,mask_self,mask_cross,save_attention=True)
                # if save_attention and transformer.save_attention:
                #     torch.save(transformer.cross_attention.get_attention_map(),'./attention/cross_attention_'+ str(i) +'.pt')
                #     i += 1
            return selfies_embeds
        
        return None
    
class BertForMaskSM(nn.Module):

    def __init__(self,config,selfies_encoder,iupac_encoder,mol_encoder,fusion_encoder):
        super().__init__()
        self.config = config
        self.selfies_encoder = selfies_encoder
        self.iupac_encoder = iupac_encoder
        self.mol_encoder = mol_encoder
        self.fusion_encoder = fusion_encoder
        self.fc = nn.Linear(config.hidden_size,config.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,selfies_ids=None,iupac_ids=None,mol_ids=None,labels=None,mode=None):

        selfies_embeds,selfies_mask = self.selfies_encoder(input_ids=selfies_ids,mode='selfies')
        iupac_embeds,iupac_mask = self.iupac_encoder(input_ids=iupac_ids,mode='iupac')
        mol_pool,mol_embeds,mol_mask = self.mol_encoder(mol_ids)

        iupac_mol_embeds = torch.cat([iupac_embeds,mol_embeds],dim=1)
        iupac_mol_mask = torch.cat([iupac_mask,mol_mask],dim=1)
        output = self.fusion_encoder(selfies_embeds=selfies_embeds,
                                     iupac_mol_embeds=iupac_mol_embeds,
                                     selfies_mask=selfies_mask,
                                     iupac_mol_mask=iupac_mol_mask,
                                     mode='fusion')
        if mode == 'output':
            return output[:,:,:]
        output = self.softmax(self.fc(output))

        if labels is not None:
            loss_fct = nn.NLLLoss(ignore_index=0)
            loss_msm = loss_fct(output.transpose(1, 2),labels)

            if mode == 'train':
                return output,loss_msm
            
            if mode == 'test':
                pred = output.transpose(1, 2).argmax(dim=1)
                num = torch.ne(labels,0).sum().item()
                for k in range(len(labels)):
                    for j in range(len(labels[k])):
                        if labels[k][j]==0: labels[k][j] = -100
                correct = torch.eq(pred,labels).sum().float().item()
                return correct,num
            
        else:
            return output,None
        

class BertForFinetune(nn.Module):

    def __init__(self,config,selfies_encoder,iupac_encoder,mol_encoder,fusion_encoder):
        super().__init__()
        self.config = config
        self.selfies_encoder = selfies_encoder
        self.iupac_encoder = iupac_encoder
        self.mol_encoder = mol_encoder
        self.fusion_encoder = fusion_encoder
        self.task_type = config.dataset.task_type
        self.metrics = config.dataset.metrics
        self.dropuout = nn.Dropout(config.dropout)
        # self.softmax = nn.LogSoftmax(dim=-1)
        if self.task_type == 'classification':
            # for p in self.parameters():
            #     p.requires_grad=False
            self.num_task = config.dataset.num_task
            self.linear1 = nn.Linear(config.hidden_size,config.hidden_size)
            self.activation = nn.Tanh()
            self.linear2 = nn.Linear(config.hidden_size,config.dataset.num_task)
        
        if self.task_type == 'regression':
            self.linear1 = nn.Linear(config.hidden_size,config.hidden_size)
            self.activation = nn.Tanh()
            self.linear2 = nn.Linear(config.hidden_size,1)


    def forward(self,selfies_ids=None,iupac_ids=None,mol_ids=None,labels=None):
        selfies_embeds,selfies_mask = self.selfies_encoder(input_ids=selfies_ids,mode='selfies')
        iupac_embeds,iupac_mask = self.iupac_encoder(input_ids=iupac_ids,mode='iupac')
        mol_pool,mol_embeds,mol_mask = self.mol_encoder(mol_ids)

        iupac_mol_embeds = torch.cat([iupac_embeds,mol_embeds],dim=1)
        iupac_mol_mask = torch.cat([iupac_mask,mol_mask],dim=1)
        output = self.fusion_encoder(selfies_embeds=selfies_embeds,
                                     iupac_mol_embeds=iupac_mol_embeds,
                                     selfies_mask=selfies_mask,
                                     iupac_mol_mask=iupac_mol_mask,
                                     mode='fusion')
        output = self.linear1(output[:,0,:])
        output = self.activation(output)
        output = self.linear2(self.dropuout(output))

        return output



    def _train(self,selfies_ids=None,iupac_ids=None,mol_ids=None,labels=None):
        output = self.forward(selfies_ids,iupac_ids,mol_ids)

        if self.task_type == 'classification':
            m = nn.Sigmoid()
            pred = m(output)
            loss_fct = nn.BCELoss(reduction = "none")
            is_valid = labels**2 > 0 
            loss_mat = loss_fct(pred,(labels+1)/2)
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat)/torch.sum(is_valid)
            return pred,loss
        elif self.task_type == 'regression':
            loss_fct = nn.MSELoss()
            loss = loss_fct(output.squeeze(-1),labels)
            return output,loss
        
        return None,None
    

    def _test(self,selfies_ids=None,iupac_ids=None,mol_ids=None,labels=None):
        output = self.forward(selfies_ids,iupac_ids,mol_ids)

        if self.task_type == 'classification':
            m = nn.Sigmoid()
            pred = m(output)
            return pred,labels
        
        if self.task_type == 'regression':
            pred = output.squeeze(-1)
            return pred,labels





        



