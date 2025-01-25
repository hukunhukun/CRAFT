import torch
import torch.nn.functional as F
from torch import nn

from model.bert import BertForMaskSM,bert_encoder
from model.gnn import GNN_graphpred

import json


class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = Config(value)
        return value
                                                 
class TMMF(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        

        self.msm_pro = config['msm_probability']
        
        mol_config = Config(json.load(open(config['gnn_config'])))
        iuapc_config = Config(json.load(open(config['iupac_config'])))
        selfies_config = Config(json.load(open(config['selfies_config'])))
        fusion_config = Config(json.load(open(config['fusion_config'])))
        train_config = Config(json.load(open(config['train_config'])))
        

        self.mol_encoder = GNN_graphpred(mol_config)
        self.iupac_encoder = bert_encoder(iuapc_config) 
        self.selfies_encoder = bert_encoder(selfies_config)
        self.fusion_encoder = bert_encoder(fusion_config)

        self.BertMaskSelfies = BertForMaskSM(config=train_config,
                                             selfies_encoder=self.selfies_encoder,
                                             iupac_encoder=self.iupac_encoder,
                                             mol_encoder=self.mol_encoder,
                                             fusion_encoder=self.fusion_encoder)


        mol_width = config['mol_width']
        iuapc_width = self.iupac_encoder.config.hidden_size
        selfies_width = self.selfies_encoder.config.hidden_size

        embed_dim = config['embed_dim']   # projection dimension


        self.mol_proj = nn.Linear(mol_width, embed_dim)
        self.iupac_proj = nn.Linear(iuapc_width, embed_dim)
        self.selfies_proj = nn.Linear(selfies_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.quene_size = config['quene_size']
        self.momentum = config['momentum']
        self.sim_head = nn.Linear(selfies_width, 2)


        # create momentum models

        self.mol_encoder_m = GNN_graphpred(mol_config)
        self.iupac_encoder_m = bert_encoder(iuapc_config)
        self.selfies_encoder_m = bert_encoder(selfies_config)
        self.fusion_encoder_m = bert_encoder(fusion_config)

        self.mol_proj_m = nn.Linear(mol_width, embed_dim)
        self.iupac_proj_m = nn.Linear(iuapc_width, embed_dim)
        self.selfies_proj_m = nn.Linear(selfies_width, embed_dim)

        self.model_pairs = [[self.mol_encoder, self.mol_encoder_m],
                            [self.iupac_encoder, self.iupac_encoder_m],
                            [self.selfies_encoder, self.selfies_encoder_m],
                            [self.mol_proj, self.mol_proj_m],
                            [self.iupac_proj, self.iupac_proj_m],
                            [self.selfies_proj, self.selfies_proj_m],
                            [self.fusion_encoder, self.fusion_encoder_m]]
        self.copy_params()

        # crerate the queue

        self.register_buffer("mol_queue", torch.randn(embed_dim, self.quene_size))
        self.register_buffer("iupac_queue", torch.randn(embed_dim, self.quene_size))
        self.register_buffer("selfies_queue", torch.randn(embed_dim, self.quene_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.mol_queue = F.normalize(self.mol_queue, dim=0)
        self.iupac_queue = F.normalize(self.iupac_queue, dim=0)
        self.selfies_queue = F.normalize(self.selfies_queue, dim=0)


    def forward(self, mol, iupac, selfies,alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)


        iupac_embeds,iupac_mask = self.iupac_encoder(input_ids=iupac)
        iupac_feat = F.normalize(self.iupac_proj(iupac_embeds[:,0,:]), dim=-1)

        selfies_embeds,selfies_mask = self.selfies_encoder(input_ids = selfies)
        selfies_feat = F.normalize(self.selfies_proj(selfies_embeds[:,0,:]), dim=-1)

        mol_pool,mol_embeds,mol_mask = self.mol_encoder(mol)

        mol_feat = F.normalize(self.mol_proj(mol_pool), dim=-1)
    

        # momentum update
        with torch.no_grad():
            self._momentum_update()
            selfies_embeds_m,_ = self.selfies_encoder_m(input_ids = selfies)
            selfies_feat_m = F.normalize(self.selfies_proj_m(selfies_embeds_m[:,0,:]), dim=-1)
            selfies_feat_all = torch.cat([selfies_feat_m.t(),self.selfies_queue.clone().detach()],dim=1)

            iupac_embeds_m,_ = self.iupac_encoder_m(input_ids=iupac)
            iupac_feat_m = F.normalize(self.iupac_proj_m(iupac_embeds_m[:,0,:]), dim=-1)
            iupac_feat_all = torch.cat([iupac_feat_m.t(),self.iupac_queue.clone().detach()],dim=1)

            mol_pool_m,mol_embeds_m,mask_mol_m = self.mol_encoder_m(mol)
            mol_feat_m = F.normalize(self.mol_proj_m(mol_pool_m), dim=-1)
            mol_feat_all = torch.cat([mol_feat_m.t(),self.mol_queue.clone().detach()],dim=1)

            sim_s2i_m = selfies_feat_m @ iupac_feat_all / self.temp
            sim_s2m_m = selfies_feat_m @ mol_feat_all / self.temp
            sim_s2im_m =  (selfies_feat_m @ iupac_feat_all + selfies_feat_m @ mol_feat_all) / self.temp

            smi_i2s_m = iupac_feat_m @ selfies_feat_all / self.temp
            smi_m2s_m = mol_feat_m @ selfies_feat_all / self.temp
            smi_im2s_m = (iupac_feat_m @ selfies_feat_all + mol_feat_m @ selfies_feat_all) / self.temp

            sim_targets = torch.zeros(sim_s2i_m.size()).to(selfies.device)
            sim_targets.fill_diagonal_(1)

            sim_s2i_targets = alpha * F.softmax(sim_s2i_m,dim=1) + (1-alpha) * sim_targets
            sim_s2m_targets = alpha * F.softmax(sim_s2m_m,dim=1) + (1-alpha) * sim_targets
            sim_s2im_targets = alpha * F.softmax(sim_s2im_m,dim=1) + (1-alpha) * sim_targets
            sim_i2s_targets = alpha * F.softmax(smi_i2s_m,dim=1) + (1-alpha) * sim_targets
            sim_m2s_targets = alpha * F.softmax(smi_m2s_m,dim=1) + (1-alpha) * sim_targets
            sim_im2s_targets = alpha * F.softmax(smi_im2s_m,dim=1) + (1-alpha) * sim_targets

        sim_s2i = selfies_feat @ iupac_feat_all / self.temp
        sim_i2s = iupac_feat @ selfies_feat_all / self.temp

        sim_s2m = selfies_feat @ mol_feat_all / self.temp
        sim_m2s = mol_feat @ selfies_feat_all / self.temp

        sim_s2im = (selfies_feat @ iupac_feat_all + selfies_feat @ mol_feat_all) / self.temp
        sim_im2s = (iupac_feat @ selfies_feat_all + mol_feat @ selfies_feat_all) / self.temp

        loss_s2i = -torch.sum(F.log_softmax(sim_s2i,dim=1)*sim_s2i_targets,dim=1).mean()
        loss_i2s = -torch.sum(F.log_softmax(sim_i2s,dim=1)*sim_i2s_targets,dim=1).mean()
        loss_s2m = -torch.sum(F.log_softmax(sim_s2m,dim=1)*sim_s2m_targets,dim=1).mean()
        loss_m2s = -torch.sum(F.log_softmax(sim_m2s,dim=1)*sim_m2s_targets,dim=1).mean()
        loss_s2im = -torch.sum(F.log_softmax(sim_s2im,dim=1)*sim_s2im_targets,dim=1).mean()
        loss_im2s = -torch.sum(F.log_softmax(sim_im2s,dim=1)*sim_im2s_targets,dim=1).mean()

        loss_sima = ((loss_s2i + loss_i2s)/2 + (loss_s2m + loss_m2s)/2 + (loss_s2im + loss_im2s)/2)/3

        self._dequeue_and_enqueue(selfies_feat_m,iupac_feat_m,mol_feat_m)


        # forward the postive selfies-iupac-mol pair
  

        iupac_mol_embeds = torch.cat([iupac_embeds,mol_embeds],dim=1)
        iupac_mol_mask = torch.cat([iupac_mask,mol_mask],dim=1)

        # mask = (batch_size,seq_len)

        # (batch_size,seq_len,embed_size)
        output_pos = self.fusion_encoder(selfies_embeds=selfies_embeds,
                                            iupac_mol_embeds=iupac_mol_embeds,
                                            selfies_mask=selfies_mask,
                                            iupac_mol_mask=iupac_mol_mask)

        with torch.no_grad():
            bs = selfies.size(0)
            weights_s2im = F.softmax(sim_s2im[:,:bs]+1e-4,dim=1)
            weights_im2s = F.softmax(sim_im2s[:,:bs]+1e-4,dim=1)

            weights_im2s.fill_diagonal_(0)
            weights_s2im.fill_diagonal_(0)

        # select a negative iupac_mol for each selfies
        # mask matrix of iupac_mol pair

        iupac_mol_embeds_neg = []
        iupac_mol_mask_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_s2im[b],1).item()
            iupac_mol_embeds_neg.append(iupac_mol_embeds[neg_idx])
            iupac_mol_mask_neg.append(iupac_mol_mask[neg_idx])
        iupac_mol_embeds_neg = torch.stack(iupac_mol_embeds_neg,dim=0)
        iupac_mol_mask_neg = torch.stack(iupac_mol_mask_neg,dim=0)

        # select a negative selfies for each iupac_mol
        selfies_embeds_neg = []
        selfies_mask_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_im2s[b],1).item()
            selfies_embeds_neg.append(selfies_embeds[neg_idx])
            selfies_mask_neg.append(selfies_mask[neg_idx])
        selfies_embeds_neg = torch.stack(selfies_embeds_neg,dim=0)
        selfies_mask_neg = torch.stack(selfies_mask_neg,dim=0)

        selfies_embeds_all = torch.cat([selfies_embeds,selfies_embeds_neg],dim=0)
        selfies_mask_all = torch.cat([selfies_mask,selfies_mask_neg],dim=0)

        iupac_mol_embeds_all = torch.cat([iupac_mol_embeds_neg,iupac_mol_embeds],dim=0)
        iuapc_mol_mask_all = torch.cat([iupac_mol_mask_neg,iupac_mol_mask],dim=0)  # to be modified


        # (2*batch_size,seq_len,hidden_size)
        output_neg = self.fusion_encoder(selfies_embeds=selfies_embeds_all,
                                        iupac_mol_embeds=iupac_mol_embeds_all,
                                        selfies_mask=selfies_mask_all,
                                        iupac_mol_mask=iuapc_mol_mask_all)
        
        sim_embeddings = torch.cat([output_pos[:,0,:],output_neg[:,0,:]],dim=0)
        sim_output = self.sim_head(sim_embeddings)
        simm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],dim=0).to(selfies.device)
        loss_simm = F.cross_entropy(sim_output,simm_labels)


    # ## mask selfies modeling loss ##
        input_ids = selfies.clone()
        labels = selfies.clone()

        pro_matrix = torch.full(labels.shape, self.msm_pro)
        input_ids, labels = self.mask(input_ids=input_ids,vocab_size=self.selfies_encoder.config.vocab_size,targets=labels,
                                      pro_matrix = pro_matrix,device=input_ids.device)  

         
        out,loss_msm = self.BertMaskSelfies(selfies_ids=input_ids,
                                                iupac_ids = iupac.clone(),
                                                mol_ids = mol.clone(),
                                                labels=labels,
                                                mode='train') 
            

        return out,loss_sima,loss_simm,loss_msm
        
    def get_fusion_feature(self,mol, iupac, selfies):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)

      
        input_ids = selfies.clone()
        labels = selfies.clone()
        out = self.BertMaskSelfies(selfies_ids=input_ids,
                                                iupac_ids = iupac.clone(),
                                                mol_ids = mol.clone(),
                                                labels=None,
                                                mode='output') 
        return out
    def test(self,selfies,iupac,mol):
        input_ids = selfies.clone()
        labels = selfies.clone()

        pro_matrix = torch.full(labels.shape, self.msm_pro)
        input_ids, labels = self.mask(input_ids=input_ids,vocab_size=self.selfies_encoder.config.vocab_size,targets=labels,
                                      pro_matrix = pro_matrix,device=input_ids.device)  
        correct,num = self.BertMaskSelfies(selfies_ids=input_ids,
                                            iupac_ids = iupac.clone(),
                                            mol_ids = mol.clone(),
                                            labels=labels,
                                            mode='test') 
        return correct,num



    def mask(self,input_ids,vocab_size,targets=None,pro_matrix=None,device=None):
        masked_indices = torch.bernoulli(pro_matrix).bool()
                                               
        masked_indices[input_ids == 0] = False
        masked_indices[input_ids == 1] = False
        
        if targets is not None:
            targets[~masked_indices] = 0          

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = 2

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids



    @torch.no_grad() 
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data) 
                param_m.requires_grad = False    

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self,selfies_feat,iupac_feat,mol_feat):
        # selfies_feat = concat_all_gather(selfies_feat)
        # iupac_feat = concat_all_gather(iupac_feat)
        # mol_feat = concat_all_gather(mol_feat)

        batch_size = selfies_feat.shape[0]

        ptr = int(self.queue_ptr)
        assert self.quene_size % batch_size == 0  # for simplicity

        self.selfies_queue[:, ptr:ptr + batch_size] = selfies_feat.T
        self.iupac_queue[:, ptr:ptr + batch_size] = iupac_feat.T
        self.mol_queue[:, ptr:ptr + batch_size] = mol_feat.T

        ptr = (ptr + batch_size) % self.quene_size  # move pointer

        self.queue_ptr[0] = ptr



# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [torch.ones_like(tensor)
#         for _ in range(torch.distributed.get_world_size())]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

#     output = torch.cat(tensors_gather, dim=0)
#     return output