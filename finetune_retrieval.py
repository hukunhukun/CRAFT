import torch
from torch import nn
from model.TMMF import TMMF
from dataset import create_dataset,create_dataloader
import ruamel.yaml as yaml
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path

class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = Config(value)
        return value
    


# calculate topk recall rate

def recall_topk(model,loader,device,topk):
    recall_s2i = []
    recall_i2s = []
    recall_s2m = []
    recall_m2s = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ids,selfies,iupac,mol = batch['ids'],batch['sfs_tokens']['selfies_tokens'],batch['ipc_tokens']['iupac_tokens'],batch['mlg']
            selfies = selfies.to(device)
            iupac = iupac.to(device)
            mol = mol.to(device)  
            
            iupac_embeds,_ = model.iupac_encoder(input_ids=iupac)
            iupac_feat = F.normalize(model.iupac_proj(iupac_embeds[:,0,:]), dim=-1)

            selfies_embeds,_ = model.selfies_encoder(input_ids = selfies)
            selfies_feat = F.normalize(model.selfies_proj(selfies_embeds[:,0,:]), dim=-1)

            mol_pool,_,_ = model.mol_encoder(mol)
            mol_feat = F.normalize(model.mol_proj(mol_pool), dim=-1)

            sim_s2i = selfies_feat @ iupac_feat.t() / model.temp
            sim_i2s = iupac_feat @ selfies_feat.t() / model.temp

            sim_s2m = selfies_feat @ mol_feat.t() / model.temp
            sim_m2s = mol_feat @ selfies_feat.t() / model.temp

            sim_s2i = F.softmax(sim_s2i, dim=-1)
            sim_i2s = F.softmax(sim_i2s, dim=-1)
            sim_s2m = F.softmax(sim_s2m, dim=-1)
            sim_m2s = F.softmax(sim_m2s, dim=-1)

            # 通过sim矩阵计算当前bacth中的topk recall rate
            # sim矩阵的每一行代表一个selfies的相似度分布, 对角线上的为正例，取相似度最高的前k个，如果正确的正例在其中，那么recall+1

            # sim_s2i 代码如下
            sim_s2i = sim_s2i.cpu().numpy()
            for i in range(sim_s2i.shape[0]):
                topk_idx = np.argsort(sim_s2i[i,:])[::-1][:topk]
                if i in topk_idx:
                    recall_s2i.append(1)
                else:
                    recall_s2i.append(0)
            # sim_i2s
            sim_i2s = sim_i2s.cpu().numpy()
            for i in range(sim_i2s.shape[0]):
                topk_idx = np.argsort(sim_i2s[i,:])[::-1][:topk]
                if i in topk_idx:
                    recall_i2s.append(1)
                else:
                    recall_i2s.append(0)
            # sim_s2m
            sim_s2m = sim_s2m.cpu().numpy()
            for i in range(sim_s2m.shape[0]):
                topk_idx = np.argsort(sim_s2m[i,:])[::-1][:topk]
                if i in topk_idx:
                    recall_s2m.append(1)
                else:
                    recall_s2m.append(0)
            # sim_m2s
            sim_m2s = sim_m2s.cpu().numpy()
            for i in range(sim_m2s.shape[0]):
                topk_idx = np.argsort(sim_m2s[i,:])[::-1][:topk]
                if i in topk_idx:
                    recall_m2s.append(1)
                else:
                    recall_m2s.append(0)


    recall_rate_s2i = sum(recall_s2i)/len(recall_s2i)
    recall_rate_i2s = sum(recall_i2s)/len(recall_i2s)
    recall_rate_s2m = sum(recall_s2m)/len(recall_s2m)
    recall_rate_m2s = sum(recall_m2s)/len(recall_m2s)

    return recall_rate_s2i,recall_rate_i2s,recall_rate_s2m,recall_rate_m2s


# zero-shot retrieval

def main(args,config):
    device = args.device
    topk = args.topk
    output_dir = args.output_dir

    print("creating dataset")
    dataset,_ = create_dataset(
    config['dataset']['ids_path'],
    config['dataset']['ipc_path'],
    config['dataset']['sfs_path'],
    config['dataset']['mlg_path'],
    1.0)

    loader = create_dataloader(dataset,1024)
    model = TMMF(config)
    model = model.to(device)

    path = './pre_train/1m_checkpoint_09.pth'
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])

    recall_rate_s2i,recall_rate_i2s,recall_rate_s2m,recall_rate_m2s = recall_topk(model,loader,device,topk)
    print("recall_rate_s2i: ",recall_rate_s2i)
    print("recall_rate_i2s: ",recall_rate_i2s)
    print("recall_rate_s2m: ",recall_rate_s2m)
    print("recall_rate_m2s: ",recall_rate_m2s)

    with open(output_dir+ 'top{}'.format(topk) +'_recall_rate.txt','w') as f:
        f.write("recall_rate_s2i: "+str(recall_rate_s2i)+'\n')
        f.write("recall_rate_i2s: "+str(recall_rate_i2s)+'\n')
        f.write("recall_rate_s2m: "+str(recall_rate_s2m)+'\n')
        f.write("recall_rate_m2s: "+str(recall_rate_m2s)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/retrieval.yaml')
    parser.add_argument('--output_dir', default='./downstream/retrieval/')
    parser.add_argument('--device', default=torch.device("cuda:0"))
    parser.add_argument('--topk', default=1)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
     
    main(args, config)





