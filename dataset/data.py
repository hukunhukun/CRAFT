import json
import torch
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from .tokenizer import Iupac_tokenizer,Selfies_tokenizer
from.atom_rep  import mol_to_graph_data_obj_simple




class ipcMapper(object):
    def __init__(self,ipc_dir,max_len):
        self.max_len = max_len
        self.ipc_dir = ipc_dir
        self.training = True 
        self.json_dict = json.load(open(ipc_dir))
        self.tokenizer = Iupac_tokenizer()
        self.cls_token = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.eos_token = self.tokenizer.convert_tokens_to_ids(['[EOS]'])[0]


    def __getitem__(self, id_):
        text = self.json_dict[id_]
            
        output = [self.get_single_ipc(text)]
        
        return output


    def get_single_ipc(self,text,max_len=None):
        output = {}
        txt_tokens = self.tokenizer.tokenize(text)
        txt_tokens = self.tokenizer.convert_tokens_to_ids(txt_tokens)
        
        txt_tokens =self.get_padded_tokens(txt_tokens,max_len)
        output['iupac_tokens'] = txt_tokens
        return output    
    
    def get_padded_tokens(self,txt_tokens,max_len=None):
        
        max_len = self.max_len if  max_len is None else max_len
        txt_tokens = txt_tokens[:max_len]


        txt_tokens = [self.cls_token] + txt_tokens + [self.eos_token]  


        txt_tokens = torch.tensor(txt_tokens, dtype=torch.long)

        output = torch.zeros(max_len + 2, dtype=torch.long)
        output[:len(txt_tokens)] = txt_tokens
        return output    
    
    def detokenize(self, ids):
        return  self.tokenizer.convert_ids_to_tokens(ids)
    

sfs_vocab =  {'[PAD]': 0, '[CLS]': 1, '[MASK]': 2, '[EOS]': 3, '[unk]': 4, '[#Branch1]': 5, '[#Branch2]': 6, '[#C]': 7, '[#N+1]': 8, '[#N]': 9, '[-/Ring1]': 10, '[-/Ring2]': 11, '[-\\Ring1]': 12, '[/Br]': 13, '[/C@@H1]': 14, '[/C@@]': 15, '[/C@H1]': 16, '[/C@]': 17, '[/C]': 18, '[/Cl]': 19, '[/F]': 20, '[/N+1]': 21, '[/N-1]': 22, '[/NH1+1]': 23, '[/NH1-1]': 24, '[/NH1]': 25, '[/NH2+1]': 26, '[/N]': 27, '[/O+1]': 28, '[/O-1]': 29, '[/O]': 30, '[/S-1]': 31, '[/S@]': 32, '[/S]': 33, '[=Branch1]': 34, '[=Branch2]': 35, '[=C]': 36, '[=N+1]': 37, '[=N-1]': 38, '[=NH1+1]': 39, '[=NH2+1]': 40, '[=N]': 41, '[=O+1]': 42, '[=OH1+1]': 43, '[=O]': 44, '[=P@@]': 45, '[=P@]': 46, '[=PH2]': 47, '[=P]': 48, '[=Ring1]': 49, '[=Ring2]': 50, '[=S+1]': 51, '[=S@@]': 52, '[=S@]': 53, '[=SH1+1]': 54, '[=S]': 55, '[Br]': 56, '[Branch1]': 57, '[Branch2]': 58, '[C@@H1]': 59, '[C@@]': 60, '[C@H1]': 61, '[C@]': 62, '[CH1-1]': 63, '[CH2-1]': 64, '[C]': 65, '[Cl]': 66, '[F]': 67, '[I]': 68, '[N+1]': 69, '[N-1]': 70, '[NH1+1]': 71, '[NH1-1]': 72, '[NH1]': 73, '[NH2+1]': 74, '[NH3+1]': 75, '[N]': 76, '[O-1]': 77, '[O]': 78, '[P+1]': 79, '[P@@H1]': 80, '[P@@]': 81, '[P@]': 82, '[PH1+1]': 83, '[PH1]': 84, '[P]': 85, '[Ring1]': 86, '[Ring2]': 87, '[S+1]': 88, '[S-1]': 89, '[S@@+1]': 90, '[S@@]': 91, '[S@]': 92, '[S]': 93, '[\\Br]': 94, '[\\C@@H1]': 95, '[\\C@H1]': 96, '[\\C]': 97, '[\\Cl]': 98, '[\\F]': 99, '[\\I]': 100, '[\\N+1]': 101, '[\\N-1]': 102, '[\\NH1+1]': 103, '[\\NH1]': 104, '[\\NH2+1]': 105, '[\\N]': 106, '[\\O-1]': 107, '[\\O]': 108, '[\\S-1]': 109, '[\\S@]': 110, '[\\S]': 111}
class sfsMapper(object):
    def __init__(self,sfs_dir,max_len):
        self.max_len = max_len
        self.sfs_dir = sfs_dir
        self.training = True 
        self.json_dict = json.load(open(sfs_dir))
        self.tokenizer = Selfies_tokenizer()
        self.sfs_vocab = self.tokenizer.vocab
        self.cls_token = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.mask_token = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        self.eos_token = self.tokenizer.convert_tokens_to_ids(['[EOS]'])[0]


    def __getitem__(self, id_):
        selfies = self.json_dict[id_]
            
        output = [self.get_single_ipc(selfies)]
        
        return output


    def get_single_ipc(self,text,max_len=None):
        output = {}
        txt_tokens = self.tokenizer.tokenize(text)
        txt_tokens = self.tokenizer.convert_tokens_to_ids(txt_tokens)
        
        txt_tokens =self.get_padded_tokens(txt_tokens,max_len)

        maksed_tokens,label = self.mask_tokens(txt_tokens)
        
        output['selfies_tokens'] = txt_tokens
        output['masked_tokens'] = maksed_tokens
        output['label'] = label
        return output    
    
    def mask_tokens(self,txt_tokens):
        sfs_tokens_masked = []
        label = [0 for _ in range(len(txt_tokens))]
        for i in range(len(txt_tokens)):
            token = txt_tokens[i]
            from random import random,randint
            if token != 0:
                if random() < 0.15:
                    label[i] = token
                    rate = random()
                    if rate < 0.8:
                        sfs_tokens_masked.append(self.mask_token)
                    elif rate < 0.9:
                        sfs_tokens_masked.append(randint(5,len(self.sfs_vocab)-1))
                    else:
                        sfs_tokens_masked.append(token)
                else:
                    sfs_tokens_masked.append(token)

            else:
                sfs_tokens_masked.append(token)
        return torch.tensor(sfs_tokens_masked),torch.tensor(label)
    
    def get_padded_tokens(self,txt_tokens,max_len=None):
        
        max_len = self.max_len if  max_len is None else max_len
        txt_tokens = txt_tokens[:max_len]


        txt_tokens = [self.cls_token] + txt_tokens + [self.eos_token]  


        txt_tokens = torch.tensor(txt_tokens, dtype=torch.long)

        output = torch.zeros(max_len + 2, dtype=torch.long)
        output[:len(txt_tokens)] = txt_tokens
        return output    
    
    def detokenize(self, ids):
        return  self.tokenizer.convert_ids_to_tokens(ids)
    


class mlgMapper(object):
    def __init__(self,smi_dir):
        self.mlg_dir = smi_dir
        self.training = True 
        self.json_dict = json.load(open(smi_dir))
        


    def __getitem__(self, id_):
        smi = self.json_dict[id_]
            
        output = self.get_single_mg(smi)
        
        return output


    def get_single_mg(self,smi,max_len=None):
        rdkit_mol = AllChem.MolFromSmiles(smi)
        if rdkit_mol != None:  # ignore invalid mol objects
            data = mol_to_graph_data_obj_simple(rdkit_mol)

        output = data

        return output    
    

    

from torch.utils.data import Dataset

class SIGDataset(Dataset):
    def __init__(self, ids_path, ipc_mapper, sfs_mapper, mlg_mapper, training):
        self.ipc_mapper = ipc_mapper
        self.sfs_mapper = sfs_mapper
        self.mlg_mapper = mlg_mapper
        self.ids = json.load(open(ids_path)) 
        self.idx = list(range(len(self.ids)))
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = list(self.ids.keys())[i]
        smi = list(self.ids.values())[i]
        ipc_tokens = None 
        sfs_tokens = None
        mlg = None


        if AllChem.MolFromSmiles(smi):
            try:
                if self.ipc_mapper is not None:
                    ipc_tokens = self.ipc_mapper[id_]


                
                if self.sfs_mapper is not None:
                    sfs_tokens = self.sfs_mapper[id_]




                if self.mlg_mapper is not None:
                    mlg = self.mlg_mapper[id_]
            except:
                ipc_tokens = None 
                sfs_tokens = None
                mlg = None   



        return id_, sfs_tokens,ipc_tokens,mlg
    

from toolz.sandbox import unzip
from torch.utils.data import DataLoader

def SIG_collate(inputs):
    

    (id_, sfs_tokens,ipc_tokens,mlg) = map(list, unzip(inputs))


    if sfs_tokens[0] is not None:
        sfs_tokens = [ j  for i in sfs_tokens for j in i]
        sfs_tokens_collate = {}
        for k in sfs_tokens[0].keys():  
            sfs_tokens_collate[k] = torch.stack([i[k] for i in sfs_tokens]) 
    else:
        sfs_tokens_collate = None 
    

    if ipc_tokens[0] is not None:
        ipc_tokens = [ j  for i in ipc_tokens for j in i]
        ipc_tokens_collate = {}
        for k in ipc_tokens[0].keys():  #### bert tokens and clip tokens
            ipc_tokens_collate[k] = torch.stack([i[k] for i in ipc_tokens]) 
    else:
        ipc_tokens_collate = None 

    if mlg[0] is not None:        
        from torch_geometric.data import Batch
        mlg_ = Batch.from_data_list(mlg)
    else:
        mlg_ = None

        

    batch =   {'ids': id_,
             'sfs_tokens': sfs_tokens_collate,
             'ipc_tokens': ipc_tokens_collate,
             'mlg': mlg_}
    
    return batch



class FinetuneDataset(Dataset):
    def __init__(self, ids_path, smi_path ,ipc_mapper, sfs_mapper, mlg_mapper, testing):
        self.ipc_mapper = ipc_mapper
        self.sfs_mapper = sfs_mapper
        self.mlg_mapper = mlg_mapper
        self.smiles = json.load(open(smi_path))
        self.ids = json.load(open(ids_path)) 
        self.idx = list(range(len(self.ids)))
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = list(self.ids.keys())[i]
        label = list(self.ids.values())[i]
        smi = list(self.smiles.values())[i]
        ipc_tokens = None 
        sfs_tokens = None
        mlg = None


   
        if self.ipc_mapper is not None:
            ipc_tokens = self.ipc_mapper[id_]


        
        if self.sfs_mapper is not None:
            sfs_tokens = self.sfs_mapper[id_]




        if self.mlg_mapper is not None:
            mlg = self.mlg_mapper[id_]



        return id_,smi, sfs_tokens,ipc_tokens,mlg,label
    
def Finetune_collate_C(inputs):
    (id_,smi,sfs_tokens,ipc_tokens,mlg,label) = map(list, unzip(inputs))

    label = torch.tensor(label, dtype=torch.long)
    if sfs_tokens[0] is not None:
        sfs_tokens = [ j  for i in sfs_tokens for j in i]
        sfs_tokens_collate = {}
        for k in sfs_tokens[0].keys():  
            sfs_tokens_collate[k] = torch.stack([i[k] for i in sfs_tokens]) 
    else:
        sfs_tokens_collate = None 
    

    if ipc_tokens[0] is not None:
        ipc_tokens = [ j  for i in ipc_tokens for j in i]
        ipc_tokens_collate = {}
        for k in ipc_tokens[0].keys():  #### bert tokens and clip tokens
            ipc_tokens_collate[k] = torch.stack([i[k] for i in ipc_tokens]) 
    else:
        ipc_tokens_collate = None 

    if mlg[0] is not None:        
        from torch_geometric.data import Batch
        mlg_ = Batch.from_data_list(mlg)

        

    batch =   {'ids': id_,
             'sfs_tokens': sfs_tokens_collate,
             'ipc_tokens': ipc_tokens_collate,
             'mlg': mlg_,
             'label': label}
    
    return batch

def Finetune_collate_R(inputs):
    (id_,smi,sfs_tokens,ipc_tokens,mlg,label) = map(list, unzip(inputs))

    label = torch.tensor(label, dtype=torch.float32)
    if sfs_tokens[0] is not None:
        sfs_tokens = [ j  for i in sfs_tokens for j in i]
        sfs_tokens_collate = {}
        for k in sfs_tokens[0].keys():  
            sfs_tokens_collate[k] = torch.stack([i[k] for i in sfs_tokens]) 
    else:
        sfs_tokens_collate = None 
    

    if ipc_tokens[0] is not None:
        ipc_tokens = [ j  for i in ipc_tokens for j in i]
        ipc_tokens_collate = {}
        for k in ipc_tokens[0].keys():  #### bert tokens and clip tokens
            ipc_tokens_collate[k] = torch.stack([i[k] for i in ipc_tokens]) 
    else:
        ipc_tokens_collate = None 

    if mlg[0] is not None:        
        from torch_geometric.data import Batch
        mlg_ = Batch.from_data_list(mlg)

        

    batch =   {'ids': id_,
             'sfs_tokens': sfs_tokens_collate,
             'ipc_tokens': ipc_tokens_collate,
             'mlg': mlg_,
             'label': label}
    
    return batch