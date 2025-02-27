from .dataset.data4csv import Data4CSV
from .dataset.data4csv import Data4CSV_collate
from .dataset.data4json import JsonDataset,JsonDataset_collate
from torch.utils.data import DataLoader
import torch
from .model.TMMF import TMMF
import ruamel.yaml as yaml
import os
from tqdm import tqdm
from dataclasses import field
from rdkit.Chem import AllChem



def get_data_loader_csv(data_path, 
                        batch_size, 
                        shuffle=False,
                        CID_name:str='CID',
                        smiles_name:str='smiles',
                        iupac_name:str='iupac name',
                        selfies_name:str=None):
    Data4CSV_dataset = Data4CSV(data_path, selfies_max_len=130, iupac_max_len=130, CID_name=CID_name, smiles_name=smiles_name, iupac_name=iupac_name, selfies_name=selfies_name)
    dataloader = DataLoader(
        Data4CSV_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=Data4CSV_collate,
        drop_last=False  
    )

    return dataloader

def get_data_loader_json(data_path, 
                        batch_size, 
                        shuffle=False,
                        CID_name:str='id',
                        smiles_name:str='input',
                        iupac_name:str='iupac name',
                        selfies_name:str=None):
    Json_dataset = JsonDataset(data_path, 
                               selfies_max_len=130, 
                               iupac_max_len=130,
                               CID_name=CID_name,
                               smiles_name=smiles_name,
                               iupac_name=iupac_name,
                               selfies_name=selfies_name)
    dataloader = DataLoader(
        Json_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=JsonDataset_collate,
        drop_last=False  
    )

    return dataloader

def get_model(model_path: str = field(default = None), 
              device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              config_path:str = './config/pre_train.yaml'):


    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

    print("Creating model...")
    model = TMMF(config)
    model = model.to(device)

    # path = './pre_train/1m_checkpoint_09.pth'
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model

class BatchFeature:
    def __init__(self, model_path: str = None, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), config_path:str = './config/pre_train.yaml'):
        self.model = get_model(model_path, device, config_path)
        self.device = device
    
    def get_feature_single(self, smiles:str, iupac_name:str, selfies:str=None, mlg_ids=None):
        from .dataset.tokenizer import Iupac_tokenizer,Selfies_tokenizer
        from .dataset.atom_rep import mol_to_graph_data_obj_simple
        ipc_tokenizer = Iupac_tokenizer()
        ipc_ids = ipc_tokenizer.convert_tokens_to_ids(ipc_tokenizer.tokenize(iupac_name))
        ipc_ids = [ipc_tokenizer.convert_tokens_to_ids(['[CLS]'])[0]] + ipc_ids + [ipc_tokenizer.convert_tokens_to_ids(['[EOS]'])[0]]
        if selfies is None:
            import selfies as sf
            selfies = sf.encoder(smiles)
        sfs_tokenizer = Selfies_tokenizer()
        sfs_ids = sfs_tokenizer.convert_tokens_to_ids(sfs_tokenizer.tokenize(selfies))
        sfs_ids = [sfs_tokenizer.convert_tokens_to_ids(['[CLS]'])[0]] + sfs_ids + [sfs_tokenizer.convert_tokens_to_ids(['[EOS]'])[0]]

        rdkit_mol = AllChem.MolFromSmiles(smiles)
        if rdkit_mol != None:  # ignore invalid mol objects
            mlg_ids = mol_to_graph_data_obj_simple(rdkit_mol)
        
        ipc_ids = [ipc_ids]
        sfs_ids = [sfs_ids]

        from torch_geometric.data import Batch
        mlg_ids = Batch.from_data_list([mlg_ids]).to(self.device)
        ipc_ids = torch.tensor(ipc_ids, dtype=torch.long).to(self.device)
        sfs_ids = torch.tensor(sfs_ids, dtype=torch.long).to(self.device)

        out = self.model.get_fusion_feature(mlg_ids,ipc_ids,sfs_ids)

        return out

    def get_feature(self, data_path:str=None, 
                    batch_size:int=32,
                    output_dir:str=None,
                    CID_name:str='CID',
                    smiles_name:str='smiles',
                    iupac_name:str='iupac name',
                    selfies_name:str=None):
        if data_path.endswith('.csv'):
            dataloader = get_data_loader_csv(data_path, 
                                             batch_size, 
                                             shuffle=False,
                                             CID_name=CID_name,
                                             smiles_name=smiles_name,
                                             iupac_name=iupac_name,
                                             selfies_name=selfies_name)
        elif data_path.endswith('.json'):
            dataloader = get_data_loader_json(data_path, 
                                              batch_size, 
                                              shuffle=False,                                             
                                              CID_name=CID_name,
                                              smiles_name=smiles_name,
                                              iupac_name=iupac_name,
                                              selfies_name=selfies_name)
        else:
            raise ValueError('Data path should be a csv or json file')
        os.makedirs(output_dir, exist_ok=True)


        for index,batch in enumerate(tqdm(dataloader, desc="Processing batches", unit="batch")):
            if data_path.endswith('.csv'):
                cids = batch['cids']
            elif data_path.endswith('.json'):
                cids = batch['ids']
            selfies_tokens = batch['selfies_tokens'].to(self.device)  
            iupac_tokens = batch['iupac_tokens'].to(self.device) 
            mol_graph = batch['mol_graph'].to(self.device) 
            with torch.no_grad():
                out = self.model.get_fusion_feature(mol_graph, iupac_tokens, selfies_tokens)

            # 将输出保存为文件
            for i, cid in enumerate(cids):
                # 提取当前样本的输出
                output_tensor = out[i].cpu()  # 将张量移动到 CPU
                # 保存为文件
                output_path = os.path.join(output_dir, f'{cid}.pt')
                torch.save(output_tensor, output_path)
            



if __name__ == '__main__':
    model_path = './pre_train/1m_checkpoint_09.pth'
    data_path = '/home/hukun/code/trimodal_code/craft_mol/data/Mol-Instructions/data/Molecule-oriented_Instructions/molecular_description_generation/test_data.json'
    output_dir = '/home/hukun/code/trimodal_code/craft_mol/data/Mol-Instructions/data/Molecule-oriented_Instructions/molecular_description_generation/test_output_features'
    batch_size = 32

    batch_feature = BatchFeature(model_path)
    batch_feature.get_feature(data_path, batch_size, output_dir)

