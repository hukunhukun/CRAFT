import torch
import pandas as pd
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch.utils.data import Dataset
import selfies as sf
from .tokenizer import Iupac_tokenizer, Selfies_tokenizer
from .atom_rep import mol_to_graph_data_obj_simple


class Data4CSV(Dataset):
    def __init__(self, 
                 csv_path:str=None, 
                 selfies_max_len:int=130, 
                 iupac_max_len:int=130, 
                 CID_name:str='CID',
                 smiles_name:str=None,
                 iupac_name:str='iupac name',
                 selfies_name:str=None
                 ):
        """
        Initialize the dataset.
        
        Parameters:
            csv_path (str): CSV path
            selfies_max_len (int): SELFIES max length
            iupac_max_len (int): IUPAC max length
        """
        self.selfies_max_len = selfies_max_len
        self.iupac_max_len = iupac_max_len
        self.CID_name = CID_name
        self.smiles_name = smiles_name
        self.iupac_name = iupac_name
        self.selfies_name = selfies_name

        self.data = pd.read_csv(csv_path)
        

        self.iupac_tokenizer = Iupac_tokenizer()
        self.selfies_tokenizer = Selfies_tokenizer()
        
 
        self.iupac_cls_token = self.iupac_tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.iupac_eos_token = self.iupac_tokenizer.convert_tokens_to_ids(['[EOS]'])[0]
        
        self.selfies_cls_token = self.selfies_tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.selfies_eos_token = self.selfies_tokenizer.convert_tokens_to_ids(['[EOS]'])[0]

    def __len__(self):
        """return the length of the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the data of the given index.
        
        return:
            dict: The data of the given index.

        """

        row = self.data.iloc[idx]
        cid = int(row[self.CID_name])
        smiles = row[self.smiles_name]
        if self.selfies_name is not None:
            selfies = row[self.selfies_name]
        else:
            selfies = None
            try:
                selfies = sf.encoder(smiles)
            except:
                pass
        iupac_name = row[self.iupac_name]
        
        # Initialize the output
        output = {
            'cid': cid,
            'selfies_tokens': None,
            'iupac_tokens': None,
            'mol_graph': None
        }
        
        # handle IUPAC
        if pd.notna(iupac_name):
            output['iupac_tokens'] = self._get_iupac_tokens(iupac_name)
        
        # handle SELFIES
        if pd.notna(selfies):
            output['selfies_tokens'] = self._get_selfies_tokens(selfies)
        
        # handle SMILES
        if pd.notna(smiles):
            output['mol_graph'] = self._get_mol_graph(smiles)
        
        return output

    def _get_iupac_tokens(self, text):
        """convert IUPAC name to tokens."""
        tokens = None
        try:
            tokens = self.iupac_tokenizer.tokenize(text)
            tokens = self.iupac_tokenizer.convert_tokens_to_ids(tokens)
            tokens = self._get_padded_tokens(tokens, self.iupac_max_len, self.iupac_cls_token, self.iupac_eos_token)
        except:
            pass
        return tokens

    def _get_selfies_tokens(self, text):
        """convert SELFIES to tokens."""
        tokens = None
        try:
            tokens = self.selfies_tokenizer.tokenize(text)
            tokens = self.selfies_tokenizer.convert_tokens_to_ids(tokens)
            tokens = self._get_padded_tokens(tokens, self.selfies_max_len, self.selfies_cls_token, self.selfies_eos_token)
        except:
            pass
        return tokens

    def _get_mol_graph(self, smiles):
        """convert SMILES to graph data object."""
        mol = AllChem.MolFromSmiles(smiles)
        if mol is not None:
            return mol_to_graph_data_obj_simple(mol)
        return None

    def _get_padded_tokens(self, tokens, max_len, cls_token, eos_token):
        """padding the tokens to the given length."""
        tokens = tokens[:max_len]
        tokens = [cls_token] + tokens + [eos_token]
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        """padding the tokens to the given length."""
        padded_tokens = torch.zeros(max_len + 2, dtype=torch.long)
        padded_tokens[:len(tokens)] = tokens
        return padded_tokens


from toolz.sandbox import unzip
from torch_geometric.data import Batch

def Data4CSV_collate(inputs):
    """
    Collate function for the Data4CSV dataset.
    """
    cid_list = []
    selfies_tokens_list = []
    iupac_tokens_list = []
    mol_graph_list = []
    
    for sample in inputs:

        if sample['selfies_tokens'] is None or sample['iupac_tokens'] is None or sample['mol_graph'] is None:
            continue  # 跳过当前样本
        
        cid_list.append(sample['cid'])
        selfies_tokens_list.append(sample['selfies_tokens'])
        iupac_tokens_list.append(sample['iupac_tokens'])
        mol_graph_list.append(sample['mol_graph'])
    

    if len(selfies_tokens_list) > 0:
        selfies_tokens_collate = torch.stack(selfies_tokens_list)  
    else:
        selfies_tokens_collate = None
    
    if len(iupac_tokens_list) > 0:
        iupac_tokens_collate = torch.stack(iupac_tokens_list)  
    else:
        iupac_tokens_collate = None
    
    if len(mol_graph_list) > 0:
        mol_graph_collate = Batch.from_data_list(mol_graph_list)  
    else:
        mol_graph_collate = None
    
    

    batch = {
        'cids': cid_list,  # CID list
        'selfies_tokens': selfies_tokens_collate,  # SELFIES tokens
        'iupac_tokens': iupac_tokens_collate,  # IUPAC tokens
        'mol_graph': mol_graph_collate  # molecule graph 
    }
    
    return batch