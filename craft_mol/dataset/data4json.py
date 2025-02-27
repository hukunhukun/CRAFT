import json
import torch
from torch.utils.data import Dataset
from rdkit.Chem import AllChem
from torch_geometric.data import Batch
import selfies as sf
from .tokenizer import Iupac_tokenizer, Selfies_tokenizer
from .atom_rep import mol_to_graph_data_obj_simple


class JsonDataset(Dataset):
    def __init__(self, 
                 json_path:str=None, 
                 selfies_max_len:int=130, 
                 iupac_max_len:int=130,
                 CID_name:str='CID',
                 smiles_name:str='input',
                 iupac_name:str='iupac name',
                 selfies_name:str=None):
        """
        Initialize the dataset.

        Parameters:
            json_path (str): JSON file path.
            selfies_max_len (int): Maximum length for SELFIES tokens.
            iupac_max_len (int): Maximum length for IUPAC tokens.
        """
        self.selfies_max_len = selfies_max_len
        self.iupac_max_len = iupac_max_len
        self.CID_name = CID_name
        self.smiles_name = smiles_name
        self.iupac_name = iupac_name
        self.selfies_name = selfies_name


        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # Initialize tokenizers
        self.iupac_tokenizer = Iupac_tokenizer()
        self.selfies_tokenizer = Selfies_tokenizer()

        # Define CLS and EOS tokens
        self.iupac_cls_token = self.iupac_tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.iupac_eos_token = self.iupac_tokenizer.convert_tokens_to_ids(['[EOS]'])[0]
        self.selfies_cls_token = self.selfies_tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.selfies_eos_token = self.selfies_tokenizer.convert_tokens_to_ids(['[EOS]'])[0]

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a data sample by index.

        Returns:
            dict: Processed data sample.
        """
        sample = self.data[idx]
        smiles = sample[self.smiles_name]
        if self.selfies_name is not None:
            selfies = sample[self.selfies_name]
        else:
            selfies = None
            try:
                selfies = sf.encoder(smiles)
            except:
                pass
        iupac_name = sample.get(self.iupac_name, None)
        sample_id = sample[self.CID_name]

        # Initialize the output dictionary
        output = {
            'id': sample_id,
            'selfies_tokens': None,
            'iupac_tokens': None,
            'mol_graph': None
        }

        # Process IUPAC name if available
        if iupac_name:
            output['iupac_tokens'] = self._get_iupac_tokens(iupac_name)

        # Convert SMILES to SELFIES


        # Process SELFIES if available
        if selfies:
            output['selfies_tokens'] = self._get_selfies_tokens(selfies)

        # Process molecular graph from SMILES
        output['mol_graph'] = self._get_mol_graph(smiles)

        return output

    def _get_iupac_tokens(self, text):
        """Convert IUPAC name to tokenized format."""
        try:
            tokens = self.iupac_tokenizer.tokenize(text)
            tokens = self.iupac_tokenizer.convert_tokens_to_ids(tokens)
            tokens = self._get_padded_tokens(tokens, self.iupac_max_len, self.iupac_cls_token, self.iupac_eos_token)
            return tokens
        except:
            return None

    def _get_selfies_tokens(self, text):
        """Convert SELFIES to tokenized format."""
        try:
            tokens = self.selfies_tokenizer.tokenize(text)
            tokens = self.selfies_tokenizer.convert_tokens_to_ids(tokens)
            tokens = self._get_padded_tokens(tokens, self.selfies_max_len, self.selfies_cls_token, self.selfies_eos_token)
            return tokens
        except:
            return None

    def _get_mol_graph(self, smiles):
        """Convert SMILES to a molecular graph."""
        mol = AllChem.MolFromSmiles(smiles)
        return mol_to_graph_data_obj_simple(mol) if mol is not None else None

    def _get_padded_tokens(self, tokens, max_len, cls_token, eos_token):
        """Pad the token sequence to the specified length."""
        tokens = tokens[:max_len]
        tokens = [cls_token] + tokens + [eos_token]
        tokens = torch.tensor(tokens, dtype=torch.long)

        padded_tokens = torch.zeros(max_len + 2, dtype=torch.long)
        padded_tokens[:len(tokens)] = tokens
        return padded_tokens


def JsonDataset_collate(inputs):
    """
    Collate function for JsonDataset.

    Args:
        inputs (list): List of processed samples.

    Returns:
        dict: Batch data.
    """
    id_list = []
    selfies_tokens_list = []
    iupac_tokens_list = []
    mol_graph_list = []

    for sample in inputs:
        if sample['selfies_tokens'] is None or sample['iupac_tokens'] is None or sample['mol_graph'] is None:
            continue  # Skip invalid samples

        id_list.append(sample['id'])
        selfies_tokens_list.append(sample['selfies_tokens'])
        iupac_tokens_list.append(sample['iupac_tokens'])
        mol_graph_list.append(sample['mol_graph'])

    # Collate token sequences
    selfies_tokens_collate = torch.stack(selfies_tokens_list) if selfies_tokens_list else None
    iupac_tokens_collate = torch.stack(iupac_tokens_list) if iupac_tokens_list else None
    mol_graph_collate = Batch.from_data_list(mol_graph_list) if mol_graph_list else None

    batch = {
        'ids': id_list,  # Sample IDs
        'selfies_tokens': selfies_tokens_collate,  # SELFIES token batch
        'iupac_tokens': iupac_tokens_collate,  # IUPAC token batch
        'mol_graph': mol_graph_collate  # Molecular graph batch
    }

    return batch
