from .data import SIGDataset, SIG_collate, ipcMapper, sfsMapper, mlgMapper, FinetuneDataset, Finetune_collate_C, Finetune_collate_R
from .data4csv import Data4CSV, Data4CSV_collate
from .data4json import JsonDataset,JsonDataset_collate
from torch.utils.data import DataLoader
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from torch.utils.data import Subset

def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    print("About to generate scaffolds")
    for ind, _ in enumerate(dataset):
        smiles = dataset[ind][1]
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: list[int] = []
    valid_inds: list[int] = []
    test_inds: list[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


def create_dataset(ids_path,ipc_path,sfs_path,mlg_path,frac,mode='pretrain'):
    ipc_mapper = ipcMapper(ipc_path,max_len=130)
    sfs_mapper = sfsMapper(sfs_path,max_len=75)
    mlg_mapper = mlgMapper(mlg_path)
    if mode == 'pretrain':
        dataset = SIGDataset(ids_path,ipc_mapper, sfs_mapper, mlg_mapper,True)
        length=len(dataset)
        train_size,test_size=(length-int(frac*length)),int(frac*length)
        train_dataset,test_dataset=torch.utils.data.random_split(dataset,[train_size,test_size])

        return train_dataset,test_dataset

    elif mode == 'finetune':
        dataset = FinetuneDataset(ids_path,mlg_path,ipc_mapper, sfs_mapper, mlg_mapper,True)
        length=len(dataset)
        train_size,val_size,test_size=(length-2*int(frac*length)),int(frac*length),int(frac*length)
        train_dataset,val_dataset,test_dataset=torch.utils.data.random_split(dataset,[train_size,val_size,test_size])        

        # train_idx, valid_idx, test_idx = scaffold_split(dataset, frac, frac)
        # train_dataset = Subset(dataset, train_idx)
        # val_dataset = Subset(dataset, valid_idx)
        # test_dataset = Subset(dataset, test_idx)
        # return train_dataset,val_dataset,test_dataset
    
        return train_dataset,val_dataset,test_dataset

def create_dataloader(dataset,batch_size,mode='pretrain',task_type='classification'):
    drop = False
    if mode == 'pretrain':
        collate = SIG_collate
        drop = True
    elif mode == 'finetune' and task_type == 'classification':
        collate = Finetune_collate_C
    elif mode == 'finetune' and task_type == 'regression':
        collate = Finetune_collate_R
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn= collate,
        drop_last=drop
    )
    return data_loader



