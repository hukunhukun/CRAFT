# smiles转为mol_id以及selfies 和 iupac表示

import pandas as pd
import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem


def check_smiles_validity(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except:
        return False

def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively
    :param mol:
    :return:
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list

def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one
    :param mol_list:
    :return:
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]

def create_standardized_mol_id(smiles):
    """
    :param smiles:
    :return: inchi
    """
    if check_smiles_validity(smiles):
        # remove stereochemistry
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles),
                                     isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)
        if mol != None: # to catch weird issue with O=C1O[al]2oc(=O)c3ccc(cn3)c3ccccc3c3cccc(c3)c3ccccc3c3cc(C(F)(F)F)c(cc3o2)-c2ccccc2-c2cccc(c2)-c2ccccc2-c2cccnc21
            if '.' in smiles: # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
        else:
            return
    else:
        return

import json

df = pd.read_csv('data/smiles_iupac_30k.csv')
smiles = df['smiles'].tolist()
iupac_names = df['iupac'].tolist()
id_data = []
ipc_data = {}
sfs_data = {}
smi_data = {}


for i in range(len(smiles)):
    try:
        id_ = str(i)
        sf_ = sf.encoder(smiles[i])
        iupac_ = iupac_names[i]
        if id_ is not None and sf_ is not None and iupac_ is not None:
            id_data.append(id_)
            ipc_data[id_] = iupac_names[i]
            sfs_data[id_] = sf.encoder(smiles[i])
            smi_data[id_] = smiles[i]
    except:
        continue


def save_json(save_path,data):
    # assert save_path.split('.')[-1] == 'json'
    f = open(save_path, 'w')
    b = json.dumps(data, indent=2)
    f.write(b)
    f.close()
 


save_json('./data/retrieval/ipc_mapper_train.json',ipc_data)
save_json('./data/retrieval/mol_id_train.json',id_data)
save_json('./data/sfs_mapper_train.json',sfs_data)
save_json('./data/smi_mapper_train.json',smi_data)



import pandas as pd
import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem
from dataset.tokenizer import Iupac_tokenizer
import json

iupac_tokenizer = Iupac_tokenizer()
df = pd.read_csv('data/smiles_iupac_30k.csv')
smiles = df['smiles'].tolist()
iupac_names = df['iupac'].tolist()
id_data = []
ipc_data = {}
sfs_data = {}
smi_data = {}


for i in range(len(smiles)):
    try:
        id_ = str(i)
        sf_ = sf.encoder(smiles[i])
        iupac_ = iupac_names[i]
        t = iupac_tokenizer.tokenize(iupac_)
        if id_ is not None and sf_ is not None and iupac_ is not None:
            id_data.append(id_)
            ipc_data[id_] = iupac_names[i]
            sfs_data[id_] = sf.encoder(smiles[i])
            smi_data[id_] = smiles[i]
    except:
        continue


def save_json(save_path,data):
    # assert save_path.split('.')[-1] == 'json'
    f = open(save_path, 'w')
    b = json.dumps(data, indent=2)
    f.write(b)
    f.close()
 


save_json('./data/downstream_tasks/retrieval/ipc_mapper_retrieval.json',ipc_data)
save_json('./data/downstream_tasks/retrieval/mol_id_retrieval.json',id_data)
save_json( './data/downstream_tasks/retrieval/sfs_mapper_retrieval.json',sfs_data)
save_json('./data/downstream_tasks/retrieval/smi_mapper_retrieval.json',smi_data)