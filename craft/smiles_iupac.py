import pandas as pd
from tqdm import tqdm
import pubchempy




def get_iupac_name(smiles):
    smiles_valid = []
    names = []
    for i in tqdm(range(len(smiles))):
        smi = smiles[i]
        compounds = pubchempy.get_compounds(smi, namespace='smiles')
        match = compounds[0]
        name = match.iupac_name
        if name != None:
            smiles_valid.append(smi)
            names.append(name)
    return smiles_valid, names


if __name__ == '__main__':
    df = pd.read_csv('./data/test.txt')

    smiles = df['SMILES']
    smiles = smiles.tolist()
    smiles_valid, names = get_iupac_name(smiles)
    df = pd.DataFrame({'SMILES': smiles_valid, 'IUPAC': names})
    df.to_csv('./data/smiles_iupac_test.csv', index=False)