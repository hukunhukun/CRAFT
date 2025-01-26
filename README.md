# CRAFT
CRAFT: Consistent Representational Fusion of Three Molecular Modalities

## 1. Environment Setup
### 1.1. Install Conda
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
chmod +x Anaconda3-2023.09-0-Linux-x86_64.sh
source ~/.bashrc
```

### 1.2. Create Conda Environment
```bash
conda create -n craft python=3.10.13
conda activate craft
conda install -c conda-forge rdkit
```

### 1.3. Install Dependencies

clone the repository
```bash
git clone https://github.com/hukunhukun/CRAFT.git
cd CRAFT
```

```bash
pip install -r requirements.txt 
```
or 

```bash
pip install -e .
```

### 1.4. Import the CRAFT package

```bash 
pip install craft_mol
```

## 2. Model weights
Download the model weights from the huggingface and place them in the `weights` directory.
```bash
git lfs install
git clone https://huggingface.co/kunkunhu/craft_mol
```




## 3. Usage
Get molecular representations from three molecular modalities (SELFIES, Molecular Graph and IUPAC names)

```python
from craft_mol.batch_feature import BatchFeature

model_path = './craft_mol/1m_checkpoint_09.pth'
config_path = './craft_mol/inference.yaml'
batch_feature = BatchFeature(model_path=model_path,config_path=config_path)

smiles = "CC(C)CC1=CC=C(C=C1)C(=O)O"
iupac_name = "4-(2-methylpropyl)benzoic acid"

mol_feature = batch_feature.get_feature_single(smiles=smiles, iupac_name=iupac_name)
print(mol_feature.shape)
```
