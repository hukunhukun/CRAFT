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
```bash
pip install -r requirements.txt 
```

or 
clone the repository
```bash
git clone https://github.com/hukunhukun/CRAFT.git
cd CRAFT
```


```bash
pip install -e .
```