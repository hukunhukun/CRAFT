from rdkit import Chem
import pandas as pd
import matplotlib.pyplot as plt
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
import matplotlib.cm as cm
from PIL import Image
import io
import argparse


def add_atom_index(mol,pros):

    for atom in mol.GetAtoms():
        atom.SetProp('atomNote', f'{pros[atom.GetIdx()]:.1f}')

    return mol


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mol_dir', type=str,default='downstream/interact/')
    parser.add_argument('--model_name', type=str,default="esol")
    parser.add_argument('--mol_name_list', type=str, nargs='+',default=['0','1'])
    args = parser.parse_args()

    mol_dir = args.mol_dir + args.model_name + '/'
    for index in args.mol_name_list:
        smi_file = mol_dir + index  + '/layer2_smi.txt'
        smi = open(smi_file, 'r').readlines()[0].strip()

        # 创建分子对象
        mol = Chem.MolFromSmiles(smi)
        num_atoms = mol.GetNumAtoms()
        # mol = add_atom_index(mol)
        print(f"Number of atoms in the molecule: {num_atoms}")
        # 输出每一个原子以及化学键






        # 获取权重大小，对于8个头，每个头都有一个权重图
        for head in range(8):
            csv_file = mol_dir + index + f'/layer2_head{head}.csv'
            df = pd.read_csv(csv_file)
            # 第一行第一列为index名称，对每一列求和得到数组
            weights = df.sum(axis=0).values[1:].tolist()


            # 创建自定义权重数组，与原子数量相同
            

            # 将weights归一化到0-1
            min_val = min(weights)
            max_val = max(weights)
            custom_weights = [(x - min_val) / (max_val - min_val) for x in weights]

            # 创建highlightAtomColor字典，颜色根据其weights决定
            dict_atom_color = {}
            dict_atom_size = {}
            dict_bond_color = {}
            atom_idx = []
            cmap = cm.get_cmap('Blues')
            for i in range(len(custom_weights)):
                color = cmap(custom_weights[i]*0.8)
                color_ = cmap(custom_weights[(i+1)%len(custom_weights)])
                dict_atom_color[i] = (color[0],color[1],color[2])
                dict_atom_size[i] = 0.3
                atom_idx.append(i)
                # bond 为平均值


            bonds = mol.GetBonds()
            bond_list = []
            for bond in bonds:
                bond_idx = bond.GetIdx()
                bond_list.append(bond_idx)
                begin_atom_idx = bond.GetBeginAtomIdx()
                end_atom_idx = bond.GetEndAtomIdx()
                # dict_bond_color是平均值
                dict_bond_color[bond_idx] = ((dict_atom_color[begin_atom_idx][0]+dict_atom_color[end_atom_idx][0])/2,
                                            (dict_atom_color[begin_atom_idx][1]+dict_atom_color[end_atom_idx][1])/2,
                                            (dict_atom_color[begin_atom_idx][2]+dict_atom_color[end_atom_idx][2])/2)
                


            d2d = rdMolDraw2D.MolDraw2DSVG(450,450)
            d2d.drawOptions().useBWAtomPalette()
            d2d.drawOptions().bondLineWidth= 4
            mol = add_atom_index(mol,custom_weights)
            d2d.DrawMolecule(mol,highlightAtoms=atom_idx,highlightAtomColors=dict_atom_color,highlightAtomRadii=dict_atom_size,
                            highlightBonds=bond_list,
                            highlightBondColors=dict_bond_color)
            d2d.FinishDrawing()
            f = SVG(d2d.GetDrawingText())

            #保存在文件中
            with open(mol_dir + index + f'/layer2_head{head}.svg','w') as svg_file:
                svg_file.write(d2d.GetDrawingText()) 