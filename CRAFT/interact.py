import torch
from torch import nn
from model.TMMF import TMMF
from dataset import create_dataset,create_dataloader
import ruamel.yaml as yaml
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw
from dataset.tokenizer import Iupac_tokenizer,Selfies_tokenizer
import selfies as sf
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt


class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = Config(value)
        return value

class BertForInteract(nn.Module):

    def __init__(self,selfies_encoder,iupac_encoder,mol_encoder,fusion_encoder):
        super().__init__()
        self.selfies_encoder = selfies_encoder
        self.iupac_encoder = iupac_encoder
        self.mol_encoder = mol_encoder
        self.fusion_encoder = fusion_encoder
        for p in self.parameters():
            p.requires_grad=False

    def forward(self,selfies_ids=None,iupac_ids=None,mol_ids=None,labels=None,save_attention=True):
        selfies_embeds,selfies_mask = self.selfies_encoder(input_ids=selfies_ids,mode='selfies')
        iupac_embeds,iupac_mask = self.iupac_encoder(input_ids=iupac_ids,mode='iupac')
        mol_pool,mol_embeds,mol_mask = self.mol_encoder(mol_ids)



        iupac_mol_embeds = torch.cat([iupac_embeds,mol_embeds],dim=1)
        iupac_mol_mask = torch.cat([iupac_mask,mol_mask],dim=1)
        output = self.fusion_encoder(selfies_embeds=selfies_embeds,
                                     iupac_mol_embeds=iupac_mol_embeds,
                                     selfies_mask=selfies_mask,
                                     iupac_mol_mask=iupac_mol_mask,
                                     mode='fusion',save_attention=save_attention)

        return output
    
def find_satisfy_iupac(dataset,iupac_token,iupac_tokenizer,sample_num=10):
    ids_sat = []        # save ids satisfied
    position = [] # record iupac_token position
    for i in range(len(dataset)):
        token_position = []
        iupac = dataset[i][2][0]['iupac_tokens']
        tokens = iupac_tokenizer.convert_ids_to_tokens(iupac.tolist())
        length = len(iupac) - iupac.tolist().count(0)
        if length < 40:
        # iupac token是一个list
            for j in range(len(iupac_token)):
                if iupac_token[j] in tokens:
                    token_position.append(tokens.index(iupac_token[j]))
            if len(token_position) == len(iupac_token):
                ids_sat.append(i)
                position.append(token_position)

            if len(ids_sat) == sample_num:
                break
    return ids_sat,position

def find_satisfy_iupac_cer(dataset,iupac_token,iupac_tokenizer,sample_num=10):
    ids_sat = []        # save ids satisfied
    position = [] # record iupac_token position
    for i in range(len(dataset)):
        token_position = []
        iupac = dataset[i][2][0]['iupac_tokens']
        iupac_token_list = iupac.tolist()
        iupac_token_list = [i for i in iupac_token_list if i!=0 and i!=1 and i!=2]
        tokens = iupac_tokenizer.convert_ids_to_tokens(iupac_token_list)
        string = ''.join(tokens)
        for j in range(len(iupac_token)):
            if string == iupac_token[j]:
                ids_sat.append(i)
                print(string)
        if len(ids_sat) == len(iupac_token):
            break
    return ids_sat,position


def get_heatmap_iupac_selfies(idx,ids_sat,token_position,dataset,attention,selfies_tokenizer,iupac_tokenizer):  
    index = ids_sat[idx]
    selfies_sample = dataset[index][1][0]['selfies_tokens']
    iupac_sample = dataset[index][2][0]['iupac_tokens']
    selfies_tokens = selfies_tokenizer.convert_ids_to_tokens(selfies_sample.tolist())   

    pos = token_position[idx]   # pos is a list
    iupac_tokens = iupac_tokenizer.convert_ids_to_tokens(iupac_sample.tolist())

    heatmap_name = []
    selfies_tokens_valid = []
    iupac_tokens_valid = []
    for j in range(1,attention[0].shape[0]):
        if selfies_tokens[j] == '[PAD]':
            break
        selfies_tokens_valid.append(selfies_tokens[j])
    try:
        smi = sf.decoder("".join(selfies_tokens_valid[0:-1]))
    except:
        smi = None
    selfies_tokens_valid.pop(-1)
    for po in pos:
        iupac_tokens_valid.append(iupac_tokens[po])
    # iupac_tokens_valid.append(iupac_tokens[pos])
    # print(selfies_tokens_valid)
    # print(iupac_tokens_valid)

    for k in range(attention.shape[0]): 
        attention_head = attention[k]
        attention_name = attention_head[1:j-1,pos]
        heatmap_name.append(attention_name.cpu())

    heatmap_name = np.array(heatmap_name)

    return heatmap_name,selfies_tokens_valid,iupac_tokens_valid,smi

def statistics_name_topk(heatmap_name,selfies_tokens_valid,iupac_tokens_valid,topk=5):
    attn_matrix = heatmap_name[-1]
    attn_matrix = np.array(attn_matrix)

    attn_rank = attn_matrix.copy()

    topk_selfies_tokens = []
    topk_coefficient = []

    # find topk selfies_tokens and record
    for i in range(topk):
        max_index = np.argmax(attn_rank)
        topk_selfies_tokens.append(selfies_tokens_valid[max_index])
        topk_coefficient.append(attn_matrix[max_index][0])
        attn_rank[max_index] = 0

    return topk_selfies_tokens,topk_coefficient
    
def statistics_atom_topk(heatmap_atom,selfies_tokens_valid,atom_tokens_valid,topk=5):
    attn_matrix = heatmap_atom[-1]
    attn_matrix = np.array(attn_matrix)
    attn_matrix = attn_matrix.T

    attn_rank = attn_matrix.copy()

    topk_atom_tokens = []
    topk_coefficient = []

    # find topk selfies_tokens and record
    for i in range(topk):
        max_index = np.argmax(attn_rank)
        topk_atom_tokens.append(atom_tokens_valid[max_index])
        topk_coefficient.append(attn_matrix[max_index][0])
        attn_rank[max_index] = 0

    return topk_atom_tokens,topk_coefficient


def plot_heatmap_name(heatmap_name,selfies_tokens_valid,iupac_tokens_valid):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.gridspec as gridspec
    # heatmap_name shape (num_heatmaps, rows, columns)


    # num_heatmaps = heatmap_name.shape[0]

    import numpy as np

    # 注意力系数矩阵
    attn_matrix = heatmap_name[-1]
    attn_matrix = np.array(attn_matrix)


    num_selfies_tokens = len(selfies_tokens_valid)
    num_iupac_tokens = len(iupac_tokens_valid)


    # 定义数据
    source_tokens = iupac_tokens_valid
    target_tokens = selfies_tokens_valid
    correlation_matrix = attn_matrix.T

    # 自动生成颜色
    cmap = plt.get_cmap('Pastel1')
    colors = [cmap(i) for i in range(len(source_tokens))]
    colors = ['rgba({},{},{},{})'.format(int(r*255), int(g*255), int(b*255), a) for r, g, b, a in colors]

    # 构建Sankey图数据结构
    source = []
    target = []
    value = []
    link_colors = []

    node_colors = colors + ['#E0E0E0'] * len(target_tokens) 

    for i, src_token in enumerate(source_tokens):
        for j, tgt_token in enumerate(target_tokens):
            source.append(i)
            target.append(j + len(source_tokens))
            value.append(correlation_matrix[i][j])
            link_colors.append(colors[i])  

    # # 绘制Sankey图
    # fig1 = go.Figure(data=[go.Sankey(
    #     node=dict(
    #         pad=25,
    #         thickness=30,  # 调整节点宽度
    #         line=dict(color="black", width=0.5),
    #         label=[],
    #         color=node_colors,
    #     ),
    #     link=dict(
    #         source=source,
    #         target=target,
    #         value=value,
    #         color=link_colors
    #     )
    # )])
    fig2 = go.Figure(data=[go.Sankey(
    node=dict(
        pad=25,
        thickness=30,  # 调整节点宽度
        line=dict(color="black", width=0.5),
        label=source_tokens + target_tokens,
        color=node_colors,
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color=link_colors
            )
        )])

    # fig1.update_layout(title_text="Sankey Diagram of Token Correlation ", font_size=10)
    # fig2.update_layout(title_text="Sankey Diagram of Token Correlation ", font_size=10)
    # 创建一个新的图
    # fig = plt.figure(figsize=(10, 15))
    # gs = gridspec.GridSpec(1, 2, width_ratios=[0.5,0.01])

    # ax = plt.subplot(gs[0, 0])
    # # 计算每个token的位置
    # positions_selfies = np.array([(x, 0) for x in range(num_selfies_tokens)])
    # positions_iupac = np.array([(x+num_selfies_tokens/2, 1) for x in range(num_iupac_tokens)])
    # # print(positions_selfies)
    # # print(positions_iupac)

    # # 画方块和token文字
    # for i, pos in enumerate(np.concatenate((positions_selfies, positions_iupac))):
    #     color = (255/255,223/255,146/255) if i < num_selfies_tokens else '#f0a0a0'  # 使用饱和度较低的颜色
    #     token = selfies_tokens_valid[i] if i < num_selfies_tokens else iupac_tokens_valid[i - num_selfies_tokens]
    #     if i < num_selfies_tokens:
    #         ax.add_patch(plt.Rectangle((pos[1] - 2, pos[0]), 0.5, 0.8, fill=True, color=color))  # 方块尺寸放大
    #         ax.text(pos[1]-1.75, pos[0]+0.4, token, ha='center', va='center',fontdict={'fontsize': 9, 'fontweight': 'bold'})
    #     else:
    #         ax.add_patch(plt.Rectangle((pos[1], pos[0]+2*(i-num_selfies_tokens)), 0.5, 0.8, fill=True, color=color))  # 方块尺寸放大
    #         ax.text(pos[1]+0.25, pos[0]+2*(i-num_selfies_tokens)+0.4, token, ha='center', va='center',fontdict={'fontsize': 10, 'fontweight': 'bold'})


        

    # # 画线
    # for i in range(num_selfies_tokens):
    #     for j in range(num_iupac_tokens):
    #         if attn_matrix[i, j] > 0.05:  # 注意力系数大于0的连接才画线
    #             # 通过注意力系数来调整线的颜色
    #             color = plt.cm.Blues(attn_matrix[i, j])  # 你可以选择你喜欢的颜色映射
    #             ax.plot(
    #                 [positions_selfies[i, 1]-1.5, positions_iupac[j, 1]],
    #                     [positions_selfies[i, 0]+0.5, positions_iupac[j, 0]+2*j+0.5],
    #                     color=color, linewidth=2.5)  # 这里固定了线宽为2，也可以通过注意力系数来调整

    # # 设置图形的界限和标签
    # ax.set_xlim([-2, 2])
    # ax.set_ylim([-1, max(num_selfies_tokens, num_iupac_tokens)])
    # ax.set_yticks([])  # 这里我们不显示y轴的刻度
    # ax.set_yticklabels([])
    # ax.set_xticks([-1.75, 1.25])
    # ax.set_xticklabels(['SELFIES', 'IUPAC'])
    # # ax.invert_yaxis()  # 反转y轴，使得第一个token位于顶部

    # # 去掉周围的框线
    # for side in ['top', 'right', 'bottom', 'left']:
    #     ax.spines[side].set_visible(False)

    # # 添加颜色条
    # cax = plt.subplot(gs[:, -1])
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=1))
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax, label='Attention Score',fraction=0.03,cax=cax)
    
    # plt.show()

    # # rows = int(np.ceil(num_heatmaps / 2))
    # fig = plt.figure(figsize=(15, 10))

    # # 定义一个GridSpec，将画布分为rows行，3列
    # gs = gridspec.GridSpec(8, 2, width_ratios=[0.5,0.01] ,height_ratios=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2])


    # for idx in range(num_heatmaps):
    #     ax = plt.subplot(gs[idx % 8, idx // 8])
    #     # 转置heatmap_name[idx]，使得行对应于iupac_tokens_valid，列对应于selfies_tokens_valid
    #     sns.heatmap(heatmap_name[idx].T,ax=ax, cmap='PuBu', cbar=False)
    #     ax.set_title(f'Head {idx + 1}',fontdict={'fontsize': 10, 'fontweight': 'bold','fontfamily':'Operator Mono'})
    #     if (idx+1) % 8 == 0:
    #         ax.set_xticks(np.arange(len(selfies_tokens_valid)) + 0.5, selfies_tokens_valid, rotation=90, fontsize=10)
    #     else:
    #         ax.set_xticks([])
    #     ax.set_yticks(np.arange(len(iupac_tokens_valid)) + 0.5, iupac_tokens_valid, rotation=0, fontsize=10)
    #     # for i in range(len(heatmap_name[idx])):
    #     #     ax.text(0.5,i+0.5,selfies_tokens_valid[i],ha='center',va='center',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    # # cloarbar third column
    # cax = plt.subplot(gs[:, -1])
    # cbar = plt.colorbar(ax.collections[0], cax=cax)
    # cbar.ax.set_xlabel('attention score', rotation=0, labelpad=15)  # add colorbar title

    # plt.tight_layout()
    
    # 将source_tokens target_tokens 以及matrix 保存为csv文件
    import pandas as pd
    df = pd.DataFrame(correlation_matrix,columns=target_tokens,index=source_tokens)
    # 保存为三列，分别是source target value
    # source = []
    # target = []
    # value = []
    # for i, src_token in enumerate(source_tokens):
    #     for j, tgt_token in enumerate(target_tokens):
    #         source.append(src_token)
    #         target.append(f'{j}_'+tgt_token)
    #         value.append(correlation_matrix[i][j])
    # df = pd.DataFrame({'IUPAC':source,'SELFIES':target,'value':value})

    return df,fig2


def find_satify_mol(dataset,selfies_token,selfies_tokenizer,atom_vocab_reverse,sample_num=10):
    ids_sat = []        # save ids satisfied
    position = [] # record iupac_token position
    # find mol contains this iupac_token
    for i in range(len(dataset)):
        token_position = []
        selfies = dataset[i][1][0]['selfies_tokens']
        mol = dataset[i][3]
        tokens = selfies_tokenizer.convert_ids_to_tokens(selfies.tolist())
        try:
            atom_tokens = [atom_vocab_reverse[k] for k in mol.x[:,0].tolist()]
            for j in range(len(selfies_token)):
                if selfies_token[j] in tokens:
                    token_position.append(tokens.index(selfies_token[j]))
            if len(token_position) == len(selfies_token) and len(atom_tokens)<30:
                ids_sat.append(i)
                position.append(token_position)  
            if len(ids_sat) == sample_num:
                break
        except:
            pass
    return ids_sat,position


def get_heatmap_mol_selfies(idx,ids_sat,token_position,dataset,attention,selfies_tokenizer,iupac_tokenizer,atom_vocab_reverse):
    index = ids_sat[idx]
    mol = dataset[index][3]
    selfies = dataset[index][1][0]['selfies_tokens']
    iupac = dataset[index][2][0]['iupac_tokens']
    iupac_tokens = iupac_tokenizer.convert_ids_to_tokens(iupac.tolist())
    selfies_tokens = selfies_tokenizer.convert_ids_to_tokens(selfies.tolist())
    pos = token_position[idx]       #pos is list
    atom_tokens = [atom_vocab_reverse[q] for q in mol.x[:,0].tolist()]
    heatmap_atom = []
    selfies_tokens_valid = []
    atom_tokens_valid = atom_tokens
    for j in range(1,attention[0].shape[0]):
        if selfies_tokens[j] == '[PAD]':
            break
        selfies_tokens_valid.append(selfies_tokens[j])
    smi = sf.decoder("".join(selfies_tokens_valid[0:-1]))
    selfies_tokens_valid.pop(-1)
    selfies_tokens_select = []
    for po in pos:
        selfies_tokens_select.append(selfies_tokens[po])
    # print(selfies_tokens_valid)
    # print(selfies_tokens_select)
    # print(atom_tokens_valid)
    for k in range(attention.shape[0]): 
        attention_head = attention[k]
        attention_atom = attention_head[pos,len(iupac_tokens):len(iupac_tokens)+len(atom_tokens)]
        heatmap_atom.append(attention_atom.cpu())
    
    heatmap_atom = np.array(heatmap_atom)
    return heatmap_atom,selfies_tokens_select,atom_tokens_valid,smi

def plot_heatmap_atom(heatmap_atom,selfies_tokens_select,atom_tokens_valid):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import matplotlib.gridspec as gridspec

    # 注意力系数矩阵
    attn_matrix = heatmap_atom[-1]
    attn_matrix = np.array(attn_matrix).T


    num_atom_tokens = len(atom_tokens_valid)
    num_selfies_tokens = len(selfies_tokens_select)

    # 创建一个新的图
    # fig = plt.figure(figsize=(10, 15))
    # gs = gridspec.GridSpec(1, 2, width_ratios=[0.5,0.01])

    # ax = plt.subplot(gs[0, 0])
    # # 计算每个token的位置
    # positions_atom = np.array([(x, 0) for x in range(num_atom_tokens)])
    # positions_selfies = np.array([(x+num_atom_tokens/2, 1) for x in range(num_selfies_tokens)])
    # # print(positions_selfies)
    # # print(positions_iupac)

    # # 画方块和token文字
    # for i, pos in enumerate(np.concatenate((positions_atom, positions_selfies))):
    #     color = (255/255,223/255,146/255) if i < num_atom_tokens else '#f0a0a0'  # 使用饱和度较低的颜色
    #     token = atom_tokens_valid[i] if i < num_atom_tokens else selfies_tokens_select[i - num_atom_tokens]
    #     if i < num_atom_tokens:
    #         ax.add_patch(plt.Rectangle((pos[1] - 2, pos[0]), 0.5, 0.8, fill=True, color=color))  # 方块尺寸放大
    #         ax.text(pos[1]-1.75, pos[0]+0.4, token, ha='center', va='center',fontdict={'fontsize': 9, 'fontweight': 'bold'})
    #     else:
    #         ax.add_patch(plt.Rectangle((pos[1], pos[0]+2*(i-num_atom_tokens)), 0.5, 0.8, fill=True, color=color))  # 方块尺寸放大
    #         ax.text(pos[1]+0.25, pos[0]+2*(i-num_atom_tokens)+0.4, token, ha='center', va='center',fontdict={'fontsize': 10, 'fontweight': 'bold'})


        

    # # 画线
    # for i in range(num_atom_tokens):
    #     for j in range(num_selfies_tokens):
    #         if attn_matrix[i, j] > 0.01:  # 注意力系数大于0的连接才画线
    #             # 通过注意力系数来调整线的颜色
    #             color = plt.cm.Blues(attn_matrix[i, j])  # 你可以选择你喜欢的颜色映射
    #             ax.plot(
    #                 [positions_atom[i, 1]-1.5, positions_selfies[j, 1]],
    #                     [positions_atom[i, 0]+0.5, positions_selfies[j, 0]+2*j+0.5],
    #                     color=color, linewidth=2.5)  # 这里固定了线宽为2，也可以通过注意力系数来调整

    # # 设置图形的界限和标签
    # ax.set_xlim([-2, 2])
    # ax.set_ylim([-1, max(num_selfies_tokens, num_atom_tokens)])
    # ax.set_yticks([])  # 这里我们不显示y轴的刻度
    # ax.set_yticklabels([])
    # ax.set_xticks([-1.75, 1.25])
    # ax.set_xticklabels(['ATOM', 'SELFIES'])
    # # ax.invert_yaxis()  # 反转y轴，使得第一个token位于顶部

    # # 去掉周围的框线
    # for side in ['top', 'right', 'bottom', 'left']:
    #     ax.spines[side].set_visible(False)

    # # 添加颜色条
    # cax = plt.subplot(gs[:, -1])
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=1))
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax, label='Attention Score',fraction=0.03,cax=cax)
    # num_heatmaps = heatmap_atom.shape[0]

    # fig = plt.figure(figsize=(10, 10))


    # gs = gridspec.GridSpec(8, 2, width_ratios=[0.5,0.01] ,height_ratios=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2])


    # for idx in range(num_heatmaps):

    #     ax = plt.subplot(gs[idx % 8, idx // 8])

    #     sns.heatmap(heatmap_atom[idx],ax=ax, cmap='PuBu', cbar=False)
    #     ax.set_title(f'Head {idx + 1}',fontdict={'fontsize': 10, 'fontweight': 'bold','fontfamily':'Operator Mono'})
    #     ax.set_yticks(np.arange(len(selfies_tokens_select)) + 0.5, selfies_tokens_select, rotation=0, fontsize=10)
    #     if (idx+1) % 8 == 0:
    #         ax.set_xticks(np.arange(len(atom_tokens_valid)) + 0.5, atom_tokens_valid, rotation=0, fontsize=10)
    #     else:
    #         ax.set_xticks([])

    #     # for i in range(len(atom_tokens_valid)):
    #     #     ax.text(0.5,i+0.5,atom_tokens_valid[i],ha='center',va='center',fontdict={'fontsize': 8, 'fontweight': 'medium'})

    # cax = plt.subplot(gs[:, -1])
    # cbar = plt.colorbar(ax.collections[0], cax=cax)
    # cbar.ax.set_ylabel('attention score', rotation=90, labelpad=15)  

    # plt.tight_layout()

    # 与plot_heatmap_name 相同，计算source target value
    source_tokens = selfies_tokens_select
    target_tokens = atom_tokens_valid
    correlation_matrix = attn_matrix.T

    # 自动生成颜色
    cmap = plt.get_cmap('Pastel1')
    colors = [cmap(i) for i in range(len(source_tokens))]
    colors = ['rgba({},{},{},{})'.format(int(r*255), int(g*255), int(b*255), a) for r, g, b, a in colors]

    # 构建Sankey图数据结构
    source = []
    target = []
    value = []
    link_colors = []

    node_colors = colors + ['#E0E0E0'] * len(target_tokens) 

    for i, src_token in enumerate(source_tokens):
        for j, tgt_token in enumerate(target_tokens):
            source.append(i)
            target.append(j + len(source_tokens))
            value.append(correlation_matrix[i][j])
            link_colors.append(colors[i])  

    fig2 = go.Figure(data=[go.Sankey(
    node=dict(
        pad=25,
        thickness=30,  # 调整节点宽度
        line=dict(color="black", width=0.5),
        label=source_tokens + target_tokens,
        color=node_colors,
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color=link_colors
            )
        )])
    
    # 保存为csv
    import pandas as pd
    df = pd.DataFrame(correlation_matrix,columns=target_tokens,index=source_tokens)


    return df,fig2

def get_heatmap_iupac_selfies_all(idx,ids_sat,token_position,dataset,attention,selfies_tokenizer,iupac_tokenizer,atom_vocab_reverse):
    index = ids_sat[idx]
    selfies_sample = dataset[index][1][0]['selfies_tokens']
    iupac_sample = dataset[index][2][0]['iupac_tokens']
    selfies_tokens = selfies_tokenizer.convert_ids_to_tokens(selfies_sample.tolist())   
    mol = dataset[index][3]

  
    iupac_tokens = iupac_tokenizer.convert_ids_to_tokens(iupac_sample.tolist())

    atom_tokens = [atom_vocab_reverse[q] for q in mol.x[:,0].tolist()]
    heatmap_atom = []
    attention_sum = []
    heatmap_name = []


    selfies_tokens_valid = []
    atom_tokens_valid = atom_tokens
    for j in range(1,attention[0].shape[0]):
        if selfies_tokens[j] == '[PAD]':
            break
        selfies_tokens_valid.append(selfies_tokens[j])
    iupac_tokens_valid = []
    for x in range(1,len(iupac_tokens)):
        if iupac_tokens[x] == '[PAD]':
            break
        iupac_tokens_valid.append(iupac_tokens[x])

    smi = sf.decoder("".join(selfies_tokens_valid[0:-1]))
    selfies_tokens_valid.pop(-1)
    iupac_tokens_valid.pop(-1)
    # print(selfies_tokens_valid)
    # print(selfies_tokens_select)
    # print(atom_tokens_valid)
    for k in range(attention.shape[0]): 
        attention_head = attention[k]
        # 对每一行求和，得到每个selfies_token对应的注意力系数
        attention_sum.append(torch.sum(attention_head[1:j-1,len(iupac_tokens):len(iupac_tokens)+len(atom_tokens)],dim=1).cpu())
        attention_name = attention_head[1:j-1,1:x-1]
        attention_atom = attention_head[1:j-1,len(iupac_tokens):len(iupac_tokens)+len(atom_tokens)]
        heatmap_atom.append(attention_atom.cpu())
        heatmap_name.append(attention_name.cpu())
    
    heatmap_atom = np.array(heatmap_atom)
    heatmap_name = np.array(heatmap_name)
    return attention_sum,heatmap_atom,heatmap_name,selfies_tokens_valid,atom_tokens_valid,iupac_tokens_valid,smi



def plot_heatmap_name_all(heatmap_atom,heatmap_name,selfies_tokens_valid,atom_tokens_valid,iupac_tokens_valid,save_path,layer):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.gridspec as gridspec
    # heatmap_name shape (num_heatmaps, rows, columns)


    num_heatmaps = heatmap_name.shape[0]
    # rows = int(np.ceil(num_heatmaps / 2))

    for idx in range(num_heatmaps):
        # 定义一个GridSpec，将画布分为rows行，3列
        fig = plt.figure(figsize=(10,10))
        gs = gridspec.GridSpec(1, 2, width_ratios=[0.95,0.01])


        ax = plt.subplot(gs[0, 0])
        # 转置heatmap_name[idx]，使得行对应于iupac_tokens_valid，列对应于selfies_tokens_valid
        sns.heatmap(heatmap_atom[idx],ax=ax, cmap='PuBu', cbar=False)
        ax.set_title(f'Head {idx + 1}',fontdict={'fontsize': 10, 'fontweight': 'bold','fontfamily':'Operator Mono'})
        ax.set_yticks(np.arange(len(selfies_tokens_valid)) + 0.5, selfies_tokens_valid, rotation=0, fontsize=20)
        ax.set_xticks(np.arange(len(atom_tokens_valid)) + 0.5, atom_tokens_valid, rotation=0, fontsize=20)
        # cloarbar third column
        cax = plt.subplot(gs[:, -1])
        cbar = plt.colorbar(ax.collections[0], cax=cax)
        cbar.ax.set_ylabel('attention score', rotation=90, labelpad=15)  # add colorbar title

        plt.tight_layout()

        fig.savefig(f'{save_path}/layer{layer}_head{idx}_atom.svg',dpi=600,format='svg')
    
    for idx in range(num_heatmaps):
        fig = plt.figure(figsize=(10,10))
        gs = gridspec.GridSpec(1, 2, width_ratios=[0.95,0.01])


        ax = plt.subplot(gs[0, 0])
        # 转置heatmap_name[idx]，使得行对应于iupac_tokens_valid，列对应于selfies_tokens_valid
        sns.heatmap(heatmap_name[idx],ax=ax, cmap='PuBu', cbar=False)
        ax.set_title(f'Head {idx + 1}',fontdict={'fontsize': 10, 'fontweight': 'bold','fontfamily':'Operator Mono'})
        ax.set_yticks(np.arange(len(selfies_tokens_valid)) + 0.5, selfies_tokens_valid, rotation=0, fontsize=20)
        ax.set_xticks(np.arange(len(iupac_tokens_valid)) + 0.5,iupac_tokens_valid, rotation=45, fontsize=15)
        # cloarbar third column
        cax = plt.subplot(gs[:, -1])
        cbar = plt.colorbar(ax.collections[0], cax=cax)
        cbar.ax.set_ylabel('attention score', rotation=90, labelpad=15)  # add colorbar title

        plt.tight_layout()

        fig.savefig(f'{save_path}/layer{layer}_head{idx}_name.svg',dpi=600,format='svg')      
    
    return None


def main(args,config):
    output_dir = args.output_dir
    device = args.device
    iupac_token = args.iupac_token      # a list
    mode = args.mode
    selfies_token = args.selfies_token
    model_name = args.model_name


    print("creating dataset")
    dataset,_ = create_dataset(
    config['dataset']['ids_path'],
    config['dataset']['ipc_path'],
    config['dataset']['sfs_path'],
    config['dataset']['mlg_path'],
    0.0)
    print(len(dataset))
    model = TMMF(config)
    model = model.to(device)

    if model_name == 'esol':
        path = './pre_train/1m_checkpoint_09.pth'
    elif model_name == 'lipo':
        path = './pre_train/lipo_checkpoint.pth'
    elif model_name == 'pre':
        path = './pre_train/1m_checkpoint_09.pth'
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    Interact_model = BertForInteract(model.selfies_encoder,model.iupac_encoder,model.mol_encoder,model.fusion_encoder)
    Interact_model = Interact_model.to(device)

    iupac_tokenizer = Iupac_tokenizer()
    selfies_tokenizer = Selfies_tokenizer()

    atom_list = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co',
                 'Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn',
                 'Sb','Te','I','Xe']
    atom_dict = {atom:i for i,atom in enumerate(atom_list)}
    atom_dict['<unk>'] = len(atom_dict)
    atom_vocab_reverse = {v:k for k,v in atom_dict.items()}

    if mode == 'iupac':
        ids_sat,token_position = find_satisfy_iupac(dataset,iupac_token,iupac_tokenizer,sample_num=args.sample_num)
        # choose one sample from ids_sat
        for i in range(len(ids_sat)):
            selfies,iupac,mol = dataset[ids_sat[i]][1][0]['selfies_tokens'],dataset[ids_sat[i]][2][0]['iupac_tokens'],dataset[ids_sat[i]][3]
            selfies,iupac = torch.unsqueeze(selfies,dim=0),torch.unsqueeze(iupac,dim=0)
            from torch_geometric.data import Batch
            mol = Batch.from_data_list([mol])
            selfies,iupac,mol = selfies.to(device),iupac.to(device),mol.to(device)
            out = Interact_model(selfies_ids=selfies,iupac_ids=iupac,mol_ids=mol,save_attention=True)
            attention = Interact_model.fusion_encoder.transformer_blocks[args.layer].cross_attention.get_attention_map()[0]
            heatmap_name,selfies_tokens_valid,iupac_tokens_valid,smi = get_heatmap_iupac_selfies(i,ids_sat,token_position,dataset,attention,selfies_tokenizer,iupac_tokenizer)
            df,fig2 = plot_heatmap_name(heatmap_name,selfies_tokens_valid,iupac_tokens_valid)
            save_path = f'{output_dir}/{iupac_token}/{i}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            df.to_csv(f'{save_path}/layer{args.layer}.csv')
            # fig1.write_image(f'{save_path}/layer{args.layer}_1.svg')
            # fig1.write_image(f'{save_path}/layer{args.layer}_1.png')
            # fig2.write_image(f'{save_path}/layer{args.layer}_2.svg')
            fig2.write_image(f'{save_path}/layer{args.layer}_2.png')
            smi_graph = Chem.MolFromSmiles(smi)
            from rdkit.Chem.Draw import rdMolDraw2D
            drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)  
            drawer.DrawMolecule(smi_graph,highlightAtoms=[], highlightBonds=[])
            drawer.FinishDrawing()

            svg = drawer.GetDrawingText().replace("svg:", "")

            with open(f'{save_path}/layer{args.layer}_smi.svg', "w") as f:
                f.write(svg)  
            with open(f'{save_path}/layer{args.layer}_smi.txt', "w") as f:
                f.write(smi)

    elif mode == 'atom':
        ids_sat,token_position = find_satify_mol(dataset,selfies_token,selfies_tokenizer,atom_vocab_reverse,sample_num=args.sample_num)
        for i in range(len(ids_sat)):
            selfies,iupac,mol = dataset[ids_sat[i]][1][0]['selfies_tokens'],dataset[ids_sat[i]][2][0]['iupac_tokens'],dataset[ids_sat[i]][3]
            selfies,iupac = torch.unsqueeze(selfies,dim=0),torch.unsqueeze(iupac,dim=0)
            from torch_geometric.data import Batch
            mol = Batch.from_data_list([mol])
            selfies,iupac,mol = selfies.to(device),iupac.to(device),mol.to(device)
            out = Interact_model(selfies_ids=selfies,iupac_ids=iupac,mol_ids=mol,save_attention=True)
            attention = Interact_model.fusion_encoder.transformer_blocks[args.layer].cross_attention.get_attention_map()[0]
            heatmap_atom,selfies_tokens_select,atom_tokens_valid,smi = get_heatmap_mol_selfies(i,ids_sat,token_position,dataset,attention,selfies_tokenizer,iupac_tokenizer,atom_vocab_reverse)
            df,fig2 = plot_heatmap_atom(heatmap_atom,selfies_tokens_select,atom_tokens_valid)
            save_path = f'{output_dir}/{selfies_token}/{i}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            df.to_csv(f'{save_path}/layer{args.layer}.csv')
            # fig1.write_image(f'{save_path}/layer{args.layer}_1.svg')
            # fig1.write_image(f'{save_path}/layer{args.layer}_1.png')
            # fig2.write_image(f'{save_path}/layer{args.layer}_2.svg')
            fig2.write_image(f'{save_path}/layer{args.layer}_2.png')
            smi_graph = Chem.MolFromSmiles(smi)
            from rdkit.Chem.Draw import rdMolDraw2D
            drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)  
            drawer.DrawMolecule(smi_graph,highlightAtoms=[], highlightBonds=[])
            drawer.FinishDrawing()

            svg = drawer.GetDrawingText().replace("svg:", "")

            with open(f'{save_path}/layer{args.layer}_smi.svg', "w") as f:
                f.write(svg)  
            with open(f'{save_path}/layer{args.layer}_smi.txt', "w") as f:
                f.write(smi)

    elif mode == 'selfies_iupac':
        iupac_list = iupac_token
        ids_sat,token_position = find_satisfy_iupac_cer(dataset,iupac_list,iupac_tokenizer,sample_num=args.sample_num)
        # choose one sample from ids_sat
        for i in range(len(ids_sat)):
            selfies,iupac,mol = dataset[ids_sat[i]][1][0]['selfies_tokens'],dataset[ids_sat[i]][2][0]['iupac_tokens'],dataset[ids_sat[i]][3]
            selfies,iupac = torch.unsqueeze(selfies,dim=0),torch.unsqueeze(iupac,dim=0)

            from torch_geometric.data import Batch
            mol = Batch.from_data_list([mol])
            selfies,iupac,mol = selfies.to(device),iupac.to(device),mol.to(device)
            out = Interact_model(selfies_ids=selfies,iupac_ids=iupac,mol_ids=mol,save_attention=True)
            attention = Interact_model.fusion_encoder.transformer_blocks[args.layer].cross_attention.get_attention_map()[0]
            try:
                attention_sum,heatmap_atom,heatmap_name,selfies_tokens_valid,atom_tokens_valid,iupac_tokens_valid,smi = get_heatmap_iupac_selfies_all(i,ids_sat,token_position,dataset,attention,selfies_tokenizer,iupac_tokenizer,atom_vocab_reverse)
                # save_path = f'{output_dir}/{iupac_list[i]}_heatmap/{i}'
                save_path = f'{output_dir}/{model_name}/{iupac_list[i]}'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                fig = plot_heatmap_name_all(heatmap_atom,heatmap_name,selfies_tokens_valid,atom_tokens_valid,iupac_tokens_valid,save_path,args.layer)
                # 将 attention_atom的8个头分别保存为csv
                import pandas as pd
                # for j in range(len(attention_sum)):
                #     df = pd.DataFrame(attention_sum[j],columns=['atten'],index=selfies_tokens_valid)
                #     df.to_csv(f'{save_path}/layer{args.layer}_head{j}.csv')

                for j in range(heatmap_atom.shape[0]):
                    df = pd.DataFrame(np.array(heatmap_atom[j]),columns=atom_tokens_valid,index=selfies_tokens_valid)
                    df.to_csv(f'{save_path}/layer{args.layer}_head{j}.csv')



                smi_graph = Chem.MolFromSmiles(smi)
                from rdkit.Chem.Draw import rdMolDraw2D
                drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)  
                drawer.DrawMolecule(smi_graph,highlightAtoms=[], highlightBonds=[])
                drawer.FinishDrawing()

                svg = drawer.GetDrawingText().replace("svg:", "")

                with open(f'{save_path}/layer{args.layer}_smi.svg', "w") as f:
                    f.write(svg)  
                # 将 smi 保存为txt
                with open(f'{save_path}/layer{args.layer}_smi.txt', "w") as f:
                    f.write(smi)
            except Exception as e:
                print(e)
                pass

    elif mode == 'iupac_statistics':
        tokens_coefficent_dict = {}
        tokens_count_dict = {}
        ids_sat,token_position = find_satisfy_iupac(dataset,iupac_token,iupac_tokenizer,sample_num=args.sample_num)
        # choose one sample from ids_sat
        topk = 3
        for i in range(len(ids_sat)):
            selfies,iupac,mol = dataset[ids_sat[i]][1][0]['selfies_tokens'],dataset[ids_sat[i]][2][0]['iupac_tokens'],dataset[ids_sat[i]][3]
            selfies,iupac = torch.unsqueeze(selfies,dim=0),torch.unsqueeze(iupac,dim=0)
            from torch_geometric.data import Batch
            mol = Batch.from_data_list([mol])
            selfies,iupac,mol = selfies.to(device),iupac.to(device),mol.to(device)
            out = Interact_model(selfies_ids=selfies,iupac_ids=iupac,mol_ids=mol,save_attention=True)
            attention = Interact_model.fusion_encoder.transformer_blocks[args.layer].cross_attention.get_attention_map()[0]
            heatmap_name,selfies_tokens_valid,iupac_tokens_valid,smi = get_heatmap_iupac_selfies(i,ids_sat,token_position,dataset,attention,selfies_tokenizer,iupac_tokenizer)
            topk_tokens,topk_coefficent = statistics_name_topk(heatmap_name,selfies_tokens_valid,iupac_tokens_valid,topk)
            for j in range(len(topk_tokens)):
                if topk_tokens[j] in tokens_coefficent_dict.keys():
                    tokens_coefficent_dict[topk_tokens[j]] = (tokens_coefficent_dict[topk_tokens[j]]*tokens_count_dict[topk_tokens[j]]+topk_coefficent[j])/(tokens_count_dict[topk_tokens[j]]+1)
                    tokens_count_dict[topk_tokens[j]] += 1
                else:
                    tokens_coefficent_dict[topk_tokens[j]] = topk_coefficent[j]
                    tokens_count_dict[topk_tokens[j]] = 1
            
        # 分为三列 tokens coefficent count
        tokens = []
        coefficent = []
        count = []
        sum_count = sum(tokens_count_dict.values())
        for key in tokens_coefficent_dict.keys():
            tokens.append(key)
            coefficent.append(tokens_coefficent_dict[key])
            count.append(tokens_count_dict[key]/sum_count)
        # 按照 count 大小排序
        tokens = [x for _,x in sorted(zip(count,tokens),reverse=True)]
        coefficent = [x for _,x in sorted(zip(count,coefficent),reverse=True)]
        count = sorted(count,reverse=True)

        import pandas as pd
        df = pd.DataFrame({'tokens':tokens,'coefficent':coefficent,'count':count})
        save_path = f'{output_dir}/{iupac_token}/statistics'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df.to_csv(f'{save_path}/top_{topk}.csv')
        
    elif mode == 'atom_statistics':
        tokens_coefficent_dict = {}
        tokens_count_dict = {}
        ids_sat,token_position = find_satify_mol(dataset,selfies_token,selfies_tokenizer,atom_vocab_reverse,sample_num=args.sample_num)
        # choose one sample from ids_sat
        topk = 1
        for i in range(len(ids_sat)):
            selfies,iupac,mol = dataset[ids_sat[i]][1][0]['selfies_tokens'],dataset[ids_sat[i]][2][0]['iupac_tokens'],dataset[ids_sat[i]][3]
            selfies,iupac = torch.unsqueeze(selfies,dim=0),torch.unsqueeze(iupac,dim=0)
            from torch_geometric.data import Batch
            mol = Batch.from_data_list([mol])
            selfies,iupac,mol = selfies.to(device),iupac.to(device),mol.to(device)
            out = Interact_model(selfies_ids=selfies,iupac_ids=iupac,mol_ids=mol,save_attention=True)
            attention = Interact_model.fusion_encoder.transformer_blocks[args.layer].cross_attention.get_attention_map()[0]
            heatmap_atom,selfies_tokens_select,atom_tokens_valid,smi = get_heatmap_mol_selfies(i,ids_sat,token_position,dataset,attention,selfies_tokenizer,iupac_tokenizer,atom_vocab_reverse)
            topk_tokens,topk_coefficent = statistics_atom_topk(heatmap_atom,selfies_tokens_select,atom_tokens_valid,topk)
            for j in range(len(topk_tokens)):
                if topk_tokens[j] in tokens_coefficent_dict.keys():
                    tokens_coefficent_dict[topk_tokens[j]] = (tokens_coefficent_dict[topk_tokens[j]]*tokens_count_dict[topk_tokens[j]]+topk_coefficent[j])/(tokens_count_dict[topk_tokens[j]]+1)
                    tokens_count_dict[topk_tokens[j]] += 1
                else:
                    tokens_coefficent_dict[topk_tokens[j]] = topk_coefficent[j]
                    tokens_count_dict[topk_tokens[j]] = 1
        
        tokens = []
        coefficent = []
        count = []
        sum_count = sum(tokens_count_dict.values())
        for key in tokens_coefficent_dict.keys():
            tokens.append(key)
            coefficent.append(tokens_coefficent_dict[key])
            count.append(tokens_count_dict[key]/sum_count)
        # 按照 count 大小排序
        tokens = [x for _,x in sorted(zip(count,tokens),reverse=True)]
        coefficent = [x for _,x in sorted(zip(count,coefficent),reverse=True)]
        count = sorted(count,reverse=True)
        import pandas as pd
        df = pd.DataFrame({'tokens':tokens,'coefficent':coefficent,'count':count})
        save_path = f'{output_dir}/{selfies_token}/statistics'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df.to_csv(f'{save_path}/top_{topk}.csv')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/retrieval.yaml')
    parser.add_argument('--layer', default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output_dir', default='downstream/interact')
    parser.add_argument('--iupac_token',type=str, nargs='+', default=['formyl'])
    parser.add_argument('--model_name',type=str, default='pre')
    parser.add_argument('--selfies_token', default=['[#N]','[Cl]','[=O]'])
    parser.add_argument('--mode', default='iupac_statistics')
    parser.add_argument('--sample_num', default=2000)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

     
    main(args, config)
    
 
