import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from model.TMMF import TMMF
from utils.utils import MetricLogger, SmoothedValue,AttrDict
from dataset import create_dataset, create_dataloader
from scheduler import create_scheduler
from optim import create_optimizer


def train(model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, config):

    model.train()
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 
        Total_params += mulValue  
        if param.requires_grad:
            Trainable_params += mulValue  # trainable
        else:
            NonTrainable_params += mulValue  


    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_msm', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_sima', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_simm', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 100   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  


    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        optimizer.zero_grad()
        selfies,iupac,mol = batch['sfs_tokens']['selfies_tokens'],batch['ipc_tokens']['iupac_tokens'],batch['mlg']
        selfies = selfies.to(device)
        iupac = iupac.to(device)
        mol = mol.to(device)  

        _,loss_sima,loss_simm,loss_msm = model(mol,iupac,selfies)
        loss = loss_sima+loss_simm+loss_msm
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss_msm=loss_msm.item())
        metric_logger.update(loss_sima=loss_sima.item())
        metric_logger.update(loss_simm=loss_simm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])         
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)        


    print("Averaged stats:", metric_logger.global_avg()) 
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    


def test(model,data_loader,device):
    i = 0
    num_total = 0
    correct_total = 0
    print_freq = 50
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            selfies,iupac,mol = batch['sfs_tokens']['selfies_tokens'],batch['ipc_tokens']['iupac_tokens'],batch['mlg']
            selfies = selfies.to(device)
            iupac = iupac.to(device)
            mol = mol.to(device)  
            correct,num = model.test(selfies,iupac,mol)
            correct_total += correct
            num_total += num
            i += 1
        
        if i%print_freq==0:
            print("iter:",i,"acc:",correct/num)
        
    print("total acc:",correct_total/num_total)


def main(args,config):

    device = torch.device(args.device)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']   

    print("creating dataset")
    train_dataset,test_dataset = create_dataset(
    config['dataset']['ids_path'],
    config['dataset']['ipc_path'],
    config['dataset']['sfs_path'],
    config['dataset']['mlg_path'],
    config['dataset']['frac'])

    train_loader = create_dataloader(train_dataset,config['dataset']['batch_size'])
    test_loader = create_dataloader(test_dataset,config['dataset']['batch_size'])

    print("creating model")
    model = TMMF(config)
    model = model.to(device)

    # 注释下面三行代码，可以从头开始训练
    # path = './pre_train/100k_checkpoint_09.pth'
    # checkpoint = torch.load(path)
    # model.load_state_dict(checkpoint['model'])

    arg_opt = AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  

    print("Start training")
    start_time = time.time()
    start_epoch = 0

    for epoch in range(start_epoch, max_epoch):
        
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)  
        
        train_stats = train(model, train_loader, optimizer, epoch, warmup_steps, device, lr_scheduler, config)
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                    }                     
        save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'config': config,
            'epoch': epoch,
        }
        if (epoch+1)%5 == 0 or epoch==max_epoch-1:
            torch.save(save_obj, os.path.join(args.output_dir, '100k_checkpoint_test%02d.pth'%epoch))  
        
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    print("Start testing")
    test(model,test_loader,device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/pre_train.yaml')
    parser.add_argument('--output_dir', default='pre_train/')
    parser.add_argument('--device', default=torch.device("cuda:0"))
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()


    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
     
    main(args, config)