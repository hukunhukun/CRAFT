from model.TMMF import TMMF
from model.bert import BertForFinetune
from utils.utils import MetricLogger, SmoothedValue,AttrDict
from dataset import create_dataset, create_dataloader
from scheduler import create_scheduler
from optim import create_optimizer
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import torch
class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = Config(value)
        return value
    

def train(model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, config):

    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', SmoothedValue(window_size=50, fmt='{value:.4f}'))

    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 5   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  


    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        optimizer.zero_grad()
        selfies,iupac,mol,label = batch['sfs_tokens']['selfies_tokens'],batch['ipc_tokens']['iupac_tokens'],batch['mlg'],batch['label']
        selfies = selfies.to(device)
        iupac = iupac.to(device)
        mol = mol.to(device)  
        label = label.to(device)

        _,loss = model._train(selfies_ids=selfies,iupac_ids=iupac,mol_ids=mol,labels=label)
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])         
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)        


    print("Averaged stats:", metric_logger.global_avg()) 
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


from sklearn.metrics import roc_auc_score
def calculate_auc(model,loader,device):
    import math
    model.eval()
    y_true = []
    y_pred = []
    total_num = 0
    total_correct = 0
    with torch.no_grad():
        for batch in loader:
            selfies,iupac,mol,label = batch['sfs_tokens']['selfies_tokens'],batch['ipc_tokens']['iupac_tokens'],batch['mlg'],batch['label']
            selfies = selfies.to(device)
            iupac = iupac.to(device)
            mol = mol.to(device)  
            label = label.to(device)
            output,labels = model._test(selfies,iupac,mol,label)
            num = label.size(0)
            total_num += num            
            pred = output.argmax(dim=1)
            correct = torch.eq(pred,label).sum().float().item()
            total_correct += correct
            output = [math.exp(output[i][1].cpu().numpy()) for i in range(len(output))]
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(output)
        print("accu:",total_correct/total_num)
    return roc_auc_score(y_true,y_pred),y_pred,y_true



def calculate_rsquared(model,loader,device):
    import math
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in loader:
            selfies,iupac,mol,label = batch['sfs_tokens']['selfies_tokens'],batch['ipc_tokens']['iupac_tokens'],batch['mlg'],batch['label']
            selfies = selfies.to(device)
            iupac = iupac.to(device)
            mol = mol.to(device)  
            label = label.to(device)
            output,labels = model._test(selfies,iupac,mol,label)


            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(output.cpu().numpy())

    return y_pred,y_true


def main(name):
    config = yaml.load(open('./config/finetune.yaml', 'r'), Loader=yaml.Loader)
    device = torch.device("cuda:1")
    seed = [42,52,62,72,32,22,12,2]
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']   

    print("creating dataset")
    dataset_config = Config(json.load(open(config['dataset'][name])))


    for i in range(len(seed)):
        train_dataset,test_dataset = create_dataset(
        dataset_config['dataset']['ids_path'],
        dataset_config['dataset']['ipc_path'],
        dataset_config['dataset']['sfs_path'],
        dataset_config['dataset']['mlg_path'],
        dataset_config['dataset']['frac'],mode="finetune")


        print("creating model")
        pretrain_config = yaml.load(open('./config/pre_train.yaml', 'r'), Loader=yaml.Loader)
        model = TMMF(pretrain_config)
        model = model.to(device)

        finetune_model = BertForFinetune(dataset_config,model.selfies_encoder,model.iupac_encoder,model.mol_encoder,model.fusion_encoder)
        finetune_model = finetune_model.to(device)
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        for param in finetune_model.parameters():
            mulValue = np.prod(param.size())  # 
            Total_params += mulValue  
            if param.requires_grad:
                Trainable_params += mulValue  # trainable
            else:
                NonTrainable_params += mulValue  


        print(f'Total params: {Total_params}')
        print(f'Trainable params: {Trainable_params}')
        print(f'Non-trainable params: {NonTrainable_params}')
        torch.manual_seed(seed[i])
        np.random.seed(seed[i])
        train_loader = create_dataloader(train_dataset,dataset_config['batch_size'],mode="finetune",task_type=dataset_config['dataset']['task_type'])
        test_loader = create_dataloader(test_dataset,dataset_config['batch_size'],mode="finetune",task_type=dataset_config['dataset']['task_type'])


        arg_opt = AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, finetune_model)
        arg_sche = AttrDict(config['schedular'])
        lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  

        print("Start training")
        start_time = time.time()
        start_epoch = 0

        for epoch in range(start_epoch, max_epoch):
            
            if epoch>0:
                lr_scheduler.step(epoch+warmup_steps)  
            
            train_stats = train(finetune_model, train_loader, optimizer, epoch, warmup_steps, device, lr_scheduler, config)
            
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch}                     


            save_path = 'downstream_without_pretrain/' + dataset_config.dataset.name
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(os.path.join(save_path + "/log"+ str(i) + ".txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str)) 

        print("Start testing")
        if dataset_config.dataset.task_type == "classification":
            roc_auc_score,y_pred,y_true = calculate_auc(finetune_model,test_loader,device)
            save_name = 'roc'
            print(roc_auc_score)

        elif dataset_config.dataset.task_type == "regression":
            y_pred,y_true = calculate_rsquared(finetune_model,test_loader,device)
            from sklearn.metrics import r2_score,mean_squared_error
            print("r2_score",r2_score(y_true,y_pred))
            print("RMSE",np.sqrt(mean_squared_error(y_true,y_pred,squared=False)))
            save_name = 'r2'
        rocdata = {'y_pred':y_pred,'label':y_true}
        from pandas.core.frame import DataFrame
        rocdata = DataFrame(rocdata)
        rocdata.to_csv(save_path + '/' +  '%d'%i + '_'+ save_name+'.csv')



if __name__ == '__main__':
    dataset_name = ['bbbp','bace' ,'clintox']
    dataset_name = ['bbbp','bace','clintox','esol','freesolv' ,'lipo']
    for name in dataset_name:
        main(name)