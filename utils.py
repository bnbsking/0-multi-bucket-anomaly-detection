import json, math, os, shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, average_precision_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, densenet121, vit_b_16


class MyDataset(Dataset):
    def __init__(self, path_list, label_fine_list, label_coarse_list, mode:str): 
        if 0:
            import random
            path_list = random.sample(path_list, 100)
            label_fine_list = random.sample(label_fine_list, 100)
            label_coarse_list = random.sample(label_coarse_list, 100)
        self.path_list = path_list
        self.label_fine_list = np.array(label_fine_list)
        self.label_coarse_list = np.array(label_coarse_list)
        if mode=="train":
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ColorJitter([0.8,1.2], [0.8,1.2], [0.8,1.2], [-0.2,0.2]),
                # brigntness, contrast, saturation, hue
                #transforms.RandomRotation([-90,90]),
                transforms.RandomHorizontalFlip(0.5),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                # RGB mean & std based on imagenet
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        x = cv2.imread(self.path_list[index])/255. # read & normalize
        x = torch.tensor(x, dtype=torch.float32).permute(2,0,1) # tensor & channel first
        x = self.transform(x)
        y_fine = torch.tensor([self.label_fine_list[index]], dtype=torch.long)
        y_coarse = torch.tensor([self.label_coarse_list[index]], dtype=torch.long)
        return x, y_fine, y_coarse # (3,224,224), (1,), (1,)


def get_loader(path_list, label_fine_list, label_coarse_list, mode:str, batch_size:int):
    dataset = MyDataset(path_list, label_fine_list, label_coarse_list, mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=mode=='train')
    return loader


class MyModel(torch.nn.Module):
    def __init__(self, backbone:str, pretrained=None, output_dim=2):
        super().__init__()
        self.backbone = self.get_backbone(backbone, pretrained)
        self.head = torch.nn.Linear(1000, output_dim)
        #self.head = torch.nn.Sequential(*[
        #    torch.nn.Linear(1000,64),
        #    torch.nn.ReLU(),
        #    torch.nn.Dropout(0.2),
        #    torch.nn.Linear(64,classes),
        #])
        print("parameters=", sum(parameter.numel() for parameter in self.parameters()))

    def get_backbone(self, backbone:str, pretrained:bool):
        if backbone=='resnet50': # 97.8MB, top-1 76.13, (B,3,224,224) -> (B,1000)
            return resnet50(weights='IMAGENET1K_V1' if pretrained else 'default')
        elif backbone=='denseet121': # 30.8MB, top-1 74.43, (B,3,224,224) -> (B,1000)
            return densenet121(weights='IMAGENET1K_V1' if pretrained else 'default')
        elif backbone=='ViT_b_16': # 330.3MB, top-1 81.07, (B,3,224,224) -> (B,1000)
            return vit_b_16(weights='IMAGENET1K_V1' if pretrained else 'default')
        else:
            raise "UnKnown Backbone"
           
    def forward(self, x): # (B,3,224,224) -> (B,output_dim)
        x = self.backbone(x)
        x = self.head(x)
        return x


class MixCE(torch.nn.Module):
    def __init__(self, fine_dim:int, coarse_dim:int, lambda_const=0.5, \
            fine_weights=None, coarse_weights=None, reduction='mean'):
        super().__init__()
        self.fine_dim, self.coarse_dim, self.lambda_const = fine_dim, coarse_dim, lambda_const
        self.loss_func_fine = torch.nn.CrossEntropyLoss(weight=fine_weights, reduction=reduction)
        self.loss_func_coarse = torch.nn.CrossEntropyLoss(weight=coarse_weights, reduction=reduction)
        self.fine_history, self.coarse_history = [], []

    def forward(self, pred, y_fine, y_coarse, train=True): # CE:(B,2), (B,)int
        # coarse
        coarse_in = pred[:,:self.coarse_dim].sum(dim=1, keepdim=True)
        coarse_ou = pred[:,self.coarse_dim:].sum(dim=1, keepdim=True)
        coarse = torch.cat((coarse_in, coarse_ou), dim=1)
        loss_coarse = self.loss_func_coarse(coarse, y_coarse)

        # fine  
        if train:
            loss_fine = self.loss_func_fine(pred, y_fine)
            # history for balancing
            self.fine_history.append( float(loss_fine.cpu().detach().numpy()) )
            self.coarse_history.append( float(loss_coarse.cpu().detach().numpy()) )
        else:
            ext = torch.zeros((pred.shape[0],pred.shape[1]-self.coarse_dim-1), dtype=torch.float32).to('cuda')
            npred = torch.cat((pred[:,:self.coarse_dim], coarse_ou, ext), dim=1)
            loss_fine = self.loss_func_fine(npred, y_fine)
        
        return loss_fine + loss_coarse * self.lambda_const


def get_optimizer(model, optim_algo:str):
    if optim_algo.lower()=='adam':
        epochs, lr_scheduler = 500, 'linear'
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        lr_schedulerD = {\
            "none": lambda x: x,
            "linear": lambda x: (1 - x / (epochs - 1)) * (1.0 - 0.1) + 0.1,
            "sine": lambda x: 1 - 0.9 * math.sin(0.5*math.pi*x/epochs)
        }
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedulerD[lr_scheduler])
        return optimizer, scheduler
    elif optim_algo.lower()=='adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=2e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        return optimizer, scheduler


class History:
    def __init__(self, save_results:str):
        self.train_loss, self.train_f1, self.train_aps, self.train_map = [], [], [], []
        self.valid_loss, self.valid_f1, self.valid_aps, self.valid_map = [], [], [], []
        self.save_results = save_results

    def __repr__(self):
        D = {}
        for key in ['train_loss', 'train_f1', 'train_map', 'valid_loss', 'valid_f1', 'valid_map']:
            D[key] = round(getattr(self,key)[-1],4) if len(getattr(self,key)) else np.nan
        return str(D)
        
    def save(self, mode='train'):
        results = {key:getattr(self,key) for key in dir(self) if 'train_' in key or 'valid_' in key}
        json.dump(results, open(os.path.join(self.save_results, f'history_{mode}.json'),'w'))


class ComputeMetrics:
    """
    label: np.array[int], shape=(N,)
    pred_probs: np.array[float], shape=(N, cls)
    # for single score -> concat 1-p and 1-p first
    # for unbounded score -> normalize first
    """
    def __init__(self, label, pred_probs, threshold_optimization=False):
        self.label = label
        self.pred_probs = pred_probs
        self.classes = pred_probs.shape[-1]
        if not threshold_optimization:
            self.pred_cls = pred_probs.argmax(axis=1)
            print(f"\ndefault_threshold={1/self.classes:.4f}")
        else:
            best_threshold = self.threshold_optimization()
            print(f"\nbest_threshold={best_threshold:.4f}")
            self.pred_cls = np.array([ row[:-1].argmax() if row.max()>=best_threshold else self.classes-1 \
                for row in self.pred_probs ])

    def threshold_optimization(self, strategy='f1'):
        best_threshold_cls = []
        for i in range(self.classes-1):
            precision, recall, thresholds = precision_recall_curve(self.label==i, self.pred_probs[:,i])
            if strategy=='f1':
                f1 = np.array([ 2*p*r/(p+r) if p+r else 0 for p,r in zip(precision,recall) ])
                best_threshold_cls.append( thresholds[f1.argmax()] )
        return sum(best_threshold_cls)/(self.classes-1)

    def get_f1(self):
        return f1_score(self.label, self.pred_cls, average='macro')
    
    def get_aps(self):
        aps = [ average_precision_score(self.label==i, self.pred_probs[:,i]) for i in range(self.classes) ]
        return aps

    def get_cls_report(self):
        return classification_report(self.label, self.pred_cls, zero_division=0)

    def get_aucs_specificities(self):
        aucs, specificities = [], []
        for i in range(self.classes):
            aucs.append( roc_auc_score(self.label==i, self.pred_probs[:,i]) )
            fpr, tpr, thresholds = roc_curve(self.label==i, self.pred_probs[:,i])
            specificities.append( 1-fpr.mean() )
        return aucs, specificities
    
    def get_confusion(self, path_list=[], losses=[]):
        confusion = [ [ [] for _ in range(self.classes) ] for _ in range(self.classes) ]
        path_list = path_list if path_list else ['']*len(self.label)
        losses = losses if losses else [-1]*len(self.label)
        for gt, pdc, path, loss in zip(self.label, self.pred_cls, path_list, losses):
            confusion[gt][pdc].append( (loss,path) )
        confusion_cnt = [ [ len(confusion[i][j]) for j in range(self.classes) ] for i in range(self.classes) ]
        return confusion, confusion_cnt
    
    def export_confusion(self, confusion, output_path, top_n=5):
        for i in range(self.classes):
            for j in range(self.classes):
                if i==j: continue
                grid_path = os.path.join(output_path, 'confusion', f"gt_{i}_pd_{j}")
                for _, path in sorted(confusion[i][j])[:top_n]:
                    os.makedirs(grid_path, exist_ok=True)
                    shutil.copy(path, grid_path)

    def export_lowest_conf(self, path_list, output_path, top_n=5):
        prob_path_list = sorted(zip(self.pred_probs.max(axis=1), path_list))
        worst_path = f"{output_path}/worst_imgs"
        os.makedirs(worst_path, exist_ok=True)
        for _, path in prob_path_list[:top_n]:
            shutil.copy(path, worst_path)


def get_prf_pr_data(label, pred_probs):
    precision_list, recall_list, f1_list, threshold_list = [], [], [], [] # 2-d list # A[class_i][val]
    for i in range(pred_probs.shape[-1]):
        precision, recall, thresholds = precision_recall_curve(label==i, pred_probs[:,i])
        refine_precision = [precision[0]]
        for j in range(1,len(precision)):
            refine_precision.append( max(refine_precision[-1],precision[j]) )
        # collect data
        precision_list.append(refine_precision)
        recall_list.append(recall)
        f1_list.append([ 2*p*r/(p+r) if p+r else 0 for p, r in zip(refine_precision, recall) ])
        threshold_list.append( np.concatenate((thresholds, np.array([1.]))) )
    return precision_list, recall_list, f1_list, threshold_list


def get_roc_data(label, pred_probs):
    roc_subplots_x, roc_subplots_y = [], [] # 3-d list # A[class_i][fpr][val], A[class_i][tpr][val] 
    for i in range(pred_probs.shape[-1]):
        fpr, tpr, thresholds = roc_curve(label==i, pred_probs[:,i])
        roc_subplots_x.append([fpr])
        roc_subplots_y.append([tpr])
    return roc_subplots_x, roc_subplots_y


def row_plot_1d(data, xlabel_list, ylabel_list, legend_list, save_path):
    """
    data: 3-d list
        1st-d subplot e.g. loss, f1, map
        2nd-d curves e.g. train, valid
    xlabel_list: list[str] e.g. ['epoch']*3
    ylabel_list: list[str] e.g. ['loss', 'f1', 'map']
    legend_list: list[list[str]] e.g. [['train','valid'] for _ in range(3)]
    save_path: str e.g. *.jpg
    """
    n = len(data)
    plt.figure(figsize=(6*n,4))
    zip_gen = zip(data, xlabel_list, ylabel_list, legend_list)
    for i, (curves, xlabel, ylabel, legend) in enumerate(zip_gen):
        plt.subplot(1,n,1+i)
        for curve in curves:
            plt.plot(curve)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(legend) if legend is not None else None
    plt.savefig(save_path)


def row_plot_2d(data_x, data_y, xlabel_list, ylabel_list, legend_list, save_path):
    """
    similar to row_plot_1d, but plot x and y
    """
    n = len(data_x)
    plt.figure(figsize=(6*n,4))
    zip_gen = zip(data_x, data_y, xlabel_list, ylabel_list, legend_list)
    for i, (curves_x, curves_y, xlabel, ylabel, legend) in enumerate(zip_gen):
        plt.subplot(1,n,1+i)
        for curve_x, curve_y, label in zip(curves_x, curves_y, legend):
            plt.plot(curve_x, curve_y, label=label)
            plt.scatter(curve_x, curve_y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(-0.1,1.1)
        plt.ylim(-0.1,1.1)
        plt.legend()
    plt.savefig(save_path)