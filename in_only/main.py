import argparse, glob, json, random, os

import numpy as np
import pandas as pd
import torch

import utils

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='train', choices=['train', 'valid', 'infer'])
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--output-dim", type=int, default=7+17, help="network output dim")
parser.add_argument("--fine-dim", type=int, default=7+0, help="in+out dim of trainset")
parser.add_argument("--coarse-dim", type=int, default=7, help="in dim")
parser.add_argument("--resume", type=str, default='', help="checkpoint path")
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--results", type=str, default="./results/exp1", help="save results and models folder")
parser.add_argument("--threshold-opt", action='store_true', help="threshold opt refers last col as bg")
args = parser.parse_args()
args.backbone = 'resnet50'
args.pretrained = True
args.lambda_const = 0
args.optim_algo = 'adamw'
print(args)

# global setting
os.makedirs(args.results, exist_ok=True)
json.dump(vars(args), open(f"{args.results}/args_{args.mode}.json", "w"))
classes = args.coarse_dim + 1

# check GPU
print(torch.cuda.is_available(), torch.backends.cudnn.is_available(), torch.cuda.get_device_name(0))
device = torch.device('cuda')

# dataset
if 1:
    """
            train_loader        valid_loader        test_loader
    train:  in:csv_fold!=5      in:csv_fold==5      -
            ou:4cls             ou:4cls_others
    valid:  -                   same as above       -
                                same as above
    infer:  -                   -                   custom
    """
    # load in
    df = pd.read_csv("/volume/my-volume/itch/my_pruritus/clinical/annotation_train_round1_7class_bypat.csv")
    df_train, df_valid = df[df['fold_index']<5], df[df['fold_index']>=5]

    # load out
    out_train_path = sorted(glob.glob("../my_pruritus/ood_data/out_rel/[0-3]-*"))
    out_valid_path = sorted(glob.glob("../my_pruritus/ood_data/out_rel/[4-7]-*"))
    out_n = [ len(glob.glob(f"../my_pruritus/ood_data/out_rel/{i}-*")) for i in range(8) ] # [100]*8

    # merge
    train_path = list(df_train['data']) #+ out_train_path
    train_fine_label = list(df_train['label']) #+ [args.coarse_dim]*out_n[0] + [args.coarse_dim+1]*out_n[1]\
        #+ [args.coarse_dim+2]*out_n[2] + [args.coarse_dim+3]*out_n[3]
    train_coarse_label = [0]*len(df_train['label']) #+ [1]*sum(out_n[:4])
    valid_path = list(df_valid['data']) #+ out_valid_path
    valid_fine_label = list(df_valid['label']) #+ [args.coarse_dim]*sum(out_n[4:])
    valid_coarse_label = [0]*len(df_valid['label']) #+ [1]*sum(out_n[4:])
    print(len(train_path), len(train_fine_label), len(train_coarse_label),\
        len(valid_path), len(valid_fine_label), len(valid_coarse_label)) # 1458, 1458, 1458, 572, 572, 572
if args.mode == 'train':
    train_loader = utils.get_loader(train_path, train_fine_label, train_coarse_label, 'train', args.batch_size)
if args.mode in ('train', 'valid'):
    valid_loader = utils.get_loader(valid_path, valid_fine_label, valid_coarse_label, 'valid', args.batch_size)
if args.mode == 'infer':
    raise

# loss weights
if args.mode == 'train':
    _, cnts = np.unique(train_loader.dataset.label_fine_list, return_counts=True)
    fine_weights_in = torch.tensor(1/cnts/(1/cnts).sum(), dtype=torch.float32)
    fine_weights_ou = torch.tensor([0]*(args.output_dim-args.fine_dim), dtype=torch.float32)
    fine_weights = torch.cat((fine_weights_in, fine_weights_ou)).to(device)
    coarse_weights = None
        #torch.tensor([cnts[args.coarse_dim:].sum()/cnts.sum(), \
        #cnts[:args.coarse_dim].sum()/cnts.sum()], dtype=torch.float32).to(device)
else:
    fine_weights = coarse_weights = None
print(f"fine_weights={fine_weights}")
print(f"coarse_weights={coarse_weights}")

# model
model = utils.MyModel(args.backbone, args.pretrained, args.output_dim)
if args.resume:
    model.load_state_dict(torch.load(args.resume))
else:
    assert args.mode=='train', f"{args.mode} needs pretrained weights"
model.to(device)

# loss
loss_func = utils.MixCE(args.fine_dim, args.coarse_dim, args.lambda_const, \
    fine_weights, coarse_weights, 'mean' if args.mode=='train' else 'none')

# optimizer
optimizer, scheduler = utils.get_optimizer(model, args.optim_algo)

# training
history = utils.History(args.results)
if args.mode=='train':
    train_label = train_loader.dataset.label_fine_list.clip(max=args.coarse_dim)
if args.mode in ('valid', 'infer'):
    loss_all = []
valid_label = valid_loader.dataset.label_fine_list.clip(max=args.coarse_dim)
for ep in range(args.epochs):
    print(f"Epoch: {ep+1}/{args.epochs}") if args.mode=='train' else None
    
    # training loop
    model.train()
    if args.mode=='train':
        pred_probs_all = []
        history.train_loss.append(0)
        for i, (x, y_fine, y_coarse) in enumerate(train_loader):
            print(f"\rbatch={i+1}/{len(train_loader)}, train_loss={history.train_loss[-1]:.5f}", end="")
            
            # basic
            x, y_fine, y_coarse = x.to(device), y_fine.to(device), y_coarse.to(device)
            y_fine, y_coarse = y_fine.reshape(-1), y_coarse.reshape(-1)
            optimizer.zero_grad()
            pred = model(x) #print(torch.nn.functional.softmax(pred[:1,:],dim=1))
            loss = loss_func(pred, y_fine, y_coarse) # CE:(B,2),(B,)int
            loss.backward()
            optimizer.step()

            # history loss
            history.train_loss[-1] += loss.item() / len(train_label)
            
            # pred collect
            pred_probs = torch.nn.functional.softmax(pred, dim=1)
            pred_probs_all += pred_probs.cpu().detach().numpy().tolist()

        # pred numpy
        scheduler.step()
        pred_probs_all = np.array(pred_probs_all)
        fg = pred_probs_all[:,:args.coarse_dim]
        bg = pred_probs_all[:,args.coarse_dim:].sum(axis=1, keepdims=True)
        pred_probs_all = np.concatenate((fg, bg), axis=1)
        
        # history f1 & aps & map
        metrics = utils.ComputeMetrics(train_label, pred_probs_all)
        history.train_f1.append(metrics.get_f1())
        history.train_aps.append(metrics.get_aps())
        history.train_map.append(sum(history.train_aps[-1])/classes)
        print("\n", history, "\n", metrics.get_cls_report())

    # validation loop
    model.eval()
    with torch.no_grad():
        pred_probs_all = []
        history.valid_loss.append(0)
        for i, (x, y_fine, y_coarse) in enumerate(valid_loader):
            print(f"\rbatch={i+1}/{len(valid_loader)}, valid_loss={history.valid_loss[-1]:.5f}", end="")
            
            # basic
            x, y_fine, y_coarse = x.to(device), y_fine.to(device), y_coarse.to(device)
            y_fine, y_coarse = y_fine.reshape(-1), y_coarse.reshape(-1)
            pred = model(x) # print(torch.nn.functional.softmax(pred[:1,:],dim=1))
            loss = loss_func(pred, y_fine, y_coarse, False) # CE:(B,2),(B,)int

            # history loss
            if args.mode != 'train':
                loss_all += loss.cpu().detach().numpy().tolist()
                loss = loss.mean(axis=0)
            history.valid_loss[-1] += loss.item() / len(valid_loader.dataset)
            
            # pred collect
            pred_probs = torch.nn.functional.softmax(pred, dim=1)
            pred_probs_all += pred_probs.cpu().detach().numpy().tolist()
        
        # pred numpy
        pred_probs_all = np.array(pred_probs_all)
        fg = pred_probs_all[:,:args.coarse_dim]
        bg = pred_probs_all[:,args.coarse_dim:].sum(axis=1, keepdims=True)
        pred_probs_all = np.concatenate((fg, bg), axis=1)

        # history f1 & aps & map
        threshold_optimization = args.mode=='valid' and args.threshold_opt
        metrics = utils.ComputeMetrics(valid_label, pred_probs_all, threshold_optimization)
        history.valid_f1.append(metrics.get_f1())
        history.valid_aps.append(metrics.get_aps())
        history.valid_map.append(sum(history.valid_aps[-1])/classes)
        print("\n", history, "\n", metrics.get_cls_report() )

        if args.mode=='train':
            # checkpoint
            if ep==0 or history.valid_map[-1]>history.valid_map[-2]:
                torch.save(model.state_dict(), os.path.join(args.results, 'model.pt'))
            history.save()
            #fcr = [ (float(t1.cpu().detach().numpy()), \
            #        float(t2.cpu().detach().numpy())) for t1,t2 in loss_func.fine_coarse_rate ]
            #batches = len(train_loader)
            #fcr = [ sum(fcr[i*batches:(i+1)*batches])/batches for i in range(len(fcr)//batches) ]
            #json.dump(fcr, open(os.path.join(args.results, 'fc.json'),'w'))
        
        elif args.mode=='valid':
            print("aps=", history.valid_aps[-1])
            history.save('valid')

            # auc & sensitivity
            aucs, specificities = metrics.get_aucs_specificities()
            print(f"aucs={aucs}, auc_mean={sum(aucs)/classes}")
            print(f"specificities={specificities}, specificity_mean={sum(specificities)/classes}")

            # confusion matrixby max prob + false_imgs_top-N_loss,
            confusion, confusion_cnt = metrics.get_confusion(valid_loader.dataset.path_list, loss_all)
            print(f"confusion_cnt={confusion_cnt}")
            metrics.export_confusion(confusion, args.results)
            break
        
        else:
            # export worst
            metrics.export_lowest_conf(valid_loader.dataset.path_list, args.results)
            break

# save prediction results
df = pd.DataFrame({
    "data": valid_loader.dataset.path_list,
    "label": valid_label, 
    "pred_probs_all": map(tuple,pred_probs_all),
    "pred_cls_all": pred_probs_all.max(axis=1),
})
df.to_csv(os.path.join(args.results, f'pred_{args.mode}.csv'), index=False)

# plot results
if args.mode=='train':
    # loss, f1, map x ep
    data_sub1 = [history.train_loss, history.valid_loss]
    data_sub2 = [history.train_f1, history.valid_f1]
    data_sub3 = [history.train_map, history.valid_map]
    utils.row_plot_1d( [data_sub1, data_sub2, data_sub3], ['epoch']*3, ['loss','f1','mAP'], \
        [['train','valid'] for i in range(3)], os.path.join(args.results, "curve_loss_map.jpg") )

elif args.mode=='valid':
    precision_list, recall_list, f1_list, threshold_list = utils.get_prf_pr_data(valid_label, pred_probs_all) # p,r,f,t | cls
    # first plot: p/r/f curve per class
    prf_subplots_x = [ [threshold_list[i]]*3 for i in range(classes) ]
    prf_subplots_y = [ [precision_list[i], recall_list[i], f1_list[i]] for i in range(classes) ]    
    utils.row_plot_2d(prf_subplots_x, prf_subplots_y, ['threshold']*classes, ['']*classes, [['precision','recall','f1'] for _ in range(classes)], \
        os.path.join(args.results, "curve_prf.jpg") )
    # second_plot: p-r curve per class
    pr_subplots_x = [ [recall_list[i]] for i in range(classes) ]
    pr_subplots_y = [ [precision_list[i]] for i in range(classes) ]
    utils.row_plot_2d(pr_subplots_x, pr_subplots_y, ['recall']*classes, ['precision']*classes, [['.'] for _ in range(classes)], \
        os.path.join(args.results, "curve_pr.jpg") )

    # third plot: roc curve
    roc_subplots_x, roc_subplots_y = utils.get_roc_data(valid_label, pred_probs_all)
    utils.row_plot_2d(roc_subplots_x, roc_subplots_y, ['fpr']*classes, ['tpr']*classes, [['.'] for _ in range(classes)], \
        os.path.join(args.results, "curve_roc.jpg") )

print("successfully finished :D")