import dgl
import torch
import torch.nn.functional as F
import numpy
import argparse
import time
from collections import Counter
from dataset import Dataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix
from BWGNN import *
from sklearn.model_selection import train_test_split


def train(model, g, args,node_lens,key_ids,key_label,train_ids,val_ids):
    features = g.ndata['feature']
    labels = g.ndata['label']
    index = list(range(len(labels)))

    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                            train_size=args.train_ratio,
                                                            random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=2, shuffle=True)
    train_mask = torch.zeros([len(labels)]).bool()
    val_mask = torch.zeros([len(labels)]).bool()
    test_mask = torch.zeros([len(labels)]).bool()

    lst = list(range(node_lens[2]))
    
    train_mask[lst[0:node_lens[0]]] = 1
    val_mask[lst[node_lens[0]:node_lens[1]]] = 1
    test_mask[lst[node_lens[1]:]] = 1

    
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.

#     weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    weight = (1-labels[train_ids]).sum().item() / labels[train_ids].sum().item()
    print('cross entropy weight: ', weight)
    time_start = time.time()
    hight=0.0
    for e in range(args.epoch):
        model.train()
        logits = model(features)
        print(logits.shape)
        loss = F.cross_entropy(logits[train_ids], labels[train_ids], weight=torch.tensor([1., weight]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        probs = logits.softmax(1)
#         f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
        f1, thres = get_best_f1(labels[val_ids], probs[val_ids])
        preds = numpy.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        
        test_preds = preds[test_mask]
        print(test_preds)

        key_pred=test_preds[key_ids].tolist()
        print("Test set predictions:", key_pred)
        print("Lebels:              ",key_label)
        total=len(key_pred)
        right_bit = sum(1 for a, b in zip(key_pred, key_label) if a == b)
#         for udx in range(total):
#             if key_pred[udx]==key_label[udx]:
#                 rigth_bit=right_bit+1
        print(right_bit)
        acc=right_bit/total
        if acc>hight:
            hight=acc
        print("accuracy: "+str(acc))
#         print("Test set probabilities:", test_probs)

        trec = recall_score(labels[test_mask], preds[test_mask])
        tpre = precision_score(labels[test_mask], preds[test_mask])
        tmf1 = f1_score(labels[test_mask], preds[test_mask], average='macro')
        tauc = roc_auc_score(labels[test_mask], probs[test_mask][:, 1].detach().numpy())

        if best_f1 < f1:
            best_f1 = f1
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc
        print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(e, loss, f1, best_f1))

    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')
    print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(final_trec*100,
                                                                     final_tpre*100, final_tmf1*100, final_tauc*100))
    print(hight)
    return final_tmf1, final_tauc


# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BWGNN')
    parser.add_argument("--dataset", type=str, default="amazon",
                        help="Dataset for this model (yelp/amazon/tfinance/tsocial)")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=1, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=1, help="Running times")
    parser.add_argument("--lockedName", type=str, default="TRLL", help="Name of the lock")
    parser.add_argument("--circuit", type=str, default="c1355", help="Circuit name")
    parser.add_argument("--test_start", type=int, required=True, help="Starting index for test set")
    parser.add_argument("--val_start", type=int, required=True, help="Starting index for validation set")
    parser.add_argument("--lenss", type=int, required=True, help="Total data length")
    parser.add_argument("--dataSetType", type=str, required=True, help="Dataset type directory")

    args = parser.parse_args()
    print(args)
    dataset_name = args.dataset
    homo = args.homo
    order = args.order
    h_feats = args.hid_dim

    node_lens=[]
    edge_lens=[]
    node_c=0
    edge_c=0
    lockedName=args.lockedName
    circuit=args.circuit   #c2670
    total_edges_1=None
    total_edges_2=None
    total_feats=None
    total_labels=None
    key_label=[]
    key_ids=[]
    key_base=0
    
    train_ids=[]
    val_ids=[]
    #处理训练数据
    test_start=args.test_start
    val_start=args.val_start
    lenss=args.lenss
    dataSetType=args.dataSetType

    for ii in range(0,lenss):
        edges = np.loadtxt(dataSetType+'/'+lockedName+'/'+circuit+'/'+str(ii)+'/edges.txt',dtype=int)
        edges_tensor = torch.tensor(edges,dtype=torch.int64)
        for x in range(edges_tensor[0].shape[0]):
            edges_tensor[0][x]=edges_tensor[0][x]+node_c
            edges_tensor[1][x]=edges_tensor[1][x]+node_c
        if total_edges_1 is None:
            total_edges_1=edges_tensor[0]
            total_edges_2=edges_tensor[1]
        else:
            total_edges_1=torch.cat((total_edges_1,edges_tensor[0]))
            total_edges_2=torch.cat((total_edges_2,edges_tensor[1]))
        x=np.loadtxt(dataSetType+'/'+lockedName+'/'+circuit+'/'+str(ii)+'/feat.txt',dtype=float)
        x=torch.tensor(x,dtype=torch.float32)
        if total_feats is None:
            total_feats=x
        else:
            total_feats=torch.cat((total_feats,x))
        labels=np.loadtxt(dataSetType+'/'+lockedName+'/'+circuit+'/'+str(ii)+'/label.txt',dtype=int)
        labels=torch.tensor(labels,dtype=torch.int64)
        if total_labels is None:
            total_labels=labels
        else:
            total_labels=torch.cat((total_labels,labels))

        if ii>=test_start:
            key=np.loadtxt(dataSetType+'/'+lockedName+'/'+circuit+'/'+str(ii)+'/keys.txt',dtype=int)
            key=torch.tensor(key,dtype=torch.int)
            for iii in range(key.shape[0]):
                key_label.append(labels[key[iii].item()].item())
                key_ids.append(key[iii].item()+key_base)
            key_base=key_base+labels.shape[0]
        elif ii>=val_start and ii<test_start:
            key=np.loadtxt(dataSetType+'/'+lockedName+'/'+circuit+'/'+str(ii)+'/keys.txt',dtype=int)
            key=torch.tensor(key,dtype=torch.int)
            for iii in range(key.shape[0]):
                val_ids.append(key[iii].item()+node_c)
        elif ii<val_start:
            key=np.loadtxt(dataSetType+'/'+lockedName+'/'+circuit+'/'+str(ii)+'/keys.txt',dtype=int)
            key=torch.tensor(key,dtype=torch.int)
            for iii in range(key.shape[0]):
                train_ids.append(key[iii].item()+node_c)
        
        node_c=node_c+labels.shape[0]
        edge_c=edge_c+edges_tensor[0].shape[0]
        
        if ii==val_start-1 or ii==test_start-1 or ii==lenss-1:
            node_lens.append(node_c)
            
    graph = dgl.graph((total_edges_1, total_edges_2))
    graph.ndata['feature']=total_feats
    graph.ndata['label']=total_labels
    
    graph.ndata['train_mask']=torch.zeros(graph.ndata['label'].shape[0], dtype=torch.int8)
    graph.ndata['val_mask']=torch.zeros(graph.ndata['label'].shape[0], dtype=torch.int8)
    graph.ndata['test_mask']=torch.zeros(graph.ndata['label'].shape[0], dtype=torch.int8)
    graph.ndata['_ID']=torch.arange(graph.ndata['label'].shape[0])
    graph.edata['_TYPE']=torch.zeros(edge_c,dtype=torch.int64)
    graph.edata['_ID']=torch.arange(edge_c)
    
    print(node_lens)
    print(node_c)
    print(graph.ndata['label'].shape)
    print(graph.ndata['feature'].shape)

    in_feats = graph.ndata['feature'].shape[1]
    num_classes = 2
    

    if args.run == 1:
        if homo:
            model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
        else:
            model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)
        train(model, graph, args,node_lens,key_ids,key_label,train_ids,val_ids)

    else:
        final_mf1s, final_aucs = [], []
        for tt in range(args.run):
            if homo:
                model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
            else:
                model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)
            mf1, auc = train(model, graph, args,node_lens,key_ids,key_label,train_ids,val_ids)
            final_mf1s.append(mf1)
            final_aucs.append(auc)
        final_mf1s = np.array(final_mf1s)
        final_aucs = np.array(final_aucs)
        print('MF1-mean: {:.2f}, MF1-std: {:.2f}, AUC-mean: {:.2f}, AUC-std: {:.2f}'.format(100 * np.mean(final_mf1s),
                                                                                            100 * np.std(final_mf1s),
                                                               100 * np.mean(final_aucs), 100 * np.std(final_aucs)))
