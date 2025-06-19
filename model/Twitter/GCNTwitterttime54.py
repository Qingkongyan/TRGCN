import sys,os
sys.path.append(os.getcwd())
from Process.process import *
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
from torch_geometric.nn import GCNConv
import copy
from multiprocessing import freeze_support





import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class TimeAwarePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(TimeAwarePositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, time_features):
        # 根据时间特征调整位置编码
        time_features = (time_features * self.pe.size(0)).long()
        pos_encoding = self.pe[time_features]
        x = x + pos_encoding
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)
        self.W_o = nn.Linear(input_dim, input_dim)

    def scaled_dot_product_attention(self, Q, K, V):
        d_k = Q.size(-1)
        scores = th.matmul(Q, K.transpose(-2, -1)) / th.sqrt(th.tensor(d_k, dtype=th.float32))
        attention_weights = F.softmax(scores, dim=-1)
        output = th.matmul(attention_weights, V)
        return output, attention_weights

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(Q, K, V)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        scaled_attention = scaled_attention.view(batch_size, seq_length, self.input_dim)

        output = self.W_o(scaled_attention)

        return output, attention_weights

class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(input_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),  # 增大FFN隐藏层维度
            nn.ReLU(),
            nn.Linear(4 * input_dim, input_dim)
        )

    def forward(self, x):
        attn_output, _ = self.attention(x)
        x = self.layer_norm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        return x

class TDrumorGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_heads):
        super(TDrumorGCN, self).__init__()
        self.embedding = nn.Linear(in_feats, hid_feats)
        self.tanh = nn.Tanh()
        self.conv1 = GCNConv(hid_feats, hid_feats)
        self.transformer_encoder = nn.ModuleList(
            [TransformerEncoderLayer(hid_feats, num_heads) for _ in range(2)]  # 增加Transformer编码器层数
        )
        self.conv2 = GCNConv(hid_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embedding(x)
        x = self.tanh(x)

        residual = x
        x = self.conv1(x, edge_index)
        x = F.relu(x, inplace=False)
        x = F.dropout(x, training=self.training, p=0.2)  # 增加Dropout率
        x = x + residual

        batch_size = data.batch.max().item() + 1
        num_nodes = data.batch.bincount().tolist()
        max_nodes = max(num_nodes)

        padded_x = th.zeros(batch_size, max_nodes, x.size(-1), device=x.device)
        start = 0
        for i, nodes in enumerate(num_nodes):
            padded_x[i, :nodes] = x[start:start + nodes]
            start += nodes

        for encoder in self.transformer_encoder:
            padded_x = encoder(padded_x)

        x = th.cat([padded_x[i, :nodes] for i, nodes in enumerate(num_nodes)], dim=0)

        residual = x
        x = self.conv2(x, edge_index)
        x = F.relu(x, inplace=False)
        x = F.dropout(x, training=self.training, p=0.2)  # 增加Dropout率
        x = x + residual

        x = scatter_mean(x, data.batch, dim=0)
        return x

class Net(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_heads=4):
        super(Net, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats, num_heads)
        self.fc = nn.Linear(out_feats, 4)
        self.layer_norm = nn.LayerNorm(4)

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        x = self.fc(TD_x)
        x = self.layer_norm(x)
        x = F.log_softmax(x, dim=1)
        return x






def train_GCN(treeDic, x_test, x_train,TDdroprate,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter):
    model = Net(5000,64,64).to(device)
    BU_params=list(map(id,model.TDrumorGCN.conv1.parameters()))
    BU_params += list(map(id, model.TDrumorGCN.conv2.parameters()))
    base_params=filter(lambda p:id(p) not in BU_params,model.parameters())
    optimizer = th.optim.Adam([
        {'params':base_params},
        {'params':model.TDrumorGCN.conv1.parameters(),'lr':lr/5},
        {'params': model.TDrumorGCN.conv2.parameters(), 'lr': lr/5}
    ], lr=lr, weight_decay=weight_decay)
    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(n_epochs):
        traindata_list, testdata_list = loadOneData(dataname, treeDic, x_train, x_test, TDdroprate)
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        avg_loss = []
        avg_acc = []
        #batch_idx = 0
        #tqdm_train_loader = tqdm(train_loader)
        for Batch_data in train_loader:
            Batch_data.to(device)
            out_labels= model(Batch_data)
            finalloss=F.nll_loss(out_labels,Batch_data.y)
            loss=finalloss
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(Batch_data.y).sum().item()
            train_acc = correct / len(Batch_data.y)
            avg_acc.append(train_acc)
            #print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,
                                                                                                 #loss.item(),
                                                                                                 #train_acc))
        print("Iter {:03d} | Epoch {:05d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch,
            loss.item(),train_acc))
            #batch_idx = batch_idx + 1

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval()
        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            Batch_data.to(device)
            val_out = model(Batch_data)
            val_loss  = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y)
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, Batch_data.y)
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), \
            temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3), \
            temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))

        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        print('results:', res)
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                       np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'BiGCN', dataname)
        accs =np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        if early_stopping.early_stop:
            print("Early stopping")
            accs=early_stopping.accs
            F1=early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break
    return train_losses , val_losses ,train_accs, val_accs,accs,F1,F2,F3,F4

def text_create(name):
    path="C:\\Users\\aaa\\Desktop\\tweet15\\"
    full_path=path+name+'.txt'
    file=open(full_path,'w')

if __name__ == '__main__':
    freeze_support()
    lr=0.0005
    weight_decay=1e-4
    patience=10
    n_epochs=100
    batchsize=64
    TDdroprate=0.2
    BUdroprate=0.2
    #datasetname=sys.argv[1] #"Twitter15"、"Twitter16"
    datasetname="Twitter15"
    #iterations=int(sys.argv[2])
    iterations=10
    model="GCN"
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    test_accs = []
    NR_F1 = []
    FR_F1 = []
    TR_F1 = []
    UR_F1 = []
    for iter in range(iterations):
        fold0_x_test, fold0_x_train, \
        fold1_x_test,  fold1_x_train,  \
        fold2_x_test, fold2_x_train, \
        fold3_x_test, fold3_x_train, \
        fold4_x_test,fold4_x_train = load5foldData(datasetname)
        treeDic=loadTree(datasetname)
        train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = train_GCN(treeDic,
                                                                                                   fold0_x_test,
                                                                                                   fold0_x_train,
                                                                                                   TDdroprate,
                                                                                                   lr, weight_decay,
                                                                                                   patience,
                                                                                                   n_epochs,
                                                                                                   batchsize,
                                                                                                   datasetname,
                                                                                                   iter)
        train_losses, val_losses, train_accs, val_accs1, accs1, F1_1, F2_1, F3_1, F4_1 = train_GCN(treeDic,
                                                                                                   fold1_x_test,
                                                                                                   fold1_x_train,
                                                                                                   TDdroprate, lr,
                                                                                                   weight_decay,
                                                                                                   patience,
                                                                                                   n_epochs,
                                                                                                   batchsize,
                                                                                                   datasetname,
                                                                                                   iter)
        train_losses, val_losses, train_accs, val_accs2, accs2, F1_2, F2_2, F3_2, F4_2 = train_GCN(treeDic,
                                                                                                   fold2_x_test,
                                                                                                   fold2_x_train,
                                                                                                   TDdroprate, lr,
                                                                                                   weight_decay,
                                                                                                   patience,
                                                                                                   n_epochs,
                                                                                                   batchsize,
                                                                                                   datasetname,
                                                                                                   iter)
        train_losses, val_losses, train_accs, val_accs3, accs3, F1_3, F2_3, F3_3, F4_3 = train_GCN(treeDic,
                                                                                                   fold3_x_test,
                                                                                                   fold3_x_train,
                                                                                                   TDdroprate, lr,
                                                                                                   weight_decay,
                                                                                                   patience,
                                                                                                   n_epochs,
                                                                                                   batchsize,
                                                                                                   datasetname,
                                                                                                   iter)
        train_losses, val_losses, train_accs, val_accs4, accs4, F1_4, F2_4, F3_4, F4_4 = train_GCN(treeDic,
                                                                                                   fold4_x_test,
                                                                                                   fold4_x_train,
                                                                                                   TDdroprate, lr,
                                                                                                   weight_decay,
                                                                                                   patience,
                                                                                                   n_epochs,
                                                                                                   batchsize,
                                                                                                   datasetname,
                                                                                                   iter)
        test_accs.append((accs0+accs1+accs2+accs3+accs4)/5)
        NR_F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
        FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
        TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
        UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
    print("Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
        sum(test_accs) / iterations, sum(NR_F1) /iterations, sum(FR_F1) /iterations, sum(TR_F1) / iterations, sum(UR_F1) / iterations))


