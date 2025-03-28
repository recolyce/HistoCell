import torch
from torch import nn
from torchvision.models import resnet18
import torch.nn.functional as F
from einops import rearrange
    

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W) # h.shape: (B, N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e) # B N N
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(-1, -2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
    

class HistoCell(nn.Module):
    def __init__(self, config) -> None:
        super(HistoCell, self).__init__()
        resnet = resnet18(pretrained=config.pretrained)
        
        if config.channels == 1:
            modules = [nn.Conv2d(config.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)] + list(resnet.children())[1:-1]
            self.resnet = nn.Sequential(*modules)

        elif config.channels == 3:
            self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        self.gat = GraphAttentionLayer(512, 512, dropout=0.5, alpha=0.2, concat=False)

        self.size_embed = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )
        self.merge = nn.Linear(512 + 16, 512)

        self.out = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)
        self.predict = nn.Sequential(
            nn.Linear(512, config.k_class),
            nn.Softmax(dim=-1)
        )
        self.tc = nn.Linear(512, config.tissue_class + 1)

    def forward(self, tissue, bag, adj, cell_size, valid_mask, raw_size): # cell number * 3 * 224 * 224
        mask_feats = self.resnet(bag).squeeze().reshape(raw_size['batch'], raw_size['cells'], -1)
        size_feats = self.size_embed(cell_size) # B C 16
        mask_feats = self.merge(torch.concat([mask_feats, size_feats], dim=-1)) # B C 512
        # import ipdb
        # ipdb.set_trace()
        dmask_feats = nn.functional.dropout(mask_feats, p=0.5, training=True)
        graph_feats = self.gat(dmask_feats, adj) # B C 512

        global_feat = self.resnet(tissue).squeeze()
        if len(global_feat.shape) <= 1:
            global_feat = global_feat.unsqueeze(0)

        global_feats = torch.stack([global_feat for _ in range(graph_feats.shape[1])], dim=1)
        seq_feats = torch.stack([global_feats, graph_feats], dim=2) # (BxC) 3 512
        seq_feats = rearrange(seq_feats, 'B C L F-> (B C) L F')
        out_feats, _ = self.out(seq_feats)
        out_feats = rearrange(out_feats, '(B C) L F-> B C L F', B=raw_size['batch'])    # B C 3 512
        dout_feats = nn.functional.dropout(out_feats, p=0.5, training=True)
        
        # tissue
        tissue_cat = self.tc(dout_feats[:, 0, 0])
        # Proportion
        probs = self.predict(dout_feats[:, :, 1]) # 16 64 cell_type
        prop_list, prob_list, cell_features = [], [], []
        for single_probs, valid_index, cell_embedding in zip(probs, valid_mask, dout_feats[:, :, 1]):
            if valid_index <= 0:
                continue
            prop_list.append(torch.mean(single_probs[:valid_index], dim=0))
            prob_list.append(single_probs[:valid_index])
            cell_features.append(cell_embedding[:valid_index])
        
        avg_probs = torch.stack(prop_list, dim=0)

        return prob_list, avg_probs, tissue_cat, cell_features
    

class HistoState(nn.Module):
    def __init__(self, config) -> None:
        super(HistoState, self).__init__()
        resnet = resnet18(pretrained=config.pretrained)
        
        if config.channels == 1:
            modules = [nn.Conv2d(config.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)] + list(resnet.children())[1:-1]
            self.resnet = nn.Sequential(*modules)

        elif config.channels == 3:
            self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        self.gat = GraphAttentionLayer(512, 512, dropout=0.5, alpha=0.2, concat=False)

        self.size_embed = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )
        self.merge = nn.Linear(512 + 16, 512)

        self.out = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)
        self.predict1 = nn.Sequential(
            nn.Linear(512, config.k_class),
            nn.Softmax(dim=-1)
        )
        self.predict2 = nn.Sequential(
            nn.Linear(512, config.k_state),
            nn.Softmax(dim=-1)
        )
        self.tc = nn.Linear(512, config.tissue_class + 1)

    def forward(self, tissue, bag, adj, cell_size, valid_mask, raw_size): # cell number * 3 * 224 * 224
        mask_feats = self.resnet(bag).squeeze().reshape(raw_size['batch'], raw_size['cells'], -1)
        size_feats = self.size_embed(cell_size) # B C 16
        mask_feats = self.merge(torch.concat([mask_feats, size_feats], dim=-1)) # B C 512
        graph_feats = self.gat(mask_feats, adj) # B C 512

        global_feat = self.resnet(tissue).squeeze()
        if len(global_feat.shape) <= 1:
            global_feat = global_feat.unsqueeze(0)

        global_feats = torch.stack([global_feat for _ in range(graph_feats.shape[1])], dim=1)
        seq_feats = torch.stack([global_feats, graph_feats, graph_feats], dim=2) # (BxC) 3 512
        seq_feats = rearrange(seq_feats, 'B C L F-> (B C) L F')
        out_feats, _ = self.out(seq_feats)
        out_feats = rearrange(out_feats, '(B C) L F-> B C L F', B=raw_size['batch'])    # B C 3 512
        
        # tissue
        tissue_cat = self.tc(out_feats[:, 0, 0])
        # Cell Proportion
        type_probs = self.predict1(out_feats[:, :, 1]) # 16 64 cell_type
        type_prop_list, type_prob_list = [], []
        for single_probs, valid_index in zip(type_probs, valid_mask):
            if valid_index <= 0:
                continue
            type_prop_list.append(torch.mean(single_probs[:valid_index], dim=0))
            type_prob_list.append(single_probs[:valid_index])
        
        avg_type_probs = torch.stack(type_prop_list, dim=0)

        state_probs = self.predict2(out_feats[:, :, 2]) # 16 64 cell_type
        state_prop_list, state_prob_list = [], []
        for single_probs, valid_index in zip(state_probs, valid_mask):
            if valid_index <= 0:
                continue
            state_prop_list.append(torch.mean(single_probs[:valid_index], dim=0))
            state_prob_list.append(single_probs[:valid_index])
        
        avg_state_probs = torch.stack(state_prop_list, dim=0)

        return {
            'tissue_compartment': tissue_cat,
            'type_prob_list': type_prob_list,
            'type_prop': avg_type_probs,
            'state_prob_list': state_prob_list,
            'state_prop': avg_state_probs
        }