import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum, scatter_max
from torch_scatter.utils import broadcast
from collections import OrderedDict


class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """

    def __init__(self, n_users, n_items, use_gate):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.use_gate = use_gate
        self.gate1 = nn.Linear(64, 64, bias=False)
        self.gate2 = nn.Linear(64, 64, bias=False)
        self.sigmoid = nn.Sigmoid()

    def KG_forward(self, entity_emb, edge_index, edge_type, weight):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index
        edge_relation_emb = weight[edge_type]
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb
        entity_agg = scatter_mean(
            src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0
        )
        return entity_agg

    def forward(
        self,
        entity_emb,
        user_emb,
        edge_index,
        edge_type,
        interact_mat,
        weight,
        fast_weights=None,
        i=0,
    ):

        entity_kg_agg = self.KG_forward(entity_emb, edge_index, edge_type, weight)
        item_kg_agg = entity_kg_agg[: self.n_items]
        att_kg_agg = entity_kg_agg[self.n_items :]

        mat_row = interact_mat._indices()[0, :]
        mat_col = interact_mat._indices()[1, :]

        item_int_agg = scatter_mean(
            src=user_emb[mat_row], index=mat_col, dim_size=self.n_items, dim=0
        )

        if self.use_gate:
            if fast_weights is None:
                gi = self.sigmoid(self.gate1(item_kg_agg) + self.gate2(item_int_agg))
            else:
                gate1_name = f"convs.{i}.gate1.weight"
                gate2_name = f"convs.{i}.gate2.weight"
                conv_w1 = fast_weights.get(gate1_name)
                conv_w2 = fast_weights.get(gate2_name)
                gi = self.sigmoid(
                    F.linear(item_kg_agg, conv_w1) + F.linear(item_int_agg, conv_w2)
                )

            item_emb_fusion = (gi * item_kg_agg) + ((1 - gi) * item_int_agg)
        else:
            item_emb_fusion = item_kg_agg + item_int_agg

        user_item_mat = torch.sparse.FloatTensor(
            torch.cat([mat_row, mat_col]).view(2, -1),
            torch.ones_like(mat_row, dtype=torch.float),
            size=[self.n_users, self.n_items],
        )
        user_agg = torch.sparse.mm(user_item_mat, item_emb_fusion)

        final_entity_agg = torch.cat([item_emb_fusion, att_kg_agg])

        return final_entity_agg, user_agg, item_kg_agg, item_int_agg


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(
        self,
        channel,
        n_hops,
        n_users,
        n_relations,
        n_items,
        use_gate,
        node_dropout_rate=0.5,
        mess_dropout_rate=0.1,
    ):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        weight = nn.init.xavier_uniform_(torch.empty(n_relations, channel))
        self.weight = nn.Parameter(weight)

        for i in range(n_hops):
            self.convs.append(
                Aggregator(n_users=n_users, n_items=n_items, use_gate=use_gate)
            )

        self.dropout = nn.Dropout(p=mess_dropout_rate)

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(
            n_edges, size=int(n_edges * rate), replace=False
        )
        return edge_index[:, random_indices], edge_type[random_indices]

    def forward(
        self,
        user_emb,
        entity_emb,
        edge_index,
        edge_type,
        interact_mat,
        fast_weights=None,
        mess_dropout=True,
        node_dropout=True,
    ):

        if node_dropout:
            edge_index, edge_type = self._edge_sampling(
                edge_index, edge_type, self.node_dropout_rate
            )

        entity_res_emb = entity_emb
        user_res_emb = user_emb

        item_kg_res_emb = torch.zeros(self.n_items, entity_emb.shape[1]).to(
            entity_emb.device
        )
        item_int_res_emb = torch.zeros(self.n_items, entity_emb.shape[1]).to(
            entity_emb.device
        )

        for i in range(len(self.convs)):

            entity_emb, user_emb, item_kg_agg, item_int_agg = self.convs[i](
                entity_emb,
                user_emb,
                edge_index,
                edge_type,
                interact_mat,
                self.weight,
                fast_weights,
                i=i,
            )

            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
                item_kg_agg = self.dropout(item_kg_agg)
                item_int_agg = self.dropout(item_int_agg)

            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)
            item_kg_agg = F.normalize(item_kg_agg)
            item_int_agg = F.normalize(item_int_agg)

            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)
            item_kg_res_emb = torch.add(item_kg_res_emb, item_kg_agg)
            item_int_res_emb = torch.add(item_int_res_emb, item_int_agg)

        return entity_res_emb, user_res_emb, item_kg_res_emb, item_int_res_emb


class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, user_pre_embed, item_pre_embed):
        super(Recommender, self).__init__()
        self.n_users = data_config["n_users"]
        self.n_items = data_config["n_items"]
        self.n_relations = data_config["n_relations"]
        self.n_entities = data_config["n_entities"]
        self.n_nodes = data_config["n_nodes"]
        self.user_pre_embed = user_pre_embed
        self.item_pre_embed = item_pre_embed
        self.num_inner_update = args_config.num_inner_update
        self.meta_update_lr = args_config.meta_update_lr
        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.use_gate = args_config.use_gate
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.device = (
            torch.device("cuda:" + str(args_config.gpu_id))
            if args_config.cuda
            else torch.device("cpu")
        )
        self.iepe_lstm = nn.LSTM(
            input_size=self.emb_size, hidden_size=self.emb_size, batch_first=True
        )
        self.iepe_attention_w = nn.Linear(self.emb_size, self.emb_size)
        self.iepe_attention_q = nn.Parameter(torch.randn(self.emb_size, 1))
        self.user_fusion_mlp = nn.Sequential(
            nn.Linear(self.emb_size * 2, self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, 1),
        )
        self.alpha = args_config.alpha
        self.beta = args_config.beta
        self.cosine_sim = nn.CosineSimilarity(dim=1)
        self.edge_index, self.edge_type = self._get_edges(graph)
        self._init_weight()
        self.gcn = self._init_model()
        self.interact_mat = None

    def _init_weight(self):
        self.all_embed = nn.init.xavier_uniform_(
            torch.empty(self.n_nodes, self.emb_size)
        )
        if self.user_pre_embed is not None and self.item_pre_embed is not None:
            entity_emb = self.all_embed[(self.n_users + self.n_items) :, :]
            self.all_embed = torch.cat(
                [self.user_pre_embed, self.item_pre_embed, entity_emb]
            )
        self.all_embed = nn.Parameter(self.all_embed)

    def _init_model(self):
        return GraphConv(
            channel=self.emb_size,
            n_hops=self.context_hops,
            n_users=self.n_users,
            n_relations=self.n_relations,
            n_items=self.n_items,
            use_gate=self.use_gate,
            node_dropout_rate=self.node_dropout_rate,
            mess_dropout_rate=self.mess_dropout_rate,
        )

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))
        index = graph_tensor[:, :-1]
        type = graph_tensor[:, -1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def get_parameter(self):
        param_dict = dict()
        for name, para in self.gcn.named_parameters():
            if name.startswith("conv"):
                param_dict[name] = para
        for name, para in self.user_fusion_mlp.named_parameters():
            param_dict["user_fusion_mlp." + name] = para
        return OrderedDict(param_dict)

    def _forward_iepe(self, user_history, item_embeddings):
        history_embeds = item_embeddings[user_history]
        lstm_out, _ = self.iepe_lstm(history_embeds)
        attention_scores = torch.matmul(
            torch.tanh(self.iepe_attention_w(lstm_out)), self.iepe_attention_q
        )
        attention_weights = F.softmax(attention_scores, dim=1)
        user_evo_emb = torch.sum(lstm_out * attention_weights, dim=1)
        return user_evo_emb

    def _fuse_user_embeddings(self, user_int_emb, user_evo_emb, fast_weights=None):
        mlp_input = torch.cat([user_int_emb, user_evo_emb], dim=1)
        if fast_weights is None:
            lambda_u = torch.sigmoid(self.user_fusion_mlp(mlp_input))
        else:
            w1 = fast_weights["user_fusion_mlp.0.weight"]
            b1 = fast_weights["user_fusion_mlp.0.bias"]
            w2 = fast_weights["user_fusion_mlp.2.weight"]
            b2 = fast_weights["user_fusion_mlp.2.bias"]
            x = F.relu(F.linear(mlp_input, w1, b1))
            lambda_u = torch.sigmoid(F.linear(x, w2, b2))
        fused_user_emb = user_int_emb + lambda_u * user_evo_emb
        return fused_user_emb

    def forward(self, batch, fast_weights=None):
        user = batch["users"]
        pos_item = batch["pos_items"]
        neg_item = batch["neg_items"]
        user_history = batch["history"]

        user_emb = self.all_embed[: self.n_users, :]
        entity_emb = self.all_embed[self.n_users :, :]

        entity_gcn_emb, user_gcn_emb, item_kg_emb, item_int_emb = self.gcn(
            user_emb,
            entity_emb,
            self.edge_index,
            self.edge_type,
            self.interact_mat,
            fast_weights=fast_weights,
            mess_dropout=self.mess_dropout,
            node_dropout=self.node_dropout,
        )

        user_int_emb = user_gcn_emb[user]

        final_item_emb = entity_gcn_emb[: self.n_items]
        pos_item_emb = final_item_emb[pos_item]
        neg_item_emb = final_item_emb[neg_item]

        user_evo_emb = self._forward_iepe(user_history, final_item_emb)
        user_fused_emb = self._fuse_user_embeddings(
            user_int_emb, user_evo_emb, fast_weights
        )

        loss, _, _, _ = self.create_bpr_loss(
            user_fused_emb,
            pos_item_emb,
            neg_item_emb,
            user_int_emb,
            user_evo_emb,
            item_kg_emb,
            item_int_emb,
            pos_item,
        )
        return loss

    def forward_meta(self, support_batch, query_batch):
        fast_weights = self.get_parameter()
        for i in range(self.num_inner_update):
            loss = self.forward(support_batch, fast_weights=fast_weights)
            gradients = torch.autograd.grad(
                loss, fast_weights.values(), create_graph=True
            )
            fast_weights = OrderedDict(
                (name, param - self.meta_update_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )
        meta_loss = self.forward(query_batch, fast_weights=fast_weights)
        return meta_loss

    def generate(self, user_history, adapt_fast_weight=None):

        user_emb = self.all_embed[: self.n_users, :]
        entity_emb = self.all_embed[self.n_users :, :]

        entity_gcn_emb, user_gcn_emb, _, _ = self.gcn(
            user_emb,
            entity_emb,
            self.edge_index,
            self.edge_type,
            self.interact_mat,
            fast_weights=adapt_fast_weight,
            mess_dropout=False,
            node_dropout=False,
        )

        final_item_emb = entity_gcn_emb[: self.n_items]
        all_user_evo_emb = self._forward_iepe(user_history, final_item_emb)
        all_user_fused_emb = self._fuse_user_embeddings(
            user_gcn_emb, all_user_evo_emb, adapt_fast_weight
        )

        return all_user_fused_emb, final_item_emb

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(
        self,
        user_fused_emb,
        pos_item_emb,
        neg_item_emb,
        user_int_emb,
        user_evo_emb,
        item_kg_emb,
        item_int_emb,
        pos_item_indices,
    ):

        batch_size = user_fused_emb.shape[0]

        pos_scores = torch.sum(torch.mul(user_fused_emb, pos_item_emb), axis=1)
        neg_scores = torch.sum(torch.mul(user_fused_emb, neg_item_emb), axis=1)
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        regularizer = (
            torch.norm(user_fused_emb) ** 2
            + torch.norm(pos_item_emb) ** 2
            + torch.norm(neg_item_emb) ** 2
        ) / 2
        emb_loss = self.decay * regularizer / batch_size

        l_align_user = 1 - self.cosine_sim(user_int_emb, user_evo_emb)
        l_align_user = torch.mean(l_align_user)

        pos_item_kg_view = item_kg_emb[pos_item_indices]
        pos_item_int_view = item_int_emb[pos_item_indices]
        l_poi_align = 1 - self.cosine_sim(pos_item_kg_view, pos_item_int_view)
        l_poi_align = torch.mean(l_poi_align)

        consistency_loss = self.alpha * l_align_user + (1 - self.alpha) * l_poi_align
        total_loss = mf_loss + emb_loss + self.beta * consistency_loss

        return total_loss, mf_loss, emb_loss, consistency_loss
