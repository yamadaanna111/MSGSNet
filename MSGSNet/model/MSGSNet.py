import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, global_mean_pool

# =========================================================
# === 层次双向注意力模块
# =========================================================
class HierarchicalBiAttention(nn.Module):
    """
    Hierarchical Bi-Attention:
    - Local Attention: 通过局部窗口关注相邻的上下文片段
    - Global Attention: 通过全局注意力汇聚整体上下文信息
    最终结果融合两者，以增强层次化表示能力。
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1, local_window=15):
        super().__init__()
        self.local_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.global_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.local_window = local_window
        self.fusion = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.size()

        # ---------- 局部注意力 ----------
        local_outputs = []
        for i in range(L):
            start = max(0, i - self.local_window // 2)
            end = min(L, i + self.local_window // 2 + 1)
            local_context = x[:, start:end, :]
            query = x[:, i:i+1, :]
            local_out, _ = self.local_attn(query, local_context, local_context)
            local_outputs.append(local_out)
        local_outputs = torch.cat(local_outputs, dim=1)  # [B, L, D]

        # ---------- 全局注意力 ----------
        global_out, _ = self.global_attn(x, x, x)  # [B, L, D]

        # ---------- 融合 ----------
        combined = torch.cat([local_outputs, global_out], dim=-1)  # [B, L, 2D]
        fused = F.relu(self.fusion(combined))  # [B, L, D]
        return fused

# =========================================================
# === SCE序列分支
# =========================================================
class SequenceBranch(nn.Module):
    def __init__(self, input_dim=21, cnn_hidden=128, lstm_hidden=128, num_heads=4, intra_heads=1, dropout=0.5):
        super().__init__()
        # 多尺度卷积
        self.conv3 = nn.Conv1d(input_dim, cnn_hidden, kernel_size=3, padding=1)
        self.conv7 = nn.Conv1d(input_dim, cnn_hidden, kernel_size=7, padding=3)
        self.conv11 = nn.Conv1d(input_dim, cnn_hidden, kernel_size=11, padding=5)
        self.bn = nn.BatchNorm1d(cnn_hidden * 3)
        self.dropout = nn.Dropout(dropout)

        # BiLSTM
        self.lstm = nn.LSTM(cnn_hidden * 3, lstm_hidden, bidirectional=True, batch_first=True)

        # 层次双向注意力
        self.hier_attn = HierarchicalBiAttention(embed_dim=lstm_hidden * 2,
                                                 num_heads=num_heads,
                                                 dropout=dropout)

        self.out_dim = lstm_hidden * 2

    def forward(self, seq_x):
        # seq_x: [B, 21, L]
        if seq_x.shape[1] != self.conv3.in_channels:
            seq_x = seq_x.permute(0, 2, 1)
        x3 = F.relu(self.conv3(seq_x))
        x7 = F.relu(self.conv7(seq_x))
        x11 = F.relu(self.conv11(seq_x))
        x_cat = torch.cat([x3, x7, x11], dim=1)
        x_cat = self.bn(x_cat)
        x_cat = self.dropout(x_cat)

        # BiLSTM
        x_seq = x_cat.permute(0, 2, 1)  # [B, L, 3*C]
        lstm_out, _ = self.lstm(x_seq)  # [B, L, 2H]

        # 层次注意力
        attn_out = self.hier_attn(lstm_out)  # [B, L, 2H]

        # 池化得到图级/序列级表示
        x_pool = torch.mean(attn_out, dim=1)  # [B, 2H]
        return x_pool

# =========================================================
# === 交叉注意力融合模块
# =========================================================
class CrossAttentionFusion(nn.Module):
    def __init__(self, graph_dim, seq_dim, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.graph_proj = nn.Linear(graph_dim, embed_dim)
        self.seq_proj = nn.Linear(seq_dim, embed_dim)

        # 图到序列的交叉注意力
        self.graph_to_seq_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 序列到图的交叉注意力
        self.seq_to_graph_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # 融合后的维度
        self.fused_dim = embed_dim * 2

    def forward(self, graph_feat, seq_feat):
        """
        graph_feat: [B, graph_dim] - 多尺度图特征
        seq_feat: [B, seq_dim] - 序列特征
        """
        # 投影到相同维度
        graph_proj = self.graph_proj(graph_feat).unsqueeze(1)  # [B, 1, embed_dim]
        seq_proj = self.seq_proj(seq_feat).unsqueeze(1)  # [B, 1, embed_dim]

        # 图到序列的注意力
        graph_enhanced, _ = self.graph_to_seq_attn(
            query=seq_proj,  # 序列作为query
            key=graph_proj,  # 图作为key
            value=graph_proj  # 图作为value
        )

        # 序列到图的注意力
        seq_enhanced, _ = self.seq_to_graph_attn(
            query=graph_proj,  # 图作为query
            key=seq_proj,  # 序列作为key
            value=seq_proj  # 序列作为value
        )

        # 残差连接和层归一化
        seq_fused = self.norm1(seq_proj + self.dropout(graph_enhanced))
        graph_fused = self.norm2(graph_proj + self.dropout(seq_enhanced))

        # 拼接融合特征
        fused = torch.cat([seq_fused.squeeze(1), graph_fused.squeeze(1)], dim=1)  # [B, 2*embed_dim]
        return fused


# =========================================================
# === 多尺度图网络组件
# =========================================================
class PairwiseCrossAttention(nn.Module):
    def __init__(self, hidden_dim, attn_dim=None, nheads=4, dropout=0.1):
        super().__init__()
        attn_dim = attn_dim or hidden_dim
        self.attn_dim = attn_dim
        self.nheads = nheads
        self.q_proj = nn.Linear(hidden_dim, attn_dim)
        self.kv_proj = nn.Linear(hidden_dim, attn_dim)
        self.mha = nn.MultiheadAttention(attn_dim, nheads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(attn_dim, hidden_dim)

    def forward_one_query(self, q, others):
        Q = self.q_proj(q).unsqueeze(1)
        KV = torch.stack([self.kv_proj(x) for x in others], dim=1)
        attn_out, attn_weights = self.mha(Q, KV, KV)
        attn_out = attn_out.squeeze(1)
        attn_out = self.out_proj(attn_out)
        return attn_out, attn_weights

    def forward(self, h5, h10, h15):
        h5_att, w5 = self.forward_one_query(h5, [h10, h15])
        h10_att, w10 = self.forward_one_query(h10, [h5, h15])
        h15_att, w15 = self.forward_one_query(h15, [h5, h10])
        h5_comb = h5 + h5_att
        h10_comb = h10 + h10_att
        h15_comb = h15 + h15_att
        fused = torch.cat([h5_comb, h10_comb, h15_comb], dim=1)
        return fused, (w5, w10, w15)


class GraphChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        r = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(r, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, batch):
        pooled = global_mean_pool(x, batch)
        attn = self.mlp(pooled)
        x = x * attn[batch]
        return x


class GraphBranch(nn.Module):
    def __init__(self, in_channels, hidden_channels, K=3, use_channel_attn=True, attn_reduction=8):
        super().__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K=K)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=K)
        self.relu = nn.ReLU()
        self.pool = global_mean_pool
        self.use_channel_attn = use_channel_attn
        if use_channel_attn:
            self.channel_attn = GraphChannelAttention(hidden_channels, reduction=attn_reduction)

    def forward(self, x, edge_index, batch):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        if self.use_channel_attn:
            x = self.channel_attn(x, batch)
        x = self.pool(x, batch)
        return x


# =========================================================
# === MultiScaleGraphNet 完整定义
# =========================================================
class MultiScaleGraphNet(nn.Module):
    def __init__(self, in_channels_list, hidden_channels=128, num_classes=2,
                 dropout=0.5, use_channel_attn=True, attn_reduction=8,
                 use_cross_attn=True, attn_heads=4, attn_dropout=0.1, attn_dim=None):
        """
        in_channels_list: list of 3 ints (for 3 scales)
        use_cross_attn: whether to use pairwise cross-attention fusion
        attn_heads, attn_dropout: hyperparams for cross-attention
        attn_dim: attention embedding dim (if None, equals hidden_channels)
        """
        super().__init__()
        assert len(in_channels_list) == 3, "in_channels_list must have length 3"
        self.branches = nn.ModuleList([
            GraphBranch(in_channels_list[0], hidden_channels, use_channel_attn=use_channel_attn,
                        attn_reduction=attn_reduction),
            GraphBranch(in_channels_list[1], hidden_channels, use_channel_attn=use_channel_attn,
                        attn_reduction=attn_reduction),
            GraphBranch(in_channels_list[2], hidden_channels, use_channel_attn=use_channel_attn,
                        attn_reduction=attn_reduction),
        ])

        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = PairwiseCrossAttention(hidden_channels, attn_dim=attn_dim or hidden_channels,
                                                     nheads=attn_heads, dropout=attn_dropout)

        # fusion: if cross-attn used, fused dim = 3*hidden (same as before)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )


    def forward(self, bat5, bat10, bat15):
        # each branch returns [B, hidden]
        h5 = self.branches[0](bat5.x, bat5.edge_index, bat5.batch)
        h10 = self.branches[1](bat10.x, bat10.edge_index, bat10.batch)
        h15 = self.branches[2](bat15.x, bat15.edge_index, bat15.batch)

        # Fusion: either pairwise cross-attention or simple concat
        if self.use_cross_attn:
            h_cat, attn_weights = self.cross_attn(h5, h10, h15)  # [B, 3*hidden], tuple of weights
        else:
            h_cat = torch.cat([h5, h10, h15], dim=1)

        out = self.mlp(h_cat)

        return out, h5, h10, h15, h_cat


# =========================================================
# === 完整的多尺度图+序列融合模型
# =========================================================
class MultiScaleGraphSeqNet(nn.Module):
    def __init__(self,
                 # 图网络参数
                 in_channels_list, hidden_channels=128,
                 # 序列分支参数
                 seq_input_dim=21, seq_cnn_hidden=128, seq_lstm_hidden=128,
                 # 融合参数
                 fusion_embed_dim=256, fusion_heads=8,
                 # 通用参数
                 num_classes=2, dropout=0.5,
                 use_channel_attn=True, attn_reduction=8,
                 use_cross_attn=True, attn_heads=4, attn_dropout=0.1):
        super().__init__()

        # 多尺度图分支
        self.graph_net = MultiScaleGraphNet(
            in_channels_list=in_channels_list,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            dropout=dropout,
            use_channel_attn=use_channel_attn,
            attn_reduction=attn_reduction,
            use_cross_attn=use_cross_attn,
            attn_heads=attn_heads,
            attn_dropout=attn_dropout
        )

        # 序列分支
        self.seq_branch = SequenceBranch(
            input_dim=seq_input_dim,
            cnn_hidden=seq_cnn_hidden,
            lstm_hidden=seq_lstm_hidden,
            num_heads=fusion_heads,
            dropout=dropout
        )

        # 交叉注意力融合
        graph_output_dim = hidden_channels * 3  # 多尺度图融合后的维度
        seq_output_dim = self.seq_branch.out_dim

        self.cross_fusion = CrossAttentionFusion(
            graph_dim=graph_output_dim,
            seq_dim=seq_output_dim,
            embed_dim=fusion_embed_dim,
            num_heads=fusion_heads,
            dropout=dropout
        )

        # 最终分类器
        self.mlp = nn.Sequential(
            nn.Linear(self.cross_fusion.fused_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )


    def forward(self, bat5, bat10, bat15, seq_x):
        """
        bat5, bat10, bat15: 多尺度图数据
        seq_x: 序列特征 [B, 21, L]
        """
        # 多尺度图特征提取
        graph_out, h5, h10, h15, graph_feat = self.graph_net(bat5, bat10, bat15)

        # 序列特征提取
        seq_feat = self.seq_branch(seq_x)

        # 交叉注意力融合
        fused_feat = self.cross_fusion(graph_feat, seq_feat)

        # 最终分类
        out = self.mlp(fused_feat)


        return out