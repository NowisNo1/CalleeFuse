import torch
from torch import nn
import torch.nn.functional as F


def build_mask(node_to_sample_ptr):
    num_nodes = node_to_sample_ptr[-1]

    starts = node_to_sample_ptr[:-1]
    ends = node_to_sample_ptr[1:]

    sub_indices = torch.arange(num_nodes, device=node_to_sample_ptr.device).unsqueeze(0)
    starts_expanded = starts.unsqueeze(1)
    ends_expanded = ends.unsqueeze(1) - 1

    is_in_range = (sub_indices >= starts_expanded) & (sub_indices <= ends_expanded)

    mask = is_in_range

    return mask


class GraphCrossFusion(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, bias=True)

    def init_params(self):
        q_weight = self.cross_attn.in_proj_weight[:self.d_model, :]
        k_weight = self.cross_attn.in_proj_weight[self.d_model: self.d_model * 2, :]
        v_weight = self.cross_attn.in_proj_weight[2 * self.d_model:, :]

        nn.init.xavier_uniform_(q_weight)
        nn.init.xavier_uniform_(k_weight)
        nn.init.xavier_uniform_(v_weight)

        nn.init.xavier_uniform_(self.cross_attn.out_proj.weight)

        if self.cross_attn.in_proj_bias is not None:
            nn.init.zeros_(self.cross_attn.in_proj_bias)
        nn.init.zeros_(self.cross_attn.out_proj.bias)

    def forward(self, hidden_state, graph_embeds, node_to_sample_ptr, attention_mask=None, graph_attention_mask=None):

        batch_size, seq_length, hidden_dim = hidden_state.shape
        num_nodes, _, _ = graph_embeds.shape

        if len(attention_mask.size()) == 4:
            attention_mask = attention_mask.squeeze(1).squeeze(1)

        graph_start = node_to_sample_ptr[:-1]
        graph_end = node_to_sample_ptr[1:]
        graph_count = graph_end - graph_start
        graph_count = graph_count.to(hidden_state.device)

        q_cross_mask = (graph_count > 0)
        q_cross_indices = q_cross_mask.nonzero().squeeze(dim=1)

        q_single_mask = (graph_count == 0)
        q_single_indices = q_single_mask.nonzero().squeeze(dim=1)

        graph_mask = torch.ones(num_nodes, dtype=torch.bool, device=hidden_state.device)

        kv_cross_indices = graph_mask.nonzero().squeeze(dim=1)

        batch_size = len(q_cross_indices)

        graph_attention_mask = graph_attention_mask[kv_cross_indices, :]

        graph_attention_mask = torch.as_tensor(graph_attention_mask, dtype=torch.bool)
        graph_attention_mask = graph_attention_mask.permute(1, 0).unsqueeze(0).unsqueeze(0)

        mask_base = build_mask(node_to_sample_ptr)
        mask_base = mask_base[q_cross_indices, :]
        mask_base = mask_base[:, kv_cross_indices]
        mask = mask_base.unsqueeze(1).unsqueeze(1).expand(-1, seq_length, seq_length, -1)

        mask = mask & graph_attention_mask


        mask = mask.reshape(batch_size, seq_length, num_nodes * seq_length)

        mask = mask.unsqueeze(1).expand(-1, self.nhead, -1, -1)
        mask = mask.reshape(batch_size * self.nhead, seq_length, num_nodes * seq_length)

        q = hidden_state[q_cross_indices, :, :]
        k = graph_embeds[graph_mask, :, :].reshape(1, num_nodes * seq_length, self.d_model).repeat(batch_size, 1, 1)
        v = k

        output, weights = self.cross_attn(
            query=q,
            key=k,
            value=v,
            attn_mask=~mask,
            need_weights=True
        )

        hidden_state_fusion = torch.zeros_like(hidden_state)
        hidden_state_fusion[q_cross_indices, :, :] = output.to(torch.float32)

        hidden_state_fusion[q_single_indices, :, :] = hidden_state[q_single_indices, :, :]

        return hidden_state_fusion, weights


class GateFusion(nn.Module):
    def __init__(self, min_size, max_size):
        super().__init__()
        self.N_min, self.N_max = min_size, max_size
        self.gate = nn.Sequential(
            nn.Linear(in_features=1, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1),
            nn.Sigmoid()
        )

    def normalize_N(self, node_counts):
        node_counts = node_counts.to(self.gate[0].weight.device)
        N_norm = (node_counts - self.N_min) / (self.N_max - self.N_min + 1e-8)
        return N_norm.unsqueeze(-1)

    def forward(self, node_counts):
        N_norm = self.normalize_N(node_counts)
        alpha = self.gate(N_norm)
        alpha = torch.clamp(alpha, min=0, max=0.1).unsqueeze(1)

        return alpha

    def init_params(self):
        for idx, module in enumerate(self.gate):
            if not isinstance(module, nn.Linear):
                continue
            if idx == 0:
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.constant_(module.bias, 0.0)
            elif idx == 2:
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.constant_(module.bias, 0.0)


class AttentionFusion(nn.Module):
    def __init__(self, d_model, min_size, max_size):
        super().__init__()
        self.d_model = d_model,
        self.N_min, self.N_max = min_size, max_size

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

        self.gate = nn.Sequential(
            nn.Linear(in_features=1, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1),
            nn.Sigmoid()
        )

    def init_params(self):
        for layer in [self.query, self.key, self.value]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

        for idx, module in enumerate(self.gate):
            if not isinstance(module, nn.Linear):
                continue
            if idx == 0:
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.constant_(module.bias, 0.0)
            elif idx == 2:
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.constant_(module.bias, 0.0)

    def normalize_N(self, graph_node_counts):
        graph_node_counts = graph_node_counts.to(self.query.weight.device)
        N_norm = (graph_node_counts - self.N_min) / (self.N_max - self.N_min + 1e-8)
        return N_norm.unsqueeze(-1)

    def forward(self, h1, h2, graph_node_counts):
        q, k, v = self.query(h1), self.key(h2), self.value(h2)

        d_k = torch.tensor(q.size(-1))
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(d_k)

        attn_weights = self.softmax(attn_scores) + 1e-8
        h2_attn = torch.matmul(attn_weights, v)

        N_norm = self.normalize_N(graph_node_counts)
        alpha = self.gate(N_norm)
        alpha = torch.clamp(alpha, min=0, max=0.5).unsqueeze(1)

        h_fused = alpha * h2_attn + (1 - alpha) * h1
        h_fused = F.layer_norm(h_fused, normalized_shape=self.d_model)
        return h_fused


class ConcatMLPFusion(nn.Module):
    def __init__(self, d_target, dropout=0.1):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(2 * d_target, 2 * d_target),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_target, d_target),
            nn.LayerNorm(d_target)
        )

    def forward(self, h_g, h_c):
        h_concat = torch.concat([h_g, h_c], dim=-1)
        return self.fusion(h_concat)

    def init_params(self):
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
