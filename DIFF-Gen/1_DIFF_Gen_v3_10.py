import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm
import pickle
import os
import time
import traceback
import re

@dataclass
class TracerouteConfig:

    seq_len: int = 30
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    
    timesteps: int = 500
    schedule: str = 'cosine'
    
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 200
    warmup_steps: int = 500
    
    max_ip_count: int = 150000
    max_completion_hops: int = 20
    max_probe_ttl: int = 60
    
    rtt_context_length: int = 3
    
    def __post_init__(self):
        self.betas = self.get_beta_schedule()
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
    
    def get_beta_schedule(self):
        if self.schedule == 'linear':
            scale = 1000 / self.timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, self.timesteps)
        elif self.schedule == 'cosine':
            steps = self.timesteps + 1
            s = 0.008
            x = torch.linspace(0, self.timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)
        elif self.schedule == 'sqrt':
            return 1 - torch.sqrt(1 / (torch.linspace(0.999, 0.001, self.timesteps) + 1e-8))
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

class TracerouteDataset(Dataset):
    def __init__(self, data: List[Dict], config: TracerouteConfig, ip_vocab=None, ip_to_idx=None):
        self.config = config
        self.sequences = self.preprocess_data(data)
        
        if ip_vocab is not None and ip_to_idx is not None:
            self.ip_vocab = ip_vocab
            self.ip_to_idx = ip_to_idx
            self.idx_to_ip = {v: k for k, v in ip_to_idx.items()}
        else:
            self.ip_vocab = self.build_ip_vocab()
            self.ip_to_idx = {}
            for i, ip in enumerate(self.ip_vocab):
                self.ip_to_idx[ip] = i + 1
            self.idx_to_ip = {v: k for k, v in self.ip_to_idx.items()}
        
        print(f"Dataset size: {len(self.sequences)} sequences")
        print(f"Number of IP addresses: {len(self.ip_vocab)}")
        
        self.analyze_rtt_patterns()
    
    def extract_ip_segments(self, ip_str):
        if not ip_str or ip_str == '0.0.0.0':
            return [0.0, 0.0, 0.0, 0.0]
        
        try:
            parts = ip_str.split('.')
            if len(parts) != 4:
                return [0.0, 0.0, 0.0, 0.0]
            
            segments = []
            for part in parts:
                if part.isdigit():
                    val = int(part)
                    segments.append(val / 255.0)
                else:
                    return [0.0, 0.0, 0.0, 0.0]
            return segments
        except:
            return [0.0, 0.0, 0.0, 0.0]
    
    def is_valid_ip(self, ip):
        if not ip or ip == '0.0.0.0':
            return False
        
        ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        if not re.match(ip_pattern, ip):
            return False
        
        parts = ip.split('.')
        for part in parts:
            if not part.isdigit() or int(part) < 0 or int(part) > 255:
                return False
        
        return True
    
    def analyze_rtt_patterns(self):
        rtt_increases = []
        rtt_abs_increases = []
        total_sequences = 0
        
        for sequence in self.sequences:
            hops = sequence['hops']
            if len(hops) >= 3:
                total_sequences += 1
                for i in range(1, len(hops)):
                    if hops[i-1]['rtt'] > 0 and hops[i]['rtt'] > 0:
                        increase_ratio = max(0, hops[i]['rtt'] - hops[i-1]['rtt']) / hops[i-1]['rtt']
                        rtt_increases.append(increase_ratio)
                        increase_abs = max(0, hops[i]['rtt'] - hops[i-1]['rtt'])
                        rtt_abs_increases.append(increase_abs)
        
        if rtt_increases:
            avg_increase = np.mean(rtt_increases)
            median_increase = np.median(rtt_increases)
            std_increase = np.std(rtt_increases)
            
            avg_abs_increase = np.mean(rtt_abs_increases)
            median_abs_increase = np.median(rtt_abs_increases)
            
            print(f"RTT growth pattern analysis (based on {total_sequences} sequences, {len(rtt_increases)} hop-to-hop comparisons):")
            print(f"  Mean RTT growth ratio: {avg_increase*100:.1f}%")
            print(f"  Median RTT growth ratio: {median_increase*100:.1f}%")
            print(f"  RTT growth standard deviation: {std_increase*100:.1f}%")
            print(f"  Mean absolute RTT growth: {avg_abs_increase:.2f}ms")
            print(f"  Median absolute RTT growth: {median_abs_increase:.2f}ms")
            
            self.rtt_stats = {
                'median_ratio': median_increase,
                'median_abs': median_abs_increase,
                'avg_ratio': avg_increase,
                'avg_abs': avg_abs_increase,
                'std_ratio': std_increase
            }
        else:
            self.rtt_stats = None
    
    def build_ip_vocab(self):
        ip_set = set()
        
        for seq in self.sequences:
            for hop in seq['hops']:
                addr = hop['addr']
                if self.is_valid_ip(addr):
                    ip_set.add(addr)
        
        ip_list = list(ip_set)[:self.config.max_ip_count]
        return ip_list
    
    def preprocess_data(self, data):
        sequences = []
        for trace in data:
            if trace.get('stop_reason') == 'COMPLETED':
                hops = trace.get('hops', [])
                if hops and len(hops) >= 3:
                    sequence = {
                        'hops': [],
                        'dst': trace.get('dst'),
                        'src': trace.get('src'),
                        'total_hops': len(hops)
                    }
                    
                    seen_ips = set()
                    has_duplicate = False
                    for hop in hops:
                        addr = hop.get('addr', '0.0.0.0')
                        if self.is_valid_ip(addr):
                            if addr in seen_ips:
                                has_duplicate = True
                                break
                            seen_ips.add(addr)
                    
                    if has_duplicate:
                        continue
                    
                    probe_ttls = []
                    for hop in hops:
                        probe_ttl = hop.get('probe_ttl', 0)
                        if probe_ttl > 0:
                            probe_ttls.append(probe_ttl)
                    
                    if probe_ttls and not all(probe_ttls[i] < probe_ttls[i+1] for i in range(len(probe_ttls)-1)):
                        continue
                    
                    for i, hop in enumerate(hops):
                        addr = hop.get('addr', '0.0.0.0')
                        if not self.is_valid_ip(addr):
                            continue
                        
                        features = {
                            'addr': addr,
                            'rtt': hop.get('rtt', 0.0),
                            'reply_ttl': hop.get('reply_ttl', 64),
                            'reply_tos': hop.get('reply_tos', 0),
                            'reply_size': hop.get('reply_size', 56),
                            'probe_ttl': hop.get('probe_ttl', i+1),
                            'icmp_type': hop.get('icmp_type', 11),
                            'is_target': (i == len(hops) - 1)
                        }
                        sequence['hops'].append(features)
                    
                    if len(sequence['hops']) >= 3:
                        sequences.append(sequence)
        
        return sequences
    
    def encode_full_sequence(self, sequence):
        hops = sequence['hops']
        seq_len = min(len(hops), self.config.seq_len)
        
        encoded_topology = torch.zeros(self.config.seq_len, 6)
        encoded_condition = torch.zeros(self.config.seq_len, 4)
        rtt_values = torch.zeros(self.config.seq_len, 1)
        probe_ttl_values = torch.zeros(self.config.seq_len, 1)
        
        ip_segments = self.extract_ip_segments(sequence['dst'])
        
        for i in range(seq_len):
            hop = hops[i]
            ip_idx = self.ip_to_idx.get(hop['addr'], 0)
            encoded_topology[i, 0] = ip_idx
            encoded_topology[i, 1] = hop['reply_ttl'] / 255.0
            encoded_topology[i, 2] = hop['reply_tos'] / 255.0
            encoded_topology[i, 3] = hop['reply_size'] / 2000.0
            encoded_topology[i, 4] = hop['icmp_type'] / 20.0
            encoded_topology[i, 5] = 1.0 if hop.get('is_target', False) else 0.0
            
            encoded_condition[i, 0] = ip_segments[0]
            encoded_condition[i, 1] = ip_segments[1]
            encoded_condition[i, 2] = ip_segments[2]
            encoded_condition[i, 3] = ip_segments[3]
            
            rtt_values[i, 0] = min(hop['rtt'], 1000) / 1000
            
            probe_ttl_values[i, 0] = min(hop['probe_ttl'], self.config.max_probe_ttl)
        
        return encoded_topology, encoded_condition, rtt_values, probe_ttl_values, seq_len
    
    def encode_partial_sequence(self, hops, target_ip=None, start_probe_ttl=1, max_len=None):
        if max_len is None:
            max_len = self.config.seq_len
        
        seq_len = min(len(hops), max_len)
        encoded_topology = torch.zeros(max_len, 6)
        encoded_condition = torch.zeros(max_len, 4)
        rtt_values = torch.zeros(max_len, 1)
        probe_ttl_values = torch.zeros(max_len, 1)
        
        ip_segments = self.extract_ip_segments(target_ip) if target_ip else [0.0, 0.0, 0.0, 0.0]
        
        for i in range(seq_len):
            hop = hops[i]
            ip_idx = self.ip_to_idx.get(hop['addr'], 0)
            encoded_topology[i, 0] = ip_idx
            encoded_topology[i, 1] = hop.get('reply_ttl', 64) / 255.0
            encoded_topology[i, 2] = hop.get('reply_tos', 0) / 255.0
            encoded_topology[i, 3] = hop.get('reply_size', 56) / 2000.0
            encoded_topology[i, 4] = hop.get('icmp_type', 11) / 20.0
            encoded_topology[i, 5] = 0.0
            
            encoded_condition[i, 0] = ip_segments[0]
            encoded_condition[i, 1] = ip_segments[1]
            encoded_condition[i, 2] = ip_segments[2]
            encoded_condition[i, 3] = ip_segments[3]
            
            rtt_values[i, 0] = min(hop.get('rtt', 0), 1000) / 1000
            
            probe_ttl_values[i, 0] = min(hop.get('probe_ttl', start_probe_ttl + i), self.config.max_probe_ttl)
        
        return encoded_topology, encoded_condition, rtt_values, probe_ttl_values, seq_len
    
    def decode_sequence(self, encoded_topology, rtt_values, probe_ttl_values=None):
        sequence = []
        
        for i in range(encoded_topology.shape[0]):
            ip_idx = int(encoded_topology[i, 0].item())
            if ip_idx == 0:
                continue
            
            ip = self.idx_to_ip.get(ip_idx, '0.0.0.0')
            
            if probe_ttl_values is not None and i < probe_ttl_values.shape[0]:
                probe_ttl = max(1, int(probe_ttl_values[i, 0].item()))
            else:
                probe_ttl = i + 1
            
            hop = {
                'addr': ip,
                'rtt': rtt_values[i, 0].item() * 1000,
                'reply_ttl': max(1, int(encoded_topology[i, 1].item() * 255)),
                'reply_tos': int(encoded_topology[i, 2].item() * 255),
                'reply_size': int(encoded_topology[i, 3].item() * 2000),
                'probe_ttl': min(probe_ttl, self.config.max_probe_ttl),
                'icmp_type': 11 if encoded_topology[i, 5].item() < 0.5 else 0,
                'is_target': encoded_topology[i, 5].item() > 0.5
            }
            sequence.append(hop)
        
        return sequence
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        encoded_topology, encoded_condition, rtt_values, probe_ttl_values, seq_len = self.encode_full_sequence(sequence)
        
        if seq_len >= 5:
            min_break = max(2, int(seq_len * 0.3))
            max_break = min(seq_len - 1, int(seq_len * 0.7))
            if min_break >= max_break:
                min_break = max(2, seq_len // 2)
                max_break = seq_len - 1
            break_point = random.randint(min_break, max_break)
        elif seq_len > 3:
            break_point = random.randint(2, seq_len - 2)
        else:
            break_point = seq_len - 1
        
        condition_hops = sequence['hops'][:break_point]
        actual_start_ttl = condition_hops[0]['probe_ttl'] if condition_hops else 1
        
        condition_topology, condition_cond, condition_rtt, condition_probe_ttl, _ = self.encode_partial_sequence(
            condition_hops, 
            target_ip=sequence['dst'], 
            start_probe_ttl=actual_start_ttl, 
            max_len=self.config.seq_len
        )
        
        condition_mask = torch.zeros(self.config.seq_len, 1)
        condition_mask[:break_point] = 1
        
        condition_full = torch.cat([condition_topology, condition_cond], dim=-1)  # [seq_len, 10]
        target_full = torch.cat([encoded_topology, encoded_condition], dim=-1)    # [seq_len, 10]
        
        return {
            'condition': condition_full,  # [seq_len, 10]
            'condition_rtt': condition_rtt,  # [seq_len, 1]
            'condition_mask': condition_mask,  # [seq_len, 1]
            'condition_probe_ttl': condition_probe_ttl,  # [seq_len, 1]
            'target_topology': encoded_topology,  # [seq_len, 6] 
            'target_condition': encoded_condition,  # [seq_len, 4]
            'target_rtt': rtt_values,  # [seq_len, 1]
            'target_probe_ttl': probe_ttl_values,  # [seq_len, 1]
            'break_point': break_point,
            'total_length': seq_len,
            'dst': sequence['dst']
        }

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.SiLU(),
            nn.Linear(in_channels, out_channels)
        )
        
        self.block2 = nn.Sequential(
            nn.LayerNorm(out_channels),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels, out_channels)
        )
        
        self.residual_connection = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        h = h + self.time_mlp(time_emb).unsqueeze(1)
        h = self.block2(h)
        return h + self.residual_connection(x)

class RTTGrowthNetwork(nn.Module):
    def __init__(self, hidden_dim: int, context_length: int = 3):
        super().__init__()
        self.context_length = context_length
        
        input_dim = context_length + 6 + 4
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        self.gru = nn.GRU(1, hidden_dim // 4, batch_first=True)
        self.gru_proj = nn.Linear(hidden_dim // 4, hidden_dim // 2)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, history_rtt: torch.Tensor, topology_features: torch.Tensor, target_features: torch.Tensor):

        batch_size = history_rtt.shape[0]
        
        history_rtt_seq = history_rtt.unsqueeze(-1)
        
        gru_out, _ = self.gru(history_rtt_seq)
        gru_features = self.gru_proj(gru_out[:, -1, :])
        
        history_flat = history_rtt.view(batch_size, -1)
        combined = torch.cat([history_flat, topology_features, target_features], dim=-1)
        
        output = self.network(combined)
        
        return output

class ConditionalTracerouteDiffusion(nn.Module):
    def __init__(self, config: TracerouteConfig):
        super().__init__()
        self.config = config
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
        
        self.input_proj = nn.Linear(10, config.hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 2,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(config.hidden_dim, config.hidden_dim, config.hidden_dim)
            for _ in range(2)
        ])
        
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.output_proj = nn.Linear(config.hidden_dim, 6)
        
        self.position_embeddings = nn.Parameter(torch.randn(1, config.seq_len, config.hidden_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x, t, condition_mask=None, condition_values=None):

        batch_size, seq_len, _ = x.shape
        
        time_emb = self.time_mlp(t)
        
        if condition_mask is not None and condition_values is not None:
            condition_cond_features = condition_values[:, :, 6:]  # [batch, seq, 4]
            
            x_full = torch.cat([x, condition_cond_features], dim=-1)  # [batch, seq, 10]
            
            x_emb = self.input_proj(x_full)
            
            condition_topology = condition_values[:, :, :6]  # [batch, seq, 6]
            condition_full = torch.cat([condition_topology, condition_cond_features], dim=-1)
            condition_emb = self.input_proj(condition_full)
            
            condition_weight = condition_mask
            x_emb = x_emb * (1 - condition_weight) + condition_emb * condition_weight
        else:
            x_full = torch.cat([x, torch.zeros(batch_size, seq_len, 4, device=x.device)], dim=-1)
            x_emb = self.input_proj(x_full)
        
        x_emb = x_emb + self.position_embeddings
        
        x_transformed = self.transformer(x_emb)
        
        for res_block in self.res_blocks:
            x_transformed = res_block(x_transformed, time_emb)
        
        x_out = self.output_norm(x_transformed)
        output = self.output_proj(x_out)
        
        return output  # [batch_size, seq_len, 6]

class TracerouteDiffusionTrainer:
    def __init__(self, config: TracerouteConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.config.betas = self.config.betas.to(self.device)
        self.config.alphas = self.config.alphas.to(self.device)
        self.config.alphas_cumprod = self.config.alphas_cumprod.to(self.device)
        self.config.sqrt_alphas_cumprod = self.config.sqrt_alphas_cumprod.to(self.device)
        self.config.sqrt_one_minus_alphas_cumprod = self.config.sqrt_one_minus_alphas_cumprod.to(self.device)
        
        self.diffusion_model = ConditionalTracerouteDiffusion(config).to(self.device)
        self.rtt_network = RTTGrowthNetwork(config.hidden_dim, config.rtt_context_length).to(self.device)
        
        self.diffusion_optimizer = optim.AdamW(self.diffusion_model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        self.rtt_optimizer = optim.AdamW(self.rtt_network.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        
        self.diffusion_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.diffusion_optimizer, T_max=config.epochs, eta_min=1e-7)
        self.rtt_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.rtt_optimizer, T_max=config.epochs, eta_min=1e-7)
        
        self.loss_fn = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []
        self.rtt_losses = []
        
        self.grad_clip = 0.5
    
    def add_noise(self, x_start, t):
        t_cpu = t.cpu()
        
        sqrt_alphas_cumprod = self.config.sqrt_alphas_cumprod[t_cpu].to(self.device).view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod = self.config.sqrt_one_minus_alphas_cumprod[t_cpu].to(self.device).view(-1, 1, 1)
        
        noise = torch.randn_like(x_start).clamp(-5, 5)
        x_noisy = sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise
        
        return x_noisy, noise
    
    def get_rtt_context_from_sequence(self, rtt_sequence, position, context_length):
        batch_size = rtt_sequence.shape[0]
        context = torch.zeros(batch_size, context_length, device=self.device)
        
        for b in range(batch_size):
            start = max(0, position - context_length)
            end = position
            available = rtt_sequence[b, start:end, 0]
            
            if len(available) > 0:
                context[b, -len(available):] = available
        
        return context
    
    def sample_timesteps(self, batch_size):
        t = torch.randint(1, self.config.timesteps - 1, (batch_size,), device=self.device).long()
        return t
    
    def train_epoch(self, dataloader, epoch):
        self.diffusion_model.train()
        self.rtt_network.train()
        
        total_diffusion_loss = 0
        total_rtt_loss = 0
        valid_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            try:
                condition = batch['condition'].to(self.device)
                condition_rtt = batch['condition_rtt'].to(self.device)
                condition_mask = batch['condition_mask'].to(self.device)
                target_topology = batch['target_topology'].to(self.device)
                target_condition = batch['target_condition'].to(self.device)
                target_rtt = batch['target_rtt'].to(self.device)
                
                batch_size = condition.shape[0]
                
                if torch.isnan(target_topology).any() or torch.isinf(target_topology).any():
                    print(f"Warning: target_topology contains NaN or Inf, skipping this batch")
                    continue
                
                t = self.sample_timesteps(batch_size)
                
                x_noisy, noise = self.add_noise(target_topology, t)
                
                if torch.isnan(x_noisy).any() or torch.isnan(noise).any():
                    print(f"Warning: noise data contains NaN, skipping this batch")
                    continue
                
                predicted_noise = self.diffusion_model(x_noisy, t, condition_mask, condition)
                
                if torch.isnan(predicted_noise).any() or torch.isinf(predicted_noise).any():
                    print(f"Warning: predicted noise contains NaN or Inf, skipping this batch")
                    continue
                
                loss_mask = 1 - condition_mask
                pred_masked = predicted_noise * loss_mask
                noise_masked = noise * loss_mask
                
                diffusion_loss = self.loss_fn(pred_masked, noise_masked)
                
                if torch.isnan(diffusion_loss) or torch.isinf(diffusion_loss):
                    print(f"Warning: diffusion_loss is NaN or Inf, skipping this batch")
                    continue
                
                self.diffusion_optimizer.zero_grad()
                diffusion_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), self.grad_clip)
                
                grad_norm = 0
                for p in self.diffusion_model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item()
                if grad_norm > 10 or torch.isnan(torch.tensor(grad_norm)):
                    print(f"Warning: abnormal gradient norm ({grad_norm:.2f}), skipping this batch")
                    self.diffusion_optimizer.zero_grad()
                    continue
                
                self.diffusion_optimizer.step()
                
                # ========== RTT Network Training ==========
                full_topology = target_topology
                target_features = target_condition
                
                rtt_loss = 0
                seq_len = target_topology.shape[1]
                condition_indices = (condition_mask.squeeze(-1) > 0.5)
                rtt_count = 0
                
                for pos in range(seq_len):
                    if condition_indices[:, pos].all():
                        continue
                    
                    history_rtt = self.get_rtt_context_from_sequence(target_rtt, pos, self.config.rtt_context_length)
                    curr_topology = full_topology[:, pos, :]
                    curr_target = target_features[:, pos, :]
                    
                    predicted_rtt = self.rtt_network(history_rtt, curr_topology, curr_target)
                    true_rtt = target_rtt[:, pos, :]
                    
                    pos_mask = ~condition_indices[:, pos]
                    if pos_mask.any():
                        rtt_loss += self.loss_fn(predicted_rtt[pos_mask], true_rtt[pos_mask])
                        rtt_count += 1
                
                if rtt_count > 0:
                    rtt_loss = rtt_loss / rtt_count
                else:
                    rtt_loss = torch.tensor(0.0, device=self.device)
                
                if not torch.isnan(rtt_loss) and not torch.isinf(rtt_loss):
                    self.rtt_optimizer.zero_grad()
                    rtt_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.rtt_network.parameters(), self.grad_clip)
                    self.rtt_optimizer.step()
                else:
                    print(f"Warning: rtt_loss is NaN or Inf, skipping RTT update")
                
                total_diffusion_loss += diffusion_loss.item()
                total_rtt_loss += rtt_loss.item() if not torch.isnan(rtt_loss) else 0
                valid_batches += 1
                
                progress_bar.set_postfix({
                    'diff_loss': diffusion_loss.item(),
                    'rtt_loss': rtt_loss.item() if not torch.isnan(rtt_loss) else 0,
                    'grad': grad_norm if 'grad_norm' in dir() else 0
                })
                
                # Warmup
                if epoch == 0 and batch_idx < self.config.warmup_steps // max(1, len(dataloader)):
                    lr_scale = min(1.0, float(batch_idx + 1) / (self.config.warmup_steps // max(1, len(dataloader))))
                    for pg in self.diffusion_optimizer.param_groups:
                        pg['lr'] = lr_scale * self.config.learning_rate
                    for pg in self.rtt_optimizer.param_groups:
                        pg['lr'] = lr_scale * self.config.learning_rate
                        
            except Exception as e:
                print(f"Error in training batch: {e}")
                traceback.print_exc()
                continue
        
        if valid_batches == 0:
            print("Error: no valid batches, training stopped")
            return 0, 0
        
        avg_diff_loss = total_diffusion_loss / valid_batches
        avg_rtt_loss = total_rtt_loss / valid_batches
        self.train_losses.append(avg_diff_loss)
        self.rtt_losses.append(avg_rtt_loss)
        
        if epoch >= 1:
            self.diffusion_scheduler.step()
            self.rtt_scheduler.step()
        
        return avg_diff_loss, avg_rtt_loss
    
    @torch.no_grad()
    def generate_completion(self, condition, condition_rtt, condition_mask, steps=100):
        self.diffusion_model.eval()
        self.rtt_network.eval()
        
        batch_size = condition.shape[0]
        seq_len = self.config.seq_len
        
        condition_cond_features = condition[:, :, 6:]
        
        noise_topology = torch.randn(batch_size, seq_len, 6, device=self.device)
        x = noise_topology.clone()
        
        for t in tqdm(reversed(range(0, steps)), desc="Generating topology", leave=False):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            predicted_noise = self.diffusion_model(x, t_batch, condition_mask, condition)
            
            alpha = self.config.alphas[t].to(self.device)
            alpha_cumprod = self.config.alphas_cumprod[t].to(self.device)
            
            if t > 0:
                beta = self.config.betas[t].to(self.device)
                new_noise = torch.randn_like(predicted_noise)
            else:
                beta = 0
                new_noise = 0
            
            denoised = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            )
            
            if t > 0:
                denoised = denoised + torch.sqrt(beta) * new_noise
            
            x = denoised
            condition_topology = condition[:, :, :6]
            x = x * (1 - condition_mask) + condition_topology * condition_mask
        
        generated_topology = x
        
        generated_rtt = torch.zeros(batch_size, seq_len, 1, device=self.device)
        
        for pos in range(seq_len):
            history_rtt = torch.zeros(batch_size, self.config.rtt_context_length, device=self.device)
            
            for b in range(batch_size):
                start = max(0, pos - self.config.rtt_context_length)
                end = pos
                
                for i in range(start, end):
                    if condition_mask[b, i, 0] > 0.5:
                        history_rtt[b, self.config.rtt_context_length - (pos - i)] = condition_rtt[b, i, 0]
                    else:
                        history_rtt[b, self.config.rtt_context_length - (pos - i)] = generated_rtt[b, i, 0]
            
            curr_topology = generated_topology[:, pos, :]
            curr_target = condition_cond_features[:, pos, :]
            
            if (condition_mask[:, pos, 0] > 0.5).all():
                generated_rtt[:, pos, :] = condition_rtt[:, pos, :]
            else:
                predicted_rtt = self.rtt_network(history_rtt, curr_topology, curr_target)
                generated_rtt[:, pos, :] = predicted_rtt
        
        return generated_topology, generated_rtt
    
    def train(self, train_dataset, val_dataset=None):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print("Starting training...")
        for epoch in range(self.config.epochs):
            train_diff_loss, train_rtt_loss = self.train_epoch(train_loader, epoch)
            print(f"Epoch {epoch+1}/{self.config.epochs}, Diffusion Loss: {train_diff_loss:.6f}, RTT Loss: {train_rtt_loss:.6f}")
            
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")
    
    def save_checkpoint(self, path):
        checkpoint = {
            'diffusion_model_state_dict': self.diffusion_model.state_dict(),
            'rtt_network_state_dict': self.rtt_network.state_dict(),
            'diffusion_optimizer_state_dict': self.diffusion_optimizer.state_dict(),
            'rtt_optimizer_state_dict': self.rtt_optimizer.state_dict(),
            'diffusion_scheduler_state_dict': self.diffusion_scheduler.state_dict(),
            'rtt_scheduler_state_dict': self.rtt_scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'rtt_losses': self.rtt_losses,
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to: {path}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.diffusion_model.load_state_dict(checkpoint['diffusion_model_state_dict'])
        self.rtt_network.load_state_dict(checkpoint['rtt_network_state_dict'])
        self.diffusion_optimizer.load_state_dict(checkpoint['diffusion_optimizer_state_dict'])
        self.rtt_optimizer.load_state_dict(checkpoint['rtt_optimizer_state_dict'])
        if 'diffusion_scheduler_state_dict' in checkpoint:
            self.diffusion_scheduler.load_state_dict(checkpoint['diffusion_scheduler_state_dict'])
        if 'rtt_scheduler_state_dict' in checkpoint:
            self.rtt_scheduler.load_state_dict(checkpoint['rtt_scheduler_state_dict'])
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'rtt_losses' in checkpoint:
            self.rtt_losses = checkpoint['rtt_losses']
        print(f"Checkpoint loaded from: {path}")

class TracerouteDiffusionSystem:
    def __init__(self, config=None):
        self.config = config or TracerouteConfig()
        self.trainer = None
        self.dataset = None
        self.all_data = None
        
    def load_data(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        trace = json.loads(line)
                        data.append(trace)
                    except json.JSONDecodeError:
                        continue
        
        print(f"Loaded {len(data)} records from {file_path}")
        self.all_data = data
        return data
    
    def preprocess_trace_for_completion(self, trace):
        if trace.get('stop_reason') == 'COMPLETED':
            return trace
        
        hops = trace.get('hops', [])
        if not hops:
            return trace
        
        if trace.get('stop_reason') == 'UNREACH' and len(hops) >= 2:
            last_hop = hops[-1]
            second_last_hop = hops[-2]
            
            last_addr = last_hop.get('addr', '0.0.0.0')
            second_last_addr = second_last_hop.get('addr', '0.0.0.0')
            
            if self.is_valid_ip(last_addr) and self.is_valid_ip(second_last_addr) and last_addr == second_last_addr:
                hops = hops[:-1]
                print(f"Processing UNREACH status: removed duplicate last hop {last_addr}")
        
        updated_trace = trace.copy()
        updated_trace['hops'] = hops
        
        if hops:
            last_hop_ttl = hops[-1].get('probe_ttl', len(hops))
            updated_trace['hop_count'] = last_hop_ttl
        else:
            updated_trace['hop_count'] = 0
        
        return updated_trace
    
    def is_valid_ip(self, ip):
        if not ip or ip == '0.0.0.0':
            return False
        
        ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        if not re.match(ip_pattern, ip):
            return False
        
        parts = ip.split('.')
        for part in parts:
            if not part.isdigit() or int(part) < 0 or int(part) > 255:
                return False
        
        return True
    
    def remove_duplicate_ips(self, existing_hops, generated_hops):
        if not generated_hops:
            return generated_hops
        
        existing_ips = set()
        for hop in existing_hops:
            addr = hop.get('addr')
            if addr and self.is_valid_ip(addr):
                existing_ips.add(addr)
        
        filtered_hops = []
        duplicate_count = 0
        
        for hop in generated_hops:
            addr = hop.get('addr', '')
            
            if not self.is_valid_ip(addr) or addr in existing_ips:
                duplicate_count += 1
                print(f"  Detected duplicate or invalid IP {addr}, removing this hop")
                continue
            
            filtered_hops.append(hop)
            existing_ips.add(addr)
        
        if duplicate_count > 0:
            print(f"  Removed {duplicate_count} duplicate or invalid IP addresses")
        
        return filtered_hops
    
    def prepare_datasets(self, data, train_ratio=0.9):
        completed_data = [t for t in data if t.get('stop_reason') == 'COMPLETED']
        print(f"Using {len(completed_data)} COMPLETED records for training")
        
        random.shuffle(completed_data)
        
        split_idx = int(len(completed_data) * train_ratio)
        train_data = completed_data[:split_idx]
        val_data = completed_data[split_idx:] if split_idx < len(completed_data) else []
        
        self.dataset = TracerouteDataset(train_data, self.config)
        
        if val_data:
            val_dataset = TracerouteDataset(
                val_data, 
                self.config,
                ip_vocab=self.dataset.ip_vocab,
                ip_to_idx=self.dataset.ip_to_idx
            )
        else:
            val_dataset = None
        
        return self.dataset, val_dataset
    
    def train_model(self, train_dataset, val_dataset=None):
        self.trainer = TracerouteDiffusionTrainer(self.config)
        self.trainer.train(train_dataset, val_dataset)
    
    def complete_single_traceroute(self, trace):
        if not self.trainer or not self.dataset:
            raise ValueError("Please train the model or load a checkpoint first")
        
        if trace.get('stop_reason') == 'COMPLETED':
            return trace
        
        trace = self.preprocess_trace_for_completion(trace)
        
        existing_hops = trace.get('hops', [])
        if not existing_hops:
            print(f"Warning: traceroute {trace.get('dst')} has no hops, cannot complete")
            trace['stop_reason'] = 'FAILED'
            return trace
        
        target_addr = trace.get('dst')
        if not target_addr or not self.is_valid_ip(target_addr):
            print(f"Warning: target address {target_addr} is invalid")
            trace['stop_reason'] = 'FAILED'
            return trace
        
        last_visible_ttl = existing_hops[-1].get('probe_ttl', len(existing_hops))
        last_visible_rtt = existing_hops[-1].get('rtt', 30.0)
        actual_start_ttl = existing_hops[0].get('probe_ttl', 1) if existing_hops else 1
        
        condition_topology, condition_cond, condition_rtt, condition_probe_ttl, condition_len = self.dataset.encode_partial_sequence(
            existing_hops, 
            target_ip=target_addr,
            start_probe_ttl=actual_start_ttl, 
            max_len=self.config.seq_len
        )
        
        condition = torch.cat([condition_topology, condition_cond], dim=-1)
        condition_mask = torch.zeros(self.config.seq_len, 1)
        condition_mask[:condition_len] = 1
        
        condition = condition.unsqueeze(0).to(self.trainer.device)
        condition_rtt = condition_rtt.unsqueeze(0).to(self.trainer.device)
        condition_mask = condition_mask.unsqueeze(0).to(self.trainer.device)
        condition_probe_ttl = condition_probe_ttl.unsqueeze(0).to(self.trainer.device)
        
        try:
            generated_topology, generated_rtt = self.trainer.generate_completion(
                condition, condition_rtt, condition_mask, steps=300
            )
            
            generated_seq_topology = generated_topology[0].cpu()
            generated_seq_rtt = generated_rtt[0].cpu()
            
            last_visible_probe_ttl = existing_hops[-1].get('probe_ttl', len(existing_hops))
            
            generated_probe_ttl = torch.zeros(self.config.seq_len, 1)
            for i in range(self.config.seq_len):
                if i < condition_len:
                    generated_probe_ttl[i, 0] = condition_probe_ttl[0, i, 0].item()
                else:
                    generated_probe_ttl[i, 0] = last_visible_probe_ttl + (i - condition_len + 1)
            
            all_hops = self.dataset.decode_sequence(generated_seq_topology, generated_seq_rtt, generated_probe_ttl)
            
            generated_hops = []
            num_to_generate = min(len(all_hops) - condition_len, self.config.max_completion_hops)
            
            for i in range(num_to_generate):
                hop_idx = condition_len + i
                if hop_idx >= len(all_hops):
                    break
                hop = all_hops[hop_idx].copy()
                hop['is_target'] = False
                generated_hops.append(hop)
            
            if not generated_hops:
                target_hop = {
                    'addr': target_addr,
                    'rtt': last_visible_rtt * 1.2,
                    'reply_ttl': 64,
                    'reply_tos': 0,
                    'reply_size': 56,
                    'probe_ttl': last_visible_probe_ttl + 1,
                    'icmp_type': 0,
                    'is_target': True
                }
                generated_hops = [target_hop]
            else:
                for i, hop in enumerate(generated_hops):
                    hop['probe_ttl'] = last_visible_probe_ttl + i + 1
                    hop['probe_ttl'] = min(hop['probe_ttl'], self.config.max_probe_ttl)
                    
                    if hop['addr'] == target_addr:
                        hop['icmp_type'] = 0
                        hop['is_target'] = True
                
                if generated_hops and generated_hops[-1]['addr'] != target_addr:
                    if len(generated_hops) >= self.config.max_completion_hops:
                        generated_hops[-1]['addr'] = target_addr
                        generated_hops[-1]['icmp_type'] = 0
                        generated_hops[-1]['is_target'] = True
            
            generated_hops = self.remove_duplicate_ips(existing_hops, generated_hops)
            
            completed_trace = trace.copy()
            completed_trace['stop_reason'] = 'COMPLETED'
            completed_trace['stop_data'] = 0
            
            all_completed_hops = existing_hops.copy()
            start_time = existing_hops[0].get('tx', {'sec': 0, 'usec': 0}) if existing_hops else {'sec': 0, 'usec': 0}
            
            for i, hop_data in enumerate(generated_hops):
                time_increment = max(0.5, hop_data['rtt'] / 1000)
                tx_sec = start_time['sec'] + int(time_increment * (len(existing_hops) + i))
                tx_usec = random.randint(0, 999999)
                
                hop_data['probe_ttl'] = min(hop_data['probe_ttl'], self.config.max_probe_ttl)
                
                hop = {
                    'addr': hop_data['addr'],
                    'probe_ttl': hop_data['probe_ttl'],
                    'probe_id': 1,
                    'probe_size': trace.get('probe_size', 44),
                    'tx': {'sec': tx_sec, 'usec': tx_usec},
                    'rtt': max(1.0, hop_data['rtt']),
                    'reply_ttl': hop_data['reply_ttl'],
                    'reply_tos': hop_data['reply_tos'],
                    'reply_size': hop_data['reply_size'],
                    'reply_ipid': random.randint(1000, 60000),
                    'icmp_type': hop_data['icmp_type'],
                    'icmp_code': 0,
                    'icmp_q_ttl': 1,
                    'icmp_q_ipl': trace.get('probe_size', 44),
                    'icmp_q_tos': trace.get('tos', 0)
                }
                all_completed_hops.append(hop)
            
            if all_completed_hops and all_completed_hops[-1]['addr'] != target_addr:
                last_ttl = min(all_completed_hops[-1]['probe_ttl'] + 1, self.config.max_probe_ttl)
                target_hop = {
                    'addr': target_addr,
                    'probe_ttl': last_ttl,
                    'probe_id': 1,
                    'probe_size': trace.get('probe_size', 44),
                    'tx': {'sec': start_time['sec'] + int(0.5 * (len(all_completed_hops))), 'usec': random.randint(0, 999999)},
                    'rtt': all_completed_hops[-1]['rtt'] * 1.1,
                    'reply_ttl': 64,
                    'reply_tos': 0,
                    'reply_size': 56,
                    'reply_ipid': random.randint(1000, 60000),
                    'icmp_type': 0,
                    'icmp_code': 0,
                    'icmp_q_ttl': 1,
                    'icmp_q_ipl': trace.get('probe_size', 44),
                    'icmp_q_tos': trace.get('tos', 0)
                }
                all_completed_hops.append(target_hop)
            
            last_probe_ttl = all_completed_hops[-1]['probe_ttl'] if all_completed_hops else 0
            completed_trace['hops'] = all_completed_hops
            completed_trace['hop_count'] = last_probe_ttl
            completed_trace['probe_count'] = last_probe_ttl
            
            return completed_trace
            
        except Exception as e:
            print(f"Completion failed: {e}")
            traceback.print_exc()
            trace['stop_reason'] = 'FAILED'
            return trace
    
    def complete_all_non_completed(self, output_file="Osaka_completed_final.txt"):
        """Complete all non-COMPLETED traceroutes"""
        if self.all_data is None:
            raise ValueError("Please load data first")
        
        print(f"\nStarting to complete all non-COMPLETED traceroutes...")
        print(f"Total records: {len(self.all_data)}")
        
        completed_count = sum(1 for t in self.all_data if t.get('stop_reason') == 'COMPLETED')
        non_completed_count = len(self.all_data) - completed_count
        print(f"COMPLETED records: {completed_count}")
        print(f"Non-COMPLETED records: {non_completed_count}")
        
        all_results = []
        non_completed_processed = 0
        success_count = 0
        
        for i, trace in enumerate(tqdm(self.all_data, desc="Completion progress")):
            if trace.get('stop_reason') == 'COMPLETED':
                all_results.append(trace)
            else:
                completed_trace = self.complete_single_traceroute(trace)
                all_results.append(completed_trace)
                non_completed_processed += 1
                
                if completed_trace.get('stop_reason') == 'COMPLETED':
                    success_count += 1
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i+1}/{len(self.all_data)} records, successfully completed: {success_count}/{non_completed_processed}")
        
        print(f"\nSaving results to {output_file}...")
        with open(output_file, 'w') as f:
            for trace in all_results:
                f.write(json.dumps(trace) + '\n')
        
        final_completed = sum(1 for t in all_results if t.get('stop_reason') == 'COMPLETED')
        final_failed = sum(1 for t in all_results if t.get('stop_reason') == 'FAILED')
        
        print(f"\nCompletion finished!")
        print(f"Original COMPLETED records: {completed_count}")
        print(f"Successfully completed non-COMPLETED records: {success_count}/{non_completed_count}")
        print(f"Total COMPLETED records: {final_completed}")
        print(f"Total FAILED records: {final_failed}")
        print(f"Total records: {len(all_results)}")
        if non_completed_count > 0:
            print(f"Completion success rate: {success_count/non_completed_count*100:.1f}%")
        
        print("\nValidating probe_ttl reasonableness...")
        invalid_ttl_count = 0
        for trace in all_results:
            prev_ttl = 0
            for hop in trace.get('hops', []):
                ttl = hop.get('probe_ttl', 0)
                if ttl > self.config.max_probe_ttl:
                    invalid_ttl_count += 1
                if ttl <= prev_ttl and prev_ttl > 0:
                    print(f"  Warning: probe_ttl not increasing: {prev_ttl} -> {ttl}")
                prev_ttl = ttl
        if invalid_ttl_count > 0:
            print(f"Warning: found {invalid_ttl_count} probe_ttl values exceeding {self.config.max_probe_ttl}")
        else:
            print(f"All probe_ttl values are <= {self.config.max_probe_ttl}")
        
        return all_results


if __name__ == "__main__":
    config = TracerouteConfig(
        seq_len=25,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        timesteps=300,
        epochs=150,
        batch_size=256,
        learning_rate=1e-4,
        warmup_steps=300,
        max_completion_hops=10,
        rtt_context_length=3
    )
    
    system = TracerouteDiffusionSystem(config)
    
    print("Loading data...")
    input_file = "Osaka_final.txt.txt"
    if not os.path.exists(input_file):
        if os.path.exists("Osaka.txt"):
            input_file = "Osaka.txt"
        else:
            print(f"Error: cannot find input file")
            exit(1)
    
    data = system.load_data(input_file)
    
    status_counts = {}
    for trace in data:
        status = trace.get('stop_reason', 'UNKNOWN')
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print("\nData status statistics:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count} records")
    
    print("\nPreparing datasets...")
    train_dataset, val_dataset = system.prepare_datasets(data, train_ratio=0.9)
    
    checkpoint_file = "traceroute_diffusion_final.pth"
    if os.path.exists(checkpoint_file):
        print(f"\nFound checkpoint file {checkpoint_file}")
        choice = input("Load checkpoint and skip training? (y/n): ").lower()
        if choice == 'y':
            system.trainer = TracerouteDiffusionTrainer(config)
            system.trainer.load_checkpoint(checkpoint_file)
            system.dataset = train_dataset
            print("Checkpoint loaded, skipping training")
        else:
            print("\nStarting diffusion model training...")
            system.train_model(train_dataset, val_dataset)
            system.trainer.save_checkpoint(checkpoint_file)
    else:
        print("\nStarting diffusion model training...")
        system.train_model(train_dataset, val_dataset)
        system.trainer.save_checkpoint(checkpoint_file)
    
    output_file = "Osaka_completed_final.txt"
    print(f"\nStarting to complete all non-COMPLETED records and save to {output_file}...")
    
    completed_traces = system.complete_all_non_completed(output_file)
    
    print(f"\nAll work completed!")
    print(f"Original file: {input_file}")
    print(f"Completed file: {output_file}")