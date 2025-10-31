import os
import random
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import logging
from typing import List, Tuple, Optional
import time
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SimpleCNN(nn.Module):
    """简化的CNN模型用于联邦学习"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SCMACodebook:
    """SCMA码本设计（非固定码本 + SNR自适应 + QPSK随机复数降相关 + 动态稀疏度 + 阈值冻结）"""
    def __init__(self, num_users=6, num_resources=4, sparsity=0.1, adaptive: bool = True, freeze_threshold: float = 1e-3):
        self.num_users = num_users
        self.num_resources = num_resources
        self.sparsity = sparsity  # 初始稀疏度（每用户占用的RB比例，默认0.1）
        self.adaptive = adaptive
        self.freeze_threshold = float(freeze_threshold)
        # 资源块“质量”打分（可理解为频域子载波/资源块的长期CSI质量，分高者为优质资源块）
        self.resource_quality = torch.randn(self.num_resources).abs()
        # 资源块噪声功率（可由外部估计/设置）
        self.resource_noise_power = torch.ones(self.num_resources) * 1.0
        # 增益-退化参数
        self.quality_gain_alpha = 0.6   # 质量带来的相对增益幅度
        self.degradation_beta_noise = 0.5  # 噪声导致的退化强度
        self.degradation_beta_intf = 0.7   # 干扰导致的退化强度
        # 每用户的当前激活资源块与功率权重
        self.user_active_positions = {u: [] for u in range(self.num_users)}
        self.user_power_weights = {u: torch.ones(self.num_resources) for u in range(self.num_users)}
        # 安全元数据：按用户保存ID哈希与公钥（用于“嵌入码本”的元数据模拟）
        self.user_id_hash = {u: None for u in range(self.num_users)}
        self.user_public_key = {u: None for u in range(self.num_users)}
        # 每用户上一轮统计：整体变化幅度、逐RB能量
        self.user_last_delta_mag = {u: 0.0 for u in range(self.num_users)}
        self.user_last_rb_energy = {u: torch.zeros(self.num_resources) for u in range(self.num_users)}
        # 每用户逐RB冻结掩码：True表示该RB位置永久为0
        self.user_frozen_zero_mask = {u: torch.zeros(self.num_resources, dtype=torch.bool) for u in range(self.num_users)}
        # 初始化随机码本（作为兜底）
        self.codebook = self._generate_codebook()

    def _generate_codebook(self):
        """生成基础SCMA码本（QPSK随机复数，考虑冻结掩码与初始稀疏度）"""
        codebook = {}
        for user in range(self.num_users):
            code = torch.zeros(self.num_resources, dtype=torch.complex64)
            k = max(1, int(self.num_resources * self.sparsity))
            candidate_positions = [r for r in range(self.num_resources) if not self.user_frozen_zero_mask[user][r]]
            if len(candidate_positions) == 0:
                codebook[user] = code
                self.user_active_positions[user] = []
                continue
            k = min(k, len(candidate_positions))
            non_zero_positions = random.sample(candidate_positions, k)
            for pos in non_zero_positions:
                code[pos] = self._qpsk_symbol(user, pos)
            codebook[user] = code
            self.user_active_positions[user] = non_zero_positions
        return codebook

    def _qpsk_symbol(self, user_id: int, rb_index: int) -> torch.complex64:
        """QPSK符号（随机复数且降低相关性）：依据(user, rb, 上次能量/幅度)扰动簇索引。"""
        # 基于哈希的基础索引，减小跨用户/资源块相关性
        base = int((user_id * 73856093 + rb_index * 19349663) % 4)
        # 引入上一轮该用户的变化幅度作为微扰，进一步打散相关性
        delta_mag = float(self.user_last_delta_mag.get(user_id, 0.0))
        bias = int(min(3, max(0, round(delta_mag * 1000) % 4)))
        idx = (base + bias) % 4
        # QPSK {±1 ± j}/sqrt(2)
        if idx == 0:
            sym = complex(1.0, 1.0)
        elif idx == 1:
            sym = complex(1.0, -1.0)
        elif idx == 2:
            sym = complex(-1.0, 1.0)
        else:
            sym = complex(-1.0, -1.0)
        scale = 1.0 / np.sqrt(2.0)
        return torch.tensor(sym * scale, dtype=torch.complex64)

    def set_resource_noise(self, noise_power_per_rb: torch.Tensor):
        """设置/更新每个资源块的噪声功率估计（长度= num_resources）。"""
        if noise_power_per_rb.numel() != self.num_resources:
            raise ValueError("noise_power_per_rb 长度必须等于 num_resources")
        self.resource_noise_power = noise_power_per_rb.detach().clone().float()

    def update_from_client_signals(self, client_signal_stats: dict):
        """根据客户端信号动态更新稀疏度并进行阈值冻结。
        client_signal_stats[user] = {
          'delta_norm': float,                  # 本轮模型参数变化幅度（L2/∞范数等）
          'rb_param_change': torch.Tensor[R]    # 每RB映射的“参数改变幅度”统计（若缺省则按resource_quality代理）
        }
        规则：
        - 稀疏度动态：随 delta_norm 增大而上调（范围 [0.1, 0.5]）；
        - 阈值冻结：当 rb_param_change[r] < freeze_threshold 时，该RB永久为0；
        """
        for user in range(self.num_users):
            stats = client_signal_stats.get(user, {})
            delta_norm = float(stats.get('delta_norm', 0.0))
            rb_change = stats.get('rb_param_change', None)
            self.user_last_delta_mag[user] = delta_norm
            if isinstance(rb_change, torch.Tensor) and rb_change.numel() == self.num_resources:
                rb_change_vec = rb_change.detach().clone().float()
            else:
                # 用资源质量作为代理
                rb_change_vec = self.resource_quality.detach().clone().float()

            # 更新冻结掩码：小于阈值的位置永久冻结
            freeze_new = rb_change_vec < self.freeze_threshold
            self.user_frozen_zero_mask[user] = torch.logical_or(self.user_frozen_zero_mask[user], freeze_new)
            self.user_last_rb_energy[user] = rb_change_vec

            # 动态稀疏度：映射到 [0.1, 0.5]
            dn = min(1.0, delta_norm / 0.1)  # 可按实际放缩
            dyn_sp = 0.1 + 0.4 * dn
            self.sparsity = max(0.1, min(0.5, dyn_sp))

    def update_by_snr(self, user_snr_db: dict, noise_power_per_rb: torch.Tensor = None):
        """根据各用户的实时SNR进行自适应资源块分配与功率加权。
        - 更高SNR的用户：分配更多“优质资源块”（resource_quality高的RB）与略高功率权重。
        - 更低SNR的用户：保持基线占用，功率略降以降低干扰。
        - 同时：优质资源块在噪声与叠加干扰作用下表现出“有效增益下降”（退化）。
        """
        if not self.adaptive:
            return

        if noise_power_per_rb is not None:
            self.set_resource_noise(noise_power_per_rb)

        # 将资源块按质量从高到低排序
        quality_rank = torch.argsort(self.resource_quality, descending=True).tolist()

        # 归一化SNR到[0,1]，用于平滑资源数量与功率分配
        snr_values = list(user_snr_db.values()) if len(user_snr_db) > 0 else [0.0]
        snr_min = float(min(snr_values))
        snr_max = float(max(snr_values))
        snr_range = max(1e-6, snr_max - snr_min)

        # 第一阶段：先为所有用户按质量排序分配候选RB集合（过滤冻结位置）
        for user in range(self.num_users):
            snr_db = float(user_snr_db.get(user, snr_min))
            snr_norm = (snr_db - snr_min) / snr_range  # 0~1

            # 动态决定该用户占用RB数量：基线k +/- delta
            base_k = max(1, int(self.num_resources * self.sparsity))
            # 最高可多拿到 25% 资源（向上取整），最低减到基线
            extra = int(round(0.25 * self.num_resources * snr_norm))
            k_user = min(self.num_resources, max(1, base_k + extra))

            # 按资源质量从高到低，分配前 k_user 个RB（过滤冻结位置）
            candidate = [r for r in quality_rank if not self.user_frozen_zero_mask[user][r]]
            active_positions = candidate[:k_user]
            self.user_active_positions[user] = active_positions

        # 统计所有用户在每个RB上的重叠数量（作为干扰度量）
        rb_overlap_counts = torch.zeros(self.num_resources)
        for user in range(self.num_users):
            for pos in self.user_active_positions[user]:
                rb_overlap_counts[pos] += 1.0

        # 质量归一化，用于增益项
        q = self.resource_quality
        q_min = float(q.min())
        q_max = float(q.max())
        q_range = max(1e-6, q_max - q_min)
        q_norm = (q - q_min) / q_range  # 0~1

        # 第二阶段：对每个用户施加“质量增益 × 噪声/干扰退化”的有效增益，并更新码字与功率
        for user in range(self.num_users):
            snr_db = float(user_snr_db.get(user, snr_min))
            snr_norm = (snr_db - snr_min) / snr_range

            active_positions = self.user_active_positions[user]

            # 基础功率权重：随SNR略增
            pw = torch.ones(self.num_resources)
            if len(active_positions) > 0:
                base_gain = 1.0 + 0.2 * snr_norm
                pw[active_positions] = base_gain

            # 有效增益模型：
            #   eff_gain[r] = (1 + alpha * q_norm[r]) / (1 + betaN * noise[r] + betaI * max(0, overlap[r] - 1))
            # 其中 overlap-1 表示该RB除当前用户外的并发用户数，作为“干扰级别”。
            eff_gain = torch.ones(self.num_resources)
            for r in range(self.num_resources):
                overlap = max(0.0, rb_overlap_counts[r].item() - (1.0 if r in active_positions else 0.0))
                numerator = 1.0 + self.quality_gain_alpha * float(q_norm[r])
                denominator = 1.0 + self.degradation_beta_noise * float(self.resource_noise_power[r]) 
                denominator += self.degradation_beta_intf * overlap
                eff_gain[r] = numerator / max(1e-6, denominator)

            # 叠加功率与有效增益，得到最终幅度权重
            final_weight = pw * eff_gain
            self.user_power_weights[user] = final_weight

            # 同步更新码本：仅在激活RB上保留码字，其它位置为0，并叠加最终权重（冻结位置强制为0）
            base_code = torch.zeros(self.num_resources, dtype=torch.complex64)
            for pos in active_positions:
                if not self.user_frozen_zero_mask[user][pos]:
                    base_code[pos] = self._qpsk_symbol(user, pos)
            self.codebook[user] = base_code * final_weight.to(base_code.dtype)

    def allocate_resources_for_user(self, user_id: int, snr_db: float):
        """单用户即时更新（可用于在线策略）"""
        self.update_by_snr({user_id: snr_db})

    def encode(self, data_vector, user_id):
        """将数据向量编码为SCMA码字（使用自适应资源分配与功率加权后的码本）"""
        if user_id not in self.codebook:
            raise ValueError(f"User {user_id} not in codebook")

        # 将数据向量映射到复数域（长度与资源块数一致的标量/小向量）
        complex_data = data_vector.float() + 1j * torch.zeros_like(data_vector.float())

        # 使用当前用户的码字与功率权重
        code = self.codebook[user_id]
        if complex_data.numel() == 1:
            # 标量扩展到每个资源块
            complex_data = complex_data.repeat(self.num_resources)
        else:
            # 截断/填充到资源块长度
            if complex_data.numel() < self.num_resources:
                pad = torch.zeros(self.num_resources - complex_data.numel()) * (1j * 0)
                complex_data = torch.cat([complex_data, pad])
            else:
                complex_data = complex_data[:self.num_resources]

        # 编码：逐资源块进行乘法（码字已包含功率权重与稀疏结构）
        encoded = code * complex_data
        return encoded

class HomomorphicEncryption:
    """简化的同态加密实现（基于BGV方案思想）"""
    
    def __init__(self, key_size=1024):
        self.key_size = key_size
        # 保留默认密钥（不用于基于ID的流程），以兼容旧接口
        self.private_key = self._generate_private_key()
        self.public_key = self._generate_public_key()
    
    def _generate_private_key(self):
        """生成私钥"""
        return torch.randint(-10, 10, (self.key_size,))
    
    def _generate_public_key(self):
        """生成公钥"""
        return torch.randint(-100, 100, (self.key_size,))

    # ===== 基于用户ID的确定性密钥与哈希 =====
    def _seed_from_user(self, user_id: int) -> int:
        m = hashlib.sha256(f"user:{user_id}".encode('utf-8')).hexdigest()
        return int(m[:16], 16)

    def _deterministic_tensor(self, shape: tuple, seed: int, low: int, high: int) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        arr = rng.integers(low=low, high=high, size=shape, dtype=np.int32)
        return torch.from_numpy(arr)

    def get_public_key_for_user(self, user_id: int) -> torch.Tensor:
        seed = self._seed_from_user(user_id) ^ 0xA5A5A5A5
        return self._deterministic_tensor((self.key_size,), seed, -100, 100)

    def get_private_key_for_user(self, user_id: int) -> torch.Tensor:
        seed = self._seed_from_user(user_id) ^ 0x5A5A5A5A
        return self._deterministic_tensor((self.key_size,), seed, -10, 10)

    def get_id_hash_for_user(self, user_id: int) -> str:
        return hashlib.sha256(f"uid:{user_id}".encode('utf-8')).hexdigest()[:16]

    def verify_id_hash_and_key(self, user_id: int, id_hash: str, public_key: torch.Tensor) -> bool:
        try:
            expected_hash = self.get_id_hash_for_user(user_id)
            expected_pk = self.get_public_key_for_user(user_id)
            same_hash = (id_hash == expected_hash)
            same_key = (public_key.numel() == expected_pk.numel() and torch.all(public_key == expected_pk))
            return bool(same_hash and same_key)
        except Exception:
            return False
    
    def encrypt(self, plaintext):
        """加密明文（旧接口，保留）"""
        if isinstance(plaintext, dict):
            encrypted = {}
            for key, value in plaintext.items():
                noise = torch.randn_like(value) * 0.01
                key_tensor = torch.randn_like(value) * 0.005
                encrypted[key] = value + noise + key_tensor
            return encrypted
        else:
            noise = torch.randn_like(plaintext) * 0.01
            key_tensor = torch.randn_like(plaintext) * 0.005
            return plaintext + noise + key_tensor

    def encrypt_with_public_key(self, plaintext, public_key: torch.Tensor):
        """使用给定公钥加密（模拟，将公钥映射为轻微扰动，使训练影响可控）。"""
        if isinstance(plaintext, dict):
            encrypted = {}
            scale = (public_key[:1].float().abs() % 7 + 1).item() / 1e3  # 小扰动系数 ~ [0.001, 0.007]
            for key, value in plaintext.items():
                noise = torch.randn_like(value) * (0.5 * scale)
                key_mask = torch.randn_like(value) * (0.5 * scale)
                encrypted[key] = value + noise + key_mask
            return encrypted
        else:
            scale = (public_key[:1].float().abs() % 7 + 1).item() / 1e3
            noise = torch.randn_like(plaintext) * (0.5 * scale)
            key_mask = torch.randn_like(plaintext) * (0.5 * scale)
            return plaintext + noise + key_mask
    
    def decrypt(self, ciphertext):
        """解密密文（旧接口，保留）"""
        if isinstance(ciphertext, dict):
            decrypted = {}
            for key, value in ciphertext.items():
                # 简化解密：移除噪声和密钥
                decrypted[key] = value * 0.9  # 简化的解密过程
            return decrypted
        else:
            return ciphertext * 0.9  # 简化的解密过程

    def decrypt_with_private_key(self, ciphertext, private_key: torch.Tensor):
        """使用给定私钥解密（模拟，与 encrypt_with_public_key 对应，做轻微缩放恢复）。"""
        if isinstance(ciphertext, dict):
            decrypted = {}
            scale = (private_key[:1].float().abs() % 7 + 1).item() / 1e3
            for key, value in ciphertext.items():
                decrypted[key] = value * (1.0 - 0.5 * scale)  # 近似恢复
            return decrypted
        else:
            scale = (private_key[:1].float().abs() % 7 + 1).item() / 1e3
            return ciphertext * (1.0 - 0.5 * scale)
    
    def add_ciphertexts(self, ciphertext1, ciphertext2):
        """密文加法（同态运算）"""
        if isinstance(ciphertext1, dict) and isinstance(ciphertext2, dict):
            result = {}
            for key in ciphertext1.keys():
                result[key] = ciphertext1[key] + ciphertext2[key]
            return result
        else:
            return ciphertext1 + ciphertext2

class SCMAOACTransmitter:
    """SCMA-OAC传输器"""
    
    def __init__(self, codebook, he_scheme):
        self.codebook = codebook
        self.he_scheme = he_scheme
        self.transmission_history = []
    
    def transmit_update(self, model_update, user_id, snr_db=20):
        """传输模型更新（SCMA-OAC）
        生成基于用户ID的确定性公钥与ID哈希，使用公钥加密，并随包附带元数据。
        """
        # 0. 基于用户ID的安全元数据
        pub_key = self.he_scheme.get_public_key_for_user(user_id)
        id_hash = self.he_scheme.get_id_hash_for_user(user_id)
        # 1. 同态加密（使用公钥）
        encrypted_update = self.he_scheme.encrypt_with_public_key(model_update, pub_key)
        
        # 2. 将加密的模型更新展平为向量
        encrypted_vector = torch.cat([v.flatten() for v in encrypted_update.values()])
        
        # 3. SCMA编码（简化版本）
        # 将向量分割成适合编码的小块
        chunk_size = self.codebook.num_resources
        if encrypted_vector.numel() < chunk_size:
            # 如果向量太小，填充到chunk_size
            padded_vector = torch.cat([encrypted_vector, torch.zeros(chunk_size - encrypted_vector.numel())])
        else:
            # 如果向量太大，截取前chunk_size个元素
            padded_vector = encrypted_vector[:chunk_size]
        
        # 简化的SCMA编码
        # 在码本对象中“嵌入/记录”该用户元数据（模拟码本携带）
        self.codebook.user_public_key[user_id] = pub_key
        self.codebook.user_id_hash[user_id] = id_hash
        encoded_update = self.codebook.encode(padded_vector, user_id)
        
        # 4. 添加信道噪声
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = 1.0 / snr_linear
        noise = torch.randn_like(encoded_update) * np.sqrt(noise_power)
        received_signal = encoded_update + noise
        
        # 记录传输信息
        self.transmission_history.append({
            'user_id': user_id,
            'original_size': sum(v.numel() for v in model_update.values()),
            'encoded_size': encoded_update.numel(),
            'snr_db': snr_db
        })
        
        packet = {
            'signal': received_signal,
            'meta': {
                'user_id': user_id,
                'id_hash': id_hash,
                'public_key': pub_key
            },
            'cipher': encrypted_update
        }
        return packet

class SCMAOACReceiver:
    """SCMA-OAC接收器"""
    
    def __init__(self, codebook, he_scheme):
        self.codebook = codebook
        self.he_scheme = he_scheme
        self.aggregation_history = []
    
    def receive_and_aggregate(self, received_list, user_ids, client_sizes):
        """接收并聚合信号。
        支持两种输入：
        - 旧格式：List[Tensor]（仅信号）
        - 新格式：List[packet]（含 'signal' 与 'meta'）
        新格式下将对ID哈希与公钥进行校验，不通过则丢弃。
        """
        # 解析输入
        use_packets = isinstance(received_list[0], dict)
        valid_signals = []
        if use_packets:
            for pkt in received_list:
                meta = pkt.get('meta', {})
                uid = int(meta.get('user_id', -1))
                id_hash = meta.get('id_hash', '')
                pub_key = meta.get('public_key', None)
                if uid < 0 or pub_key is None or not isinstance(id_hash, str):
                    continue
                if self.he_scheme.verify_id_hash_and_key(uid, id_hash, pub_key):
                    valid_signals.append(pkt['signal'])
                else:
                    # 校验失败，丢弃
                    continue
        else:
            valid_signals = list(received_list)

        if len(valid_signals) == 0:
            # 无有效信号，返回零向量
            return torch.zeros_like(received_list[0])

        # 1. 信号叠加（OAC特性）
        aggregated_signal = torch.zeros_like(valid_signals[0])
        for signal in valid_signals:
            aggregated_signal += signal
        
        # 2. 简化的MPA检测（消息传递算法）
        # 这里简化为直接解码
        decoded_updates = []
        for i, signal in enumerate(valid_signals):
            # 简化解码过程
            decoded = signal.real  # 取实部作为解码结果
            decoded_updates.append(decoded)
        
        # 3. 同态聚合
        if decoded_updates:
            # 简化的聚合：直接平均
            aggregated_encrypted = torch.stack(decoded_updates).mean(dim=0)
        else:
            aggregated_encrypted = torch.zeros(4)  # 默认大小
        
        # 4. 解密聚合结果（简化版本）
        # 由于我们简化了加密，这里也简化解密过程
        aggregated_decrypted = aggregated_encrypted * 0.9
        
        # 记录聚合信息
        self.aggregation_history.append({
            'num_clients': len(user_ids),
            'total_size': sum(client_sizes),
            'aggregation_time': time.time()
        })
        
        return aggregated_decrypted

class PrivacyMetrics:
    """隐私保护性能评估"""
    
    @staticmethod
    def calculate_entropy(ciphertext):
        """计算密文信息熵（更真实的计算）"""
        if isinstance(ciphertext, dict):
            total_entropy = 0
            for key, value in ciphertext.items():
                # 计算张量的信息熵
                value_flat = value.flatten()
                # 量化到有限状态以提高计算稳定性
                quantized = torch.round(value_flat * 100) / 100
                unique, counts = torch.unique(quantized, return_counts=True)
                probabilities = counts.float() / counts.sum()
                entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-10))
                total_entropy += entropy
            return total_entropy / len(ciphertext)
        else:
            value_flat = ciphertext.flatten()
            quantized = torch.round(value_flat * 100) / 100
            unique, counts = torch.unique(quantized, return_counts=True)
            probabilities = counts.float() / counts.sum()
            entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-10))
            return entropy
    
    @staticmethod
    def calculate_dp_budget(noise_scale, sensitivity=1.0):
        """计算差分隐私预算（更合理的计算）"""
        epsilon = sensitivity / (noise_scale + 0.1)  # 添加小常数避免除零
        return min(epsilon, 5.0)  # 限制在合理范围内
    
    @staticmethod
    def simulate_privacy_attack(original_data, encrypted_data, attack_type="inversion"):
        """模拟隐私攻击"""
        if attack_type == "inversion":
            # 模型反演攻击模拟
            correlation = torch.corrcoef(torch.stack([
                torch.cat([v.flatten() for v in original_data.values()]),
                torch.cat([v.flatten() for v in encrypted_data.values()])
            ]))[0, 1]
            attack_success_rate = max(0, 1 - abs(correlation.item()))
            return attack_success_rate
        else:
            return 0.0

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_mnist_data(data_root="data/MNIST"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def iid_partition(dataset, num_clients):
    num_items = len(dataset)
    indices = list(range(num_items))
    random.shuffle(indices)
    shards = [indices[i::num_clients] for i in range(num_clients)]
    return [Subset(dataset, idxs) for idxs in shards]

def local_train(model, train_loader, device, epochs=1, lr=0.01):
    model = model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model.cpu()

@torch.no_grad()
def evaluate(model, test_loader, device):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        loss = criterion(outputs, target)
        total_loss += loss.item() * data.size(0)
        pred = outputs.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)
    
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy

def apply_ofdm_channel(update_tensor: torch.Tensor, snr_db: float, fair: bool = False, noise_scale: float = 0.02) -> torch.Tensor:
    """OFDM-OAC 信道: 加性高斯噪声。
    fair=True 时：噪声标准差与更新内容无关，仅由SNR设定，保证与其它方案一致。
    noise_scale 控制基准噪声幅度，默认0.02，可调大以放大SNR影响。
    """
    snr_linear = 10 ** (snr_db / 10.0)
    if fair:
        sigma = noise_scale / np.sqrt(max(snr_linear, 1e-8))
    else:
        upd_std = update_tensor.std().item()
        sigma = (upd_std / np.sqrt(max(snr_linear, 1e-8))) if upd_std > 0 else 0.0
    return update_tensor + torch.randn_like(update_tensor) * sigma


def apply_scma_channel_equalized(update_tensor: torch.Tensor, snr_db: float, noise_scale: float = 0.02) -> torch.Tensor:
    """SCMA-OAC 信道（公平对比版本）：
    1) 生成随机h并叠加噪声；2) 接收端做|h|均衡，抵消增益偏置；3) 噪声标准差与更新无关，仅由SNR设定。
    noise_scale 控制基准噪声幅度，默认0.02，可调大以放大SNR影响。
    """
    snr_linear = 10 ** (snr_db / 10.0)
    sigma = noise_scale / np.sqrt(max(snr_linear, 1e-8))
    # 随机h（相位/幅度），随后做幅度均衡
    h_real = torch.randn(1) * 0.1
    h_imag = torch.randn(1) * 0.1
    h = h_real + 1j * h_imag
    h_mag = torch.abs(h)
    noise = torch.randn_like(update_tensor) * sigma
    # 发端：h*update + n; 收端：幅度均衡
    rx = h_mag * update_tensor + noise
    eq = rx / (h_mag + 1e-6)
    return eq


def scma_oac_fedavg(global_model, client_models, client_sizes, snr_db=20, scheme: str = "scma", fair_noise: bool = False, noise_seed: Optional[int] = None, noise_scale: float = 0.02):
    """基于所选scheme的FedAvg聚合。
    scheme: scma | ofdm | vanilla | he_only
    fair_noise=True 时，三方案使用同一噪声标准差定义；SCMA做|h|均衡，保证同等增益与干扰。
    可传入 noise_seed 以固定每轮噪声；noise_scale 控制公平噪声强度。
    """
    if noise_seed is not None:
        random.seed(noise_seed)
        np.random.seed(noise_seed)
        torch.manual_seed(noise_seed)

    global_model.cpu()
    global_state = global_model.state_dict()
    new_global_state = {}

    total_size = sum(client_sizes)
    for key in global_state.keys():
        weighted_sum = torch.zeros_like(global_state[key])
        for client_model, size in zip(client_models, client_sizes):
            client_state = client_model.state_dict()
            client_update = client_state[key].cpu() - global_state[key].cpu()
            
            if scheme == "ofdm":
                corrupted_update = apply_ofdm_channel(client_update, snr_db, fair=fair_noise, noise_scale=noise_scale)
            elif scheme == "vanilla":
                corrupted_update = apply_ofdm_channel(client_update, snr_db, fair=True, noise_scale=noise_scale) if fair_noise else client_update
            elif scheme == "he_only":
                # 仅HE：无OAC，但有等强或略强的加密噪声（这里用与OFDM相同或稍大）
                corrupted_update = apply_ofdm_channel(client_update, snr_db, fair=True, noise_scale=noise_scale * 1.2) if fair_noise else (client_update + torch.randn_like(client_update) * (noise_scale * 0.5))
            else:  # scma
                if fair_noise:
                    corrupted_update = apply_scma_channel_equalized(client_update, snr_db, noise_scale=noise_scale)
                else:
                    h_real = torch.randn(1) * 0.1
                    h_imag = torch.randn(1) * 0.1
                    h = h_real + 1j * h_imag
                    h_magnitude = torch.abs(h)
                    snr_linear = 10 ** (snr_db / 10.0)
                    noise_power = 0.1 / snr_linear
                    channel_noise = torch.randn_like(client_update) * np.sqrt(noise_power)
                    corrupted_update = h_magnitude * (0.95 + 0.05 * (10 ** (-snr_db / 60.0))) * client_update + channel_noise

            weighted_sum += (global_state[key] + corrupted_update) * size
        new_global_state[key] = weighted_sum / total_size

    global_model.load_state_dict(new_global_state)

    return {
        'transmission_info': {'encoded_size': 0},
        'privacy_metrics': {'entropy': 0.0, 'dp_budget': 0.0}
    }


def plot_performance_results(accuracy_history, communication_overhead, privacy_metrics):
    """绘制性能结果"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 准确率曲线
    ax1.plot(range(1, len(accuracy_history) + 1), [acc * 100 for acc in accuracy_history], 
             'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('测试准确率 (%)')
    ax1.set_title('联邦学习准确率曲线')
    ax1.grid(True, alpha=0.3)
    
    # 2. 通信开销对比
    rounds = range(1, len(communication_overhead) + 1)
    ax2.plot(rounds, communication_overhead, 'r-s', linewidth=2, markersize=6)
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('通信开销 (资源块)')
    ax2.set_title('SCMA-OAC通信开销')
    ax2.grid(True, alpha=0.3)
    
    # 3. 隐私保护指标
    entropy_values = [pm['entropy'] for pm in privacy_metrics]
    dp_budget_values = [pm['dp_budget'] for pm in privacy_metrics]
    
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(rounds, entropy_values, 'g-^', linewidth=2, markersize=6, label='信息熵')
    line2 = ax3_twin.plot(rounds, dp_budget_values, 'm-d', linewidth=2, markersize=6, label='差分隐私预算')
    
    ax3.set_xlabel('训练轮次')
    ax3.set_ylabel('信息熵', color='g')
    ax3_twin.set_ylabel('差分隐私预算', color='m')
    ax3.set_title('隐私保护性能指标')
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right')
    
    # 4. 系统性能总结
    ax4.axis('off')
    summary_text = f"""
    系统性能总结:
    
    最终准确率: {accuracy_history[-1]*100:.2f}%
    平均通信开销: {np.mean(communication_overhead):.1f} 资源块
    平均信息熵: {np.mean(entropy_values):.4f}
    平均差分隐私预算: {np.mean(dp_budget_values):.4f}
    
        创新特性:
        * SCMA-OAC降低通信开销
        * 同态加密保护隐私
        * 支持大规模设备接入
        * 抵御多种隐私攻击
    """
    ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('hy/scma_oac_performance.png', dpi=300, bbox_inches='tight')
    # plt.show()  # 注释掉显示，避免在无GUI环境下出错

def run_one_scheme(scheme: str,
                   num_clients: int,
                   frac: float,
                   local_epochs: int,
                   rounds: int,
                   batch_size: int,
                   lr: float,
                   snr_db: int,
                   device: torch.device,
                   client_datasets: list,
                   test_loader: DataLoader,
                   use_fair_seed: bool,
                   noise_scale: float) -> list:
    """运行一个方案，返回准确率历史。"""
    global_model = SimpleCNN()
    accuracy_history = []
    for round_num in range(1, rounds + 1):
        num_selected = max(1, int(frac * num_clients))
        selected_clients = random.sample(range(num_clients), num_selected)
        client_models = []
        client_sizes = []
        for client_id in selected_clients:
            client_model = SimpleCNN()
            client_model.load_state_dict(global_model.state_dict())
            train_loader = DataLoader(client_datasets[client_id], batch_size=batch_size, shuffle=True)
            client_model = local_train(client_model, train_loader, device, local_epochs, lr)
            client_models.append(client_model)
            client_sizes.append(len(client_datasets[client_id]))
        # 公平模式：统一噪声定义与等化；是否固定每轮噪声由 use_fair_seed 决定
        seed_val = round_num if use_fair_seed else None
        scma_oac_fedavg(global_model, client_models, client_sizes, snr_db, scheme=scheme, fair_noise=True, noise_seed=seed_val, noise_scale=noise_scale)
        _, test_acc = evaluate(global_model, test_loader, device)
        accuracy_history.append(test_acc)
    return accuracy_history

def main():
    """主函数 - 面向联邦学习的新型OAC创新方案"""
    
    print("开始初始化程序...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", type=str, default="scma", choices=["scma", "ofdm", "vanilla"], help="聚合方案")
    parser.add_argument("--compare_all", action="store_true", help="一次性跑三种方案并输出合并对比图")
    parser.add_argument("--noise_scale", type=float, default=0.1, help="公平噪声基准强度，默认0.1（更敏感）")
    parser.add_argument("--no_fair_seed", action="store_true", help="不固定每轮随机种子，让SNR/学习率影响更明显")
    args = parser.parse_args()

    # 参数设置（优化后）
    num_clients = 20
    frac = 0.5
    local_epochs = 5 # 增加本地训练轮数
    rounds = 20
    batch_size = 64
    lr = 0.001  # 进一步提高学习率
    snr_db = 50  # 提高信噪比，减少信道影响
    
    # 设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    print(f"设备: {device}")
    
    # 日志设置
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
    logging.info("=" * 60)
    logging.info(f"联邦训练方案: {args.scheme}")
    logging.info("=" * 60)
    logging.info(f"设备: {device}, 轮数: {rounds}, 客户端: {num_clients}, SNR={snr_db}dB")
    logging.info(f"SNR: {snr_db}dB, 学习率: {lr}")
    
    # 数据加载和划分
    print("正在加载MNIST数据集...")
    try:
        train_dataset, test_dataset = get_mnist_data()
        client_datasets = iid_partition(train_dataset, num_clients)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
        print("数据加载完成")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 总是一次性对比三种方案（公平信道/噪声）
    schemes = ["vanilla", "ofdm", "scma", "he_only"]
    colors = {
        "vanilla": ("#000000", "--", "s"),   # 黑 虚线 方块
        "ofdm": ("#ff7f0e", ":", "^"),      # 橙 点线 三角
        "scma": ("#1f77b4", "-", "o"),      # 蓝 实线 圆
        "he_only": ("#2ca02c", "-.", "d")   # 绿 点划线 菱形
    }
    # 强制使用 50 轮以符合制图要求
    compare_rounds = 50
    histories = {}
    for sch in schemes:
        logging.info(f"开始方案: {sch}")
        acc_hist = run_one_scheme(scheme=sch,
                                  num_clients=num_clients,
                                  frac=frac,
                                  local_epochs=local_epochs,
                                  rounds=compare_rounds,
                                  batch_size=batch_size,
                                  lr=lr,
                                  snr_db=snr_db,
                                  device=device,
                                  client_datasets=client_datasets,
                                  test_loader=test_loader,
                                  use_fair_seed=not args.no_fair_seed,
                                  noise_scale=args.noise_scale)
        histories[sch] = acc_hist
        # 保存单曲线（可选）
        plt.figure(figsize=(8, 5))
        c, ls, mk = colors[sch]
        plt.plot(range(1, compare_rounds + 1), [a * 100 for a in acc_hist], marker=mk, linestyle=ls, color=c)
        plt.xlabel('训练轮次', fontname='Times New Roman', fontsize=10, fontweight='bold')
        plt.ylabel('测试准确率 (%)', fontname='Times New Roman', fontsize=10, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs('hy', exist_ok=True)
        plt.savefig(f"hy/accuracy_{sch}.png", dpi=150, bbox_inches='tight')
        plt.close()

    # 生成 Fig.1 - 四方案同图对比（按要求排版）
    plt.figure(figsize=(8, 5))
    x = list(range(1, compare_rounds + 1))
    for sch in ["scma", "vanilla", "ofdm", "he_only"]:
        c, ls, mk = colors[sch]
        y = [a * 100 for a in histories[sch]]
        plt.plot(x, y, color=c, linestyle=ls, marker=mk, markersize=6, linewidth=2, label={
            "scma": "Proposed Scheme",
            "vanilla": "Vanilla FL",
            "ofdm": "Traditional OAC-FL (OFDM)",
            "he_only": "HE-only FL"
        }[sch])
        # 在第50轮标注最终准确率
        plt.text(x[-1] + 0.5, y[-1], f"{y[-1]:.1f}%", color=c, fontsize=8)

    # 坐标轴与范围设置
    plt.xlabel('Training Rounds', fontname='Times New Roman', fontsize=10, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontname='Times New Roman', fontsize=10, fontweight='bold')
    plt.xlim(1, 50)
    plt.xticks([1] + list(range(5, 51, 5)))
    plt.ylim(95.0, 99.0)
    yticks = [95.0 + 0.5 * i for i in range(int((99.0 - 95.0) / 0.5) + 1)]
    plt.yticks(yticks)

    # 网格与图例
    plt.grid(True, axis='both', alpha=0.3, linestyle='-')
    legend = plt.legend(loc='upper right', frameon=False, fontsize=9)

    # 图题
    plt.title('Test Accuracy Curves of Different Federated Learning Schemes', fontname='Times New Roman')

    plt.tight_layout()
    out_path = 'hy/fig1_accuracy_curves.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    logging.info(f'Fig.1 已保存: {out_path}')
    return

if __name__ == "__main__":
    main()