from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
except Exception as exc:
    raise ImportError(f"torch is required for cnn_gru_qcvv_model.py: {exc}")

try:
    from .utilities import QCVVBatch, ensure_torch, make_length_mask
except ImportError:
    from utilities import QCVVBatch, ensure_torch, make_length_mask


@dataclass
class CNNGRUQCVVConfig:
    in_channels: int = 2
    coord_dim: int = 3
    family_vocab_size: int = 8
    run_vocab_size: int = 16
    id_embed_dim: int = 8
    conv_channels: int = 64
    gru_hidden: int = 128
    gru_layers: int = 1
    embedding_dim: int = 128
    num_states: int = 4
    dropout: float = 0.10
    eps: float = 1e-5
    use_family_meta: bool = False
    use_run_meta: bool = False


class CNNGRUQCVVModel(nn.Module):
    def __init__(self, config: CNNGRUQCVVConfig | None = None):
        super().__init__()
        self.config = config or CNNGRUQCVVConfig()

        c = self.config.conv_channels
        g = 8 if c % 8 == 0 else 4 if c % 4 == 0 else 1
        self.conv = nn.Sequential(
            nn.Conv1d(self.config.in_channels, c, kernel_size=7, padding=3),
            nn.GroupNorm(g, c),
            nn.GELU(),
            nn.Conv1d(c, c, kernel_size=5, padding=2),
            nn.GroupNorm(g, c),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )

        self.gru = nn.GRU(
            input_size=c,
            hidden_size=self.config.gru_hidden,
            num_layers=self.config.gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.family_emb = nn.Embedding(self.config.family_vocab_size, self.config.id_embed_dim)
        self.run_emb = nn.Embedding(self.config.run_vocab_size, self.config.id_embed_dim)

        context_in = (
            2 * self.config.gru_hidden
            + self.config.coord_dim
            + 1  # sample_dt
            + 1  # valid_length ratio
            + self.config.id_embed_dim
            + self.config.id_embed_dim
        )

        self.fuse = nn.Sequential(
            nn.Linear(context_in, self.config.embedding_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.embedding_dim, self.config.embedding_dim),
            nn.GELU(),
        )

        self.state_head = nn.Linear(self.config.embedding_dim, self.config.num_states)
        self.switch_head = nn.Linear(self.config.embedding_dim, 1)
        self.quality_head = nn.Linear(self.config.embedding_dim, 1)
        self.reconstruction_head = nn.Linear(2 * self.config.gru_hidden, self.config.in_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.7)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def masked_pool(self, x: torch.Tensor, valid_length: torch.Tensor) -> torch.Tensor:
        mask = make_length_mask(valid_length, max_len=x.shape[1]).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (x * mask).sum(dim=1) / denom

    def _normalize_traces(self, traces: torch.Tensor, valid_length: torch.Tensor) -> torch.Tensor:
        traces = torch.nan_to_num(traces, nan=0.0, posinf=0.0, neginf=0.0)
        mask = make_length_mask(valid_length, max_len=traces.shape[1]).float().unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (traces * mask).sum(dim=1, keepdim=True) / denom
        var = (((traces - mean) * mask) ** 2).sum(dim=1, keepdim=True) / denom
        std = torch.sqrt(var.clamp_min(self.config.eps))
        traces = (traces - mean) / std
        traces = torch.nan_to_num(traces, nan=0.0, posinf=0.0, neginf=0.0)
        traces = torch.clamp(traces, min=-8.0, max=8.0)
        return traces

    def forward(self, batch: QCVVBatch) -> dict[str, torch.Tensor]:
        traces = ensure_torch(batch.traces, dtype=torch.float32)
        valid_length = ensure_torch(batch.valid_length, device=traces.device, dtype=torch.long)
        valid_length = valid_length.clamp(min=1, max=traces.shape[1])

        coords = ensure_torch(batch.coord_normalized, device=traces.device, dtype=torch.float32)
        coords = torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)

        sample_dt = ensure_torch(batch.sample_dt, device=traces.device, dtype=torch.float32)
        sample_dt = torch.nan_to_num(sample_dt, nan=1e-6, posinf=1e-3, neginf=1e-6).clamp_min(1e-8)
        sample_dt = torch.log10(sample_dt).unsqueeze(-1)

        if batch.family_id is None or not self.config.use_family_meta:
            family_id = torch.zeros(traces.shape[0], device=traces.device, dtype=torch.long)
        else:
            family_id = ensure_torch(batch.family_id, device=traces.device, dtype=torch.long)

        if batch.run_id is None or not self.config.use_run_meta:
            run_id = torch.zeros(traces.shape[0], device=traces.device, dtype=torch.long)
        else:
            run_id = ensure_torch(batch.run_id, device=traces.device, dtype=torch.long)

        traces = self._normalize_traces(traces, valid_length)

        x = traces.transpose(1, 2).contiguous()
        x = self.conv(x)
        x = x.transpose(1, 2)

        gru_out, _ = self.gru(x)
        gru_out = torch.nan_to_num(gru_out, nan=0.0, posinf=0.0, neginf=0.0)
        reconstruction = self.reconstruction_head(gru_out)
        mask = make_length_mask(valid_length, max_len=traces.shape[1]).float().unsqueeze(-1)
        recon_err = ((reconstruction - traces) ** 2) * mask
        recon_loss = recon_err.sum() / mask.sum().clamp_min(1.0)

        pooled = self.masked_pool(gru_out, valid_length)

        valid_ratio = (valid_length.float() / traces.shape[1]).unsqueeze(-1)
        fam_e = self.family_emb(family_id)
        run_e = self.run_emb(run_id)

        fused = torch.cat([pooled, coords, sample_dt, valid_ratio, fam_e, run_e], dim=-1)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
        embedding = self.fuse(fused)
        embedding = torch.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)

        state_logits = self.state_head(embedding)
        switch_logits = self.switch_head(embedding)
        quality_score = self.quality_head(embedding)

        state_logits = torch.nan_to_num(state_logits, nan=0.0, posinf=0.0, neginf=0.0)
        switch_logits = torch.nan_to_num(switch_logits, nan=0.0, posinf=0.0, neginf=0.0)
        quality_score = torch.nan_to_num(quality_score, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "embedding": embedding,
            "state_logits": state_logits,
            "switch_logits": switch_logits,
            "quality_score": quality_score,
            "reconstruction": torch.nan_to_num(reconstruction, nan=0.0, posinf=0.0, neginf=0.0),
            "recon_loss": torch.nan_to_num(recon_loss, nan=0.0, posinf=0.0, neginf=0.0),
        }
