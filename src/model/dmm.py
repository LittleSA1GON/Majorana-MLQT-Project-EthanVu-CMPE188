from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:
    raise ImportError(f"torch is required for dmm.py: {exc}")

try:
    from .utilities import QCVVBatch, ensure_torch, make_length_mask
except ImportError:
    from utilities import QCVVBatch, ensure_torch, make_length_mask


@dataclass
class DMMQCVVConfig:
    in_channels: int = 2
    coord_dim: int = 3
    family_vocab_size: int = 8
    run_vocab_size: int = 16
    id_embed_dim: int = 8
    conv_channels: int = 48
    encoder_hidden: int = 64
    latent_dim: int = 24
    static_dim: int = 32
    embedding_dim: int = 128
    num_states: int = 4
    temporal_downsample: int = 4
    dropout: float = 0.10
    eps: float = 1e-5
    use_family_meta: bool = False
    use_run_meta: bool = False


class DeepMarkovQCVVModel(nn.Module):
    def __init__(self, config: DMMQCVVConfig | None = None):
        super().__init__()
        self.config = config or DMMQCVVConfig()

        c = self.config.conv_channels
        g = 8 if c % 8 == 0 else 4 if c % 4 == 0 else 1
        self.frontend = nn.Sequential(
            nn.Conv1d(self.config.in_channels, c, kernel_size=7, padding=3),
            nn.GroupNorm(g, c),
            nn.GELU(),
            nn.Conv1d(c, c, kernel_size=5, padding=2),
            nn.GroupNorm(g, c),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )

        self.encoder_rnn = nn.GRU(
            input_size=c,
            hidden_size=self.config.encoder_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.family_emb = nn.Embedding(self.config.family_vocab_size, self.config.id_embed_dim)
        self.run_emb = nn.Embedding(self.config.run_vocab_size, self.config.id_embed_dim)

        static_in = self.config.coord_dim + 1 + 1 + self.config.id_embed_dim + self.config.id_embed_dim
        self.static_proj = nn.Sequential(
            nn.Linear(static_in, self.config.static_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )

        prior_in = self.config.latent_dim + self.config.static_dim
        post_in = 2 * self.config.encoder_hidden + self.config.latent_dim + self.config.static_dim
        self.prior_net = nn.Sequential(
            nn.Linear(prior_in, 2 * self.config.encoder_hidden),
            nn.GELU(),
            nn.Linear(2 * self.config.encoder_hidden, 2 * self.config.latent_dim),
        )
        self.post_net = nn.Sequential(
            nn.Linear(post_in, 2 * self.config.encoder_hidden),
            nn.GELU(),
            nn.Linear(2 * self.config.encoder_hidden, 2 * self.config.latent_dim),
        )
        self.emit_net = nn.Sequential(
            nn.Linear(self.config.latent_dim + self.config.static_dim, 64),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(64, self.config.in_channels),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.config.latent_dim + self.config.static_dim, self.config.embedding_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.state_head = nn.Linear(self.config.embedding_dim, self.config.num_states)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        for name, param in self.encoder_rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _normalize_traces(self, traces: torch.Tensor, valid_length: torch.Tensor) -> torch.Tensor:
        traces = torch.nan_to_num(traces, nan=0.0, posinf=0.0, neginf=0.0)
        mask = make_length_mask(valid_length, max_len=traces.shape[1]).float().unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (traces * mask).sum(dim=1, keepdim=True) / denom
        var = (((traces - mean) * mask) ** 2).sum(dim=1, keepdim=True) / denom
        std = torch.sqrt(var.clamp_min(self.config.eps))
        traces = (traces - mean) / std
        traces = torch.nan_to_num(traces, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.clamp(traces, min=-8.0, max=8.0)

    def _downsample_time(self, x_btc: torch.Tensor, valid_length: torch.Tensor):
        ds = max(1, int(self.config.temporal_downsample))
        if ds == 1:
            return x_btc, valid_length

        x = x_btc.transpose(1, 2)
        x = F.avg_pool1d(x, kernel_size=ds, stride=ds, ceil_mode=True)
        x = x.transpose(1, 2)
        valid_ds = torch.div(valid_length + ds - 1, ds, rounding_mode="floor")
        valid_ds = valid_ds.clamp(min=1, max=x.shape[1])
        return x, valid_ds

    @staticmethod
    def _split_mu_logvar(stats: torch.Tensor):
        return torch.chunk(stats, 2, dim=-1)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar.clamp(min=-12.0, max=12.0))
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _kl_diag_gaussian(q_mu, q_logvar, p_mu, p_logvar):
        qv = torch.exp(q_logvar)
        pv = torch.exp(p_logvar)
        term = p_logvar - q_logvar + (qv + (q_mu - p_mu) ** 2) / (pv + 1e-8) - 1.0
        return 0.5 * term.sum(dim=-1)

    def _masked_mean(self, x: torch.Tensor, valid_length: torch.Tensor) -> torch.Tensor:
        mask = make_length_mask(valid_length, max_len=x.shape[1]).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (x * mask).sum(dim=1) / denom

    def forward(self, batch: QCVVBatch, sample_latent: bool | None = None) -> dict[str, torch.Tensor]:
        if sample_latent is None:
            sample_latent = self.training

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

        traces_norm = self._normalize_traces(traces, valid_length)
        x_target, valid_ds = self._downsample_time(traces_norm, valid_length)

        x = traces_norm.transpose(1, 2).contiguous()
        x = self.frontend(x)
        x = x.transpose(1, 2)
        x, valid_ds = self._downsample_time(x, valid_length)

        enc_out, _ = self.encoder_rnn(x)
        enc_out = torch.nan_to_num(enc_out, nan=0.0, posinf=0.0, neginf=0.0)

        valid_ratio = (valid_length.float() / traces.shape[1]).unsqueeze(-1)
        fam_e = self.family_emb(family_id)
        run_e = self.run_emb(run_id)
        static = self.static_proj(torch.cat([coords, sample_dt, valid_ratio, fam_e, run_e], dim=-1))

        B, T, _ = enc_out.shape
        prev_z = torch.zeros(B, self.config.latent_dim, device=enc_out.device)
        z_seq = []
        recon_seq = []
        kl_terms = []

        for t in range(T):
            p_stats = self.prior_net(torch.cat([prev_z, static], dim=-1))
            p_mu, p_logvar = self._split_mu_logvar(p_stats)
            p_logvar = p_logvar.clamp(min=-8.0, max=8.0)

            q_stats = self.post_net(torch.cat([enc_out[:, t], prev_z, static], dim=-1))
            q_mu, q_logvar = self._split_mu_logvar(q_stats)
            q_logvar = q_logvar.clamp(min=-8.0, max=8.0)

            z_t = self._reparameterize(q_mu, q_logvar, sample=sample_latent)
            x_hat_t = self.emit_net(torch.cat([z_t, static], dim=-1))

            z_seq.append(z_t)
            recon_seq.append(x_hat_t)
            kl_terms.append(self._kl_diag_gaussian(q_mu, q_logvar, p_mu, p_logvar))
            prev_z = z_t

        z_seq = torch.stack(z_seq, dim=1)
        recon_seq = torch.stack(recon_seq, dim=1)

        mask = make_length_mask(valid_ds, max_len=T).float()
        mask_exp = mask.unsqueeze(-1)

        recon_err = ((recon_seq - x_target[:, :T, :]) ** 2).sum(dim=-1)
        recon_loss = (recon_err * mask).sum() / mask.sum().clamp_min(1.0)

        kl = torch.stack(kl_terms, dim=1)
        kl_loss = (kl * mask).sum() / mask.sum().clamp_min(1.0)

        z_pool = self._masked_mean(z_seq, valid_ds)
        embedding = self.classifier(torch.cat([z_pool, static], dim=-1))
        state_logits = self.state_head(embedding)

        return {
            "embedding": torch.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0),
            "state_logits": torch.nan_to_num(state_logits, nan=0.0, posinf=0.0, neginf=0.0),
            "latent_sequence": torch.nan_to_num(z_seq, nan=0.0, posinf=0.0, neginf=0.0),
            "reconstruction": torch.nan_to_num(recon_seq, nan=0.0, posinf=0.0, neginf=0.0),
            "recon_loss": torch.nan_to_num(recon_loss, nan=0.0, posinf=0.0, neginf=0.0),
            "kl_loss": torch.nan_to_num(kl_loss, nan=0.0, posinf=0.0, neginf=0.0),
        }
