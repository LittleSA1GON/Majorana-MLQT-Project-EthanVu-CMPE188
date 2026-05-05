from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


def _log_gaussian_diag(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    # x: [T, D]
    var = torch.clamp(var, min=1e-6)
    diff = x - mean.unsqueeze(0)
    return -0.5 * (
        torch.sum(torch.log(2.0 * torch.pi * var))
        + torch.sum((diff ** 2) / var.unsqueeze(0), dim=1)
    )


@dataclass
class HSMMConfig:
    num_states: int = 4
    min_duration: int = 1
    max_duration: int = 16
    duration_smoothing: float = 1.0
    transition_smoothing: float = 1e-3
    random_state: int = 42


class GaussianHSMMQCVV:
    def __init__(self, config: HSMMConfig | None = None):
        self.config = config or HSMMConfig()
        self.pi = None                # [K]
        self.A = None                 # [K, K]
        self.means = None             # [K, D]
        self.vars = None              # [K, D]
        self.duration_logprob = None  # [K, max_duration+1]

    def fit_supervised(
        self,
        X: np.ndarray,            # [N, D]
        state_labels: np.ndarray, # [N]
        sequence_id: np.ndarray | None = None,
    ):
        X = np.asarray(X, dtype=np.float64)
        z = np.asarray(state_labels, dtype=np.int64)
        K = self.config.num_states
        D = X.shape[1]

        self.means = np.zeros((K, D), dtype=np.float64)
        self.vars = np.ones((K, D), dtype=np.float64)

        for k in range(K):
            mask = z == k
            if np.any(mask):
                xk = X[mask]
                self.means[k] = np.mean(xk, axis=0)
                self.vars[k] = np.var(xk, axis=0) + 1e-6

        self._fit_initial_and_transitions(z, sequence_id)
        self._fit_durations(z, sequence_id)
        return self

    def fit_unsupervised_init(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        kmeans = KMeans(
            n_clusters=self.config.num_states,
            random_state=self.config.random_state,
            n_init=10,
        )
        z = kmeans.fit_predict(X)
        return self.fit_supervised(X, z, sequence_id=None)

    def _fit_initial_and_transitions(self, z: np.ndarray, sequence_id: np.ndarray | None):
        K = self.config.num_states
        smooth = self.config.transition_smoothing

        pi = np.full(K, smooth, dtype=np.float64)
        A = np.full((K, K), smooth, dtype=np.float64)

        if sequence_id is None:
            pi[z[0]] += 1.0
            for i in range(len(z) - 1):
                A[z[i], z[i + 1]] += 1.0
        else:
            seq = np.asarray(sequence_id)
            unique_seq = np.unique(seq)
            for sid in unique_seq:
                idx = np.flatnonzero(seq == sid)
                if idx.size == 0:
                    continue
                pi[z[idx[0]]] += 1.0
                for a, b in zip(idx[:-1], idx[1:]):
                    A[z[a], z[b]] += 1.0

        self.pi = pi / pi.sum()
        self.A = A / A.sum(axis=1, keepdims=True)

    def _fit_durations(self, z: np.ndarray, sequence_id: np.ndarray | None):
        K = self.config.num_states
        max_d = self.config.max_duration
        smooth = self.config.duration_smoothing

        counts = np.full((K, max_d + 1), smooth, dtype=np.float64)

        def consume_run(labels: np.ndarray):
            if labels.size == 0:
                return
            cur = int(labels[0])
            dur = 1
            for t in range(1, labels.size):
                nxt = int(labels[t])
                if nxt == cur and dur < max_d:
                    dur += 1
                else:
                    counts[cur, min(dur, max_d)] += 1.0
                    cur = nxt
                    dur = 1
            counts[cur, min(dur, max_d)] += 1.0

        if sequence_id is None:
            consume_run(z)
        else:
            seq = np.asarray(sequence_id)
            for sid in np.unique(seq):
                idx = np.flatnonzero(seq == sid)
                consume_run(z[idx])

        probs = counts / counts.sum(axis=1, keepdims=True)
        self.duration_logprob = np.log(np.clip(probs, 1e-12, None))

    def emission_loglik(self, X: np.ndarray) -> np.ndarray:
        if self.means is None or self.vars is None:
            raise ValueError("Model must be fit before calling emission_loglik.")
        X = np.asarray(X, dtype=np.float64)
        K = self.config.num_states
        out = np.zeros((X.shape[0], K), dtype=np.float64)
        for k in range(K):
            out[:, k] = _log_gaussian_diag(X, self.means[k], self.vars[k])
        return out

    def viterbi_decode(self, X: np.ndarray) -> np.ndarray:
        """
        Approximate HSMM decode using dynamic programming over state durations.

        Input:
            X: [T, D]
        Output:
            states: [T]
        """
        if self.pi is None or self.A is None or self.duration_logprob is None:
            raise ValueError("Model must be fit before decode.")

        X = np.asarray(X, dtype=np.float64)
        T = X.shape[0]
        K = self.config.num_states
        max_d = self.config.max_duration

        emit = self.emission_loglik(X)  # [T, K]
        seg_emit = np.full((T, K, max_d + 1), -np.inf, dtype=np.float64)

        for t in range(T):
            for d in range(1, min(max_d, t + 1) + 1):
                seg = emit[t - d + 1:t + 1]  # [d, K]
                seg_emit[t, :, d] = np.sum(seg, axis=0)

        dp = np.full((T, K), -np.inf, dtype=np.float64)
        back = [[None for _ in range(K)] for _ in range(T)]

        for d in range(1, min(max_d, T) + 1):
            t = d - 1
            score = np.log(self.pi) + self.duration_logprob[:, d] + seg_emit[t, :, d]
            better = score > dp[t]
            dp[t, better] = score[better]
            for k in np.flatnonzero(better):
                back[t][k] = (-1, -1, d)

        for t in range(T):
            for k in range(K):
                best_score = dp[t, k]
                best_prev = back[t][k]
                for d in range(1, min(max_d, t + 1) + 1):
                    prev_t = t - d
                    if prev_t < 0:
                        continue
                    trans = np.log(np.clip(self.A[:, k], 1e-12, None))
                    cand = dp[prev_t] + trans + self.duration_logprob[k, d] + seg_emit[t, k, d]
                    j = int(np.argmax(cand))
                    if cand[j] > best_score:
                        best_score = float(cand[j])
                        best_prev = (prev_t, j, d)
                dp[t, k] = best_score
                back[t][k] = best_prev

        states = np.zeros(T, dtype=np.int64)
        t = T - 1
        k = int(np.argmax(dp[T - 1]))
        while t >= 0:
            prev_t, prev_k, d = back[t][k]
            start = max(0, t - d + 1)
            states[start:t + 1] = k
            t = prev_t
            k = prev_k if prev_k >= 0 else 0

        return states


    def summarize_states(self, states: np.ndarray, sample_dt: float | None = None) -> dict:
        states = np.asarray(states, dtype=np.int64)
        if states.size == 0:
            return {"num_segments": 0, "state_counts": {}}

        segments = []
        cur = int(states[0])
        dur = 1
        for z in states[1:]:
            z = int(z)
            if z == cur:
                dur += 1
            else:
                segments.append((cur, dur))
                cur = z
                dur = 1
        segments.append((cur, dur))

        out = {
            "num_segments": int(len(segments)),
            "state_counts": {str(k): int(v) for k, v in zip(*np.unique(states, return_counts=True))},
            "duration_steps_mean_by_state": {},
            "duration_steps_median_by_state": {},
        }
        for k in np.unique(states):
            durs = np.asarray([d for s, d in segments if s == int(k)], dtype=np.float64)
            if durs.size:
                out["duration_steps_mean_by_state"][str(int(k))] = float(np.mean(durs))
                out["duration_steps_median_by_state"][str(int(k))] = float(np.median(durs))
                if sample_dt is not None and sample_dt > 0:
                    out.setdefault("duration_seconds_mean_by_state", {})[str(int(k))] = float(np.mean(durs) * sample_dt)
        return out

    def predict_states(self, X: np.ndarray, sequence_id: np.ndarray | None = None) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if sequence_id is None:
            return self.viterbi_decode(X)

        seq = np.asarray(sequence_id)
        out = np.zeros(X.shape[0], dtype=np.int64)
        for sid in np.unique(seq):
            idx = np.flatnonzero(seq == sid)
            out[idx] = self.viterbi_decode(X[idx])
        return out


class NeuralGaussianHSMMQCVV(nn.Module):
    """A GPU-capable HSMM-style model with Gaussian emissions and duration priors."""

    def __init__(self, config: HSMMConfig | None = None, feature_dim: int = 1):
        super().__init__()
        self.config = config or HSMMConfig()
        K = self.config.num_states
        self.log_pi = nn.Parameter(torch.zeros(K))
        self.log_A = nn.Parameter(torch.zeros(K, K))
        self.means = nn.Parameter(torch.randn(K, feature_dim))
        self.log_vars = nn.Parameter(torch.zeros(K, feature_dim))
        self.duration_logprob = nn.Parameter(torch.zeros(K, self.config.max_duration + 1))

    def emission_loglik(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(dtype=torch.float32)
        var = torch.exp(self.log_vars).clamp(min=1e-6)
        diff = X.unsqueeze(1) - self.means.unsqueeze(0)
        log_norm = -0.5 * torch.sum(torch.log(2.0 * torch.pi * var), dim=1)
        quad = -0.5 * torch.sum(diff * diff / var.unsqueeze(0), dim=2)
        return log_norm.unsqueeze(0) + quad

    def transition_logprob(self) -> torch.Tensor:
        A = torch.softmax(self.log_A, dim=1)
        return torch.log(A + 1e-12)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        emit = self.emission_loglik(X)
        logpi = torch.log_softmax(self.log_pi, dim=0)
        logA = self.transition_logprob()
        T = X.shape[0]
        dp = torch.full((T, self.config.num_states), -1e9, dtype=torch.float32, device=X.device)
        dp[0] = logpi + emit[0]
        for t in range(1, T):
            dp[t] = torch.logsumexp(dp[t - 1].unsqueeze(1) + logA, dim=0) + emit[t]
        return torch.logsumexp(dp[-1], dim=0)

    def predict_states(self, X: torch.Tensor) -> torch.Tensor:
        # Simple posterior decoding via emission logits and transition priors
        emit = self.emission_loglik(X)
        logpi = torch.log_softmax(self.log_pi, dim=0)
        logA = self.transition_logprob()
        T = X.shape[0]
        delta = torch.zeros((T, self.config.num_states), device=X.device)
        psi = torch.zeros((T, self.config.num_states), dtype=torch.long, device=X.device)
        delta[0] = logpi + emit[0]
        for t in range(1, T):
            vals = delta[t - 1].unsqueeze(1) + logA
            delta[t], psi[t] = torch.max(vals, dim=0)
            delta[t] = delta[t] + emit[t]
        states = torch.empty(T, dtype=torch.long, device=X.device)
        states[-1] = torch.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states
