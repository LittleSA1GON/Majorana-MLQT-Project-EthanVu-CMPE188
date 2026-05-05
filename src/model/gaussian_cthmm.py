from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class CTHMMConfig:
    num_states: int = 2
    num_features: int = 2
    max_em_iters: int = 25
    tol: float = 1e-5
    min_variance: float = 1e-6
    min_rate: float = 1e-12
    max_rate: float = 1e12
    max_emission_points: int = 250_000
    random_state: int = 42
    verbose: bool = True


@dataclass
class CTHMMFitResult:
    log_likelihoods: list[float]
    converged: bool
    n_iter: int
    n_sequences: int
    n_observations: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NeuralCTHMM(nn.Module):
    def __init__(self, config: CTHMMConfig):
        super().__init__()
        self.config = config
        self.num_states = config.num_states
        self.num_features = config.num_features
        
        # Learnable parameters
        self.means = nn.Parameter(torch.randn(self.num_states, self.num_features))
        self.log_vars = nn.Parameter(torch.zeros(self.num_states, self.num_features))
        self.log_rates = nn.Parameter(torch.randn(self.num_states, self.num_states) - 3)  # Initialize low
        
    def forward(self, sequences: list[torch.Tensor]) -> torch.Tensor:
        # Compute log-likelihood for sequences
        total_ll = 0.0
        for seq in sequences:
            ll = self._sequence_log_likelihood(seq)
            total_ll += ll
        return total_ll / len(sequences)  # Average
    
    def _sequence_log_likelihood(self, y: torch.Tensor) -> torch.Tensor:
        # Simplified: assume uniform initial, compute emission and transition
        T = y.shape[0]
        log_alpha = torch.zeros(T, self.num_states, device=y.device)
        
        # Initial
        log_alpha[0] = self._emission_log_prob(y[0]) + torch.log(torch.ones(self.num_states) / self.num_states)
        
        for t in range(1, T):
            trans = self._transition_matrix(1.0)  # Assume dt=1
            log_alpha[t] = self._emission_log_prob(y[t]) + torch.logsumexp(log_alpha[t-1].unsqueeze(1) + torch.log(trans), dim=0)
        
        return torch.logsumexp(log_alpha[-1], dim=0)
    
    def _emission_log_prob(self, y: torch.Tensor) -> torch.Tensor:
        var = torch.exp(self.log_vars)
        diff = y.unsqueeze(0) - self.means
        return -0.5 * (torch.sum(torch.log(2 * torch.pi * var), dim=1) + torch.sum(diff**2 / var, dim=1))
    
    def _transition_matrix(self, dt: float) -> torch.Tensor:
        rates = torch.exp(self.log_rates)
        # Simple exponential for CTMC
        return torch.exp(-rates * dt)


def logsumexp(a: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    max_a = np.max(a, axis=axis, keepdims=True)
    max_a[~np.isfinite(max_a)] = 0.0
    out = max_a + np.log(np.sum(np.exp(a - max_a), axis=axis, keepdims=True) + 1e-300)
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def _sanitize_y(y: np.ndarray, num_features: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError(f"Expected y shaped [T, C], got {y.shape}")
    if y.shape[1] < num_features:
        raise ValueError(f"Expected at least {num_features} feature columns, got {y.shape[1]}")
    y = y[:, :num_features]
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def _sample_points(sequences: list[np.ndarray], max_points: int, seed: int, num_features: int) -> np.ndarray:
    parts = []
    for y in sequences:
        y = _sanitize_y(y, num_features)
        if y.shape[0] > 0:
            parts.append(y)
    if not parts:
        raise ValueError("No observations available for CT-HMM initialization.")
    X = np.concatenate(parts, axis=0)
    if X.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        X = X[idx]
    return X


def _pca_projection(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xc = X - np.mean(X, axis=0, keepdims=True)
    if Xc.shape[1] == 1:
        direction = np.ones(1, dtype=np.float64)
    else:
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        direction = vt[0]
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    score = Xc @ direction
    return score, direction


def _transition_matrix_to_rates(A: np.ndarray, dt: float, min_rate: float, max_rate: float) -> tuple[float, float]:
    A = np.asarray(A, dtype=np.float64)
    dt = float(max(dt, 1e-30))
    a01 = float(np.clip(A[0, 1], 1e-12, 1.0 - 1e-12))
    a10 = float(np.clip(A[1, 0], 1e-12, 1.0 - 1e-12))
    s = float(np.clip(a01 + a10, 1e-12, 1.0 - 1e-12))
    total_rate = -math.log(max(1.0 - s, 1e-12)) / dt
    gamma01 = total_rate * (a01 / s)
    gamma10 = total_rate * (a10 / s)
    return (
        float(np.clip(gamma01, min_rate, max_rate)),
        float(np.clip(gamma10, min_rate, max_rate)),
    )


def _rates_to_transition_matrix(gamma01: float, gamma10: float, dt: float) -> np.ndarray:
    gamma01 = max(float(gamma01), 0.0)
    gamma10 = max(float(gamma10), 0.0)
    dt = max(float(dt), 1e-30)
    lam = gamma01 + gamma10
    if lam <= 0:
        return np.eye(2, dtype=np.float64)
    e = math.exp(-lam * dt) if lam * dt < 700 else 0.0
    p01 = gamma01 / lam * (1.0 - e)
    p10 = gamma10 / lam * (1.0 - e)
    return np.asarray([[1.0 - p01, p01], [p10, 1.0 - p10]], dtype=np.float64)


class GaussianContinuousTimeHMM:
    """Two-state Gaussian CT-HMM with diagonal covariances."""

    def __init__(self, config: CTHMMConfig | None = None):
        self.config = config or CTHMMConfig()
        if self.config.num_states != 2:
            raise ValueError("This implementation currently supports exactly two states.")
        self.pi = np.asarray([0.5, 0.5], dtype=np.float64)
        self.means = np.zeros((2, self.config.num_features), dtype=np.float64)
        self.vars = np.ones((2, self.config.num_features), dtype=np.float64)
        self.gamma01 = 1.0
        self.gamma10 = 1.0
        self.fit_result: CTHMMFitResult | None = None
        self.projection_direction: np.ndarray | None = None

    @property
    def rates(self) -> dict[str, float]:
        return {"gamma_01": float(self.gamma01), "gamma_10": float(self.gamma10)}

    @property
    def lifetimes(self) -> dict[str, float]:
        return {
            "tau_0": float(1.0 / max(self.gamma01, 1e-300)),
            "tau_1": float(1.0 / max(self.gamma10, 1e-300)),
            "tau_mean": float(0.5 / max(self.gamma01, 1e-300) + 0.5 / max(self.gamma10, 1e-300)),
        }

    def transition_matrix(self, dt: float) -> np.ndarray:
        A = _rates_to_transition_matrix(self.gamma01, self.gamma10, dt)
        A = np.clip(A, 1e-300, 1.0)
        A = A / A.sum(axis=1, keepdims=True)
        return A

    def initialize(self, sequences: list[np.ndarray], dts: list[float] | np.ndarray) -> None:
        cfg = self.config
        X = _sample_points(sequences, cfg.max_emission_points, cfg.random_state, cfg.num_features)
        score, direction = _pca_projection(X)
        self.projection_direction = direction
        threshold = float(np.median(score))
        labels = (score > threshold).astype(np.int64)
        if len(np.unique(labels)) < 2:
            labels = np.arange(X.shape[0]) % 2
        means = []
        vars_ = []
        for k in range(2):
            Xk = X[labels == k]
            if Xk.shape[0] == 0:
                Xk = X
            means.append(np.mean(Xk, axis=0))
            vars_.append(np.maximum(np.var(Xk, axis=0), cfg.min_variance))
        means = np.asarray(means, dtype=np.float64)
        vars_ = np.asarray(vars_, dtype=np.float64)

        # Order states along the readout projection for stable naming.
        projected_means = means @ direction
        order = np.argsort(projected_means)
        self.means = means[order]
        self.vars = vars_[order]
        self.pi = np.asarray([0.5, 0.5], dtype=np.float64)

        # Crude rate initialization from nearest-mean decoded traces.
        trans = np.ones((2, 2), dtype=np.float64) * 1e-3
        dts_arr = np.asarray(dts, dtype=np.float64)
        median_dt = float(np.median(dts_arr[dts_arr > 0])) if np.any(dts_arr > 0) else 1.0
        for y in sequences[: min(len(sequences), 500)]:
            y = _sanitize_y(y, cfg.num_features)
            dist = ((y[:, None, :] - self.means[None, :, :]) ** 2 / np.maximum(self.vars[None, :, :], cfg.min_variance)).sum(axis=2)
            states = np.argmin(dist, axis=1)
            if states.shape[0] >= 2:
                for a, b in zip(states[:-1], states[1:]):
                    trans[int(a), int(b)] += 1.0
        A = trans / trans.sum(axis=1, keepdims=True)
        self.gamma01, self.gamma10 = _transition_matrix_to_rates(A, median_dt, cfg.min_rate, cfg.max_rate)

    def emission_logprob(self, y: np.ndarray) -> np.ndarray:
        y = _sanitize_y(y, self.config.num_features)
        vars_ = np.maximum(self.vars, self.config.min_variance)
        log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * vars_), axis=1)
        diff = y[:, None, :] - self.means[None, :, :]
        quad = -0.5 * np.sum(diff * diff / vars_[None, :, :], axis=2)
        return log_norm[None, :] + quad

    def forward_backward(self, y: np.ndarray, dt: float) -> tuple[float, np.ndarray, np.ndarray]:
        logB = self.emission_logprob(y)
        T = logB.shape[0]
        K = 2
        A = self.transition_matrix(dt)
        logA = np.log(np.clip(A, 1e-300, 1.0))
        logpi = np.log(np.clip(self.pi, 1e-300, 1.0))

        alpha = np.zeros((T, K), dtype=np.float64)
        beta = np.zeros((T, K), dtype=np.float64)
        alpha[0] = logpi + logB[0]
        for t in range(1, T):
            alpha[t] = logB[t] + logsumexp(alpha[t - 1][:, None] + logA, axis=0)
        loglik = float(logsumexp(alpha[-1], axis=0))

        beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            beta[t] = logsumexp(logA + logB[t + 1][None, :] + beta[t + 1][None, :], axis=1)

        log_gamma = alpha + beta - loglik
        gamma = np.exp(log_gamma)
        gamma = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

        xi_sum = np.zeros((K, K), dtype=np.float64)
        for t in range(T - 1):
            log_xi = alpha[t][:, None] + logA + logB[t + 1][None, :] + beta[t + 1][None, :] - loglik
            xi = np.exp(log_xi)
            xi_sum += xi / max(np.sum(xi), 1e-300)
        return loglik, gamma, xi_sum

    def fit(self, sequences: list[np.ndarray], dts: list[float] | np.ndarray) -> CTHMMFitResult:
        if len(sequences) == 0:
            raise ValueError("No sequences provided to GaussianContinuousTimeHMM.fit().")
        cfg = self.config
        dts = np.asarray(dts, dtype=np.float64)
        if dts.shape[0] != len(sequences):
            raise ValueError("dts length must match sequences length.")
        dts = np.where(np.isfinite(dts) & (dts > 0), dts, np.nanmedian(dts[dts > 0]) if np.any(dts > 0) else 1.0)
        median_dt = float(np.median(dts[dts > 0])) if np.any(dts > 0) else 1.0
        clean_sequences = [_sanitize_y(y, cfg.num_features) for y in sequences if np.asarray(y).shape[0] >= 2]
        clean_dts = np.asarray([d for y, d in zip(sequences, dts) if np.asarray(y).shape[0] >= 2], dtype=np.float64)
        if not clean_sequences:
            raise ValueError("All sequences were too short after filtering.")

        self.initialize(clean_sequences, clean_dts)
        log_likelihoods: list[float] = []
        converged = False
        n_obs = int(sum(y.shape[0] for y in clean_sequences))

        for it in range(1, cfg.max_em_iters + 1):
            pi_acc = np.zeros(2, dtype=np.float64)
            gamma_sum = np.zeros(2, dtype=np.float64)
            y_sum = np.zeros((2, cfg.num_features), dtype=np.float64)
            y2_sum = np.zeros((2, cfg.num_features), dtype=np.float64)
            xi_sum = np.zeros((2, 2), dtype=np.float64)
            total_loglik = 0.0

            for y, dt in zip(clean_sequences, clean_dts):
                loglik, post, xi = self.forward_backward(y, float(dt))
                total_loglik += loglik
                pi_acc += post[0]
                gamma_sum += post.sum(axis=0)
                y_sum += post.T @ y
                y2_sum += post.T @ (y * y)
                xi_sum += xi

            self.pi = pi_acc / max(np.sum(pi_acc), 1e-300)
            self.pi = np.clip(self.pi, 1e-8, 1.0)
            self.pi = self.pi / self.pi.sum()

            for k in range(2):
                denom = max(gamma_sum[k], 1e-300)
                self.means[k] = y_sum[k] / denom
                var = y2_sum[k] / denom - self.means[k] ** 2
                self.vars[k] = np.maximum(var, cfg.min_variance)

            A = xi_sum + 1e-6
            A = A / A.sum(axis=1, keepdims=True)
            self.gamma01, self.gamma10 = _transition_matrix_to_rates(A, median_dt, cfg.min_rate, cfg.max_rate)

            # Keep stable state ordering along the learned projection / mean separation.
            if self.projection_direction is not None:
                order = np.argsort(self.means @ self.projection_direction)
                if not np.array_equal(order, np.arange(2)):
                    self.means = self.means[order]
                    self.vars = self.vars[order]
                    self.pi = self.pi[order]
                    # Swapping state labels swaps rates.
                    self.gamma01, self.gamma10 = self.gamma10, self.gamma01

            log_likelihoods.append(float(total_loglik))
            if cfg.verbose:
                per_obs = total_loglik / max(n_obs, 1)
                print(
                    f"[cthmm em {it:03d}] loglik={total_loglik:.3f} per_obs={per_obs:.6f} "
                    f"gamma01={self.gamma01:.6g} gamma10={self.gamma10:.6g}",
                    flush=True,
                )
            if it >= 2:
                prev = log_likelihoods[-2]
                rel = abs(total_loglik - prev) / max(1.0, abs(prev))
                if rel < cfg.tol:
                    converged = True
                    break

        self.fit_result = CTHMMFitResult(
            log_likelihoods=log_likelihoods,
            converged=converged,
            n_iter=len(log_likelihoods),
            n_sequences=len(clean_sequences),
            n_observations=n_obs,
        )
        return self.fit_result

    def score(self, sequences: list[np.ndarray], dts: list[float] | np.ndarray) -> dict[str, float]:
        total = 0.0
        n_obs = 0
        for y, dt in zip(sequences, dts):
            y = _sanitize_y(y, self.config.num_features)
            if y.shape[0] < 2:
                continue
            ll, _, _ = self.forward_backward(y, float(dt))
            total += ll
            n_obs += y.shape[0]
        return {
            "log_likelihood": float(total),
            "log_likelihood_per_observation": float(total / max(n_obs, 1)),
            "n_observations": int(n_obs),
        }

    def posterior(self, y: np.ndarray, dt: float) -> np.ndarray:
        _, gamma, _ = self.forward_backward(y, dt)
        return gamma

    def predict_states(self, y: np.ndarray, dt: float, method: str = "viterbi") -> np.ndarray:
        if method == "posterior":
            return np.argmax(self.posterior(y, dt), axis=1).astype(np.int64)
        return self.viterbi(y, dt)

    def viterbi(self, y: np.ndarray, dt: float) -> np.ndarray:
        logB = self.emission_logprob(y)
        T = logB.shape[0]
        A = self.transition_matrix(dt)
        logA = np.log(np.clip(A, 1e-300, 1.0))
        logpi = np.log(np.clip(self.pi, 1e-300, 1.0))
        delta = np.zeros((T, 2), dtype=np.float64)
        psi = np.zeros((T, 2), dtype=np.int64)
        delta[0] = logpi + logB[0]
        for t in range(1, T):
            vals = delta[t - 1][:, None] + logA
            psi[t] = np.argmax(vals, axis=0)
            delta[t] = logB[t] + np.max(vals, axis=0)
        states = np.zeros(T, dtype=np.int64)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def sample(self, T: int, dt: float, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.config.random_state if seed is None else seed)
        A = self.transition_matrix(dt)
        states = np.zeros(int(T), dtype=np.int64)
        states[0] = int(rng.choice(2, p=self.pi))
        for t in range(1, int(T)):
            states[t] = int(rng.choice(2, p=A[states[t - 1]]))
        y = np.zeros((int(T), self.config.num_features), dtype=np.float32)
        for k in range(2):
            idx = states == k
            if np.any(idx):
                y[idx] = rng.normal(self.means[k], np.sqrt(np.maximum(self.vars[k], self.config.min_variance)), size=(int(np.sum(idx)), self.config.num_features))
        return y, states

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "pi": self.pi.tolist(),
            "means": self.means.tolist(),
            "vars": self.vars.tolist(),
            "gamma01": float(self.gamma01),
            "gamma10": float(self.gamma10),
            "rates": self.rates,
            "lifetimes": self.lifetimes,
            "projection_direction": None if self.projection_direction is None else self.projection_direction.tolist(),
            "fit_result": None if self.fit_result is None else self.fit_result.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GaussianContinuousTimeHMM":
        cfg = CTHMMConfig(**data.get("config", {}))
        model = cls(cfg)
        model.pi = np.asarray(data["pi"], dtype=np.float64)
        model.means = np.asarray(data["means"], dtype=np.float64)
        model.vars = np.asarray(data["vars"], dtype=np.float64)
        model.gamma01 = float(data["gamma01"])
        model.gamma10 = float(data["gamma10"])
        if data.get("projection_direction") is not None:
            model.projection_direction = np.asarray(data["projection_direction"], dtype=np.float64)
        return model

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "GaussianContinuousTimeHMM":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
