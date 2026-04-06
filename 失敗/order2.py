# train_regretnet_assignment_init_DSD_with_updatecheck_jp.py
# ------------------------------------------------------------
# 目的：
#   - Pの初期値を DSD（deterministic serial dictatorship）に近づけてから本学習
#   - 「本当に損失の減少方向に更新されているか」を update-check で可視化
#
# 仕様：
#   - ログは基本 OEvio, NE, SP（mean）だけ
#   - ただし print_every のときだけ、最初のバッチで update-check を追加表示
#   - 最後に代表入力の P を表示
#
# 注意：
#   - update-check は “同じバッチ” で forward をやり直すので計算コストが増えます
#     （print_every の epoch かつ先頭バッチのみ）
# ------------------------------------------------------------

import os
import random
from dataclasses import dataclass
from typing import Tuple, List

import torch
import torch.nn as nn


# ============================================================
# 選好（順位）生成・表現
# ============================================================
def all_perms(n: int) -> torch.Tensor:
    import itertools
    perms = list(itertools.permutations(range(n)))
    return torch.tensor(perms, dtype=torch.long)


def random_profile(n: int) -> torch.Tensor:
    prof = []
    goods = list(range(n))
    for _ in range(n):
        g = goods[:]
        random.shuffle(g)
        prof.append(g)
    return torch.tensor(prof, dtype=torch.long)


def canonical_profile(n: int) -> torch.Tensor:
    return torch.tensor([list(range(n)) for _ in range(n)], dtype=torch.long)


def reverse_profile(n: int) -> torch.Tensor:
    return torch.tensor([list(reversed(range(n))) for _ in range(n)], dtype=torch.long)


def encode_profile(profile_2d: torch.Tensor) -> torch.Tensor:
    n = profile_2d.shape[0]
    x = torch.zeros((n, n, n), dtype=torch.float32, device=profile_2d.device)
    for i in range(n):
        for t in range(n):
            a = int(profile_2d[i, t].item())
            x[i, t, a] = 1.0
    return x


# ============================================================
# DSD (Deterministic Serial Dictatorship) 教師割当
#   - 優先順位は 0,1,2,...,n-1 固定
# ============================================================
@torch.no_grad()
def dsd_allocation(profile: torch.Tensor) -> torch.Tensor:
    """
    profile: (B, n, n) long
    return : (B, n, n) float32 0/1 permutation matrix
    """
    device = profile.device
    B, n, _ = profile.shape
    P = torch.zeros((B, n, n), dtype=torch.float32, device=device)

    for b in range(B):
        remaining = set(range(n))
        for i in range(n):
            for t in range(n):
                a = int(profile[b, i, t].item())
                if a in remaining:
                    P[b, i, a] = 1.0
                    remaining.remove(a)
                    break
    return P


# ============================================================
# Sinkhorn（log-space で数値安定化）
# ============================================================
def sinkhorn_logspace(
    logits: torch.Tensor,
    n_iters: int = 80,
    eps: float = 1e-12,
) -> torch.Tensor:
    logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    logK = logits.to(torch.float64)

    for _ in range(n_iters):
        logK = logK - torch.logsumexp(logK, dim=-1, keepdim=True)  # row normalize
        logK = logK - torch.logsumexp(logK, dim=-2, keepdim=True)  # col normalize
        logK = torch.nan_to_num(logK, nan=-100.0, posinf=0.0, neginf=-100.0)

    x = torch.exp(logK).to(torch.float32)
    x = torch.clamp(x, min=eps)
    x = x / (x.sum(dim=-1, keepdim=True) + eps)
    x = x / (x.sum(dim=-2, keepdim=True) + eps)
    x = torch.clamp(x, min=eps)
    return x


# ============================================================
# メカニズムネット
# ============================================================
class MechanismNet(nn.Module):
    def __init__(self, n: int, hidden: int = 256, sinkhorn_iters: int = 80):
        super().__init__()
        self.n = n
        self.sinkhorn_iters = sinkhorn_iters
        inp = n * n * n
        out = n * n
        self.mlp = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out),
        )

    def forward(self, profile: torch.Tensor) -> torch.Tensor:
        b, n, _ = profile.shape
        x = torch.stack([encode_profile(profile[k]).reshape(-1) for k in range(b)], dim=0)
        logits = self.mlp(x).reshape(b, n, n)
        return sinkhorn_logspace(logits, n_iters=self.sinkhorn_iters)


# ============================================================
# 累積和（選好順）
# ============================================================
def cumdiff_along_pref(diff: torch.Tensor, pref: torch.Tensor) -> torch.Tensor:
    b, one, n = diff.shape
    idx = pref.unsqueeze(-1)
    gathered = torch.gather(
        diff.unsqueeze(2).expand(b, one, n, n),
        3,
        idx
    ).squeeze(-1)
    return torch.cumsum(gathered, dim=-1)


# ============================================================
# NE/SP（|x|-x = 2*max(-x,0)）
# ============================================================
def neg_part_penalty(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.clamp(-x, min=0.0)


def NEvio(P: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
    b, n, _ = P.shape
    total = torch.zeros((b,), device=P.device)
    for i in range(n):
        Pi = P[:, i, :]
        pref_i = profile[:, i, :].unsqueeze(1)
        for j in range(n):
            if j == i:
                continue
            Pj = P[:, j, :]
            diff = (Pi - Pj).unsqueeze(1)
            cum = cumdiff_along_pref(diff, pref_i)
            total += neg_part_penalty(cum).sum(dim=-1).squeeze(1)
    return total


def SPvio(P_truth: torch.Tensor, P_mis: torch.Tensor, profile_truth: torch.Tensor) -> torch.Tensor:
    b, n, _ = P_truth.shape
    total = torch.zeros((b,), device=P_truth.device)
    for i in range(n):
        diff = (P_truth[:, i, :] - P_mis[:, i, :]).unsqueeze(1)
        pref_i = profile_truth[:, i, :].unsqueeze(1)
        cum = cumdiff_along_pref(diff, pref_i)
        total += neg_part_penalty(cum).sum(dim=-1).squeeze(1)
    return total


# ============================================================
# P' 列挙（0.2刻み二重確率）
# ============================================================
def enumerate_discrete_bistochastic(n: int = 4, denom: int = 5) -> torch.Tensor:
    MATS: List[torch.Tensor] = []
    colrem0 = [denom] * n

    def gen_row_compositions(total: int, bounds: List[int]) -> List[List[int]]:
        out = []

        def rec(i: int, rem: int, cur: List[int]):
            if i == n - 1:
                x = rem
                if 0 <= x <= bounds[i]:
                    out.append(cur + [x])
                return
            for x in range(min(bounds[i], rem) + 1):
                rec(i + 1, rem - x, cur + [x])

        rec(0, total, [])
        return out

    def backtrack(r: int, colrem: List[int], rows: List[List[int]]):
        if r == n:
            if all(c == 0 for c in colrem):
                mat = torch.tensor(rows, dtype=torch.float32) / float(denom)
                MATS.append(mat)
            return
        for row in gen_row_compositions(denom, colrem):
            new_colrem = [colrem[j] - row[j] for j in range(n)]
            backtrack(r + 1, new_colrem, rows + [row])

    backtrack(0, colrem0, [])
    return torch.stack(MATS, dim=0)


# ============================================================
# OE（softmin + min(s) gate）
# ============================================================
def smoothstep5_poly(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    return 10.0 * x3 - 15.0 * x4 + 6.0 * x5


def OEvio_softmin_smoothstep5(
    P: torch.Tensor,
    profile: torch.Tensor,
    Pprime_set: torch.Tensor,
    eps_c: float,
    tau_softmin: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = P.device
    b, n, _ = P.shape
    M = Pprime_set.shape[0]
    Pp = Pprime_set.to(device)

    diff = Pp.unsqueeze(0).expand(b, M, n, n) - P.unsqueeze(1)
    s = torch.zeros((b, M), device=device)
    gain = torch.zeros((b, M), device=device)

    w = torch.arange(n, 0, -1, device=device, dtype=torch.float32)

    for i in range(n):
        pref_i = profile[:, i, :]
        idx = pref_i.unsqueeze(1).unsqueeze(-1)

        diff_i = torch.gather(
            diff[:, :, i, :].unsqueeze(2).expand(b, M, n, n),
            3,
            idx.expand(b, M, n, 1)
        ).squeeze(-1)

        cum = torch.cumsum(diff_i, dim=-1)

        s += (2.0 * torch.clamp(-cum, min=0.0)).sum(dim=-1)
        gain += (diff_i * w.view(1, 1, n)).sum(dim=-1)

    min_s = torch.min(s, dim=1).values
    scaled = -(s - min_s.unsqueeze(1)) / float(tau_softmin)
    weights = torch.softmax(scaled, dim=1)
    sum_gain_soft = torch.sum(gain * weights, dim=1)

    x = (min_s / float(eps_c))
    x = torch.clamp(x, 0.0, 1.0)
    gate = 1.0 - smoothstep5_poly(x)

    OE_raw = gate * sum_gain_soft
    OE_raw = torch.nan_to_num(OE_raw, nan=0.0, posinf=0.0, neginf=0.0)

    OE_pos = torch.clamp(OE_raw, min=0.0)
    return OE_pos.mean(), OE_raw.mean()


# ============================================================
# 診断：パラメータノルム
# ============================================================
@torch.no_grad()
def param_vector_norm(net: nn.Module) -> float:
    s = 0.0
    for p in net.parameters():
        if p.requires_grad:
            v = p.detach()
            s += float(torch.sum(v * v).item())
    return s ** 0.5


# ============================================================
# 学習設定
# ============================================================
@dataclass
class TrainConfig:
    n: int = 4
    train_profiles: int = 30
    test_profiles: int = 100

    batch_size: int = 10
    epochs: int = 600

    # --- DSD pretrain ---
    dsd_pretrain_epochs: int = 300
    dsd_pretrain_lr: float = 2e-3
    dsd_pretrain_print_every: int = 50

    # --- main training ---
    lr: float = 3e-4
    warmup_epochs: int = 200
    post_warmup_lr_mult: float = 0.3

    rho_init: float = 5_000.0
    rho_growth: float = 1.15
    rho_max: float = 1.0e5
    lambda_lr: float = 1.0
    lam_max: float = 1.0e4

    sinkhorn_iters: int = 80
    grad_clip: float = 1.0
    param_clip: float = 50.0

    eps_c: float = 1e-6
    tau_softmin: float = 5e-2

    print_every: int = 25

    # update-check を出すか
    enable_update_check: bool = True

    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir: str = "./outputs"
    model_name: str = "regretnet_mechanism.pt"


@torch.no_grad()
def eval_means(
    net: nn.Module,
    dataset: torch.Tensor,
    Pprime_set: torch.Tensor,
    misreports: torch.Tensor,
    eps_c: float,
    tau_softmin: float,
) -> Tuple[float, float, float]:
    device = dataset.device
    b, n, _ = dataset.shape
    R = misreports.shape[0]

    P = net(dataset)
    if not torch.isfinite(P).all():
        return float("nan"), float("nan"), float("nan")

    NE_mean = NEvio(P, dataset).mean().item()

    sp_acc = torch.zeros((b,), device=device)
    for r in range(R):
        rep = misreports[r].view(1, -1).expand(b, -1)
        for i in range(n):
            prof_mis = dataset.clone()
            prof_mis[:, i, :] = rep
            P_mis = net(prof_mis)
            sp_acc += SPvio(P, P_mis, dataset)
    SP_mean = (sp_acc / float(n * R)).mean().item()

    _, OE_raw_mean = OEvio_softmin_smoothstep5(
        P, dataset, Pprime_set, eps_c=eps_c, tau_softmin=tau_softmin
    )
    OE_mean = OE_raw_mean.item()
    return OE_mean, NE_mean, SP_mean


def dsd_pretrain(net: nn.Module, train_set: torch.Tensor, cfg: TrainConfig) -> None:
    device = train_set.device
    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=cfg.dsd_pretrain_lr)
    indices = list(range(train_set.shape[0]))
    mse = torch.nn.MSELoss()

    for ep in range(1, cfg.dsd_pretrain_epochs + 1):
        random.shuffle(indices)
        for start in range(0, len(indices), cfg.batch_size):
            batch_idx = indices[start:start + cfg.batch_size]
            prof = train_set[batch_idx]

            target = dsd_allocation(prof)
            pred = net(prof)

            loss = mse(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()

            with torch.no_grad():
                for p in net.parameters():
                    if p.requires_grad:
                        p.clamp_(min=-cfg.param_clip, max=cfg.param_clip)

        if ep % cfg.dsd_pretrain_print_every == 0 or ep == 1:
            net.eval()
            with torch.no_grad():
                prof0 = train_set[: min(cfg.batch_size, train_set.shape[0])]
                P_pred = net(prof0)
                P_tgt = dsd_allocation(prof0)
                err = torch.mean(torch.abs(P_pred - P_tgt)).item()
            net.train()
            print(f"[DSD pretrain {ep:4d}] mean|P-DSD|={err:.6e}")


def train():
    cfg = TrainConfig()

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(cfg.device)
    n = cfg.n

    train_set = torch.stack([random_profile(n) for _ in range(cfg.train_profiles)], dim=0).to(device)
    test_set = torch.stack([random_profile(n) for _ in range(cfg.test_profiles)], dim=0).to(device)

    Pprime_set = enumerate_discrete_bistochastic(n=n, denom=5).to(device)
    misreports = all_perms(n).to(device)
    R = misreports.shape[0]

    net = MechanismNet(n=n, hidden=256, sinkhorn_iters=cfg.sinkhorn_iters).to(device)

    # ============================================================
    # 1) DSD 事前学習
    # ============================================================
    print("=== DSD pretraining: initialize P near DSD ===")
    dsd_pretrain(net, train_set, cfg)

    net.eval()
    with torch.no_grad():
        P_tr = net(train_set)
        P_tr_dsd = dsd_allocation(train_set)
        P_te = net(test_set)
        P_te_dsd = dsd_allocation(test_set)
        tr_err = torch.mean(torch.abs(P_tr - P_tr_dsd)).item()
        te_err = torch.mean(torch.abs(P_te - P_te_dsd)).item()
    print(f"[After DSD pretrain] train mean|P-DSD|={tr_err:.6e}  test mean|P-DSD|={te_err:.6e}")
    net.train()

    # ============================================================
    # 2) 本学習
    # ============================================================
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    lam_NE = torch.tensor(0.0, device=device)
    lam_SP = torch.tensor(0.0, device=device)
    rho = cfg.rho_init

    indices = list(range(train_set.shape[0]))

    for epoch in range(1, cfg.epochs + 1):
        random.shuffle(indices)

        if epoch == cfg.warmup_epochs + 1:
            for pg in opt.param_groups:
                pg["lr"] = pg["lr"] * cfg.post_warmup_lr_mult

        for start in range(0, len(indices), cfg.batch_size):
            batch_idx = indices[start:start + cfg.batch_size]
            prof = train_set[batch_idx]
            B = prof.shape[0]

            # ===== forward (before) =====
            P = net(prof)
            if not torch.isfinite(P).all():
                print("P has NaN/Inf! stopping.")
                return

            ne_stat = NEvio(P, prof).max()

            sp_vec = torch.zeros((B,), device=device)
            for i in range(n):
                prof_rep = prof.unsqueeze(1).expand(B, R, n, n).contiguous().view(B * R, n, n)
                rep = misreports.unsqueeze(0).expand(B, R, n).contiguous().view(B * R, n)
                prof_rep[:, i, :] = rep
                P_mis = net(prof_rep).view(B, R, n, n)
                if not torch.isfinite(P_mis).all():
                    print("P_mis has NaN/Inf! stopping.")
                    return
                for r in range(R):
                    sp_vec += SPvio(P, P_mis[:, r, :, :], prof)
            sp_stat = (sp_vec / float(n * R)).max()

            oe_pos_mean, _ = OEvio_softmin_smoothstep5(
                P, prof, Pprime_set, eps_c=cfg.eps_c, tau_softmin=cfg.tau_softmin
            )

            if epoch <= cfg.warmup_epochs:
                loss = (ne_stat ** 2) + (sp_stat ** 2)
            else:
                loss = (
                    oe_pos_mean
                    + lam_NE * ne_stat + 0.5 * rho * (ne_stat ** 2)
                    + lam_SP * sp_stat + 0.5 * rho * (sp_stat ** 2)
                )

            if not torch.isfinite(loss):
                print("loss is NaN/Inf! stopping.")
                return

            loss_before = loss.detach()
            pnorm_before = param_vector_norm(net)

            # ===== update =====
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()

            with torch.no_grad():
                for p in net.parameters():
                    if p.requires_grad:
                        p.clamp_(min=-cfg.param_clip, max=cfg.param_clip)

            pnorm_after = param_vector_norm(net)

            # λ 更新
            if epoch > cfg.warmup_epochs:
                with torch.no_grad():
                    lam_NE = torch.clamp(lam_NE + cfg.lambda_lr * ne_stat, min=0.0, max=cfg.lam_max)
                    lam_SP = torch.clamp(lam_SP + cfg.lambda_lr * sp_stat, min=0.0, max=cfg.lam_max)

            # ===== update-check (print_every の epoch かつ先頭バッチのみ) =====
            do_check = (
                cfg.enable_update_check
                and start == 0
                and (epoch % cfg.print_every == 0 or epoch == 1)
            )
            if do_check:
                with torch.no_grad():
                    P_after = net(prof)
                    ne_after = NEvio(P_after, prof).max()

                    sp_vec_a = torch.zeros((B,), device=device)
                    for i in range(n):
                        prof_rep_a = prof.unsqueeze(1).expand(B, R, n, n).contiguous().view(B * R, n, n)
                        rep_a = misreports.unsqueeze(0).expand(B, R, n).contiguous().view(B * R, n)
                        prof_rep_a[:, i, :] = rep_a
                        P_mis_a = net(prof_rep_a).view(B, R, n, n)
                        for r in range(R):
                            sp_vec_a += SPvio(P_after, P_mis_a[:, r, :, :], prof)
                    sp_after = (sp_vec_a / float(n * R)).max()

                    oe_pos_after, _ = OEvio_softmin_smoothstep5(
                        P_after, prof, Pprime_set, eps_c=cfg.eps_c, tau_softmin=cfg.tau_softmin
                    )

                    if epoch <= cfg.warmup_epochs:
                        loss_after = (ne_after ** 2) + (sp_after ** 2)
                    else:
                        loss_after = (
                            oe_pos_after
                            + lam_NE * ne_after + 0.5 * rho * (ne_after ** 2)
                            + lam_SP * sp_after + 0.5 * rho * (sp_after ** 2)
                        )

                    delta = float(loss_after.item() - loss_before.item())
                    step = abs(pnorm_after - pnorm_before)
                    print(
                        f"  [update-check] loss_before={loss_before.item():.6e} "
                        f"loss_after={loss_after.item():.6e} delta={delta:.3e} | "
                        f"param_norm={pnorm_before:.3e}->{pnorm_after:.3e} step~{step:.3e}"
                    )

        if epoch > cfg.warmup_epochs:
            rho = min(rho * cfg.rho_growth, cfg.rho_max)

        # ===== epoch log (mean only) =====
        if epoch % cfg.print_every == 0 or epoch == 1:
            net.eval()
            with torch.no_grad():
                tr_OE, tr_NE, tr_SP = eval_means(
                    net, train_set, Pprime_set, misreports, cfg.eps_c, cfg.tau_softmin
                )
                te_OE, te_NE, te_SP = eval_means(
                    net, test_set, Pprime_set, misreports, cfg.eps_c, cfg.tau_softmin
                )
                print(
                    f"[epoch {epoch:4d}] "
                    f"train OEvio={tr_OE:.6e} NE={tr_NE:.6e} SP={tr_SP:.6e} | "
                    f"test OEvio={te_OE:.6e} NE={te_NE:.6e} SP={te_SP:.6e}"
                )
            net.train()

    # 保存
    os.makedirs(cfg.out_dir, exist_ok=True)
    save_path = os.path.join(cfg.out_dir, cfg.model_name)
    torch.save({"state_dict": net.state_dict(), "config": cfg.__dict__}, save_path)

    # 代表入力に対する P
    net.eval()
    with torch.no_grad():
        profiles = {
            "canonical_all_same_[0..n-1]": canonical_profile(n),
            "reverse_all_same_[n-1..0]": reverse_profile(n),
            "random_profile_1": random_profile(n),
            "random_profile_2": random_profile(n),
        }

        print("\n--- 学習済みメカニズムの出力確率行列 P（代表プロファイル入力）---")
        for name, prof in profiles.items():
            P0 = net(prof.unsqueeze(0).to(device))[0].detach().cpu()
            print(f"\n[{name}] profile=")
            print(prof)
            print(f"[{name}] P=")
            print(P0)

    print("\nSaved model:", save_path)


if __name__ == "__main__":
    train()
