"""
fdd.py — Online Frequency-Aware Domain Discriminator (Section 3.3).
Extracts low-frequency Fourier descriptors from input images and performs
Bayesian Gaussian domain matching with online covariance updates.
"""

import torch
import numpy as np


class FrequencyDomainDiscriminator:
    """
    Lightweight, training-free domain detector using low-frequency
    spectral features (Eq. 6-15 in the paper).
    """

    def __init__(self, freq_radius=16, threshold=1.5, shrinkage=0.1,
                 init_var=1.0, diagonal=True, device="cuda"):
        self.l = freq_radius              # frequency radius
        self.L = 2 * freq_radius + 1      # patch side length
        self.df = self.L ** 2             # descriptor dimension
        self.tau = threshold              # τ: new-domain threshold
        self.eps = shrinkage              # ε: covariance shrinkage
        self.sigma2_0 = init_var          # σ²₀: initial variance
        self.diagonal = diagonal          # use diagonal covariance
        self.device = device

        # domain bank: list of dicts {mu, sigma, count}
        self.domains = []

    @property
    def num_domains(self):
        return len(self.domains)

    # ─── Low-frequency feature extraction (Eq. 16-17) ─────────────────
    @torch.no_grad()
    def extract_descriptor(self, images):
        """
        images: [B, 3, H, W] tensor (float, 0-1 or 0-255)
        returns: [B, df] low-frequency descriptors
        """
        # convert to grayscale
        if images.shape[1] == 3:
            # standard ITU-R BT.601 weights
            gray = (0.2989 * images[:, 0] +
                    0.5870 * images[:, 1] +
                    0.1140 * images[:, 2])   # [B, H, W]
        else:
            gray = images[:, 0]

        B, H, W = gray.shape

        # 2D DFT + shift DC to centre
        F = torch.fft.fft2(gray)                       # [B, H, W] complex
        F_shifted = torch.fft.fftshift(F, dim=(-2, -1))
        magnitude = F_shifted.abs()                    # [B, H, W]

        # log-scale to compress dynamic range (raw magnitudes ~10k,
        # log1p brings them to ~10, making Mahalanobis distances meaningful)
        magnitude = torch.log1p(magnitude)

        # crop low-frequency patch centred at DC
        cr, cc = H // 2, W // 2
        l = self.l
        low_freq = magnitude[:, cr - l: cr + l + 1,
                                cc - l: cc + l + 1]     # [B, L, L]

        # flatten
        descriptors = low_freq.reshape(B, -1)           # [B, df]
        return descriptors

    # ─── Mahalanobis distance (Eq. 6, 9, Appendix B) ─────────────────
    def _mahalanobis(self, z, domain_id):
        """
        Compute shrinkage-regularised Mahalanobis distance,
        normalised by dimensionality so that within-distribution
        batch means score ~1.0 and the threshold τ=1.5 is meaningful.
        z: [df] descriptor (batch mean)
        Returns scalar distance.
        """
        d = self.domains[domain_id]
        diff = z - d["mu"]                              # [df]

        if self.diagonal:
            # Σ_reg = (1-ε)diag(σ²) + εI
            sigma_reg = (1 - self.eps) * d["sigma"] + self.eps
            dist = (diff ** 2 / sigma_reg).sum()
        else:
            cov = d["sigma"]                             # [df, df]
            cov_reg = (1 - self.eps) * cov + self.eps * torch.eye(
                self.df, device=cov.device)
            dist = diff @ torch.linalg.solve(cov_reg, diff)

        # normalise: raw Mahalanobis in d dimensions has E[m] ≈ d
        # dividing by df makes within-distribution distances ≈ 1
        return (dist / self.df).item()

    # ─── Domain matching (Eq. 8-9) ────────────────────────────────────
    @torch.no_grad()
    def detect_domain(self, images):
        """
        Given a batch of images, determine which domain they belong to.
        Returns (domain_id, is_new_domain).
        """
        descs = self.extract_descriptor(images)         # [B, df]
        z_mean = descs.mean(dim=0)                      # [df]  (Eq. 8)

        if self.num_domains == 0:
            # first batch ever — initialise domain 0
            self._init_domain(z_mean, descs)
            return 0, True

        # compute Mahalanobis distance to each domain
        distances = [self._mahalanobis(z_mean, i)
                     for i in range(self.num_domains)]
        d_min = min(distances)
        i_star = distances.index(d_min)

        if d_min < self.tau:
            # known domain — update statistics
            self._update_domain(i_star, descs)
            return i_star, False
        else:
            # new domain detected
            self._init_domain(z_mean, descs)
            return self.num_domains - 1, True

    # ─── Initialise new domain (Eq. 15) ───────────────────────────────
    def _init_domain(self, z_mean, descriptors=None):
        """
        Initialise a new domain. If descriptors are provided, compute
        the initial variance from the batch (data-driven). Otherwise
        fall back to sigma2_0.
        """
        if descriptors is not None and descriptors.shape[0] > 1:
            # data-driven: use batch variance + floor for stability
            if self.diagonal:
                sigma = descriptors.var(dim=0).clamp(min=self.sigma2_0)
            else:
                centered = descriptors - z_mean.unsqueeze(0)
                sigma = (centered.T @ centered) / descriptors.shape[0]
                sigma = sigma + self.sigma2_0 * torch.eye(
                    self.df, device=sigma.device)
        else:
            if self.diagonal:
                sigma = torch.full((self.df,), self.sigma2_0,
                                   device=z_mean.device)
            else:
                sigma = self.sigma2_0 * torch.eye(self.df,
                                                   device=z_mean.device)
        self.domains.append({
            "mu": z_mean.clone(),
            "sigma": sigma,
            "count": 1,
        })

    # ─── Online update of domain statistics (Eq. 10-14) ───────────────
    @torch.no_grad()
    def _update_domain(self, domain_id, descriptors):
        """
        Robust weighted online update using soft responsibilities.
        descriptors: [B, df]
        """
        d = self.domains[domain_id]
        B = descriptors.shape[0]

        # compute per-sample Mahalanobis distances for soft weights (Eq. 10)
        diffs = descriptors - d["mu"].unsqueeze(0)      # [B, df]
        if self.diagonal:
            sigma_reg = (1 - self.eps) * d["sigma"] + self.eps
            dists = (diffs ** 2 / sigma_reg.unsqueeze(0)).sum(dim=1)  # [B]
        else:
            cov_reg = ((1 - self.eps) * d["sigma"] +
                       self.eps * torch.eye(self.df, device=d["sigma"].device))
            dists = (diffs @ torch.linalg.solve(cov_reg, diffs.T)
                     ).diag()                            # [B]

        # normalised weights (Eq. 10)
        log_w = -0.5 * dists
        log_w = log_w - log_w.max()                     # numerical stability
        w = torch.softmax(log_w, dim=0)                 # [B], sums to 1

        c = d["count"]

        # update mean (Eq. 12)
        weighted_mean = (w.unsqueeze(1) * descriptors).sum(dim=0)  # [df]
        mu_new = (c * d["mu"] + weighted_mean) / (c + 1)

        # update covariance (Eq. 13, simplified with μ_old approx)
        if self.diagonal:
            weighted_var = (w.unsqueeze(1) *
                            (diffs ** 2)).sum(dim=0)     # [df]
            sigma_new = (c * d["sigma"] + weighted_var) / (c + 1)
        else:
            weighted_scatter = sum(
                w[b] * diffs[b:b+1].T @ diffs[b:b+1]
                for b in range(B))                       # [df, df]
            sigma_new = (c * d["sigma"] + weighted_scatter) / (c + 1)

        # commit (Eq. 14)
        d["mu"] = mu_new
        d["sigma"] = sigma_new
        d["count"] = c + 1

    # ─── Get all domain statistics (for diagnostics) ──────────────────
    def get_summary(self):
        return [(i, d["count"],
                 d["mu"].norm().item(),
                 d["sigma"].mean().item() if self.diagonal
                 else d["sigma"].diag().mean().item())
                for i, d in enumerate(self.domains)]

    # ─── Distance to all domains (for teacher selection) ──────────────
    @torch.no_grad()
    def distances_to_all_domains(self, images):
        """
        Compute Mahalanobis distance from batch to every known domain.
        Returns dict {domain_id: distance}.
        Used by pseudo_labels.py to decide which experts to include as teachers.
        """
        if self.num_domains == 0:
            return {}
        descs = self.extract_descriptor(images)
        z_mean = descs.mean(dim=0)
        return {i: self._mahalanobis(z_mean, i) for i in range(self.num_domains)}
