# qFitLatent

SE(3)-invariant model that predicts per-residue sidechain torsion distributions from backbone structure and sequence, trained on qFit multiconformer crystal structures.

---

## Architecture

### Inputs

| Symbol | Shape | Description |
|--------|-------|-------------|
| $a_i$ | $(N,)$ | amino acid tokens, $a_i \in \{0, \ldots, 20\}$ |
| $R_i$ | $(N, 3, 3)$ | backbone rotation matrix (to local frame, primary altloc) |
| $t_i$ | $(N, 3)$ | $C_\alpha$ position (frame origin) |

Local frames are constructed from $N$, $C_\alpha$, $C$ atoms of the primary (blank) altloc:

$$\hat{x} = \frac{C - C_\alpha}{\| C - C_\alpha \|}, \quad \hat{z} = \frac{\hat{x} \times (N - C_\alpha)}{\| \hat{x} \times (N - C_\alpha) \|}, \quad \hat{y} = \hat{z} \times \hat{x}$$

so that $R_i = [\hat{x} \; \hat{y} \; \hat{z}]$ is a proper orthonormal frame that puts the $C_\alpha$ at the origin, the $C$ along the positive x-axis, and the $N$ in the xy-plane with positive y.

### Output head

The final single representation $s_i$ is projected to a $k$-component **von Mises mixture** over sidechain $\chi$ angles:

$$\pi_i = \mathrm{softmax}(W_\pi\, s_i)$$

$$\mu_{ik} = \pi \tanh(W_\mu\, s_i) \in (-\pi, \pi]^{D_\chi}$$

$$\kappa_{ik} = \mathrm{softplus}(W_\kappa\, s_i) + \kappa_{\min} \in \mathbb{R}_{>0}^{D_\chi}$$

with $D_\chi = 4$ ($\chi_1, \chi_2, \chi_3, \chi_4$). Each component $k$ places a von Mises distribution on every $\chi$ angle: a mean direction $\mu_{ik}$ (the projection is passed through $\tanh$ and scaled to bound it to a full turn) and a concentration $\kappa_{ik}$ — the circular analogue of inverse variance ($\kappa \to 0$ is uniform on the circle, large $\kappa$ is sharply peaked). The von Mises mixture replaces an earlier wrapped-Gaussian formulation, removing the variance-floor hack and giving a properly normalized density on the torus.

---

## Training

**Data.** qFit multiconformer PDB files. Named altlocs (A, B, $\ldots$) with crystallographic occupancies provide the discrete ground-truth ensemble; the blank altloc provides the input backbone frame.

**Loss.** Occupancy-weighted negative log-likelihood of the observed altloc $\chi$ angles under the predicted von Mises mixture. For a single residue with $A$ observed altlocs:

$$\mathcal{L} = - \sum_{a=1}^{A} o_a \log \left[ \sum_{k=1}^{K} \pi_k \prod_{d=1}^{D} \frac{1}{2\pi I_0(\kappa_{dk})} \exp\left( \kappa_{dk}\cos(\chi_{ad}-\chi_{dk}) \right) \right]$$

The training loss is this quantity averaged over the valid residue set $\mathcal{R}$.

where:

- $\mathcal{R} = \{ i : n_{\mathrm{obs},i} \geq 2 \text{ and } d_{\mathrm{eff},i} \geq 1 \}$ is the set of residues with at least two observed altlocs and at least one defined $\chi$ angle.
- $o_a$ is the crystallographic occupancy of altloc $a$, normalized so $\sum_a o_a = 1$.
- $\pi_k$ and $\kappa_{dk}$ are the predicted mixture weight and concentration; $\chi_{dk} \equiv \mu_{kd}$ is the predicted von Mises mean direction of component $k$ on $\chi$ angle $d$; $\chi_{ad}$ is the observed angle of altloc $a$.
- $D$ is the number of defined $\chi$ angles for the residue (i.e. the per-$\chi$ validity mask restricts the product), and $K$ the number of mixture components.
- $\frac{1}{2\pi I_0(\kappa)} \exp(\kappa \cos \Delta)$ is the von Mises density on the circle, with $I_0$ the modified Bessel function of the first kind, order $0$. The log-normalizer is evaluated stably as $\log I_0(\kappa) = \log\,\mathrm{i0e}(\kappa) + \kappa$ (`torch.special.i0e`), so large $\kappa$ stays finite.
- The residual $\Delta = \chi_{ad}-\chi_{dk}$ is taken on the circle. For $\chi$ angles with $\pi$-rotational symmetry (ASP $\chi_2$, GLU $\chi_3$, PHE $\chi_2$, TYR $\chi_2$) it is folded to $(-\pi/2, \pi/2]$ via $\tfrac{1}{2}\, \mathrm{atan2}(\sin 2\Delta, \cos 2\Delta)$.

**Optimization.** AdamW with $\eta = 3 \times 10^{-4}$, cosine annealing to $\eta / 100$ over the full training run. Batch size 1 (one full protein per step).

---

## Default hyperparameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| single representation | $d_s$ | $128$ |
| pair representation | $d_z$ | $32$ |
| IPA blocks | $L$ | $8$ |
| attention heads | $H$ | $4$ |
| head dim | $c$ | $8$ |
| query/key points | $N_{qk}$ | $4$ |
| value points | $N_v$ | $8$ |
| von Mises components | $k$ | $5$ |
| chi dimensions | $D_\chi$ | $4$ |
| concentration floor | $\kappa_{\min}$ | $0.01$ |
| dropout | $p$ | $0.2$ |
| optimizer | --- | AdamW |
| learning rate | $\eta$ | $3 \times 10^{-4}$ |
| schedule | --- | cosine to $\eta / 100$ |
| batch size | --- | $1$ |
