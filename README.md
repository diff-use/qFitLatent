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

The final single representation $s_i$ is projected to a $k$-component Gaussian mixture over sidechain $\chi$ angles:

$$\pi_i = \mathrm{softmax}(W_\pi\, s_i)$$

$$\mu_{ik} = W_\mu\, s_i \in \mathbb{R}^{D_\chi}$$

$$\sigma_{ik} = \mathrm{softplus}(W_\sigma\, s_i) + \sigma_{\min} \in \mathbb{R}_{>0}^{D_\chi}$$

with $D_\chi = 4$ ($\chi_1, \chi_2, \chi_3, \chi_4$), $k = 5$ mixture components.

---

## Training

**Data.** qFit multiconformer PDB files. Named altlocs (A, B, $\ldots$) with crystallographic occupancies provide the discrete ground-truth ensemble; the blank altloc provides the input backbone frame.

**Loss.** Occupancy-weighted negative log-likelihood of observed altloc $\chi$ angles under the predicted mixture:

$$\mathcal{L} = -\frac{1}{|\mathcal{R}|} \sum_{i \in \mathcal{R}} \sum_{a} w_{ia} \log \sum_{k=1}^{K} \pi_{ik} \prod_{j=1}^{D_\chi} m_{ij}\, \mathcal{N}_{\mathrm{circ}}\bigl( \chi_{iaj};\, \mu_{ikj},\, \sigma_{ikj}^{2} \bigr)$$

where:

- $\mathcal{R} = \{ i : n_{\mathrm{obs},i} \geq 2 \text{ and } d_{\mathrm{eff},i} \geq 1 \}$ is the set of residues with at least two observed altlocs and at least one defined $\chi$ angle.
- $w_{ia} = \mathrm{occ}_{ia} / \sum_{a'} \mathrm{occ}_{ia'}$ are normalized occupancy weights.
- $m_{ij} \in \{0, 1\}$ is the chi-validity mask.
- $\mathcal{N}_{\mathrm{circ}}$ uses a wrapped Gaussian on the circle. Residual errors are wrapped to $(-\pi, \pi]$ via $\mathrm{atan2}(\sin \Delta, \cos \Delta)$. For chi angles with $\pi$-rotational symmetry (ASP $\chi_2$, GLU $\chi_3$, PHE $\chi_2$, TYR $\chi_2$), errors are folded to $(-\pi/2, \pi/2]$ via $\tfrac{1}{2}\, \mathrm{atan2}(\sin 2\Delta, \cos 2\Delta)$.

**Optimization.** AdamW with $\eta = 3 \times 10^{-4}$, cosine annealing to $\eta / 100$ over the full training run. Batch size 1 (one full protein per step).

---

## Validation

`validation/validate.py` runs inference on a train and a val structure and produces:

- A predicted multiconformer PDB with GMM component means written as altlocs (for PyMOL overlay).
- A per-structure figure showing per-residue loss and RMSF, observed vs predicted per-atom RMSF, and observed RMSF vs predicted $\sigma_k$.

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
| GMM components | $k$ | $5$ |
| chi dimensions | $D_\chi$ | $4$ |
| sigma floor | $\sigma_{\min}$ | $0.3$ rad |
| dropout | $p$ | $0.2$ |
| optimizer | --- | AdamW |
| learning rate | $\eta$ | $3 \times 10^{-4}$ |
| schedule | --- | cosine to $\eta / 100$ |
| batch size | --- | $1$ |
