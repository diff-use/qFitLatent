# created by clay
''' 
The loss function for learning qfit latent dynamics through
ground truth multiconformer models refit to better match electron density. 
The model learns a probaility density function for each residue in a protein
from the backbone structure and its sequence. Each residue is treated like a
gaussian mixture model in torsional space.
'''

import torch.nn as nn
import torch
import math

class ChiGMMLoss(nn.Module):
    '''
    Weighted negative log likelihood of observed chi angles in the ground truth
    qfit multiconformer model. Gaussian mixture is defined over the torsion 
    space with up to k components per residue. 

    Averaged over residies for a altlocs and k components and j chi angles:
    L = -mean (residues) Sum(a) w_a log Sum(k) π_k ∀j N_circ(χ_aj ; mu_kj, sigma_kj^2)
    '''
    def forward(
        self,
        pi, 
        mu,
        sigma, 
        qfit_chis,
        occupancies,
        mask,
        symmetry_mask
    ):
        N, k, D = mu.shape
        device = mu.device

        # masks for valid reisdues 
        n_obs = (occupancies > 0).sum(-1) # number of qfit altlocs that were observed
        d_eff = mask.float().sum(-1) # number of chi angles that are observed 
        # valid if it has observed occupancy & a chi angle
        valid_res = (n_obs >= 2) & (d_eff >= 1)

        # set up the loss dict which also sums per residue and per chi angle
        out = {
            "loss": pi.new_zeros(()),
            "per_residue_loss": torch.full((N,), float("nan"), device=device),
            "per_chi_loss": torch.full((N, D), float("nan"), device=device)
        }

        # if no valid residues, return default loss
        if not valid_res.any():
            return out
        
        # get the per chi angle losses as raw radians 
        error = qfit_chis[:, :, None, :].float() - mu[:, None, :, :].float()
        # wrap the radian errors around the 2pi circle
        error_2pi = wrap_2pi(error)

        if symmetry_mask is not None and symmetry_mask.any():
            # wrap distances to pi and return the wrapped those for sym residues
            error_pi = wrap_pi(error)
            sm = symmetry_mask[:, None, None, :].to(torch.bool)
            # pi wrapped for pi sym residue/angles 2pi wrapped for all else
            error = torch.where(sm, error_pi, error_2pi)
        else: # otherwise all 2pi wrapped
            error = error_2pi

        # reshape the mask to broadcast over the m/k pairs
        mask_f = mask[:, None, None, :].float()

        # precompute constants for the gaussian term(s)
        sigma2 = sigma.float().pow(2).clamp(min=1e-6)
        log_sigma2 = sigma2.log()
        log2pi = math.log(2*math.pi)  

        # evaluate each altloc under each component for each residue and chi ang
        per_dim = (log2pi + log_sigma2[:, None, :, :]
                   + error.pow(2) / sigma2[:, None, :, :])
        # multiply by mask and sum over chi dimensions
        log_norm = -0.5 * (per_dim * mask_f).sum(-1)
        # multiply the component loss by the log of component weight
        log_pi = pi.float().clamp(min=1e-8).log()
        # stay in log space when summing to prevent underflow
        log_p = torch.logsumexp(log_pi[:, None, :] + log_norm, dim=-1)

        # weight the term losses per occupancies (normalize first) and sum to be
        # per residue
        obs_mask = occupancies > 0 # get valid altloc spots
        occ_norm = occupancies / occupancies.sum(-1, keepdim=True).clamp(min=1e-8)
        log_prob = -(occ_norm * log_p * obs_mask).sum(-1) # [N] total loss per res

        # get total loss per residue type and per protein
        out["loss"] = log_prob[valid_res].mean()
        out["per_residue_loss"][valid_res] = log_prob[valid_res]

        return out
    
# chi angles are symmetric about 2pi so the distances need to wrap
def wrap_2pi(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

# some residues (PHE, etc.) are symmetric about pi for a given angle 
def wrap_pi(x):
    return 0.5 * torch.atan2(torch.sin(2.0 * x), torch.cos(2.0 * x))
        
    