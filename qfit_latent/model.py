# created by clay
''' 
The model to learn qfit multiconformer dynamics with an
IPA backbone
'''

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _checkpoint
import torch.nn.functional as F
import math
from .data.data import N_CHI, N_CHI_PER_AA

# global parameters
# minimum variance allowed to prevent sigma collapse
SIG_FLOOR = 0.15

class qFitLatent(nn.Module):
    ''' 
    SE(3) invariant model for learning protein dynamics
    from qfit multiconformer models
    
    Inputs:
        - aa_tokens: (N)
        - R: (N, 3, 3)
        - t: (N, 3)
    
    Outputs:
        pi: (N, k)
        mu: (N, k, d_chi)
        sigma: (N, k, d_chi)
    '''

    # initialize with hyperparams
    def __init__(
        self,
        n_aa = 21,
        d_single = 128,
        d_pair = 32,
        n_ipa = 8,
        n_heads = 8,
        n_geom_attn_qpts = 4,
        n_geom_attn_vpts = 8,
        c = 8,
        k = 5,
        dropout = 0.0,
    ):
        super().__init__() # inherit nn 
        # set representation sizes
        self.d_s = d_single
        self.d_z = d_pair

        # set architecture sizes:
        # discrete vars:
        # embedding table for 21 amino acids to the residue rep dimension
        self.aa_embed = nn.Embedding(n_aa, d_single) # [21, d_single]
        # embedding table for primary seq diatance (capped at +- 64 distance)
        self.rel_pos_embed = nn.Embedding(129, d_pair) # [129, d_pair]
        # continous vars:
        # linear layer for pairwise distance projection CA-CA
        self.dist_embed = nn.Linear(1, d_pair, bias=False) # [1, d_pair]
        # projection of a single res representation to pair dim
        self.env_proj = nn.Linear(d_single, d_pair) # [d_single, d_pair]
        # projection of the pair representation to d_pair
        self.env_pair_proj = nn.Linear(d_pair, d_pair) # [d_pair, d_pair]

        # IPA block(s) hyperparameters
        ipa_hyperparams = dict(
            n_heads = n_heads,
            n_geom_attn_qpts = n_geom_attn_qpts,
            n_geom_attn_vpts = n_geom_attn_vpts,
            c=c
        )
        # build the IPA layers
        self.ipa_blocks = nn.ModuleList(
            [IPABlock(d_single, d_pair, dropout=dropout, **ipa_hyperparams)
            for _ in range(n_ipa)]
        )

        # build the chi angle GMM prediction head (output layer)
        self.chigmm_head = ChiGMMHead(
            d_single, 
            k=k,
            d_chi=N_CHI,
        )

    # initialize the pair representation track
    def pair_init(self, s, t):
        # number of residues
        N = s.shape[0]
        # sequential indexes
        idx = torch.arange(N, device=t.device)

        # the relative position distances (clamped to +-64)
        rel_pos = (idx[:, None] - idx[None, :]).clamp(-64, 64) + 64
        # embed the relative positions
        z = self.rel_pos_embed(rel_pos) # [N, N, 129] to [N, N, d_pair]
        # get the pairwise distances of CA's (L2 Norm) keep dim for proj
        dist = (t[:, None] - t[None, :]).norm(dim=-1, keepdim=True)
        # add the distance embeddings to the rel pos embeddings
        z = z + self.dist_embed(dist / 10.0) # angstroms to nm
        # project the single representations
        env = self.env_proj(s) # [N, d_single] to [N, d_pair]
        # add the pair projection products to the pair rep
        z = z + self.env_pair_proj(env[:, None, :]*env[None, :, :])
        return z # return the pairwise representation [N, N, d_pair]
    
    def chi_mask(self, aa_tokens):
        counts = torch.tensor(N_CHI_PER_AA, device=aa_tokens.device)
        thresholds = torch.arange(N_CHI, device=aa_tokens.device)
        return thresholds[None, :] < counts[aa_tokens, None] # [N, N_CHI] bool
    
    # a forward pass
    def forward(self, aa_tokens, R, t):
        # embed sequence tokens and prepare pair track
        s = self.aa_embed(aa_tokens)
        z = self.pair_init(s, t)

        # iterate through blocks and pass info; pair track is now updated
        # in-place by each block via the triangle multiplication
        for block in self.ipa_blocks:
            if self.training: # load from checkpoint and resume
                s, z = _checkpoint(block, s, z, R, t, use_reentrant=False)
            else: # forward pass
                s, z = block(s, z, R, t)
        
        chi_mask = self.chi_mask(aa_tokens)
        return self.chigmm_head(s, chi_mask) # pi [N,k], mu [N, k, N_CHI], sigma [N, k, N_CHI]


class ChiGMMHead(nn.Module):
    '''
    This is the readout layer for the qfit latent dynamics GMM
    model. It takes a hidden representation [N, d_single] and 
    predicts k gaussian components to represent the distribution
    of residue rotamers as a gaussian misxture.

    p(𝝌 | h) = Sum(k<K) pi_k * 𝒩(𝝌 | mu_k, sigma_k^2)

    Inputs:
        - h: [N, d_single] learned representation from IPA blocks
        - k: number of components
    
    Outputs:
        - pi: [N, k] predicted weights for each component
        - mu: [N, k * d_chi] predicted average chi angles per component
        - sigma: [N, k * d_chi] predicted variance per chi angle per component
    '''
    def __init__(self, d_single, k = 5, d_chi = N_CHI):
        super().__init__() # inherit module
        # dims of outputs
        self.k = k
        self.d_chi = d_chi

        # the output layers
        self.pi_proj = nn.Linear(d_single, k)
        self.mu_proj = nn.Linear(d_single, k*d_chi)
        self.sigma_proj = nn.Linear(d_single, k*d_chi)

    # forward pass for learning and prediction
    def forward(self, h, chi_mask=None):
        *batch, _ = h.shape
        k, d = self.k, self.d_chi
        pi = torch.softmax(self.pi_proj(h).float(), dim=-1)
        # wrap the prediction of chi angles to the circle
        mu    = math.pi * torch.tanh(self.mu_proj(h).float()).view(*batch, k, d)
        sigma = F.softplus(self.sigma_proj(h)).float().view(*batch, k, d) + SIG_FLOOR
        # apply chi mask to hide chi angles that dont exist in residues
        if chi_mask is not None:
            m     = chi_mask[..., None, :]              
            mu    = mu * m.to(mu.dtype)
            sigma = torch.where(m, sigma, torch.ones_like(sigma))

        return pi, mu, sigma
    
class IPA(nn.Module):
    '''
    Invariant point attention to make use of standard multi head
    self attention  plus geometric attention allowing the model to 
    learn a combined representation with structure and sequence priors

    For a query residue i and key residue j:

    a_ij = standard attention + geometric attention + pair bias
    a_ij = q_i * k_j / sqrt(c) +
           (-𝛾_h / 2) * Sum(pts) dist(qi_pts, kj_pts) + 
           b_ij

    Inputs:
        - s: sequence representations from embedding or previous layer
        - z: pair representation
        - R: local frame rotation
        - t: local frame translation

    Outputs:
        - s: updated features from attn mechanism
    '''

    def __init__(
        self, 
        d_single, 
        d_pair, 
        n_heads = 8, 
        n_geom_attn_qpts = 4, 
        n_geom_attn_vpts = 8, 
        c = 16
    ):
        super().__init__() # inherit module
        # set vars
        self.H, self.n_qk, self.n_v, self.c = n_heads, n_geom_attn_qpts, n_geom_attn_vpts, c

        # the mechanism of IPA (project representations and combine pairwise)
        # standard
        self.q_s = nn.Linear(d_single, n_heads*c, bias=False)
        self.k_s = nn.Linear(d_single, n_heads*c, bias=False)
        self.v_s = nn.Linear(d_single, n_heads*c, bias=False)
        # geometric
        self.q_pts = nn.Linear(d_single, n_heads * n_geom_attn_qpts * 3, bias=False)
        self.k_pts = nn.Linear(d_single, n_heads * n_geom_attn_qpts * 3, bias=False)
        self.v_pts = nn.Linear(d_single, n_heads * n_geom_attn_vpts * 3, bias=False)

        # the pair bias
        self.b_ij = nn.Linear(d_pair, n_heads, bias=False)
        # the head weights
        self.head_weight = nn.Parameter(torch.zeros(n_heads))

        # output dimension and projection back to single rep dim
        d_out = n_heads*c + n_heads * n_geom_attn_vpts * (3 + 1)
        self.out = nn.Linear(d_out, d_single)

    # global transformations (local to global)
    def _global(self, pts, R, t):
        return torch.einsum("nab,nhpb->nhpa", R, pts) + t[:, None, None, :]
    
    # forward pass
    def forward(self, s, z, R, t):
        N, H, nq, nv, c = s.shape[0], self.H, self.n_qk, self.n_v, self.c

        # project seqeunce representations and reshape for multihead attn
        # standard attn
        q = self.q_s(s).view(N, H, c)
        k = self.k_s(s).view(N, H, c)
        v = self.v_s(s).view(N, H, c)
        # global attention
        q_p = self._global(self.q_pts(s).view(N, H, nq, 3), R, t)
        k_p = self._global(self.k_pts(s).view(N, H, nq, 3), R, t)
        v_p = self._global(self.v_pts(s).view(N, H, nv, 3), R, t)

        # add attention terms
        attn = torch.einsum("ihc,jhc->ijh", q, k) / math.sqrt(c)
        diff = q_p[:, None] - k_p[None, :]
        dist = diff.pow(2).sum(-1).sum(-1) # squared distances
        attn = attn - 0.5 * F.softplus(self.head_weight)[None, None] * dist
        attn = F.softmax(attn + self.b_ij(z), dim=1) # normalize over the j residues

        # project the multihead scalar attention onto v and reshape back to dim
        o_scalar = torch.einsum("ijh,jhc->ihc", attn, v).reshape(N, -1)

        # aggreagate the point attention and transform back to local frame of i
        v_agg = torch.einsum("ijh,jhpd->ihpd", attn, v_p)
        v_local= torch.einsum("nba,nhpb->nhpa", R, v_agg - t[:, None, None, :])
        # flatten of local points
        o_point = v_local.reshape(N, H*nv*3)
        # get norms of coords
        o_n = v_local.norm(dim=-1).reshape(N, H*nv)

        # return scalar attn features, local attention points, their magnitudes
        return self.out(torch.cat([o_scalar, o_point, o_n], dim=-1))
    
class IPABlock(nn.Module):
    '''
    A full IPA block that passes the protein through multihead IPA and a feed 
    forward nn to represent the residues with understanding of each other. The
    IPA backbone allows the residues to understand global distnace and 
    orientation, the standard attention allows the sequence representations to
    talk to one another, and the ffn with gelu allows the attention mechanism to 
    capture nonlinear relationships.

    Inputs:
        - s: sequence representations from embedding or previous layer
        - z: pair representation
        - R: local frame rotation
        - t: local frame translation

    Outputs:
        - s: updated features from attn and ffn
    '''
    def __init__(self, d_single, d_pair, dropout, **ipa_args):
        super().__init__() # inherit
        # pass through layernorms, ipa, then ff, applying dropout
        self.norm1 = nn.LayerNorm(d_single)
        self.ipa = IPA(d_single, d_pair, **ipa_args)
        self.norm2 = nn.LayerNorm(d_single)
        self.ff = nn.Sequential(
            nn.Linear(d_single, d_single*4),
            nn.GELU(),
            nn.Linear(d_single*4, d_single)
        )
        self.dropout = nn.Dropout(dropout)
        # AF-style triangle multiplicative update on the pair rep
        self.triangle = TriangleUpdate(d_pair)

    # a pass through the full IPA with dropout
    def forward(self, s, z, R, t):
        s = s + self.dropout(self.ipa(self.norm1(s), z, R, t))
        s = s + self.dropout(self.ff(self.norm2(s)))
        # update the pair representation with one outgoing triangle multiply
        z = z + self.dropout(self.triangle(z))
        return s, z


class TriangleUpdate(nn.Module):
    '''
    AlphaFold-style triangle multiplicative update (outgoing direction). For
    each pair (i,j), aggregates information from all (i,k) and (j,k) edges
    via an outer contraction over k:

        z_ij ← Out( Sum_k a_ik * b_jk )

    This injects structural triangle-inequality priors into the pair track
    so it can carry geometric relations between residues, rather than being
    a static bias.

    Inputs:
        - z: [N, N, d_pair] pair representation

    Outputs:
        - delta: [N, N, d_pair] update to add residually to z
    '''
    def __init__(self, d_pair):
        super().__init__() # inherit
        # layernorm the pair input before projecting
        self.norm = nn.LayerNorm(d_pair)
        # the two triangle edge projections (a for i-k, b for j-k)
        self.a = nn.Linear(d_pair, d_pair)
        self.b = nn.Linear(d_pair, d_pair)
        # output projection after the triangle contraction
        self.out = nn.Linear(d_pair, d_pair)

    def forward(self, z):
        z_n = self.norm(z)
        # outgoing triangle: sum over k of a_ik * b_jk -> [N, N, d_pair]
        return self.out(torch.einsum("ikd,jkd->ijd", self.a(z_n), self.b(z_n)))