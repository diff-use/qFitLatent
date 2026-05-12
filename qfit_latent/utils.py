# created by clay
'''
utility functions used for training 
'''

import torch
import csv
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from .data.data import N_AA, N_CHI, SYM_TBL, AA_NAMES
from .loss import ChiGMMLoss

def run_epoch(
    model,
    loader,
    device,
    loss_function,
    optimizer=None,
    scheduler=None
):
    # set the boolean for whether its training or not
    training = optimizer is not None
    model.train(training)

    # keep track of loss function
    run_loss = 0.0
    # number of proteins 
    n = 0
    # learning rate storage
    lr_history = []

    aa_sum = torch.zeros(N_AA, dtype=torch.float64)
    aa_count = torch.zeros(N_AA, dtype=torch.float64)

    for sample in tqdm(loader):
        if sample is None:
            continue

        R = sample["R"].to(device)
        t = sample["t"].to(device)
        aa = sample["aa_tokens"].to(device)
        chi_angles = sample["chi_angles"].to(device)
        occupancies = sample["occupancies"].to(device)
        chi_mask = sample["chi_mask"].to(device)

        # skip n=1 rotamers
        if (occupancies > 0).sum(dim=-1).max() < 2:
            continue

        if training:
            optimizer.zero_grad()
        sym_tbl = SYM_TBL.to(device)
        sym_mask = sym_tbl[aa]

        with torch.set_grad_enabled(training):
            pi, mu, sigma = model(aa, R, t)

            # calculate loss
            out = loss_function(
                pi, mu, sigma, 
                chi_angles, occupancies, chi_mask, sym_mask
            )
            loss = out["loss"]

        if not loss.isfinite():
            continue
        if training and not loss.requires_grad:
            continue

        if training:
            # backprop and step
            loss.backward()
            optimizer.step()
            scheduler.step()

            # track learning rate
            current_step = scheduler.last_epoch
            current_lr = scheduler.get_last_lr()
            lr_history.append((current_step, current_lr[0]))

        # iterate
        run_loss += loss.item()
        n += 1

        # store losses per residue
        with torch.no_grad():
            residue_log_prob = out["per_residue_loss"].detach().float().cpu()
            aa_cpu = aa.detach().cpu()

            valid_r = torch.isfinite(residue_log_prob)
            if valid_r.any():
                a = aa_cpu[valid_r].clamp(max=N_AA - 1)
                v = residue_log_prob[valid_r].double()

                in_range = aa_cpu[valid_r] < N_AA
                if in_range.any():
                    aa_sum.scatter_add_(0, a[in_range], v[in_range])
                    aa_count.scatter_add_(
                        0, a[in_range], torch.ones_like(v[in_range])
                    )

    # if none passed return no losses
    if n == 0:
        return None
    
    # avg the losses
    avg_loss = run_loss / n
    # accumulate per aa
    per_aa = (aa_sum / aa_count.clamp(min=1).where(aa_count > 0,
              torch.full_like(aa_sum, float("nan")))).tolist()
    
    return {
        "total": avg_loss,
        "per_aa": per_aa,
        "learning_rates": lr_history
    }

def train(
    model,
    train_dataset,
    val_dataset,
    *,
    lr              = 3e-4,
    epochs          = 100,
    device          = "cpu",
    checkpoint_dir  = Path("checkpoints"),
    start_epoch     = 0,
    save_every      = 10,
    loss_csv        = Path("logs/loss.csv"),
    optimizer_state = None,
    sched_state     = None
    ):

    # create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # create csv for logging
    loss_csv = Path(loss_csv)
    loss_csv.parent.mkdir(parents=True, exist_ok=True)

    # gpu or cpu
    model.to(device)

    # set up the optimizer (load from previous state if passed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    # the loss function
    loss_fn = ChiGMMLoss()

    # data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1, 
        shuffle=True,
        collate_fn=collate,
        num_workers=4
    )
    val_loader   = (DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False,
        collate_fn=collate,
        num_workers=4) if val_dataset is not None else None
    )
    has_val = val_loader is not None

    # set up the scheduler to go per step (load if passed)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max = len(train_loader) * epochs, eta_min=lr / 100
    )
    if sched_state is not None:
        sched.load_state_dict(sched_state)

    # determine whether loss csv already exists or not and write header if not
    write_header = not loss_csv.exists() or loss_csv.stat().st_size == 0
    csv_fh = open(loss_csv, "a", newline="")
    writer = csv.writer(csv_fh)
    if write_header:
        writer.writerow(_csv_header(has_val))

    # the learning rate csv tracker:
    lr_csv = Path("logs/lr_schedule.csv")
    write_lr_header = not lr_csv.exists() or lr_csv.stat().st_size == 0
    lr_fh = open(lr_csv, "a", newline="")
    lr_writer = csv.writer(lr_fh)
    if write_lr_header:
        lr_writer.writerow(["step", "learning rate"])

    # iterate through epochs and train
    for epoch in range(start_epoch, start_epoch + epochs):

        # run an epoch and return the train loss dict
        tr = run_epoch(model, train_loader, device, loss_fn,
                        optimizer=optimizer, scheduler=sched) #scaler=scaler)
        if tr is None:
            continue

        # run a validation forward pass and return loss
        val = None
        if has_val:
            val = run_epoch(model, val_loader, device, loss_fn)

        # print the losses to console
        if val is not None:
            print(
                f"epoch {epoch+1:>4}  "
                f"total {tr['total']:.4f}  val_total {val['total']:.4f}"
            )
        else:
            print(f"epoch {epoch+1:>4}  total {tr['total']:.4f}")

        # log the losses in the csv
        writer.writerow(_row(epoch + 1, tr, val))
        csv_fh.flush()

        # log the learning rate 
        if tr["learning_rates"]:
            for step_num, step_lr in tr["learning_rates"]:
                lr_writer.writerow([step_num, step_lr])
            lr_fh.flush()

        # save the stat of the model/optimizer/scheduler to checkpoint 
        if (epoch + 1) % save_every == 0:
            ckpt = {"epoch": epoch + 1, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": sched.state_dict()}
            torch.save(ckpt, checkpoint_dir / f"epoch_{epoch+1:04d}.pt")
            torch.save(ckpt, checkpoint_dir / "latest.pt")

    # close csv 
    csv_fh.close()
    
def collate(batch):
    valid = [x for x in batch if x is not None]
    return valid[0] if valid else None

# format the csv header based on the inputs and whether or not validation is
# being used
def _csv_header(has_val):
    cols = ["epoch", "total"]
    if has_val:
        cols += ["val_total"]
    cols += [f"aa_{a}" for a in AA_NAMES]
    if has_val:
        cols += [f"val_aa_{a}" for a in AA_NAMES]
    return cols

# create a row entry for a given epoch from the loss dicts
def _row(epoch, tr, val):
    row = [epoch, tr["total"]]
    if val is not None:
        row += [val["total"]]
    row += list(tr["per_aa"])
    if val is not None:
        row += list(val["per_aa"])
    return row