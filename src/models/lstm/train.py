import torch


# -------------------------------------------------------------
#  Epoch helpers
# -------------------------------------------------------------
def train_epoch(model, loader, loss_fn, opt, device):
    model.train()
    running = 0.0
    for seq, stat, y in loader:
        seq, stat, y = seq.to(device), stat.to(device), y.to(device)
        opt.zero_grad()
        pred = model(seq, stat)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        running += loss.item() * seq.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate_epoch(model, loader, loss_fn, device):
    model.eval()
    running = 0.0
    for seq, stat, y in loader:
        seq, stat, y = seq.to(device), stat.to(device), y.to(device)
        pred = model(seq, stat)
        loss = loss_fn(pred, y)
        running += loss.item() * seq.size(0)
    return running / len(loader.dataset)



