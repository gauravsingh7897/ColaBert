import torch
from tqdm import tqdm


def train_fn(model, data_loader, optimizer, scheduler, device):
    model.train()

    total_train_loss = 0

    for _, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = data['ids']
        masks = data['masks']
        labels = data['labels']

        ids = ids.to(device)
        masks = masks.to(device)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()

        loss, logits = model(
            ids, masks, labels
        )

        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(data_loader)

    print(f"Average Train Loss : {avg_train_loss}")
