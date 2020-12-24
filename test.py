import torch
from tqdm import tqdm


def test_fn(model, data_loader, device):
    model.eval()
    final_targets = []
    final_outputs = []
    total_test_loss = 0

    for _, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = data['ids']
        masks = data['masks']
        labels = data['labels']

        ids = ids.to(device)
        masks = masks.to(device)
        labels = labels.to(device, dtype=torch.long)

        with torch.no_grad():
            loss, logits = model(
                ids, masks, labels
            )
        final_targets.extend(labels.cpu().detach().numpy().tolist())
        final_outputs.extend(logits.cpu().detach().numpy().tolist())

        total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(data_loader)

    print(f"Average Test Loss : {avg_test_loss}")

    return final_targets, final_outputs
