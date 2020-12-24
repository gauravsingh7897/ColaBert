import test
import train
import torch
import config
import preprocess
import numpy as np
from model import ColaBert
from sklearn import metrics
from transformers import AdamW
from dataset import ColaDataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup


def main():

    device = torch.device('cuda')

    train_x, train_y = preprocess.load()
    train_dataset = ColaDataset(train_x, train_y)
    test_x, test_y = preprocess.load(False)
    test_dataset = ColaDataset(test_x, test_y)

    train_data_loader = DataLoader(
        train_dataset,
        num_workers=0,
        batch_size=config.TRAIN_BATCH_SIZE
    )

    test_data_loader = DataLoader(
        test_dataset,
        num_workers=0,
        batch_size=config.VALID_BATCH_SIZE
    )

    model = ColaBert()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    total_steps = int(len(train_x) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    best_accuracy = 0

    for _ in range(config.EPOCHS):
        train.train_fn(model, train_data_loader, optimizer, scheduler, device)
        targets, outputs = test.test_fn(model, test_data_loader, device)
        outputs = np.argmax(outputs, axis=1)
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy : {accuracy}")

        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    main()
