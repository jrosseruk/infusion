import numpy as np
import torch
from tqdm import tqdm
import os

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


import matplotlib.pyplot as plt

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, ckpt_dir, random_seed=None):
    # Get the device model parameters are on
    device = next(model.parameters()).device

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs), desc=f"Training for {epochs} epochs..."):
        # Training phase
        model.train()
        batch_train_losses = []
        batch_train_nums = []
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            loss, n = loss_batch(model, loss_func, xb, yb, opt)
            batch_train_losses.append(loss)
            batch_train_nums.append(n)
        train_loss = np.sum(np.multiply(batch_train_losses, batch_train_nums)) / np.sum(batch_train_nums)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        with torch.no_grad():
            batch_val_losses, batch_val_nums = zip(
                *[
                    loss_batch(
                        model, loss_func, xb.to(device), yb.to(device)
                    )
                    for xb, yb in valid_dl
                ]
            )
        val_loss = np.sum(np.multiply(batch_val_losses, batch_val_nums)) / np.sum(batch_val_nums)
        val_losses.append(val_loss)

        print(epoch, train_loss, val_loss)

        # Save model checkpoint every 10 epochs (including the first epoch)
        if (epoch + 1) % 1 == 0:
            checkpoint_path = os.path.join(ckpt_dir, f'ckpt_epoch_{epoch + 1}.pth')
            # Save model state dict and training metadata
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
            }
            if random_seed is not None:
                checkpoint['random_seed'] = random_seed
            torch.save(checkpoint, checkpoint_path)

    # Plot train and validation loss in two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(range(epochs), train_losses, marker='o', color='tab:blue')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Train Loss')
    axs[0].set_title('Training Loss')
    axs[0].grid(True)

    axs[1].plot(range(epochs), val_losses, marker='o', color='tab:orange')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Validation Loss')
    axs[1].set_title('Validation Loss')
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()

