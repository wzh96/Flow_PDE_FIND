import torch.nn as nn
import torch
import numpy as np
from utils_LSTM import params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def LSTM_Train(model, dataloader, num_epochs, learning_rate, device):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    loss_history = []

    for epoch in range(num_epochs):

        epoch_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs= model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return loss_history








