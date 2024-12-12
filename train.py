import torch
import time
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn


def accuracy(logits, labels):
    _, predicted_indices = torch.max(logits, dim=1)
    correct = torch.sum(predicted_indices == labels)
    return correct.item() / len(labels)


def evaluate(model, feature_list, labels, loss_fn, idx):
    model.eval()
    with torch.no_grad():
        logits = model(feature_list)[0]
    return loss_fn(logits[idx], labels[idx]).item(), accuracy(logits[idx], labels[idx])


def entropy_of_row(row, eps=1e-9):
    row_sum = row.sum()
    if row_sum == 0:
        return 0
    p = row / row_sum
    p = p[p > 0]
    p = torch.clamp(p, min=eps)
    return -torch.sum(p * torch.log2(p))


def total_entropy_of_matrix(matrix):
    return sum(entropy_of_row(row) for row in matrix) / matrix.shape[0]


def loss_H(S_list):
    return sum(total_entropy_of_matrix(S) for S in S_list) / len(S_list)


def train(model, feature_vector, labels, epochs, train_idx, test_idx, device, lr,
          weight_decay, alpha, loss_fn=nn.CrossEntropyLoss()):
    labels = torch.LongTensor(labels).to(device)
    train_idx = torch.LongTensor(train_idx).to(device)
    test_idx = torch.LongTensor(test_idx).to(device)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_acc_all, test_acc_all = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(feature_vector)
        logits, S = output[0], output[1]

        loss_0 = loss_fn(logits[train_idx], labels[train_idx])
        loss_1 = loss_H(S)
        loss = loss_0 + alpha * loss_1

        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits[train_idx], labels[train_idx])
        loss_test, test_acc = evaluate(model, feature_vector, labels, loss_fn, test_idx)

        train_acc_all.append(train_acc)
        test_acc_all.append(test_acc)

        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f} | Loss: {loss.item():.4f}")

    return train_acc_all, test_acc_all
