"""this is the code for the training loop and testing loop which 
consist the def of the relevant functions
 """

import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model:torch.nn.Module,
               dataloader : torch.utils.data.DataLoader,
               loss_fn : torch.nn.Module,
               optimizer : torch.optim.Optimizer,
               devise : torch.device) -> Tuple[float, float]:
    
    """trains pytorch model a single epoch"""
    model.train()
    train_loss, train_acc  = 0.0, 0.0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(devise), y.to(devise)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        y_pred_classed = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred == y).sum().item() / len(y_pred)

## to get teh average loss
    train_loss = train_loss /len(dataloader)
    train_acc = train_acc / len(dataloader)


    return train_loss, train_acc


def test_step(model : torch.nn.Module,
              dataloader : torch.utils.data.DataLoader,
              loss_fn : torch.nn.Module,
              device : torch.device) -> Tuple[float, float]:
    model.eval()
    test_loss, test_acc = 0.0, 0.0

    with torch.inference_mode():
        for bath, X, y in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            model.eval()
            test_pred_logits = model(X)

            loss = loss_fn(X, y)
            test_loss += loss.item()

            test_pred_labels = torch.argmax(torch.softmax(test_pred_logits, dim=1), dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    test_loss = test_loss/len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc





    