import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import time
import copy
from tqdm import tqdm
import numpy as np


def plotify(train_losses, val_losses):

    plt.plot(train_losses, label = 'Train Loss')
    plt.plot(val_losses, label = 'Test Loss')
    plt.show()


def train(criterion, model, optimizer, n_epochs, device, train_loader, val_loader):
    
    since = time.time()

    train_losses = []
    valid_losses = []
    valid_loss_min = np.Inf

    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in tqdm(range(1, n_epochs+1)):
        print ('\n')
        print ("="*30)
        print (f'\nEpoch : {epoch}') 
        train_loss = 0.0
        valid_loss = 0.0
        model.train()
        for images, labels in train_loader:
            images, labels = Variable(images.to(device)), Variable(labels.to(device))
            labels = labels.view(-1,1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels, cat='train')
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*images.size(0)
        else:
            with torch.no_grad():
                model.eval()
                for images, labels in val_loader:
                    images, labels = Variable(images.to(device)), Variable(labels.to(device))
                    labels = labels.view(-1,1)

                    output = model(images)

                    loss = criterion(output, labels, cat='valid')

                    valid_loss += loss.item()*images.size(0)
        train_loss = train_loss/len(train_loader.sampler)
        train_losses.append(train_loss)
        valid_loss = valid_loss/len(val_loader.sampler)
        valid_losses.append(valid_loss)
        print (f"\nTraining Loss : {train_loss} \nValidation Loss : {valid_loss}")

        if valid_loss < valid_loss_min:
            print (f"Validation Loss decreased from {valid_loss} to  {valid_loss_min} ....Saving model")
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)
    plotify(train_losses, valid_losses)

    return model