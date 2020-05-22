import os
from typing import List, Union
import pickle
import random

import torch

from torch.utils.data.dataloader import DataLoader
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
### END YOUR LIBRARIES

import utils

from bpe import BytePairEncoding
from model import IMDBmodel
from data import IMDBdataset

# You can use tqdm to check your progress
from tqdm import tqdm, trange

def training(
    model: IMDBmodel,
    model_name: str,
    train_dataset: IMDBdataset,
    val_dataset: IMDBdataset,
    pretrained_model_path: Union[str, None]
):
    """ IMDB classification trainer
    Implement IMDB sentiment classification trainer with the given model and datasets.
    If the pretrained model is given, please load the model before training.

    Note 1: Don't forget setting model.train() and model.eval() before training / validation.
            It enables / disables the dropout layers of our model.

    Note 2: Use (TRUE_POSITIVES + TRUE_NEGATIVES) / (TOTAL_SAMPLES) as accuracy.

    Note 3: There are useful tools for your implementation in utils.py

    Note 4: Training takes less than 10 minutes per a epoch on TITAN RTX.

    Memory tip 1: If you delete the output tensors explictly after every loss calculation like "del out, loss",
                  tensors are garbage-collected before next loss calculation so you can cut memory usage.

    Memory tip 2: If you use torch.no_grad when inferencing the model for validation,
                  you can save memory space of gradients. 

    Memory tip 3: If you want to keep batch_size while reducing memory usage,
                  creating a virtual batch is a good solution.
    Explanation: https://medium.com/@davidlmorton/increasing-mini-batch-size-without-increasing-memory-6794e10db672

    Useful readings: https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/ 

    Arguments:
    model -- IMDB model which need to be trained
    model_name -- The model name. You can use this name to save your model per a epoch
    train_dataset -- IMDB dataset for training
    val_dataset -- IMDB dataset for validation
    pretrained_model_path -- the pretrained model file path.
                             You have to load the pretrained model properly
                             None if pretraining is disabled

    Variables:
    batch_size -- Batch size
    learning_rate -- Learning rate for the optimizer
    epochs -- The number of epochs

    Returns:
    train_losses -- List of average training loss per a epoch
    val_losses -- List of average validation loss per a epoch
    train_accuracies -- List of average training accuracy per a epoch
    val_accuracies -- List of average validation accuracy per a epoch
    """
    # Below options are just our recommendation. You can choose different options if you want.
    batch_size = 16
    epochs = 10
    if pretrained_model_path is None:
        learning_rate = 1e-4
    else:
        learning_rate = 3e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ### YOUR CODE HERE
    train_losses: List[float] = None
    val_losses: List[float] = None
    train_accuracies: List[float] = None
    val_accuracies: List[float] = None
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Load pretrained model
    if pretrained_model_path is not None:
        state_dict = torch.load(pretrained_model_path)
        own_state = model.state_dict()

        for name, param in state_dict.items():
            if name not in own_state:
                continue
            elif isinstance(param, torch.nn.Parameter):
                param = param.data
            own_state[name].copy_(param)

    # Train
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    eval_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    loss_log = tqdm(bar_format='{desc}')

    model.to(device)
    ceLoss = torch.nn.CrossEntropyLoss()
    for epoch in trange(epochs, desc="Epoch", position=0):
        train_loss_in_epoch = 0
        train_acc_in_epoch = 0
        model.train()
        for sentences, label in tqdm(train_data):
            optimizer.zero_grad()
            sentences = torch.stack(sentences,0).to(device)
            gt_label = torch.stack([torch.LongTensor([0]) for i in range(batch_size)]).squeeze(1).to(device)
            for i in range(label.shape[0]):
                if label[i]:
                    gt_label[i] = torch.LongTensor([1])
            predict = model(sentences)
            loss = ceLoss(predict, gt_label)
            loss.backward()
            optimizer.step()
            predict_bool = torch.argmax(predict, dim=-1)
            true_positive = (predict_bool * gt_label).sum(0)
            true_negative = ((~(predict_bool.to(torch.bool)))*(~(gt_label.to(torch.bool)))).sum()
            acc = (float(true_positive) + float(true_negative)) / float(batch_size)
            train_loss_in_epoch += float(loss.data)
            train_acc_in_epoch += acc
            des = 'Acc: {:06.4f}'.format(acc)
            loss_log.set_description_str(des)

        # Log training loss
        train_loss_in_epoch = train_loss_in_epoch / float(len(train_data))
        train_losses.append(train_loss_in_epoch)
        train_acc_in_epoch = train_acc_in_epoch / float(len(train_data))
        train_accuracies.append(train_acc_in_epoch)

        # Validate
        model.eval()
        valid_loss_in_epoch = 0.
        valid_acc_in_epoch = 0.
        for j, (sentences, label) in enumerate(eval_data):
            with torch.no_grad():
                sentences = torch.stack(sentences, 0).to(device)
                gt_label = torch.stack([torch.LongTensor([0]) for i in range(batch_size)]).squeeze(1).to(device)
                for i in range(label.shape[0]):
                    if label[i]:
                        gt_label[i] = torch.LongTensor([1])
                predict = model(sentences)
                loss = ceLoss(predict, gt_label)
                valid_loss_in_epoch += float(loss.data)
                predict_bool = torch.argmax(predict, dim=-1)
                true_positive = (predict_bool * gt_label).sum(0)
                true_negative = ((~(predict_bool.to(torch.bool))) * (~(gt_label.to(torch.bool)))).sum()
                acc = (float(true_positive) + float(true_negative)) / float(batch_size)
                valid_acc_in_epoch += acc
        valid_loss_in_epoch = valid_loss_in_epoch / float(len(eval_data))
        valid_acc_in_epoch = valid_acc_in_epoch / float(len(eval_data))
        val_losses.append(valid_loss_in_epoch)
        val_accuracies.append(valid_acc_in_epoch)
        print('Epoch: {}'.format(epoch))
        print('train_loss: {:06.4f}'.format(train_loss_in_epoch))
        print('train_acc: {:06.4f}'.format(train_acc_in_epoch))
        print('valid_loss: {:06.4f}'.format(valid_loss_in_epoch))
        print('valid_acc: {:06.4f}'.format(valid_acc_in_epoch))
        # Log validation loss



    ### END YOUR CODE

    assert len(train_losses) == len(val_losses) == len(train_accuracies) == len(val_accuracies) == epochs

    assert all(isinstance(loss, float) for loss in train_losses) and \
           all(isinstance(loss, float) for loss in val_losses) and \
           all(isinstance(accuracy, float) for accuracy in train_accuracies) and \
           all(isinstance(accuracy, float) for accuracy in val_accuracies)

    return train_losses, val_losses, train_accuracies, val_accuracies

#############################################################
# Testing functions below.                                  #
#                                                           #
# We do not tightly check the correctness of your trainer.  #
# You should attach the loss & accuracy plot to the report  #
# and submit the trained model to validate your trainer.    #
# We will grade the score by running your saved model.      #
#############################################################

def train_model():
    print("======IMDB Training======")
    """ IMDB Training 
    You can modify this function by yourself.
    This function does not affects your final score.
    """
    train_dataset = IMDBdataset(os.path.join('data', 'imdb_train.csv'))
    val_dataset = IMDBdataset(os.path.join('data', 'imdb_val.csv'))
    model = IMDBmodel(train_dataset.token_num)

    model_name = 'imdb'

    # You can choose whether to enable fine-tuning
    fine_tuning = True

    if fine_tuning:
        model_name += '_fine_tuned'
        pretrained_model_path = 'models/pretrained146.pth'

        # You can use a model which has been pretrained over 200 epochs by TA
        # If you use this saved model, you should mention it in the report
        #
        # pretrained_model_path = 'pretrained_byTA.pth' 

    else:
        model_name += '_no_fine_tuned'
        pretrained_model_path = None
        
    train_losses, val_losses, train_accuracies, val_accuracies \
            = training(model, model_name, train_dataset, val_dataset, \
                       pretrained_model_path=pretrained_model_path)

    torch.save(model.state_dict(), model_name+'_final.pth')

    with open(model_name+'_result.pkl', 'wb') as f:
        pickle.dump((train_losses, val_losses, train_accuracies, val_accuracies), f)

    utils.plot_values(train_losses, val_losses, title=model_name + "_losses")
    utils.plot_values(train_accuracies, val_accuracies, title=model_name + "_accuracies")

    print("Final training loss: {:06.4f}".format(train_losses[-1]))
    print("Final validation loss: {:06.4f}".format(val_losses[-1]))
    print("Final training accuracy: {:06.4f}".format(train_accuracies[-1]))
    print("Final validation accuracy: {:06.4f}".format(val_accuracies[-1]))

if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    random.seed(1234)
    torch.manual_seed(1234)

    train_model()
