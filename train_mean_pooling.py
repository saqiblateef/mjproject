import os
import pickle
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import warnings
warnings.filterwarnings('ignore')

# Import Path,Vocabulary, utility, evaluator and datahandler module
from config import Path, ConfigMP
from dictionary import Vocabulary
from utils import Utils
from evaluate import Evaluator
from data import DataHandler
from models.mean_pooling.model import MeanPooling

def main():
    # Set seed for reproducibility
    utils = Utils()
    utils.set_seed(1)

    # Create Mean pooling object
    cfg = ConfigMP()
    # specifying the dataset in configuration object from {'msvd','msrvtt'}
    cfg.dataset = 'msvd'
    # creation of path object
    path = Path(cfg, os.getcwd())

    # Hyperparameters
    cfg.batch_size = 100  # training batch size
    cfg.n_layers = 1     # number of layers in decoder rnn
    cfg.decoder_type = 'lstm'  # from {'lstm','gru'}
    cfg.vocabulary_min_count = 1

    # Vocabulary object
    voc = Vocabulary(cfg)
    # If vocabulary is already saved or downloaded the saved file
    try:
        voc.load()
        print('Loaded existing vocabulary')
    except:
        print('Creating new vocabulary')
        text_dict = {}
        data_handler = DataHandler(cfg, path, voc)
        text_dict.update(data_handler.train_dict)
        text_dict.update(data_handler.val_dict)
        text_dict.update(data_handler.test_dict)
        for k, v in text_dict.items():
            for anno in v:
                voc.addSentence(anno)
        voc.save()

    print('Vocabulary Size : ', voc.num_words)

    # Filter rare words
    min_count = cfg.vocabulary_min_count  # remove all words below count min_count
    voc.trim(min_count=min_count)
    print('Vocabulary Size after trimming: ', voc.num_words)

    # Datasets and dataloaders
    data_handler = DataHandler(cfg, path, voc)
    train_dset, val_dset, test_dset = data_handler.getDatasets()
    train_loader, val_loader, test_loader = data_handler.getDataloader(train_dset, val_dset, test_dset)

    # Model object
    model = MeanPooling(voc, cfg, path)
    
    # Evaluator object on test data
    test_evaluator_greedy = Evaluator(model, test_loader, path, cfg, data_handler.test_dict)
    test_evaluator_beam = Evaluator(model, test_loader, path, cfg, data_handler.test_dict, decoding_type='beam')

    # Training configuration
    cfg.encoder_lr = 1e-4
    cfg.decoder_lr = 1e-3
    cfg.teacher_forcing_ratio = 1.0
    model.update_hyperparameters(cfg)

    print("Starting training...")
    val_losses = []
    best_val_loss = float('inf')
    for e in range(1, 21):
        loss = model.train_epoch(train_loader, utils)
        if e % 10 == 0:
            print('Epoch:', e, 'Loss:', loss)
            print('Greedy decoding metrics:', test_evaluator_greedy.evaluate(utils, model, e, loss))
            val_loss = model.loss_calculate(val_loader, utils)
            val_losses.append(val_loss)
            print('Validation loss:', val_losses[-1])
            
            # Save model if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Saving model with validation loss: {val_loss}")
                torch.save(model.state_dict(), os.path.join('Saved', 'mean_pooling_model.pt'))

if __name__ == "__main__":
    main() 