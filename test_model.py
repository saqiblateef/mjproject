import os
import torch
import warnings
warnings.filterwarnings('ignore')

from config import Path, ConfigMP
from dictionary import Vocabulary
from utils import Utils
from data import DataHandler
from models.mean_pooling.model import MeanPooling

def main():
    # Set seed for reproducibility
    utils = Utils()
    utils.set_seed(1)

    # Create Mean pooling object
    cfg = ConfigMP()
    cfg.dataset = 'msvd'
    cfg.n_layers = 1
    cfg.decoder_type = 'lstm'
    cfg.vocabulary_min_count = 1
    cfg.batch_size = 10  # Smaller batch size for testing
    cfg.val_batch_size = 5  # Smaller batch size for testing
    
    # Set device
    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {cfg.device}")
    
    # Create path object
    path = Path(cfg, os.getcwd())
    
    # Load vocabulary
    voc = Vocabulary(cfg)
    voc.load()
    print('Vocabulary Size:', voc.num_words)
    
    # Create data handler and get test loader
    data_handler = DataHandler(cfg, path, voc)
    train_dset, val_dset, test_dset = data_handler.getDatasets()
    train_loader, val_loader, test_loader = data_handler.getDataloader(train_dset, val_dset, test_dset)
    
    # Create and load model
    model = MeanPooling(voc, cfg, path)
    model = model.to(cfg.device)
    
    # Load the trained model
    model_path = os.path.join('Saved', 'mean_pooling_model.pt')
    if os.path.exists(model_path):
        print("Loading trained model...")
        model.load_state_dict(torch.load(model_path, map_location=cfg.device))
        model.eval()  # Set to evaluation mode
    else:
        print("Error: No saved model found at", model_path)
        return
    
    print("\nGenerating captions for test videos...")
    # Test for 5 videos
    dataiter = iter(test_loader)
    for i in range(5):
        try:
            features, targets, mask, max_length, _, motion_batch, object_batch = next(dataiter)
            features = features.to(cfg.device)
            
            print(f"\nTest Video {i+1}:")
            print("Ground Truth:", utils.target_tensor_to_caption(voc, targets))
            
            # Greedy decoding
            _, greedy_caption = model.GreedyDecoding(features)
            print("Greedy Caption:", greedy_caption)
            
            # Only use features for beam search (mean pooling model doesn't use motion/object features)
            _, beam_caption = model.GreedyDecoding(features)
            print("Beam Search Caption:", beam_caption)
            
        except StopIteration:
            break

if __name__ == "__main__":
    main() 