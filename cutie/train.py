from torch.optim.lr_scheduler import ReduceLROnPlateau
from cutie.model import Cutie
from cutie.utils import CutieDataset, write_scores
from cutie.utils import FocalTverskyLoss, FocalLoss
import re
import os
from datetime import datetime
import argparse
import torch
import torch.optim as optim

#from torchviz import make_dot
#os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

parser = argparse.ArgumentParser(description='CUTIE parameters')

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--embedding_size', type=int, default=768) # Not used for DistilBERT as embedding size is 768.
parser.add_argument('--grid_size', type=int, default=64)
parser.add_argument('--n_class', type=int, default=5)
parser.add_argument('--embedding', type=str, default='distilbert') # 'distilbert' or 'w2vec'
parser.add_argument('--train_path', type=str, default='cutie/data/train') 
parser.add_argument('--val_path', type=str, default='cutie/data/val') 
parser.add_argument('--metrics_path', type=str, default='cutie/outputs/metrics')
parser.add_argument('--save_path', type=str, default='cutie/outputs/models')  
parser.add_argument('--model_save', type=int, default=0) # Number of epochs between model saving,\
                                                            # in addition to the best model
parser.add_argument('--epochs', type=int, default=10) # Number of epochs
parser.add_argument('--learning_rate', type=float, default=0.0001) # Number of iterations
parser.add_argument('--checkpoint', type=str, default='None') # Path to the last checkpoint of the training

params = parser.parse_args()

BATCH_SIZE = params.batch_size
NUM_WORKERS = params.num_workers
GRID_SIZE = params.grid_size
N_CLASS = params.n_class
TRAIN_PATH = params.train_path
VAL_PATH = params.val_path
TIME = str(datetime.now())
TIME = re.sub(r'\W+', '', TIME) 
METRICS_PATH = params.metrics_path + '/' + TIME
SAVE_PATH = params.save_path
MODEL_SAVE = params.model_save
EPOCHS = params.epochs
LEARNING_RATE = params.learning_rate

if params.embedding == 'distilbert':
    from tools.cutie.preprocessing import distilbert_embedding
    from transformers import DistilBertTokenizer, DistilBertModel, logging
    logging.set_verbosity_error()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    bertmodel = DistilBertModel.from_pretrained('distilbert-base-uncased')
    def EMBEDDING_FUN(text):
        vec = distilbert_embedding(text, tokenizer, bertmodel)
        return vec
    EMBEDDING_SIZE = 768

elif params.embedding == 'w2vec':
    from cutie.utils import word2vec_embedding
    from tools.cutie.preprocessing import generate_vocab
    EMBEDDING_SIZE = params.embedding_size
    vocab = generate_vocab(TRAIN_PATH, GRID_SIZE)
    embedding = word2vec_embedding(vocab, EMBEDDING_SIZE)
    EMBEDDING_FUN = embedding.embed  

if __name__ == '__main__':
    
    print(params)
    train_dataset = CutieDataset(TRAIN_PATH, EMBEDDING_FUN, GRID_SIZE, EMBEDDING_SIZE, N_CLASS)
    val_dataset = CutieDataset(VAL_PATH, EMBEDDING_FUN, GRID_SIZE, EMBEDDING_SIZE, N_CLASS)
    #test_dataset = CutieDataset('cutie/data/test', EMBEDDING_FUN, GRID_SIZE, EMBEDDING_SIZE, N_CLASS)
    train_loader = torch.utils.data.DataLoader( \
        train_dataset,\
        batch_size=BATCH_SIZE, \
        shuffle=True, \
        num_workers=NUM_WORKERS, \
        pin_memory=False, \
        drop_last=True )

    val_loader = torch.utils.data.DataLoader( \
        val_dataset, \
        batch_size=BATCH_SIZE, \
        shuffle=False,  \
        num_workers=NUM_WORKERS, \
        pin_memory=False, \
        drop_last=True)

    #test_loader = torch.utils.data.DataLoader( \
        #test_dataset, \
        #batch_size=BATCH_SIZE, \
        #shuffle=False,  \
        #num_workers=NUM_WORKERS, \
        #drop_last=False)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print('Using PyTorch version:', torch.__version__, ' Device:', device)
    model = Cutie(EMBEDDING_SIZE, GRID_SIZE, N_CLASS, BATCH_SIZE, EMBEDDING_FUN).to(device)
    
    #batch = next(iter(train_loader))
    #yhat = model(batch[0].to(device)) # Give dummy batch to forward().
    #make_dot(yhat, params=dict(list(model.named_parameters()))).render("cutie_torchviz", format="png")
    
    if params.checkpoint != 'None':
        model.load_state_dict(torch.load(params.checkpoint))
    print(model.count_parameters())
    criterion = FocalLoss()
    #optimizer = optim.SGD(model.parameters(),lr=LEARNING_RATE, momentum = 0.9)
    optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = 5, verbose = True)

    history = []
    for epoch in range(1,EPOCHS+1):
        print('training')
        train_scores = model.learn(optimizer, criterion, epoch, train_loader, device, 4)
        print('validating')
        val_scores = model.validate(criterion, val_loader, device)
        print(epoch,' : ',train_scores['loss'], val_scores['loss'])
        if epoch == 1:
            torch.save(model.state_dict(), os.path.join(SAVE_PATH,'best_model.pt'))
        elif val_scores['loss'] < min(history):
            torch.save(model.state_dict(), os.path.join(SAVE_PATH,'best_model.pt'))
        if MODEL_SAVE !=0:
            if epoch % MODEL_SAVE == 0:
                torch.save(model.state_dict(), os.path.join(SAVE_PATH,'model_epoch' + str(epoch) + '.pt'))
        if epoch == EPOCHS:
            torch.save(model.state_dict(), os.path.join(SAVE_PATH,'last_model.pt'))  
        history.append(val_scores['loss'])
        scheduler.step(train_scores['loss'])

        if not os.path.isdir(METRICS_PATH):
            os.mkdir(METRICS_PATH)
        write_scores(train_scores,epoch,N_CLASS,os.path.join(METRICS_PATH,'train_metrics.csv'))
        write_scores(val_scores,epoch,N_CLASS,os.path.join(METRICS_PATH,'val_metrics.csv'))

