import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from gensim.models import Word2Vec
import csv
from tools.cutie.preprocessing import convert_json_to_tensors

class FocalTverskyLoss(nn.Module):
    """
    Pytorch implementation of the FocalTversky Loss function
    """
    def __init__(self, smooth=1, alpha=0.5, beta=0.5, gamma=1):
        super(FocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)  
        FocalTversky = (1 - Tversky)**self.gamma
                       
        return FocalTversky

class TverskyLoss(nn.Module):
    """
    Pytorch implementation of the Tversky Loss function
    """
    def __init__(self, smooth=1, alpha=0.5, beta=0.5):
        super(TverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + self.smoothth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)  
        
        return 1 - Tversky

class FocalLoss(nn.Module):
    """
    Pytorch implementation of the Focal Loss function
    """
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
                       
        return focal_loss

class CutieDataset(torch.utils.data.Dataset):
    """
    Dataset class for cutie model training
    """
    def __init__(self, root, embedding_fun: typing.Callable, grid_size: int = 64, embedding_size: int = 128, N_class: int = 5):
        self.root = root
        self.files = [os.path.join(root,file) for file in os.listdir(root) if file.endswith('.json')]
        self.embedding = embedding_fun
        self.grid_size = grid_size
        self.embedding_size = embedding_size
        self.N_class = N_class

    def __getitem__(self, index):
        
        grid_tensor, classes_tensor = convert_json_to_tensors(self.files[index], embedding_fun = self.embedding, grid_size = self.grid_size, embedding_size = self.embedding_size, N_class = self.N_class)
        
        return grid_tensor, classes_tensor
    
    def __len__(self):
        return len(self.files)

class word2vec_embedding():
    """
    Word2vec embedding class for cutie model training experimentation
    """
    def __init__(self, vocab, embedding_size):
        self.vocab = vocab
        self.embedding_size = embedding_size
    def embed(self, text):
        w2v = Word2Vec(sentences=self.vocab, vector_size=self.embedding_size, window=3, min_count=1, workers=4)
        if text not in w2v.wv.key_to_index.keys():
            w2v.build_vocab([[text]], update=True)
            w2v.train([[text]], total_examples=1, epochs =2)
        vector = torch.tensor(w2v.wv[text])
        return vector

def init_stats(N_class: int) -> dict:
    """
    Initialize a dictionary to store cutie model training or validation statistics
    
    :param N_class: Number of classes to predict
    :type N_class: int
    :return: dictionary to store the statistics
    :rtype: dict
    """
    stats = {'TP':{},'TN':{},'FP':{},'FN':{},'softAP':{},'AP':{},'running_loss':0}
    for class_id in range(N_class):
        for key in stats.keys():
            if key != 'running_loss':
                stats[key][class_id] = 0
    return stats

def init_scores(N_class: int) -> dict:
    """
    Initialize a dictionary to store cutie model training or validation scores
    
    :param N_class: Number of classes to predict
    :type N_class: int
    :return: dictionary to store the scores
    :rtype: dict
    """
    scores = {'Acc':{},'Prec':{},'Rec':{},'F1':{},'softAP':{},'AP':{},'loss':0}
    for class_id in range(N_class):
        for key in scores.keys():
            if key != 'loss':
                scores[key][class_id] = 0
    return scores

def save_stats(stats: dict, N_class: int, output: torch.tensor, target: torch.tensor, loss: torch.tensor):
    """
    Calculates and saves the statistics after each batch during cutie training or validation.
    
    :param stats: Dictionnary storing the training or validation statistics
    :type stats: dict
    :param N_class: Number of classes to predict
    :type N_class: int
    :param output: output predictions of the model
    :type output: torch.tensor
    :param target: Ground truth classes
    :type target: torch.tensor
    :param loss: Loss calculated on the current batch
    :type loss: torch.tensor
    """
    with torch.no_grad():
        for class_id in range(N_class):
            TP_tensor = ((output.data.max(1)[1] == class_id) * target[:,class_id,:])
            TN_tensor = ((output.data.max(1)[1] != class_id) * (1-target[:,class_id,:]))
            FP_tensor = ((output.data.max(1)[1] == class_id) * (1-target[:,class_id,:]))
            FN_tensor = ((output.data.max(1)[1] != class_id) * target[:,class_id,:])
            stats['TP'][class_id] += torch.sum(TP_tensor).item()  
            stats['TN'][class_id] += torch.sum(TN_tensor).item()
            stats['FP'][class_id] += torch.sum(FP_tensor).item()
            stats['FN'][class_id] += torch.sum(FN_tensor).item()
            stats['softAP'][class_id] += torch.sum(torch.sum(FN_tensor,dim=1) == 0).item()
            stats['AP'][class_id] += torch.sum((torch.sum(FN_tensor,dim=1) == 0) * (torch.sum(FP_tensor,dim=1) == 0)).item()
    stats['running_loss'] += loss.item()

def save_scores(scores: dict, stats: dict, dataloader :typing.Callable, N_class: int):
    """
    Calculates and saves the scores after each epoch during cutie training
    
    :param scores: Dictionnary storing the training or validation scores
    :type scores: dict
    :param stats: Dictionnary storing the training or validation statistics
    :type stats: dict
    :param dataloader: Dataloader function used to train or validate the model
    :type dataloader: torch.tensor
    :param N_class: Number of classes to predict
    :type N_class: int
    """
    with torch.no_grad():
        for class_id in range(N_class):
            TN = stats['TN'][class_id]
            TP = stats['TP'][class_id]
            FN = stats['FN'][class_id]
            FP = stats['FP'][class_id]
            scores['softAP'][class_id] = stats['softAP'][class_id] / (dataloader.batch_size * len(dataloader))
            scores['softAP'][class_id] = round(scores['softAP'][class_id],3)
            scores['AP'][class_id] = stats['AP'][class_id] / (dataloader.batch_size * len(dataloader))
            scores['AP'][class_id] = round(scores['AP'][class_id],3)
            if (TN + TP + FN + FP) != 0:
                scores['Acc'][class_id] = round((TN + TP) / (TN + TP + FN + FP),3)
            else:
                scores['Acc'][class_id] = 0
            if (TP + FP) != 0:
                scores['Prec'][class_id] = round(TP / (TP + FP),3)
            else:
                scores['Prec'][class_id] = 0
            if (TP + FN) != 0:
                scores['Rec'][class_id] = round(TP / (TP + FN),3)
            else:
                scores['Rec'][class_id] = 0
            if (scores['Prec'][class_id] + scores['Rec'][class_id]) != 0:
                scores['F1'][class_id] = 2 * (scores['Prec'][class_id] * scores['Rec'][class_id]) / (scores['Prec'][class_id] + scores['Rec'][class_id])
                scores['F1'][class_id] = round(scores['F1'][class_id],3)      
            else:
                scores['F1'][class_id] = 0
        scores['loss'] = round(stats['running_loss'] / len(dataloader),5)

def write_scores(scores: dict, epoch: int, N_class: int, file_path: str):
    """
    Append the scores to a csv file after each epoch during cutie training
    
    :param scores: Dictionnary storing the training or validation scores
    :type scores: dict
    :param epoch: Current epoch of the training
    :type epoch: int
    :param N_class: Number of classes to predict
    :type N_class: int
    :param file_path: Path to the output file
    :type file_path: str
    """
    if not os.path.isfile(file_path):
        open(file_path,'w').close()
    
    filesize = os.path.getsize(file_path)
    if filesize == 0:
        headers = ['epoch','loss']
        for topic in ['AP','softAP','Acc','Prec','Rec','F1']:
            for class_id in range(N_class):
                headers.append(f'{topic}_{class_id}')
        with open(file_path, 'w',newline='') as outfile:
            writer = csv.writer(outfile, delimiter='|')
            writer.writerow(headers)
    
    new_line = [epoch,scores["loss"]]
    for topic in ['AP','softAP','Acc','Prec','Rec','F1']:
            for class_id in range(N_class):
                new_line.append(scores[topic][class_id])
    with open(file_path, 'a',newline='') as outfile:
        writer = csv.writer(outfile, delimiter='|')
        writer.writerow(new_line)




