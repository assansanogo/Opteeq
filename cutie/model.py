import torch
import torch.nn as nn
import torch.nn.functional as F
from cutie.utils import init_stats, save_stats, init_scores, save_scores
from tools.cutie.preprocessing import convert_json_to_tensors, generate_grids
import typing

class Cutie(nn.Module):
    """
    Pytorch implementation of Cutie model
    https://arxiv.org/pdf/1903.12363.pdf
    """
    def __init__(self, EMBEDDING_SIZE: int, GRID_SIZE: int, N_CLASS: int, BATCH_SIZE: int, EMBEDDING_FUN: typing.Callable,):
      """
      :param EMBEDDING_SIZE: dimension of the word embedding output
      :type EMBEDDING_SIZE: int
      :param GRID_SIZE: dimension of the grid of word
      :type GRID_SIZE: int
      :param N_CLASS: Number of classes to predict
      :type N_CLASS: int
      :param BATCH_SIZE: Batch size for the model training
      :type BATCH_SIZE: int
      :param EMBEDDING_FUN: Word embedding function
      :type EMBEDDING_FUN: callable
      """
      super(Cutie, self).__init__()
      # Parameters
      self.embedding_size = EMBEDDING_SIZE
      self.embedding = EMBEDDING_FUN
      self.grid_size = GRID_SIZE
      self.n_class = N_CLASS
      self.batch_size = BATCH_SIZE

      # conv Block
      self.conv1 = nn.Conv2d(self.embedding_size, 256, (3,5), stride = 1, padding = 'same')
      self.conv2 = nn.Conv2d(256, 256, (3,5), stride = 1, padding = 'same')
      self.conv3 = nn.Conv2d(256, 256, (3,5), stride = 1, padding = 'same')
      self.conv4 = nn.Conv2d(256, 256, (3,5), stride = 1, padding = 'same')
      # atrous conv Block
      self.atrousconv1 = nn.Conv2d(256, 256, (3,5), stride = 1, padding = 'same', dilation = 2)
      self.atrousconv2 = nn.Conv2d(256, 256, (3,5), stride = 1, padding = 'same', dilation = 2)
      self.atrousconv3 = nn.Conv2d(256, 256, (3,5), stride = 1, padding = 'same', dilation = 2)
      self.atrousconv4 = nn.Conv2d(256, 256, (3,5), stride = 1, padding = 'same', dilation = 2)
      # ASPP
      self.aspp1 = nn.Conv2d(256, 256, (3,5), stride = 1, padding = 'same', dilation = 4)
      self.aspp2 = nn.Conv2d(256, 256, (3,5), stride = 1, padding = 'same', dilation = 8)
      self.aspp3 = nn.Conv2d(256, 256, (3,5), stride = 1, padding = 'same', dilation = 16)
      self.aspp4 = nn.AvgPool2d(self.grid_size)
      self.asppout = nn.Conv2d(256*4, 256, 1, stride = 1, padding = 'same')
      # SHORTCUT LAYER
      self.shortcutconv = nn.Conv2d(256*2, 64, (3,5), stride = 1, padding = 'same')
      #OUT LAYER
      self.outconv = nn.Conv2d(64, self.n_class, 1, stride = 1, padding = 'same')

      self.initialize_weights()
      
    def forward(self, x):
      # Conv block
      x_lowlevel = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x_lowlevel))
      x = F.relu(self.conv3(x))
      x = F.relu(self.conv4(x))

      # atrous conv block :
      x = F.relu(self.atrousconv1(x))
      x = F.relu(self.atrousconv2(x))
      x = F.relu(self.atrousconv3(x))
      x = F.relu(self.atrousconv4(x))

      # ASPP block :
      x1 = F.relu(self.aspp1(x))
      x2 = F.relu(self.aspp2(x))
      x3 = F.relu(self.aspp3(x))
      x4 = F.interpolate(self.aspp4(x), size=(self.grid_size), mode='nearest')
      x = torch.cat((x1,x2,x3,x4),1)
      x = F.relu(self.asppout(x))
      
      # SHORTCUT LAYER
      x = torch.cat((x,x_lowlevel),1)
      x = F.relu(self.shortcutconv(x))

      # output layer
      x = F.relu(self.outconv(x))
      x = x.view(self.batch_size, self.n_class, self.grid_size * self.grid_size)
      out = nn.Softmax(dim=1)(x)

      return out
      
    def initialize_weights(self):
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          #nn.init.xavier_normal_(m.weight)
          nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    def count_parameters(self):
      return sum(p.numel() for p in self.parameters() if p.requires_grad)
      
    def learn(self, optimizer, criterion, epoch, train_loader, device, log_interval=10):
      # Initialize statistics and scores
      stats = init_stats(self.n_class)
      train_scores = init_scores(self.n_class)
      
      # Set model to training mode
      self.train()

      # Loop over each batch from the training set
      for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad() 
        
        # Pass data through the network
        output = self(data)

        # Calculate loss
        loss = criterion(output, target)
        # Backpropagate
        loss.backward()
        # Update weights
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

        save_stats(stats, self.n_class, output, target, loss)
      save_scores(train_scores, stats, train_loader, self.n_class)
      return(train_scores)

    def validate(self, criterion, val_loader, device):
      # Initialize statistics and scores
      stats = init_stats(self.n_class)
      val_scores = init_scores(self.n_class)
      
      # Set model to training mode
      self.eval()

      # Loop over each batch from the training set
      for batch_idx, (data, target) in enumerate(val_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)
        
        # Pass data through the network
        output = self(data)

        # Calculate loss
        loss = criterion(output, target)

        save_stats(stats, self.n_class, output, target, loss)
      save_scores(val_scores, stats, val_loader, self.n_class)
      return(val_scores)
    
    def predict(self, json_file, device):
      result = {'PLACE':[],'DATE':[],'TOTAL_AMOUNT':[]} 
      input, _ = convert_json_to_tensors(json_file, self.embedding, self.grid_size, self.embedding_size, self.n_class)
      input = input.reshape((1,self.embedding_size,self.grid_size,self.grid_size))
      input = input.to(device)
      output = self(input)
      grid = generate_grids(json_file, grid_size = self.grid_size, known_class=False)
      class_map = {0: 'NOTHING',1: 'PLACE', 2: 'DATE',3: 'TOTAL_TEXT',4: 'TOTAL_AMOUNT'}
      for line in range(1,self.grid_size + 1):
        for col in  range(1,self.grid_size + 1):
            if (col, line) in grid['grid'].keys():
                class_name = class_map[output.data.max(1)[1][0,line * self.grid_size + col].item()]
                if class_name in result.keys():
                    result[class_name].append(grid['grid'][(col, line)][0])
      return(result)