import argparse
import os
import torch as pt
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
from modules import SparseLinear
from utils import mkdir_exp

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default='datasets')
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--test-batch-size', type=int, default=1000)
parser.add_argument('--num-epochs', type=int, default=20)
parser.add_argument('--coef', type=float, default=0.1)
args = parser.parse_args()

class SparseMNISTClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = SparseLinear(784, 300)
    self.fc2 = SparseLinear(300, 100)
    self.fc3 = SparseLinear(100, 10)

  def forward(self, x):
    x, lc1 = self.fc1(x)
    x = F.relu_(x)
    x, lc2 = self.fc2(x)
    x = F.relu_(x)
    x, lc3 = self.fc3(x)
    F.log_softmax(x, dim=-1)
    return x, (lc1, lc2, lc3)

exp_path = mkdir_exp('L0_MNIST')

writer = SummaryWriter(log_dir=exp_path)

data_root = os.path.join(exp_path, args.data_root)
if not os.path.isdir(data_root):
  os.makedirs(data_root)

train_loader = DataLoader(
  MNIST(
    data_root, 
    train=True, 
    download=True,
    transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
    ])
  ), 
  shuffle=True, 
  batch_size=args.batch_size
)

test_loader = DataLoader(
  MNIST(
    data_root, 
    train=False, 
    download=True,
    transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
    ])
  ), 
  batch_size=args.test_batch_size
)

device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
model = SparseMNISTClassifier().to(device)
optimizer = Adam(model.parameters(), lr=args.lr)

for epoch in tqdm(range(args.num_epochs)):
  model.train()

  ep_error_loss = 0.0
  ep_complexity_loss = 0.0
  ep_train_loss = 0.0

  for data, target in train_loader:
    data, target = data.to(device), target.to(device)
    output, complexity_losses = model(data.flatten(start_dim=1))

    lam = args.coef / len(train_loader.dataset)
    complexity_loss = lam * sum(complexity_losses)
    error_loss = F.cross_entropy(output, target)
    loss = error_loss + complexity_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ep_error_loss += error_loss.item()
    ep_complexity_loss += complexity_loss.item()
    ep_train_loss += loss.item()

  ep_error_loss = ep_error_loss / len(train_loader)
  ep_complexity_loss = ep_complexity_loss / len(train_loader)
  ep_train_loss = ep_train_loss / len(train_loader)

  writer.add_scalar('Loss/train/error_loss', ep_error_loss, epoch)
  writer.add_scalar('Loss/train/complexity_loss', ep_complexity_loss, epoch)
  writer.add_scalar('Loss/train/loss', ep_train_loss, epoch)
    
  model.eval()
  test_loss = 0.0
  correct = 0.0
  with pt.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output, _ = model(data.flatten(start_dim=1))
      test_loss += F.cross_entropy(output, target, reduction='sum').item()
      pred = output.argmax(dim=-1, keepdim=True)
      correct += pred.eq(target.view_as(pred)).sum().item()
  test_loss = test_loss / len(test_loader.dataset)
  accuracy = 100 * correct / len(test_loader.dataset)

  writer.add_scalar('Loss/eval/loss', test_loss, epoch)
  writer.add_scalar('Accuracy/eval/accuracy', accuracy, epoch)
  
