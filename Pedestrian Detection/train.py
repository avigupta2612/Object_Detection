import torch
from engine import train_one_epoch, evaluate
from penndataset import PennDataset
from model import get_model_instance
import transforms as T
import utils

def get_transform(train):
    transform = []
    transform.append(T.ToTensor())
    if train:
        transform.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transform)

import utils
data_dir= '../PennFudanPed'
dataset = PennDataset(data_dir, get_transform(train=True))
dataset_test = PennDataset(data_dir, get_transform(train=False))
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
data_loader= torch.utils.data.DataLoader(dataset, shuffle= True, 
                                       batch_size=2, num_workers= 4,
                                       collate_fn = utils.collate_fn)
data_loader_test= torch.utils.data.DataLoader(dataset_test, shuffle=False, batch_size=2,
                                      num_workers=4, collate_fn= utils.collate_fn)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_classes = 2
model = get_model_instance(num_classes)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr = 0.005,
                            momentum = 0.9, weight_decay = 0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                        step_size =3,
                                        gamma= 0.1)
                                        
num_epochs = 5

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()
    evaluate(model, data_loader_test, device=device)