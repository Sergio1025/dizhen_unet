
import segmentation_models_pytorch as smp
import os
import time
import warnings
import random
import numpy as np
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
import albumentations as A

warnings.filterwarnings("ignore")
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

num_classes = 3
model = smp.Unet(encoder_name="resnet50",encoder_weights="imagenet", in_channels=3, classes=7)

train_root = r'C:\Users\63037\Desktop\jpg\train'
val_root = r'C:\Users\63037\Desktop\jpg\val'

class data_load(Dataset):
    def __init__(self, root, mode='val'):
        self.root = root
        self.mode = mode
        self.images, self.mask = self.read_file(self.root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, mask = self.load_img_and_mask(index)
        mask = np.array(mask)
        image = np.array(image)
        if self.mode == "train":
            trans_results = self.transform(image, mask)
            image = trans_results['image']
            mask = trans_results['mask']
        else:
            trans_results = self.test_transform(image, mask)
            image = trans_results['image']
            mask = trans_results['mask']
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(mask)
        return image, mask

    def read_file(self, root):
        image_path = root
        mask_path = root.replace('image','label')
        images = []
        masks = []
        image_files_list = os.listdir(image_path)
        for img in image_files_list:
            img_save_path = os.path.join(image_path, img)
            images.append(img_save_path)
            mask = os.path.join(mask_path, img.replace('JPG','png'))
            masks.append(mask)  
        images.sort()
        masks.sort()
        return images, masks

    def transform(self, image, mask):
        trans = A.Compose([
            A.Resize(1024,1024),
            A.HorizontalFlip(p=0.5),
        ])
        trans_results = trans(image=image, mask=mask)
        return trans_results
    
    def test_transform(self, image, mask):
        trans = A.Compose([
            A.Resize(1024,1024),
        ])
        trans_results = trans(image=image, mask=mask)

        return trans_results
    
    def load_img_and_mask(self, index):
        img_name = self.images[index]
        mask_name = self.mask[index]
        img = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name)
        return img, mask

train_set = data_load(train_root)
val_set = data_load(val_root)
n_val = len(train_set)
n_train = len(val_set)
loader_args = dict(batch_size=4, num_workers=4, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

def predict(X, threshold):
    '''X is sigmoid output of the model'''
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds

def metric(probability, truth, threshold=0.5, reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.reshape(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos

class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, phase, epoch):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.extend(dice.tolist())
        self.dice_pos_scores.extend(dice_pos.tolist())
        self.dice_neg_scores.extend(dice_neg.tolist())
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.nanmean(self.base_dice_scores)
        dice_neg = np.nanmean(self.dice_neg_scores)
        dice_pos = np.nanmean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou

def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print("Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" % (epoch_loss, iou, dice, dice_neg, dice_pos))
    return dice, iou

def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]

def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(outputs) # copy is imp
    labels = np.array(labels) # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou

class Trainer(object):
    '''This class takes care of training and validation of our model'''
    def __init__(self, model):
        self.num_workers = 6
        self.batch_size = {"train": 4, "val": 4}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 5e-4
        self.num_epochs = 50
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        self.net = model
        self.net = self.net.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        
        cudnn.benchmark = True
        self.dataloaders = {'train':train_loader,'val':val_loader}
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        
    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device,dtype=torch.long)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader): # replace `dataloader` with `tk0` for tqdm
            
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            targets = targets.to(dtype=torch.long)
            targets = torch.eye(num_classes)[targets.squeeze(1)]

            targets = targets.permute(0, 3, 1, 2)
            targets = targets[:,1:,:,:]
            outputs = outputs[:,1:,:,:]
            meter.update(targets, outputs)

        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "./model.pth")
            print()

if __name__ == '__main__':
    model_trainer = Trainer(model)
    model_trainer.start()

    losses = model_trainer.losses
    dice_scores = model_trainer.dice_scores
    iou_scores = model_trainer.iou_scores

    def plot(scores, name):
        plt.figure(figsize=(15,5))
        plt.plot(range(len(scores["train"])), scores["train"], label=f'train {name}')
        plt.plot(range(len(scores["train"])), scores["val"], label=f'val {name}')
        plt.title(f'{name} plot'); plt.xlabel('Epoch'); plt.ylabel(f'{name}');
        plt.legend(); 
        plt.show()

    plot(losses, "BCE loss")
    plot(dice_scores, "Dice score")
    plot(iou_scores, "IoU score")
    with open('./result.txt','a') as f:
        f.write(str(losses)+'\n')
        f.write(str(dice_scores)+'\n')
        f.write(str(iou_scores)+'\n')
