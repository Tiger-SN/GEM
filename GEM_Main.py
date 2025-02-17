import os
import sys
import time
import pickle
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import clip
from clip.config import CFG
from clip.tokenizer import Tokenizer
from clip.models.utils.hungarian import hungarian
from transformers import AdamW
from utils import cosine_lr_schedule

# Log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Path
data_json = './Eay_Gaze/Eay_gaze_Caption_Train.json'
locate_data_json = './Eay_Gaze/Eay_gaze_Gaze_Train.json'
data_json = './Eay_Gaze/Eay_gaze_Caption_Val.json'
locate_data_json = './Eay_Gaze/Eay_gaze_Gaze_Val.json'


# Data processing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


class MIMICXrayDataSet(Dataset):
    def __init__(self, image_dir, data_json, locate_data_json, transform=None):
        self.image_dir = image_dir
        self.data = JsonReader(data_json)
        self.axis_data = JsonReader(locate_data_json)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Read Iamge label Caption
        frontal_img_name = self.data[index][0].split('g_')[0] + 'g'
        frontal_img = Image.open(frontal_img_name).convert('RGB')
        ##Caption

        ##Gaze point
        if self.transform:
            frontal_img = self.transform(frontal_img)
        return frontal_img, Caption, Gaze_point


train_dataset = MIMICXrayDataSet(image_dir, data_json, locate_data_json, vocab, transform)
val_dataset = MIMICXrayDataSet(image_dir, Val_data_json, '', vocab, transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)


CLIP_model, preprocess = clip.load('RN50', device)
##Load GEM model
Predictor_model = GEM_model().to(device)
Decoder_model = Decoder(vocab_size=227, encoder_length=CFG.num_patches, dim=256, num_heads=8, num_layers=6).to(device)


optimizer = AdamW(Predictor_model.parameters(), lr=1e-6)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 训练函数
def train(epoch):
    #CLIP_model.train()
    Predictor_model.train()
    total_loss = 0
    for i, (images, captions,GazePoint) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        captions = clip.tokenize(captions).to(device)
        
      
        image_features, text_features, x_text = CLIP_model(images, captions)
        Landmarks_Pred, out_heatmap_Graph, _, _, _ = Predictor_model(image_features, text_features, x_text, captions)
        
 
        loss = Regress_loss(Landmarks_Pred, Heatmap_landmarks) + Graph_loss(out_heatmap_Graph, Diag_Relation) * 0.1
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if i % 10 == 0:
            logging.info(f'Epoch:{epoch:02d} -> Iter:{i:02d} Loss:{loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    logging.info(f'Epoch:{epoch:02d} -> Average Loss:{avg_loss:.4f}')


start_epoch = 0
max_epoch = 10000
for epoch in range(start_epoch, max_epoch):
    train(epoch)
    if epoch % 5 == 0:
        torch.save(CLIP_model.state_dict(), f'./models/CLIP_model_epoch_{epoch:02d}.pth')
        torch.save(Predictor_model.state_dict(), f'./models/Predictor_model_epoch_{epoch:02d}.pth')