from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pickle
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import argparse


class DatasetNCFiles(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.frames = []
        self.transform = transform

        # Check if root directory exists
        if not os.path.exists(self.root):
            raise ValueError(f"Root directory {self.root} does not exist.")

        # Load datasets paths
        self.crop_dirs = os.listdir(self.root)

        for crop_dir in self.crop_dirs:
            full_dir_path = os.path.join(self.root, crop_dir)

            if not os.path.isdir(full_dir_path):
                continue

            crops = os.listdir(full_dir_path)
            crops.sort()
            s = len(crops)
            valid_count = s - s % 7
            crops = crops[:valid_count]
            crops = [os.path.join(crop_dir, crop) for crop in crops]
            self.frames.extend(crops)

    def __len__(self):
        return len(self.frames) // 7

    def __getitem__(self, idx):
        # compute start index
        start_idx = idx * 7

        if start_idx + 6 >= len(self.frames):
            raise IndexError("Index out of range")
        
        # At least 7 frames left in the dataset
        T = self.frames[start_idx:start_idx+7]

        I0 = torch.load(os.path.join(self.root, T[0]), map_location='cpu')
        I1 = torch.load(os.path.join(self.root, T[3]), map_location='cpu')
        I2 = torch.load(os.path.join(self.root, T[6]), map_location='cpu')

        if self.transform:
            I0 = self.transform(I0) if self.transform else I0
            I1 = self.transform(I1) if self.transform else I1
            I2 = self.transform(I2) if self.transform else I2

        return I0, I1, I2
    

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
    

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x = self.up(x1)
        x = torch.cat([x, x2], dim=1)
        return self.double_conv(x)
    

class SEBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // r, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.size()
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)
    

class UNetSE(nn.Module):
    def __init__(self, channels):
        super(UNetSE, self).__init__()
        self.channels = channels

        # Encoder Layers
        self.enc = DoubleConv(self.channels, 64)
        self.down_se_enc = SEBlock(64)
        self.down1 = DoubleConv(64, 128)
        self.down_se1 = SEBlock(128)
        self.down2 = DoubleConv(128, 256)
        self.down_se2 = SEBlock(256)
        self.down3 = DoubleConv(256, 512)
        self.down_se3 = SEBlock(512)
        self.down4 = DoubleConv(512, 1024)
        self.down_se4 = SEBlock(1024)


        # Decoder Layers
        self.dec1 = Up(1024, 512)
        self.up_se1 = SEBlock(512)
        self.dec2 = Up(512, 256)
        self.up_se2 = SEBlock(256)
        self.dec3 = Up(256, 128)
        self.up_se3 = SEBlock(128)
        self.dec4 = Up(128, 64)
        self.up_se4 = SEBlock(64)
        self.outc = nn.Conv2d(64, self.channels, kernel_size=1)

    def forward(self, I0, I1):
        #print("Forward pass UNetSE")
        # I0: batch, channel, width, height
        x = torch.cat([I0, I1], dim=1) # We concatenate along channel dimension
        #print(x.shape)
        # Encoder
        x1 = self.down_se_enc(self.enc(x))
        x2 = self.down_se1(self.down1(x1))
        x3 = self.down_se2(self.down2(x2))
        x4 = self.down_se3(self.down3(x3))
        x5 = self.down_se4(self.down4(x4))

        # Decoder
        out = self.up_se1(self.dec1(x5, x4))
        out = self.up_se2(self.dec2(out, x3))
        out = self.up_se3(self.dec3(out, x2))
        out = self.up_se4(self.dec4(out, x1))
        out = self.outc(out)

        return torch.chunk(out, chunks=2, dim=1)
    

class LCC(nn.Module):
    def __init__(self):
        super(LCC, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, I1, s1_pred2, s1_pred3, s2_pred1, s2_pred2, s2_pred3, s2_pred4, alpha):
        LCC1 = self.criterion(s2_pred2, I1) + self.criterion(s2_pred3, I1)
        LCC2 = 0.5 * (self.criterion(s2_pred1, s1_pred2) + self.criterion(s2_pred4, s1_pred3))

        return alpha * LCC1 + (1-alpha) * LCC2


# Advanced training loop

def train_model(model, loader, val_loader, criterion, optimizer, num_epochs, interval, alpha, device):
    
    writer = SummaryWriter(f'logs/alpha_{alpha}')

    train_losses = []
    val_losses = []

    global_step = 0

    print(f"Starting training for {num_epochs} epochs with alpha={alpha}")

    for epoch in range(num_epochs):

        model.train()
        current_train_loss = 0 
        train_batches = 0 

        for i, (batch_I0, batch_I1, batch_I2) in enumerate(loader):

            batch_I0 = batch_I0.to(device).float()
            batch_I1 = batch_I1.to(device).float()
            batch_I2 = batch_I2.to(device).float()

            optimizer.zero_grad()

            # Forward passs
            s1_pred1, s1_pred2 = model(batch_I0, batch_I1)
            s1_pred3, s1_pred4 = model(batch_I1, batch_I2)

            s2_pred1, s2_pred2 = model(s1_pred1, s1_pred3)
            s2_pred3, s2_pred4 = model(s1_pred2, s1_pred4)

            loss = criterion(batch_I1, s1_pred2, s1_pred3, s2_pred1, s2_pred2, s2_pred3, s2_pred4, alpha)

            # Backward pass
            loss.backward()
            optimizer.step()
    
            current_train_loss += loss.item()
            train_batches += 1

            # Write to tensorboard
            writer.add_scalar('Loss/Train_Step', loss.item(), global_step)

            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] - Batch {i}/{len(loader)} - Loss: {loss.item():.6f}")

            global_step += 1

        avg_train_loss = current_train_loss / train_batches # Average train loss for current epoch
        train_losses.append(avg_train_loss)

        writer.add_scalar('Loss/Train_Epoch_Avg', avg_train_loss, epoch)
            
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for j, (val_I0, val_I1, val_I2) in enumerate(val_loader):

                val_I0 = val_I0.to(device).float()
                val_I1 = val_I1.to(device).float()
                val_I2 = val_I2.to(device).float()

                s1_pred1, s1_pred2 = model(val_I0, val_I1)
                s1_pred3, s1_pred4 = model(val_I1, val_I2)

                s2_pred1, s2_pred2 = model(s1_pred1, s1_pred3)
                s2_pred3, s2_pred4 = model(s1_pred2, s1_pred4)

                loss = criterion(val_I1, s1_pred2, s1_pred3, s2_pred1, s2_pred2, s2_pred3, s2_pred4, alpha)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
            
        writer.add_scalar('Loss/Validation', avg_val_loss, global_step)

        writer.add_scalars('Loss/Epochs', {
            'Train': avg_train_loss,
            'Validation': avg_val_loss
        }, epoch)

        print(f"End of Epoch {epoch} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        if epoch % interval == 0:
            PATH = f"model_alpha_{alpha}_epoch_{epoch}.pth"
            torch.save(model.state_dict(), PATH)
            print(f"Model at epoch {epoch} saved in {PATH}")

    print("Training complete.")
    PATH = f"model_alpha_{alpha}.pth"
    torch.save(model.state_dict(), PATH)
    print(f"Model saved in {PATH}")

    with open(f'train_losses_alpha_{alpha}.pkl', 'wb') as fp:
        pickle.dump(train_losses, fp)

    with open(f'val_losses_alpha_{alpha}.pkl', 'wb') as fp:
        pickle.dump(val_losses, fp)

    writer.close()


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_train_root", default="../DATA/CMENS_DATA_PATCHES_V2/TRAIN/")
    parser.add_argument("--data_val_root", default="../DATA/CMENS_DATA_PATCHES_V2/VAL/")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=10, help="After how many epochs we save the model")
    parser.add_argument("--model_save_path", type=str, default="model.pth")
    parser.add_argument("--train_losses_path", type=str, default="train_losses.pkl")
    parser.add_argument("--val_losses_path", type=str, default="val_losses.pkl")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transformation
    DATA_MIN = 10.0
    DATA_MAX = 30.0
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float()), # From float64 to float32 (less memory usage)
        transforms.Lambda(lambda x: (x - DATA_MIN) / (DATA_MAX - DATA_MIN)),
        transforms.Lambda(lambda x: x.reshape(1, 128, 128)),
        
    ])

    dataset = DatasetNCFiles(root=args.data_train_root, transform=transform)
    val_dataset = DatasetNCFiles(root=args.data_val_root, transform=transform)
    print("Dataset length:", len(dataset))
    print("Validation Dataset length:", len(val_dataset))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    model = UNetSE(2)
    model.to(device)

    criterion = LCC()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_model(model, loader, val_loader, criterion, optimizer, args.num_epochs, args.save_interval, args.alpha, device)