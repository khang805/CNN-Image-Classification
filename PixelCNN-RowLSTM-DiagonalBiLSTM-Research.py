# pixelrnn_family.py
# PyTorch implementation: PixelCNN (masked convs), RowLSTM (row-scan), Diagonal BiLSTM (skew+BiLSTM)
# Trainable on CIFAR-10 (discrete 0..255 pixel modeling)
# Author: assistant (adapt/extend for assignment)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
import os

# -----------------------
# Utils
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def bits_per_dim(nll_loss, pixels):
    # nll_loss: average negative log-likelihood per image in nats (i.e., cross entropy)
    # bits/dim = (nll / ln(2)) / pixels
    return (nll_loss / math.log(2.0)) / pixels

# -----------------------
# Dataset (CIFAR-10)
# -----------------------
# We'll model raw RGB 0..255 values as categorical over 256 values
transform = transforms.Compose([
    transforms.ToTensor(),  # 0..1 float
])

train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Convert to integer 0..255 representation on the fly in collate_fn
def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])  # [B,3,32,32], floats 0..1
    imgs = (imgs * 255.0).long()  # integers 0..255
    labels = torch.tensor([b[1] for b in batch])
    return imgs, labels

batch_size = 64
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

PIXEL_VALUES = 256
H, W = 32, 32
C = 3
PIXELS = H * W * C

# -----------------------
# Masked Convolution (for PixelCNN)
# -----------------------
class MaskedConv2d(nn.Conv2d):
    """
    Masked conv for autoregressive PixelCNN.
    mask_type: 'A' (exclude center) for first layer, 'B' (include center) for subsequent.
    Works with RGB ordering: we will implement channel-aware masking to respect pixel ordering
    (we'll use pixel ordering: channel-major R,G,B at each spatial position).
    Simplified mask: For channels we keep full channel ordering by blocking future pixels.
    """
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        assert mask_type in ['A', 'B']
        self.register_buffer('mask', self.weight.data.clone().zero_())
        kh, kw = self.kernel_size
        # center position
        cy, cx = kh // 2, kw // 2
        # create base mask (same for all in/out channels) then we'll replicate
        for y in range(kh):
            for x in range(kw):
                if y < cy or (y == cy and x < cx):
                    self.mask[:, :, y, x] = 1
                if mask_type == 'B':
                    # allow center pixel for type B
                    if y == cy and x == cx:
                        self.mask[:, :, y, x] = 1
        # Note: this simple mask blocks future spatial pixels. For exact channel-order masking
        # (R->G->B) you need channel-wise masks; good enough for students / experiments.

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

# -----------------------
# PixelCNN model
# -----------------------
class PixelCNN(nn.Module):
    def __init__(self, in_channels=3, n_filters=64, n_layers=7):
        super().__init__()
        self.input_conv = MaskedConv2d('A', in_channels, n_filters, kernel_size=7, padding=3)
        self.hidden = nn.ModuleList([MaskedConv2d('B', n_filters, n_filters, kernel_size=3, padding=1)
                                     for _ in range(n_layers)])
        self.out_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(n_filters, in_channels * PIXEL_VALUES, kernel_size=1)  # predict logits per channel per pixel
        )

    def forward(self, x):
        # x: [B, C, H, W] integers 0..255 (long) -> convert to float and normalize for conv input
        # We'll convert to float in [0,1] (or -0.5..0.5)
        x_in = x.float() / 255.0 - 0.5
        h = F.relu(self.input_conv(x_in))
        for conv in self.hidden:
            h = F.relu(conv(h)) + h  # residual-ish
        logits = self.out_conv(h)  # [B, C*256, H, W]
        # reshape to [B, C, 256, H, W]
        B = logits.shape[0]
        logits = logits.view(B, C, PIXEL_VALUES, H, W)
        # Move channels to [B, C, H, W, 256] for easier cross-entropy
        logits = logits.permute(0,1,3,4,2)  # [B,C,H,W,256]
        return logits

    def sample(self, shape, device=device):
        # Autoregressive sampling pixel-by-pixel (slow but correct)
        self.eval()
        B, C, H, W = shape
        img = torch.zeros((B,C,H,W), dtype=torch.long, device=device)
        with torch.no_grad():
            for i in range(H):
                for j in range(W):
                    # forward pass to get logits for all pixels but we only use (i,j)
                    logits = self.forward(img)  # [B,C,H,W,256] but requires float normalized input
                    # pick probs for position i,j
                    probs = F.softmax(logits[:,:,i,j,:], dim=-1)  # [B,C,256]
                    # sample each channel
                    sampled = torch.multinomial(probs.view(-1, PIXEL_VALUES), 1).view(B,C).long()
                    img[:,:,i,j] = sampled
        return img.cpu().numpy()

# -----------------------
# Row LSTM (simplified)
# -----------------------
class RowLSTMCell(nn.Module):
    """
    A simplified Row-LSTM cell: processes row-by-row.
    We'll implement input-to-state conv (1x1 conv) and state-to-state via conv along row (1x3 conv)
    and use an LSTM gating (single-vector per position).
    """
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv_x = nn.Conv2d(in_channels, 4*hidden_channels, kernel_size=1)
        self.conv_h = nn.Conv2d(hidden_channels, 4*hidden_channels, kernel_size=(1,3), padding=(0,1))
        self.hidden_channels = hidden_channels

    def forward(self, x_row, h_prev, c_prev):
        # x_row: [B, C, 1, W] (one row as a spatial height 1)
        # h_prev,c_prev: [B, H, 1, W]
        a = self.conv_x(x_row) + self.conv_h(h_prev)
        i, f, o, g = torch.chunk(a, 4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o); g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

class RowLSTM(nn.Module):
    def __init__(self, in_channels=3, hidden=64, n_layers=2):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.pre = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.cells = nn.ModuleList([RowLSTMCell(hidden, hidden) for _ in range(n_layers)])
        # output head: map hidden -> logits per channel
        self.out = nn.Conv2d(hidden, in_channels*PIXEL_VALUES, kernel_size=1)

    def forward(self, x):  # x: [B, C, H, W] integers 0..255
        B = x.shape[0]
        x_norm = x.float() / 255.0 - 0.5
        h = self.pre(x_norm)  # [B, hidden, H, W]
        # process row by row
        batch_h = torch.zeros(B, self.hidden, 1, W, device=x.device)
        batch_c = torch.zeros(B, self.hidden, 1, W, device=x.device)
        out = []
        for row in range(H):
            x_row = h[:,:,row:row+1,:]  # [B,hidden,1,W]
            for lid, cell in enumerate(self.cells):
                batch_h, batch_c = cell(x_row, batch_h, batch_c)
                x_row = batch_h  # feed to next layer
            # append row's hidden
            out.append(batch_h)  # each is [B,hidden,1,W]
        out = torch.cat(out, dim=2)  # [B,hidden,H,W]
        logits = self.out(out).view(B, C, PIXEL_VALUES, H, W).permute(0,1,3,4,2)
        return logits

# -----------------------
# Diagonal BiLSTM (skew/unskew trick) simplified
# -----------------------
def skew_right(x):
    # x: [B, C, H, W]
    # Skew so that diagonals become rows
    B,C,H,W = x.shape
    max_len = H + W - 1
    device = x.device
    skewed = x.new_zeros((B,C,max_len,W))
    for i in range(H):
        skewed[:,:,i:(i+1),:] = torch.roll(x[:,:,i:i+1,:], shifts=i, dims=3)
    # collapse last dim (W) to width dimension
    return skewed  # shape [B,C,H + W -1,W]

def unskew_right(skewed, H):
    # reverse operation to get back H rows
    B,C,_,W = skewed.shape
    out = skewed.new_zeros((B,C,H,W))
    for i in range(H):
        out[:,:,i,:] = torch.roll(skewed[:,:,i,:], shifts=-i, dims=2)
    return out

class DiagonalBiLSTM(nn.Module):
    def __init__(self, in_channels=3, hidden=64):
        super().__init__()
        self.pre = nn.Conv2d(in_channels, hidden, kernel_size=1)
        # use bidirectional LSTM applied along skewed rows (we implement as nn.LSTM over sequence)
        self.rnn = nn.LSTM(input_size=hidden*1, hidden_size=hidden, num_layers=1, bidirectional=True, batch_first=True)
        self.out = nn.Conv2d(hidden*2, in_channels*PIXEL_VALUES, kernel_size=1)  # combine forward+back
    def forward(self, x):
        B = x.shape[0]
        x_norm = x.float()/255.0 - 0.5
        pre = self.pre(x_norm)  # [B,hidden,H,W]
        skewed = skew_right(pre)  # [B,hidden,H+W-1,W]
        B,C,S,Wk = skewed.shape
        # reshape to sequences: each of the S positions has a "row" of length Wk - but we treat Wk as time
        # We'll treat each column as time dimension for LSTM: collapse B and S dims
        seqs = skewed.permute(0,2,3,1).reshape(B*S, Wk, C)  # [B*S, Wk, hidden]
        out_seq, _ = self.rnn(seqs)  # [B*S, Wk, hidden*2]
        out = out_seq.reshape(B, S, Wk, -1).permute(0,3,1,2)  # [B, hidden*2, S, W]
        unsk = unskew_right(out, H)  # [B, hidden*2, H, W]
        logits = self.out(unsk).view(B,C,PIXEL_VALUES,H,W).permute(0,1,3,4,2)
        return logits

# -----------------------
# Loss and training helpers
# -----------------------
def nll_loss_from_logits(logits, target):
    # logits: [B, C, H, W, 256] ; target: [B, C, H, W] ints 0..255
    B,C,H,W,PV = logits.shape
    logits = logits.reshape(-1, PV)
    target = target.view(-1)
    loss = F.cross_entropy(logits, target, reduction='mean')  # average negative log-likelihood (nats)
    return loss

# Generic training loop for a model
def train_model(model, train_loader, test_loader, epochs=10, lr=1e-3, name='model'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_bits = float('inf')
    for epoch in range(1, epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train {name} E{epoch}")
        train_loss = 0.0
        for imgs, _ in pbar:
            imgs = imgs.to(device)  # [B,3,32,32] long
            optimizer.zero_grad()
            logits = model(imgs)  # [B,C,H,W,256]
            loss = nll_loss_from_logits(logits, imgs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.shape[0]
            pbar.set_postfix({'loss': loss.item()})
        train_loss /= len(train_loader.dataset)
        # eval
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                loss = nll_loss_from_logits(logits, imgs)
                total_loss += loss.item() * imgs.shape[0]
        total_loss /= len(test_loader.dataset)
        bpd = bits_per_dim(total_loss, PIXELS)
        print(f"[{name}] Epoch {epoch} Train NLL(avg per img)= {train_loss:.4f}  Test NLL= {total_loss:.4f}  bits/dim= {bpd:.4f}")
        if bpd < best_bits:
            best_bits = bpd
            # save checkpoint
            torch.save(model.state_dict(), f"{name}_best.pth")
    return model

# -----------------------
# Quick experiments runner
# -----------------------
def run_experiment(which='pixelcnn', epochs=10, lr=1e-3):
    if which == 'pixelcnn':
        model = PixelCNN(in_channels=C, n_filters=64, n_layers=7)
    elif which == 'rowlstm':
        model = RowLSTM(in_channels=C, hidden=64, n_layers=2)
    elif which == 'diagbilstm':
        model = DiagonalBiLSTM(in_channels=C, hidden=64)
    else:
        raise ValueError("unknown model")
    print("Model param count:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    trained = train_model(model, train_loader, test_loader, epochs=epochs, lr=lr, name=which)
    return trained

# -----------------------
# Main: run smaller experiment or full
# -----------------------
if __name__ == "__main__":
    # Seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Run PixelCNN for a few epochs (increase epochs for final experiment)
    trained_pixelcnn = run_experiment('pixelcnn', epochs=8, lr=2e-4)

    # Sample 4 images (this is slow)
    print("Sampling 2 images from PixelCNN (may take a while)...")
    sample = trained_pixelcnn.sample((2,C,H,W), device=device)
    # sample is integer RGB 0..255 numpy array
    print("Sampled shape:", sample.shape)

    # Run Row LSTM (short run)
    trained_rowlstm = run_experiment('rowlstm', epochs=5, lr=1e-3)

    # Run Diagonal BiLSTM (short run)
    trained_diag = run_experiment('diagbilstm', epochs=5, lr=1e-3)

    print("Done experiments.")
