# streamlit_app.py
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Hyperparams
latent_dim = 100
device = torch.device('cpu')

# Generator model (same as training)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 10, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_onehot = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)
        x = torch.cat([z, label_onehot], dim=1)
        out = self.net(x)
        out = out.view(-1, 1, 28, 28)
        return out

# Load model
generator = Generator().to(device)
generator.load_state_dict(torch.load('generator_mnist.pth', map_location=device))
generator.eval()

# Streamlit UI
st.title("MNIST Digit Generator")
digit = st.selectbox("Select digit to generate (0-9):", list(range(10)))

if st.button("Generate Images"):
    z = torch.randn(5, latent_dim).to(device)
    labels = torch.full((5,), digit, dtype=torch.long).to(device)
    gen_imgs = generator(z, labels)
    gen_imgs = gen_imgs.squeeze().detach().cpu().numpy()

    # Display 5 images
    cols = st.columns(5)
    for i in range(5):
        img = ((gen_imgs[i] + 1) / 2.0 * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
        cols[i].image(img_pil, caption=f'Digit: {digit}')
