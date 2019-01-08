import os
import numpy as np
from PIL import Image

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

if not os.path.exists('./reconstruction'):
    os.mkdir('./reconstruction')

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epochs = 10
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), 
            nn.Linear(64, 12), 
            nn.ReLU(True), 
            nn.Linear(12, 2))
        
        self.decoder = nn.Sequential(
            nn.Linear(2, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DecoderPart(nn.Module):
    def __init__(self):
        super(DecoderPart, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), 
            nn.Linear(64, 12), 
            nn.ReLU(True), 
            nn.Linear(12, 2))
        
        self.decoder = nn.Sequential(
            nn.Linear(2, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.decoder(x)
        return x
      
# Train autoencoder
model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

print('Training autoencoder...')
for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()

        output = model(img)
        loss = criterion(output, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.item()))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './reconstruction/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './autoencoder.pth')

# Initialize decoder
print('Initializing decoder...')
model_decoder = DecoderPart().cuda()
model_decoder.load_state_dict(torch.load('./autoencoder.pth'))

# Generate images by tuning two dimensions
print('Generating...')
num_x = 20
num_y = 20
generated_images = np.zeros((28*num_x, 28*num_y))

upper_bound = 16
lower_bound = -16

for idx_i, i in enumerate(np.linspace(lower_bound,upper_bound,num_x)):
  for idx_j, j in enumerate(np.linspace(lower_bound,upper_bound,num_y)):
    out_gen = model_decoder(Variable(torch.tensor([[i, j]])).cuda())

    im_gen = out_gen.detach().cpu().numpy().reshape(28,28)
    generated_images[idx_i*28:(idx_i+1)*28, idx_j*28:(idx_j+1)*28] = im_gen
    
# normalize and save  
generated_images = 255 * (generated_images-np.min(generated_images)) / (np.max(generated_images) - np.min(generated_images))
im = Image.fromarray(generated_images)
im = im.convert("L")
im.save("generated_images.png")

# import matplotlib.pyplot as plt
# im = Image.open('generated_images.png')
# plt.imshow(im)
print("Saved to <generated_images.png>")
print("Finished")