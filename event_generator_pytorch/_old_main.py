import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from epochsviz import Epochsviz
from hist import hist
from load_data import load_data

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

### CONSTANTS
### CONSTANTS
### CONSTANTS
from models import train_discriminator, train_generator, Discriminator, Generator

HIST_POINTS = 100
BATCH_SIZE = 128 # 256
PADDING = 400 # 2000
FEATURES = 15
INP_RAND_SIZE = 512

### CONSTANTS

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

data, max_length = load_data(device=device)

# PADDING = max_length



padded_data: torch.Tensor = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, ).split(split_size=PADDING, dim=1)[0]
#padded_data = torch.zeros(size=(len(padded_data), PADDING, FEATURES))


data_train = DataLoader(
    padded_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

num_batches = len(data_train)


def noise(size, device) -> torch.Tensor:
    # Generates a 1-d vector of gaussian sampled random values
    return torch.randn(size, INP_RAND_SIZE).to(device=device)


try:
    gen = torch.load('gen_model_')
except:
    gen = Generator(inp=INP_RAND_SIZE, out_length=PADDING, out_features=FEATURES, device=device)  # .to(device=device)

try:
    dis = torch.load('dis_model_')
except:
    dis = Discriminator(samples=PADDING, features=FEATURES, out=1, device=device)  # .to(device=device)

# optimizerd1 = optim.SGD(dis.parameters(), lr=0.001, momentum=0.9)
# optimizerd2 = optim.SGD(gen.parameters(), lr=0.001, momentum=0.9)

d_optimizer = optim.Adam(dis.parameters(), lr=0.005)
g_optimizer = optim.Adam(gen.parameters(), lr=0.005)


def mae_loss(input, target):
    return torch.sum((input - target) ** 2)


loss = nn.BCELoss()

### TRAINING

# logger = Logger(model_name='GAN', data_name='Pythia')

eviz = Epochsviz(title='figure', plot_width=1200, plot_height=600)


def train():
    num_epochs = 200

    for epoch in range(num_epochs):
        d_error = 0
        g_error = 0
        fake_data = torch.Tensor()
        real_data = torch.Tensor()
        for n_batch, batch in enumerate(data_train):

            print(str(n_batch))
            N = len(batch)

            # 1. Train Discriminator
            # real_data = Variable(images_to_vectors(real_batch))
            real_data: torch.Tensor = batch.to(device=device)

            # Generate fake data and detach
            # (so gradients are not calculated for generator)
            fake_data: torch.Tensor = gen(noise(N, device=device)).detach()

            # Train D
            d_error, d_pred_real, d_pred_fake = train_discriminator(
                dis=dis,
                optimizer=d_optimizer,
                loss=loss,
                real_data=real_data,
                fake_data=fake_data,
                device=device
            )

            # 2. Train Generator
            # Generate fake data
            fake_data = gen(noise(N, device=device))

            # Train G
            g_error = train_generator(
                dis=dis,
                optimizer=g_optimizer,
                loss=loss,
                fake_data=fake_data,
                device=device
            )

            # Log batch error
            # logger.log(d_error, g_error, epoch, n_batch, num_batches)

            # Display Progress every few batches

            if (n_batch) % 30 == 0:
                print('epoch: ' + str(epoch) + ', ' +
                      'n_batch: ' + str(n_batch) + ', ' +
                      'd_error: ' + str(d_error.item()) + ', ' +
                      'g_error: ' + str(g_error.item())
                      )

            plt.pause(0.001)

        eviz.send_data(epoch, d_error.item(), g_error.item())
        if epoch%10==0:
            torch.save(dis.state_dict(), './dis_model_' + str(epoch))
            torch.save(gen.state_dict(), './gen_model_' + str(epoch))

        '''
        print('REAL DATA 11')
        plt.imshow(real_data.cpu()[11, :, :], )
        plt.colorbar()
        plt.show()
        '''

        img_smpls_to_shw = 80

        print('REAL DATA')
        for i in range(0, real_data.shape[0], 100):
            plt.figure('Real img:' + str(i))
            plt.imshow(real_data[i, :, :].split(split_size=img_smpls_to_shw, dim=0)[0].cpu())
            plt.ion()
            plt.show()
            plt.figure('Real hist:' + str(i))
            plt.hist(real_data[i, :, 6].cpu().flatten(), 100)
            plt.show()
        plt.pause(0.001)

        # plt.figure(1)
        # plt.hist(real_data[:, :, 6].detach().cpu().numpy().flatten(), 100)


        print('FAKE DATA')
        for i in range(0, fake_data.shape[0], 100):
            plt.figure('Fake img:' + str(i))
            plt.imshow(fake_data[i, :, :].split(split_size=img_smpls_to_shw, dim=0)[0].detach().cpu())
            plt.ion()
            plt.show()
            plt.figure('Fake hist:' + str(i))
            plt.hist(fake_data[i, :, 6].detach().cpu().flatten(), 100)
            plt.show()

        plt.pause(0.001)

        # plt.figure(2)
        # plt.hist(fake_data[:, :, 6].detach().cpu().numpy().flatten(), 100)



# After the train function
torch.cuda.empty_cache()
eviz.start_thread(train_function=train)
