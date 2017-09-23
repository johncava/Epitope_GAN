import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from process.py import *

prot_set = ['A','R','N','D','C','E','Q','G','H','I','L','K','M','F',
            'P','S','T','W','Y','V']

def create_epitope():
    epitope = ''
    for aa in xrange(20):
        epitope += random.choice(prot_set)
    print epitope

class Generator(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
                super(Generator, self).__init__()
                self.weight1 = nn.Linear(input_size, hidden_size)
                self.weight2 = nn.Linear(hidden_size, hidden_size)
                self.weight3 = nn.Linear(hidden_size, output_size)

        def forward(self, input):
                layer1 = F.relu(self.weight1(input))
                layer2 = F.relu(self.weight2(layer1))
                output_layer = F.sigmoid(self.weight3(layer2))

class Discriminator(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
                super(Discriminator, self).__init__()
                self.weight1 = nn.Linear(input_size, hidden_size)
                self.weight2 = nn.Linear(hidden_size, hidden_size)
                self.weight3 = nn.Linear(hidden_size, output_size)

        def forward(self, input):
                layer1 = F.relu(self.weight1(input))
                layer2 = F.relu(self.weight2(layer1))
                output_layer = F.sigmoid(self.weight3(layer2))

iterations = 1000
debug = 100
discriminator_steps = 100
generator_steps = 100
discriminator = Discriminator(input_size=20, hidden_size=50, output_size=1)
generator = Generator(input_size=20, hidden_size=50, output_size=20)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=g_learning_rate, betas=optim_betas)
generator_optimizer = optim.Adam(generator.parameters(), lr=g_learning_rate, betas=optim_betas)

for iteration in xrange(iterations):
    for d_steps in xrange(discriminator_steps):
        discriminator.zero_grad()

        # Sample Real Data
        real_data = sample_data()

        # Train Discriminator on Real Data
        discriminator_decision = discriminator(Variable(real_data))
        discriminator_error = nn.BCELoss(discriminator_decision, Variable(torch.ones(1)))
        discriminator_error.backward()

        # Create Fake Data
        fake_data = create_fake_data()
        fake_data = torch.from_numpy(fake_data)
        fake_data = fake_data.float()

        # Train Discriminator on Fake Data
        generator_data = generator(Variable(fake_data)).detach()
        discriminator_decision = discriminator(generator_data.t())
        discriminator_error = nn.BCELoss(discriminator_decision, Variable(torch.zeros(1)))
        discriminator_error.backward()
        discriminator_optimizer.step()

    for g_steps in xrange(generator_steps):
        
        generator.zero_grad()

        # Create Fake Data
        fake_data = create_fake_data()
        fake_data = torch.from_numpy(fake_data)
        fake_data = fake_data.float()

        #Train Generator from Discriminator
        generator_data = generator(Variable(fake_data)).detach()
        discriminator_decision = discriminator(generator_data.t())
        discriminator_error = nn.BCELoss(discriminator_decision, Variable(torch.ones(1)))
        discriminator_error.backward()
        discriminator_optimizer.step()

