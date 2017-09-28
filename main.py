import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from process import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')

prot_set = ['A','R','N','D','C','E','Q','G','H','I','L','K','M','F',
            'P','S','T','W','Y','V']

def create_epitope():
    epitope = ''
    for aa in xrange(20):
        epitope += random.choice(prot_set)
    print epitope
'''
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
'''
Generator = torch.nn.Sequential(
	torch.nn.Linear(20,50),
	torch.nn.LeakyReLU(0.2),
	torch.nn.Dropout(p=0.2),
	torch.nn.Linear(50,50),
	torch.nn.LeakyReLU(0.2),
	torch.nn.Dropout(p=0.2),
	torch.nn.Linear(50,20),
	torch.nn.Dropout(p=0.2),
	torch.nn.Sigmoid()
)
Discriminator = torch.nn.Sequential(
	torch.nn.Linear(20, 50),
	torch.nn.LeakyReLU(0.2),
	torch.nn.Dropout(p=0.2),
	torch.nn.Linear(50, 150),
	torch.nn.LeakyReLU(0.2),
	torch.nn.Dropout(p=0.2),
	torch.nn.Linear(150,50),
	torch.nn.LeakyReLU(0.2),
	torch.nn.Dropout(p=0.1),
	torch.nn.Linear(50,50),
	torch.nn.LeakyReLU(0.1),
	torch.nn.Dropout(p=0.1),
	torch.nn.Linear(50,1),
	torch.nn.Dropout(p=0.1),
	torch.nn.Sigmoid()
)

iterations = 100
debug = 100
discriminator_steps = 100
generator_steps = 100
optim_betas = (0.9, 0.999)
d_learning_rate = 2e-4
g_learning_rate = 2e-4
#discriminator = Discriminator(input_size=20, hidden_size=50, output_size=1)
#generator = Generator(input_size=20, hidden_size=50, output_size=20)
discriminator_optimizer = optim.Adam(Discriminator.parameters(), lr=g_learning_rate, betas=optim_betas)
generator_optimizer = optim.Adam(Generator.parameters(), lr=g_learning_rate, betas=optim_betas)

loss_fn = torch.nn.BCELoss()

pos_data = create_pos_data()
neg_data = create_neg_data()

iteration_list = []
error_list = []

for iteration in xrange(iterations):
    for d_steps in xrange(discriminator_steps):
        Discriminator.zero_grad()

        # Sample Real Data
        real_data = sample_data(pos_data)
	real_data = torch.from_numpy(real_data)
	real_data = real_data.float()
        
	# Train Discriminator on Real Positive Data
        discriminator_decision = Discriminator(Variable(real_data))
        discriminator_error = loss_fn(discriminator_decision, Variable(torch.ones(1)))
	#print discriminator_error[0]
	discriminator_error.backward()

        # Create Fake Data
        fake_data = create_fake_data(20)
        fake_data = torch.from_numpy(fake_data)
        fake_data = fake_data.float()

        # Train Discriminator on Fake Data
        generator_data = Generator(Variable(torch.normal(means=torch.zeros(20))).view(1,20)).detach()
        discriminator_decision = Discriminator(generator_data)
        discriminator_error = loss_fn(discriminator_decision, Variable(torch.zeros(1)))
        #print discriminator_error[0]
	discriminator_error.backward()
        discriminator_optimizer.step()

    for g_steps in xrange(generator_steps):
        
        Generator.zero_grad()

        # Create Fake Data
        fake_data = create_fake_data(20)
        fake_data = torch.from_numpy(fake_data)
        fake_data = fake_data.float()

        #Train Generator from Discriminator
        generator_data = Generator(Variable(torch.normal(means=torch.zeros(20))).view(1,20)).detach()
        discriminator_decision = Discriminator(generator_data)
        discriminator_error = loss_fn(discriminator_decision, Variable(torch.ones(1)))
        #error_list.append(discriminator_error.data[0])
	#iteration_list.append(iteration)
	discriminator_error.backward()
        discriminator_optimizer.step()

    error_list.append(discriminator_error.data[0])
    iteration_list.append(iteration)

for iteration in xrange(10):
        fake_data = create_fake_data(20)
        fake_data = torch.from_numpy(fake_data)
        fake_data = fake_data.float()
        synthetic_data = Generator(Variable(fake_data))
        #synthetic_data = synthetic_data.data.numpy()
        print synthetic_data

f = plt.figure()
plt.plot(iteration_list, error_list)
plt.title("GAN Discriminator Error from Generator")
plt.show()
f.savefig("error.pdf")
