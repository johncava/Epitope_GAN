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
        def __init__(self):
                super(Generator, self).__init__()
                self.weight1 = nn.Linear(input_size, hidden_size)
                self.weight2 = nn.Linear(hidden_size, hidden_size)
                self.weight3 = nn.Linear(hidden_size, output_size)
                self.conv1 = nn.Conv1d(1,10,3)
                self.conv2 = nn.Conv1d(10,100,3)
                self.conv

        def forward(self, input):
                layer1 = F.relu(self.weight1(input))
                layer2 = F.relu(self.weight2(layer1))
                output_layer = F.sigmoid(self.weight3(layer2))
'''
class Discrim(nn.Module):
        def __init__(self):
                super(Discrim, self).__init__()
                self.d1 = torch.nn.Conv1d(1,40,3)
                self.d2 = torch.nn.Conv1d(40,30,3)
                self.d3 = torch.nn.Conv1d(30,20,3)
                self.d4 = torch.nn.Conv1d(20,10,3)
                self.d5 = torch.nn.Conv1d(10,1,3)
                self.linear = torch.nn.Linear(10,1)
		self.relu = torch.nn.ReLU()
                self.sig = torch.nn.Sigmoid()
        
	def forward(self, input):
                conv = self.relu(self.d5(self.relu(self.d4(self.relu(self.d3(self.relu(self.d2(self.relu(self.d1(input))))))))))
                output_layer = self.sig(self.linear(conv.view(1,10)))
                return output_layer
'''
input_ = Variable(torch.randn(1,10)).view(1,1,10)
c1 = torch.nn.ConvTranspose1d(1,10,3)
c2 = torch.nn.ConvTranspose1d(10,20,3)
c3 = torch.nn.ConvTranspose1d(20,40,3)
c4 = torch.nn.ConvTranspose1d(40,40,3)
c5 = torch.nn.ConvTranspose1d(40,1,3)

d1 = torch.nn.Conv1d(1,40,3)
d2 = torch.nn.Conv1d(40,30,3)
d3 = torch.nn.Conv1d(30,20,3)
d4 = torch.nn.Conv1d(20,10,3)
d5 = torch.nn.Conv1d(10,1,3)

linear = torch.nn.Linear(10,1)
sig = torch.nn.Sigmoid()
'''
Generator = torch.nn.Sequential(
    torch.nn.ConvTranspose1d(1,10,3),
    torch.nn.ReLU(),
    torch.nn.ConvTranspose1d(10,20,3),
    torch.nn.ReLU(),    
    torch.nn.ConvTranspose1d(20,40,3),
    torch.nn.ReLU(),
    torch.nn.ConvTranspose1d(40,40,3),
    torch.nn.ReLU(),    
    torch.nn.ConvTranspose1d(40,1,3),
    torch.nn.Sigmoid()
)

'''
Discriminator = torch.nn.Sequential(
    torch.nn.Conv1d(1,40,3),
    torch.nn.Conv1d(40,30,3),
    torch.nn.Conv1d(30,20,3),
    torch.nn.Conv1d(20,10,3),
    torch.nn.Conv1d(10,1,3),
    torch.nn.BatchNorm1d(1),
    torch.nn.Linear(10,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,1),
    torch.nn.Sigmoid()
)
'''
Discriminator = Discrim()
iterations = 100
debug = 100
discriminator_steps = 50
generator_steps = 50
optim_betas = (0.9, 0.999)
d_learning_rate = 2e-4
g_learning_rate = 2e-4
discriminator_optimizer = optim.Adam(Discriminator.parameters(), lr=g_learning_rate, betas=optim_betas)
generator_optimizer = optim.Adam(Generator.parameters(), lr=g_learning_rate, betas=optim_betas)

loss_fn = torch.nn.BCELoss()

data_pos = create_pos_data()
data_neg = create_neg_data()

iteration_list = []
error_list = []

for iteration in xrange(iterations):
    for d_steps in xrange(discriminator_steps):
        Discriminator.zero_grad()

        # Sample Real Data
        real_data = sample_data(data_pos)
        real_data = torch.from_numpy(real_data)
        real_data = real_data.float()
        
	# Train Discriminator on Real Data
        discriminator_decision = Discriminator(Variable(real_data).view(1,1,20))
        discriminator_error = loss_fn(discriminator_decision, Variable(torch.ones(1)))
        #print discriminator_error[0]
        discriminator_error.backward()

        # Sample Real Data
        real_data = sample_data(data_neg)
        real_data = torch.from_numpy(real_data)
        real_data = real_data.float()

        # Train Discriminator on Real Data
        discriminator_decision = Discriminator(Variable(real_data).view(1,1,20))
        discriminator_error = loss_fn(discriminator_decision, Variable(torch.zeros(1)))
        #print discriminator_error[0]
        discriminator_error.backward()

        # Create Fake Data
        fake_data = create_fake_data(10)
        fake_data = torch.from_numpy(fake_data)
        fake_data = fake_data.float()

        # Train Discriminator on Fake Data
        generator_data = Generator(Variable(fake_data).view(1,1,10)).detach()
	discriminator_decision = Discriminator(generator_data)
        discriminator_error = loss_fn(discriminator_decision, Variable(torch.zeros(1)))
        #print discriminator_error[0]
	discriminator_error.backward()
        discriminator_optimizer.step()

    for g_steps in xrange(generator_steps):
        
        Generator.zero_grad()

        # Create Fake Data
        fake_data = create_fake_data(10)
        fake_data = torch.from_numpy(fake_data)
        fake_data = fake_data.float()

        #Train Generator from Discriminator
        generator_data = Generator(Variable(fake_data).view(1,1,10)).detach()
        discriminator_decision = Discriminator(generator_data)
        discriminator_error = loss_fn(discriminator_decision, Variable(torch.ones(1)))
	discriminator_error.backward()
        discriminator_optimizer.step()

    error_list.append(discriminator_error.data[0])
    iteration_list.append(iteration)

print "max: ", max(error_list)
print "min: ", min(error_list)

for iteration in xrange(10):
        fake_data = create_fake_data(10)
        fake_data = torch.from_numpy(fake_data)
        fake_data = fake_data.float()

	synthetic_data = Generator(Variable(fake_data).view(1,1,10))
	#synthetic_data = synthetic_data.view(1,20)
	#synthetic_data = synthetic_data.data.numpy()
	print synthetic_data
f = plt.figure()
plt.plot(iteration_list, error_list)
plt.title("DGCAN Discriminator Error from Generator")
plt.show()
f.savefig("dcgan_error.pdf")

