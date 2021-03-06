from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from dataset import HDF5Dataset
from hdf5_io import save_hdf5
import conditional_dcgan_1 as dcgan
import numpy as np
np.random.seed(43)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='3D')
parser.add_argument('--datapath', required=True, help='path to dataset')
parser.add_argument('--labelpath', required=True, help='path to labels')
parser.add_argument('--workdir', default="./")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--samples', type=int, default=128, help='number of example images to generate')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

opt = parser.parse_args()
print(opt)

#Change workdir to where you want the files output
work_dir = str(opt.workdir) + "/"

try:
    os.makedirs(opt.outf)
except OSError:
    pass
opt.manualSeed = 43 # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['3D']:
    dataset = np.load(str(opt.datapath))
    labels = np.load(str(opt.labelpath))
    
ncl = labels.shape[1]
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
ns = int(opt.samples)
nc = 1

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# custom iteration function for use with numpy array
def iterate(inds_range, batch_size):
    np.random.shuffle(inds_range)
    for i in range(inds_range.shape[0] // batch_size):
        yield inds_range[i*batch_size : (i+1)*batch_size]
    if (i+1)*batch_size < inds_range.shape[0]:
        yield inds_range[(i+1)*batch_size:]

netG = dcgan.DCGAN3D_G(opt.imageSize, nz, nc, ngf, ngpu, ncl)

netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = dcgan.DCGAN3D_D(opt.imageSize, nz, nc, ndf, ngpu, ncl)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

input, noise, fixed_noise = None, None, None
input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1, 1)
fixed_noise = torch.FloatTensor(ns, nz, 1, 1, 1).normal_(0, 1)

label = torch.FloatTensor(opt.batchSize)
real_label = 0.9
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

input = Variable(input)
label = Variable(label)
noise = Variable(noise)

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))

gen_iterations = 0
for epoch in range(opt.niter):
    
    for i, data in enumerate(iterate(dataset, int(opt.batchSize))):
        f = open(work_dir+"training_curve.csv", "a")
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        
        real_cpu = torch.from_numpy(data)
            
        batch_size = real_cpu.size(0)
        input.data.resize_(real_cpu.size()).copy_(real_cpu)
        label.data.resize_(batch_size).fill_(real_label)
        
        output = netD(input)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.data.resize_(batch_size, nz, 1, 1, 1)
        noise.data.normal_(0, 1)
        fake = netG(noise).detach()
        label.data.fill_(fake_label)
        output = netD(fake)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()
        
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        g_iter = 1
        while g_iter != 0:
            netG.zero_grad()
            label.data.fill_(1.0) # fake labels are real for generator cost
            noise.data.normal_(0, 1)
            fake = netG(noise)
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()
            g_iter -= 1
        
        gen_iterations += 1

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, dataset.shape[0] // int(opt.batchSize),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        f.write('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, dataset.shape[0] // int(opt.batchSize),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        f.write('\n')
        f.close()
        
        if gen_iterations % 500 == 0:
            fake = netG(fixed_noise).cpu().data.numpy()
            np.save(work_dir+'fake_samples_{0}.npy'.format(gen_iterations), fake)
	
    # do checkpointing
    if epoch % 5 == 0:
        torch.save(netG.state_dict(), work_dir+'netG_epoch_%d.pth' % (epoch))
        torch.save(netD.state_dict(), work_dir+'netD_epoch_%d.pth' % (epoch))
