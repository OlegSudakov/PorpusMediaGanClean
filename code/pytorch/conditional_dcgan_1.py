import torch
import torch.nn as nn
import torch.nn.parallel

class DCGAN3D_D(nn.Container):
    def __init__(self, isize, nz, nc, ndf, ngpu, ncl, n_extra_layers=0):
        super(DCGAN3D_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        main = nn.Sequential(
            # input is nc x isize x isize
            nn.Conv3d(nc + ncl, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        i, csize, cndf = 3, isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(str(i),
                            nn.Conv3d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module(str(i+1),
                            nn.BatchNorm3d(cndf))
            main.add_module(str(i+2),
                            nn.LeakyReLU(0.2, inplace=True))
            i += 3

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module(str(i),
                            nn.Conv3d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module(str(i+1),
                            nn.BatchNorm3d(out_feat))
            main.add_module(str(i+2),
                            nn.LeakyReLU(0.2, inplace=True))
            i+=3
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4 x 4
        main.add_module(str(i),
                        nn.Conv3d(cndf, 1, 4, 1, 0, bias=False))
        main.add_module(str(i+1), nn.Sigmoid())
        
        self.main = main


    def forward(self, input, label):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        input = torch.cat([input, label], 1)
        output = self.main(input)
        return output.view(-1, 1)

class DCGAN3D_G(nn.Container):
    def __init__(self, isize, nz, nc, ngf, ngpu, ncl, n_extra_layers=0):
        super(DCGAN3D_G, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose3d(nz + ncl, cngf, 4, 1, 0, bias=False),
            nn.BatchNorm3d(cngf),
            nn.ReLU(True),
        )

        i, csize, cndf = 3, 4, cngf
        while csize < isize//2:
            main.add_module(str(i),
                nn.ConvTranspose3d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module(str(i+1),
                            nn.BatchNorm3d(cngf//2))
            main.add_module(str(i+2),
                            nn.ReLU(True))
            i += 3
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(str(i),
                            nn.Conv3d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module(str(i+1),
                            nn.BatchNorm3d(cngf))
            main.add_module(str(i+2),
                            nn.ReLU(True))
            i += 3

        main.add_module(str(i),
                        nn.ConvTranspose3d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module(str(i+1), nn.Tanh())
        self.main = main

    def forward(self, input, label):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        input = torch.cat([input, label], 1)
        return self.main(input)