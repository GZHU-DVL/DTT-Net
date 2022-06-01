"""DTT-Net model, rain rempval and generation in two cycles.
In this code, subsript r is for for "real", s is for "synthetic",
which means, for example, O_r is the rainy image in real dataset, B_s is the background in synthetic dataset.
For the sub-networks, we denote netG1 for synthetic rain removal, netG2 for real rain generation, 
netG3 for real rain removal, netG4 for synthetic rain generation.

Our code is inspired by https://github.com/junyanz/CycleGAN. Thanks for their contribution.
"""
import torch
from .base_model import BaseModel
from . import networks
from util.image_pool import ImagePool



class RainCycleModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='rain')  # You can rewrite default values for this model.
        if is_train:
            parser.add_argument('--lambda_MSE', type=float, default=40.0, help='weight for the mse loss')  # You can define new arguments for this model.
            parser.add_argument('--lambda_GAN', type=float, default=4.0, help='weight for the gan loss')
            parser.add_argument('--lambda_Cycle', type=float, default=40.0, help='weight for the cycle loss')

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['MSE','GAN','Cycle','G_total','D_Or', 'D_Os','D_B']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        if self.isTrain:
            self.visual_names = ['Os','Or','Bs',
                                 'pred_Bs','pred_Or','pred_pred_Bs','pred_pred_Os',
                                 'pred_Br','pred_Os','pred_pred_Br','pred_pred_Or',
                                 'pred_Rs', 'pred_Rsr', 'pred_pred_Rr', 'pred_pred_Rrs',
                                 'pred_Rr', 'pred_Rrs', 'pred_pred_Rs', 'pred_pred_Rsr']
        else:
            self.visual_names = ['Or', 'Br', 'pred_Br','Os', 'Bs', 'pred_Bs', 'Rs',
                                 'pred_Or', 'pred_pred_Bs', 'pred_pred_Os',
                                 'pred_Os', 'pred_pred_Br', 'pred_pred_Or',
                                 'pred_Rs', 'pred_Rsr', 'pred_pred_Rr', 'pred_pred_Rrs',
                                 'pred_Rr', 'pred_Rrs', 'pred_pred_Rs', 'pred_pred_Rsr']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        self.model_names = ['G1', 'G2', 'G3', 'G4', 'D_B', 'D_Os', 'D_Or']


        # netG1 for synthetic rain removal
        # netG2 for real rain generation
        # netG3 for real rain removal
        # netG4 for synthetic rain generation
        self.netG2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'resnet_9blocks', gpu_ids=self.gpu_ids)
        self.netG4 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'resnet_9blocks', gpu_ids=self.gpu_ids)
        self.netG1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'restormer_dim10', gpu_ids=self.gpu_ids)
        self.netG3 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'restormer_dim10', gpu_ids=self.gpu_ids)


        self.netD_Or = networks.define_D(opt.output_nc, opt.ndf, opt.netD,opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_Os = networks.define_D(opt.output_nc, opt.ndf, opt.netD,opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD,opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.feature_dim = networks.DnCNN(channels=3).out_feature_dim
        self.netD_feature = networks.define_D(self.feature_dim, opt.ndf, opt.netD,
                                              opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # only defined during training time

            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
            self.criterion_MSE = torch.nn.MSELoss()
            self.criterion_GAN = self.criterion_GAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()


            self.Pool_fake_Os = ImagePool(opt.pool_size)
            self.Pool_fake_Or = ImagePool(opt.pool_size)
            self.Pool_pred_Bs = ImagePool(opt.pool_size)
            self.Pool_pred_pred_Bs = ImagePool(opt.pool_size)
            self.Pool_pred_B_r = ImagePool(opt.pool_size)
            self.Pool_pred_pred_B_r = ImagePool(opt.pool_size)
            self.Pool_real_B = ImagePool(opt.pool_size)

            # define and initialize optimizers. You can define one optimizer for each network.
            parameters_list = [dict(params=self.netG1.parameters(), lr=opt.lr)]
            parameters_list.append(dict(params=self.netG3.parameters(), lr=opt.lr))
            parameters_list.append(dict(params=self.netG2.parameters(), lr=opt.lr))
            parameters_list.append(dict(params=self.netG4.parameters(), lr=opt.lr))
            self.optimizer_G = torch.optim.Adam(parameters_list, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_Os = torch.optim.Adam(self.netD_Os.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_Or = torch.optim.Adam(self.netD_Or.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = [self.optimizer_G, self.optimizer_D_Os, self.optimizer_D_Or, self.optimizer_B]
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.Os = input['O_s'].to(self.device)
        self.Bs = input['B_s'].to(self.device)
        self.Or = input['O_r'].to(self.device)
        if not self.isTrain:
            self.Br = input['B_r'].to(self.device)
        self.image_paths = input['path']  # for test


    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        if not self.isTrain:
            self.Rr = self.Or - self.Br
        self.Rs = self.Os - self.Bs
        # syn - real
        self.pred_Bs = self.netG1(self.Os)
        self.pred_Rs = self.Os - self.pred_Bs
        self.pred_Rsr = self.netG2(self.pred_Rs)
        self.pred_Or = self.pred_Bs + self.pred_Rsr
        # syn - real - syn
        self.pred_pred_Bs = self.netG3(self.pred_Or)
        self.pred_pred_Rr = self.pred_Or - self.pred_pred_Bs
        self.pred_pred_Rrs =  self.netG4(self.pred_pred_Rr)
        self.pred_pred_Os = self.pred_pred_Bs + self.pred_pred_Rrs
        # real - syn
        self.pred_Br = self.netG3(self.Or)
        self.pred_Rr = self.Or - self.pred_Br
        self.pred_Rrs = self.netG4(self.pred_Rr)
        self.pred_Os = self.pred_Br + self.pred_Rrs
        # real - syn - real
        self.pred_pred_Br = self.netG1(self.pred_Os)
        self.pred_pred_Rs = self.pred_Os - self.pred_pred_Br
        self.pred_pred_Rsr = self.netG2(self.pred_pred_Rs)
        self.pred_pred_Or = self.pred_pred_Br + self.pred_pred_Rsr

    def backward_G(self):
        lambda_MSE = self.opt.lambda_MSE
        lambda_GAN = self.opt.lambda_GAN
        lambda_Cycle = self.opt.lambda_Cycle
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results

        # # Identity Loss
        # if lambda_Idt >0:
        #     self.Idt_Bs = self.netG1(self.Bs)


        # Cycle Loss
        self.Cycle_Os = self.criterionCycle(self.Os, self.pred_pred_Os)
        self.Cycle_Or = self.criterionCycle(self.Or, self.pred_pred_Or)
        self.Cycle_Bs = self.criterionCycle(self.pred_Bs, self.pred_pred_Bs)
        self.Cycle_Br = self.criterionCycle(self.pred_Br, self.pred_pred_Br)
        self.loss_Cycle = self.Cycle_Os + self.Cycle_Or + self.Cycle_Bs + self.Cycle_Br

        # GAN Loss
        self.GAN_Or = self.criterion_GAN(self.netD_Or(self.pred_Or), True)
        self.GAN_Os = self.criterion_GAN(self.netD_Os(self.pred_Os), True)
        self.GAN_pred_Bs = self.criterion_GAN(self.netD_B(self.pred_Bs), True)
        self.GAN_pred_pred_Bs = self.criterion_GAN(self.netD_B(self.pred_pred_Bs), True)
        self.GAN_pred_Br = self.criterion_GAN(self.netD_B(self.pred_Br), True)
        self.GAN_pred_pred_Br = self.criterion_GAN(self.netD_B(self.pred_pred_Br), True)
        self.loss_GAN = self.GAN_Or + self.GAN_Os + self.GAN_pred_Bs + self.GAN_pred_pred_Bs + self.GAN_pred_Br + self.GAN_pred_pred_Br

        # MSE Loss
        self.MSE_pred_Bs = self.criterion_MSE(self.pred_Bs, self.Bs)
        self.MSE_pred_pred_Bs = self.criterion_MSE(self.pred_pred_Bs, self.Bs)
        self.loss_MSE = self.MSE_pred_Bs + self.MSE_pred_pred_Bs

        self.loss_G_total = lambda_MSE * self.loss_MSE + lambda_GAN * self.loss_GAN + lambda_Cycle * self.loss_Cycle
        self.loss_G_total.backward()

    def cal_GAN_loss_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.criterion_GAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def backward_D_Ot(self):
        fake_Or = self.Pool_fake_Or.query(self.pred_Or)
        self.loss_D_Or = self.cal_GAN_loss_D_basic(self.netD_Or, self.Or, fake_Or)
        self.loss_D_Or.backward()


    def backward_D_Os(self):
        fake_Os = self.Pool_fake_Os.query(self.pred_Os)
        self.loss_D_Os = self.cal_GAN_loss_D_basic(self.netD_Os, self.Os, fake_Os)
        self.loss_D_Os.backward()

    def backward_D_B(self):
        pred_Bs = self.Pool_pred_Bs.query(self.pred_Bs)
        pred_pred_Bs = self.Pool_pred_pred_Bs.query(self.pred_pred_Bs)
        pred_Br = self.Pool_pred_B_r.query(self.pred_Br)
        pred_pred_Br =self.Pool_pred_pred_B_r.query(self.pred_pred_Br)
        real_images = self.Pool_real_B.query(self.Bs)
        real_images = self.Pool_real_B.query(return_num=4)
        self.loss_D_B = self.cal_GAN_loss_D_basic(self.netD_B, real_images[0], pred_Bs) + self.cal_GAN_loss_D_basic(self.netD_B, real_images[1], pred_pred_Bs) + self.cal_GAN_loss_D_basic(self.netD_B, real_images[2],pred_Br) + self.cal_GAN_loss_D_basic(self.netD_B, real_images[3], pred_pred_Br)
        self.loss_D_B.backward()


    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results

        self.set_requires_grad([self.netD_B, self.netD_Or, self.netD_Os], False)
        self.optimizer_G.zero_grad()   # clear network G's existing gradients
        self.backward_G()              # calculate gradients for network G
        self.optimizer_G.step()        # update gradients for network G

        self.set_requires_grad([self.netD_B, self.netD_Or, self.netD_Os], True)
        self.optimizer_B.zero_grad()
        self.backward_D_B()
        self.optimizer_B.step()

        self.optimizer_D_Or.zero_grad()
        self.backward_D_Ot()
        self.optimizer_D_Or.step()

        self.optimizer_D_Os.zero_grad()
        self.backward_D_Os()
        self.optimizer_D_Os.step()



