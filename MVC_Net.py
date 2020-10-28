import os
import numpy as np
import torch
import torch.nn as nn

from datetime import datetime


def get_matrixS(n):
    """
    Manually define a constant matrix S.
    (Paper: Automated comprehensive Adolescent Idiopathic Scoliosis assessment using MVC-Net)
    :param n: n=1/2 * #landmarks  (refer to the paper for details)
    :return: matrix S (with size 2n-by-2n)
    """

    mat_nxn = np.zeros([n, n], dtype=int)
    for row_num in range(1, n + 1):
        i = row_num - 1
        if row_num == 1:
            mat_nxn[i][i + 1] = 1
            mat_nxn[i][i + 2] = 1
        elif row_num == 2:
            mat_nxn[i][i - 1] = 1
            mat_nxn[i][i + 2] = 1
        elif row_num == n - 1:
            mat_nxn[i][i + 1] = 1
            mat_nxn[i][i - 2] = 1
        elif row_num == n:
            mat_nxn[i][i - 1] = 1
            mat_nxn[i][i - 2] = 1
        elif row_num % 2 == 1:
            mat_nxn[i][i + 1] = 1
            mat_nxn[i][i + 2] = 1
            mat_nxn[i][i - 2] = 1
        elif row_num % 2 == 0:
            mat_nxn[i][i - 1] = 1
            mat_nxn[i][i + 2] = 1
            mat_nxn[i][i - 2] = 1
    mat_nxn = mat_nxn + np.eye(n, dtype=int)
    mat_2nx2n = np.repeat(np.repeat(mat_nxn, 2, 0), 2, 1)
    return torch.as_tensor(mat_2nx2n)


# the landmark loss described in paper: Automated comprehensive Adolescent Idiopathic Scoliosis assessment using MVC-Net
def loss_landmark(x, y, theta=5):
    """
    Define Joint Regression Loss (JRL) as the landmark loss.
    (Paper: Automated comprehensive Adolescent Idiopathic Scoliosis assessment using MVC-Net)
    :param x: predicted landmark coordinates (2 * #landmarks) of a mini-batch, dim(x)=[N, #coordinates]
    :param y: ground truth, dim(y)=[N, #coordinates]
    :param theta: default to 5, weight of regression loss
    :return: landmark loss
    """
    reg_loss = x.sub(y).cosh().log().mean(1)  # size: [N,]
    corr_loss = 1 - x.mul(y).mean(1).sub(x.mean(1).mul(y.mean(1))).true_divide(x.std(1).mul(y.std(1)))  # size: [N,]
    loss = corr_loss + theta * reg_loss
    mean_loss = loss.mean(0)  # mean loss of the batch, # size: [1,]
    return mean_loss


# the angle loss described in paper: Automated comprehensive Adolescent Idiopathic Scoliosis assessment using MVC-Net
def loss_angle(x, y, theta=5):
    """
    Define circular Joint Regression Loss (cJRL) as the angle loss.
    (Paper: Automated comprehensive Adolescent Idiopathic Scoliosis assessment using MVC-Net)
    :param x: predicted angles of a mini-batch, dim(x)=[N, #angles]
    :param y: ground truth, dim(y)=[N, #angles]
    :param theta: default to 0.5, weight of regression loss
    :return: angle loss
    """

    # cReg loss
    log_cosh_err = x.sub(y).cosh().log()
    x_bar_log_cosh_err = log_cosh_err.cos().mean(1)
    y_bar_log_cosh_err = log_cosh_err.sin().mean(1)
    cReg_loss = y_bar_log_cosh_err.true_divide(x_bar_log_cosh_err).atan()  # circular regression loss, # size: [N,]

    # cCorr_loss
    xy = x.mul(y)
    x_bar_xy = xy.cos().mean(1)
    y_bar_xy = xy.sin().mean(1)
    cmean_xy = y_bar_xy.true_divide(x_bar_xy).atan()

    x_bar_x = x.cos().mean(1)
    y_bar_x = x.sin().mean(1)
    cmean_x = y_bar_x.true_divide(x_bar_x).atan()

    x_bar_y = y.cos().mean(1)
    y_bar_y = y.sin().mean(1)
    cmean_y = y_bar_y.true_divide(x_bar_y).atan()

    std_x = x.std(1)
    std_y = y.std(1)
    stdX_x_stdY = std_x.mul(std_y)

    cmeanX_x_cmeanY = cmean_x.mul(cmean_y)
    c_rho = cmean_xy.sub(cmeanX_x_cmeanY).true_divide(stdX_x_stdY)
    cCorr_loss = 1 - c_rho  # circular correlation loss, # size: [N,]

    loss = cCorr_loss + theta * cReg_loss
    mean_loss = loss.mean(0)  # mean loss of the batch, # size: [1,]

    return mean_loss


# replace CMAE by absolute CMAE
def loss_angle_absCMAE(x, y, theta=5):
    """
    Define circular Joint Regression Loss (cJRL) as the angle loss.
    (Paper: Automated comprehensive Adolescent Idiopathic Scoliosis assessment using MVC-Net)
    This loss is based on the one defined in the paper above, yet we use the absolute CMAE instead of CMAE.
    :param x: predicted angles of a mini-batch, dim(x)=[N, #angles]
    :param y: ground truth, dim(y)=[N, #angles]
    :param theta: default to 0.5, weight of regression loss
    :return: angle loss
    """

    # cReg loss
    # TODO: check: should x be degree or radian
    log_cosh_err = x.sub(y).cosh().log()
    x_bar_log_cosh_err = log_cosh_err.cos().mean(1)
    y_bar_log_cosh_err = log_cosh_err.sin().mean(1)
    cReg_loss = y_bar_log_cosh_err.true_divide(x_bar_log_cosh_err).atan()  # circular regression loss, # size: [N,]
    abs_cReg_loss = cReg_loss.abs()

    # cCorr_loss
    xy = x.mul(y)
    x_bar_xy = xy.cos().mean(1)
    y_bar_xy = xy.sin().mean(1)
    cmean_xy = y_bar_xy.true_divide(x_bar_xy).atan()

    x_bar_x = x.cos().mean(1)
    y_bar_x = x.sin().mean(1)
    cmean_x = y_bar_x.true_divide(x_bar_x).atan()

    x_bar_y = y.cos().mean(1)
    y_bar_y = y.sin().mean(1)
    cmean_y = y_bar_y.true_divide(x_bar_y).atan()

    std_x = x.std(1)
    std_y = y.std(1)
    stdX_x_stdY = std_x.mul(std_y)

    cmeanX_x_cmeanY = cmean_x.mul(cmean_y)
    c_rho = cmean_xy.sub(cmeanX_x_cmeanY).true_divide(stdX_x_stdY)
    cCorr_loss = 1 - c_rho  # circular correlation loss, # size: [N,]

    loss = cCorr_loss + theta * abs_cReg_loss  # use absolute CMAE as circular regression loss
    mean_loss = loss.mean(0)  # mean loss of the batch, # size: [1,]

    return mean_loss


# angle loss adopted by Yongcheng Yao
def loss_angle_nonCircular(x, y, theta=0.2):
    reg_loss = x.sub(y).cosh().log().mean(1)  # size: [N,]
    corr_loss = 1 - x.mul(y).mean(1).sub(x.mean(1).mul(y.mean(1))).true_divide(x.std(1).mul(y.std(1)))  # size: [N,]
    loss = corr_loss + theta * reg_loss
    mean_loss = loss.mean(0)  # mean loss of the batch, # size: [1,]
    return mean_loss


# draw histogram of network gradient via tensorboard
def plot_MVCNet_grad(model, writer, iter, flag_CAE):
    # add histogram to tensorboard
    if flag_CAE:
        # CAE
        writer.add_histogram('Histogram for gradients of weights/CAE_cor/dense2', model.CAE_cor.dense2.weight.grad,
                             iter)
        writer.add_histogram('Histogram for gradients of weights/CAE_cor/dense1', model.CAE_cor.dense1.weight.grad,
                             iter)
        writer.add_histogram('Histogram for gradients of weights/CAE_sag/dense2', model.CAE_sag.dense2.weight.grad,
                             iter)
        writer.add_histogram('Histogram for gradients of weights/CAE_sag/dense1', model.CAE_sag.dense1.weight.grad,
                             iter)
    # xModule3
    writer.add_histogram('Histogram for gradients of weights/xModule3/convModule_cor/PReLU',
                         model.xModule3.convModule_cor.PRelu.weight.grad, iter)
    writer.add_histogram('Histogram for gradients of weights/xModule3/convModule_cor/bn',
                         model.xModule3.convModule_cor.bn.weight.grad, iter)
    writer.add_histogram('Histogram for gradients of weights/xModule3/convModule_cor/conv',
                         model.xModule3.convModule_cor.conv.weight.grad, iter)
    writer.add_histogram('Histogram for gradients of weights/xModule3/convModule_sag/PReLU',
                         model.xModule3.convModule_sag.PRelu.weight.grad, iter)
    writer.add_histogram('Histogram for gradients of weights/xModule3/convModule_sag/bn',
                         model.xModule3.convModule_sag.bn.weight.grad, iter)
    writer.add_histogram('Histogram for gradients of weights/xModule3/convModule_sag/conv',
                         model.xModule3.convModule_sag.conv.weight.grad, iter)
    # xModule2
    writer.add_histogram('Histogram for gradients of weights/xModule2/convModule_cor/PReLU',
                         model.xModule2.convModule_cor.PRelu.weight.grad, iter)
    writer.add_histogram('Histogram for gradients of weights/xModule2/convModule_cor/bn',
                         model.xModule2.convModule_cor.bn.weight.grad, iter)
    writer.add_histogram('Histogram for gradients of weights/xModule2/convModule_cor/conv',
                         model.xModule2.convModule_cor.conv.weight.grad, iter)
    writer.add_histogram('Histogram for gradients of weights/xModule2/convModule_sag/PReLU',
                         model.xModule2.convModule_sag.PRelu.weight.grad, iter)
    writer.add_histogram('Histogram for gradients of weights/xModule2/convModule_sag/bn',
                         model.xModule2.convModule_sag.bn.weight.grad, iter)
    writer.add_histogram('Histogram for gradients of weights/xModule2/convModule_sag/conv',
                         model.xModule2.convModule_sag.conv.weight.grad, iter)
    # xModule1
    writer.add_histogram('Histogram for gradients of weights/xModule1/convModule_cor/PReLU',
                         model.xModule1.convModule_cor.PRelu.weight.grad, iter)
    writer.add_histogram('Histogram for gradients of weights/xModule1/convModule_cor/bn',
                         model.xModule1.convModule_cor.bn.weight.grad, iter)
    writer.add_histogram('Histogram for gradients of weights/xModule1/convModule_cor/conv',
                         model.xModule1.convModule_cor.conv.weight.grad, iter)
    writer.add_histogram('Histogram for gradients of weights/xModule1/convModule_sag/PReLU',
                         model.xModule1.convModule_sag.PRelu.weight.grad, iter)
    writer.add_histogram('Histogram for gradients of weights/xModule1/convModule_sag/bn',
                         model.xModule1.convModule_sag.bn.weight.grad, iter)
    writer.add_histogram('Histogram for gradients of weights/xModule1/convModule_sag/conv',
                         model.xModule1.convModule_sag.conv.weight.grad, iter)
    # convM1
    writer.add_histogram('Histogram for gradients of weights/convM1_cor/PReLU', model.convM1_cor.PRelu.weight.grad,
                         iter)
    writer.add_histogram('Histogram for gradients of weights/convM1_cor/bn', model.convM1_cor.bn.weight.grad, iter)
    writer.add_histogram('Histogram for gradients of weights/convM1_cor/conv', model.convM1_cor.conv.weight.grad, iter)
    writer.add_histogram('Histogram for gradients of weights/convM1_sag/PReLU', model.convM1_sag.PRelu.weight.grad,
                         iter)
    writer.add_histogram('Histogram for gradients of weights/convM1_sag/bn', model.convM1_sag.bn.weight.grad, iter)
    writer.add_histogram('Histogram for gradients of weights/convM1_sag/conv', model.convM1_sag.conv.weight.grad, iter)


class wSummation(nn.Module):
    """
    The spatial weighted summation layer.
    """

    def __init__(self, input_dim):
        """
        :param input_dim: input dimension [C,H,W]
        """
        super(wSummation, self).__init__()
        # Note: must register Q as parameter to enable auto-grad
        self.Q = nn.Parameter(torch.rand(input_dim))
        # Note: the weighting matrix Q should be trainable
        self.Q.requires_grad = True

    def forward(self, x1, x2):
        """
        Calculate the weighted summation of 2 inputs.
        :param x1: input 1
        :param x2: input 2
        :return: the weighted summation
        """
        return x1 * self.Q + (1 - self.Q) * x2


class mul_matrixS(nn.Module):
    """
    A layer that multiplies the input with a constant matrix S.
    (Paper: Automated comprehensive Adolescent Idiopathic Scoliosis assessment using MVC-Net)
    """

    def __init__(self, n_lm):
        """
        :param n_lm: #landmarks
        """
        super(mul_matrixS, self).__init__()
        matrixS = get_matrixS(int(n_lm / 2)).float()
        # NOTE: we can register matrix S as parameter, though we will not update it
        self.matrixS = nn.Parameter(matrixS)
        self.matrixS.requires_grad = False  # sets the tensor to constant

    def forward(self, x):
        """
        Multiplies the input with a constant matrix S
        :param x: input
        :return: output of matrix multiplication
        """
        return torch.mm(x, self.matrixS)


class conv_bn_prelu_dropout(nn.Module):
    """
    Define a convolution Module: conv2d -> batchNorm -> PReLu -> Dropout
    (Paper: Automated comprehensive Adolescent Idiopathic Scoliosis assessment using MVC-Net)
    """

    def __init__(self, conv_in, conv_out, conv_ker, conv_stri, pad, bn_C, prelu_a, drop_rate):
        """
        Initialize the following building blocks:
            - Conv2d
            - BatchNorm2d
            - PReLU
            - Dropout2d
        :param conv_in: #input channels (for Conv2d)
        :param conv_out: #out channels (for Conv2d)
        :param conv_ker: kernel size (for Conv2d)
        :param conv_stri: kernel strike (for Conv2d)
        :param pad: padding (for Conv2d)
        :param bn_C: #channels (for BatchNorm2d)
        :param prelu_a: #alphas, default=1, prelu_a=#Channels enable using separate alpha for each channel
        :param drop_rate: dropout rate, default=0.5 (for Dropout2d)
        """
        super(conv_bn_prelu_dropout, self).__init__()
        self.conv = nn.Conv2d(conv_in, conv_out, conv_ker, conv_stri, pad)
        self.bn = nn.BatchNorm2d(bn_C)
        self.PRelu = nn.PReLU(prelu_a)
        self.Dropout = nn.Dropout2d(drop_rate)

    def forward(self, x):
        """
        The forwarding pass of convolution Module.
        :param x: input
        :return y: output of convolution Module
        """
        y = self.Dropout(self.PRelu(self.bn(self.conv(x))))
        return y


class xModule(nn.Module):
    """
    Define X-Module which has 2 inputs and 2 outputs.
    (Paper: Automated comprehensive Adolescent Idiopathic Scoliosis assessment using MVC-Net)
    """

    def __init__(self, input_dim, conv_in, conv_ker, conv_stri, pad, bn_C, prelu_a, drop_rate):
        """
        Initialize the X-Module, which contains the following building blocks:
            - wSum: a spatial weighted summation layer
            - convModule_sag: a convolution module for sagittal image
            - convModule_cor: a convolution module for coronal image
        :param input_dim: input dimension in the form of [C,H,W]
        :param conv_in: #input channels
        :param conv_ker: kernel size (for Conv2d)
        :param conv_stri: kernel strike (for Conv2d)
        :param pad: padding (for Conv2d)
        :param bn_C: #channels (for BatchNorm2d)
        :param prelu_a: #alphas, default=1 (for PReLu)
        :param drop_rate: dropout rate, default=0.5 (for Dropout2d)
        """
        super(xModule, self).__init__()

        # weighted spatial summation layer
        self.wSum = wSummation(input_dim)

        # two convolution-Module share the same setting, e.g. kernel number and size...
        self.convModule_sag = conv_bn_prelu_dropout(2 * conv_in, 2 * conv_in, conv_ker, conv_stri, pad, bn_C, prelu_a,
                                                    drop_rate)
        self.convModule_cor = conv_bn_prelu_dropout(2 * conv_in, 2 * conv_in, conv_ker, conv_stri, pad, bn_C, prelu_a,
                                                    drop_rate)

    def forward(self, x_sag, x_cor):
        """
        The forwarding pass of xModule.
        :param x_sag: sagittal input with size  (N,C,H,W)
        :param x_cor: coronal input with size  (N,C,H,W)
        :return:
            - y_sag: sagittal output with size (N,2C,H/2,W/2)
            - y_cor: coronal output with size (N,2C,H/2,W/2)
        """
        wSum_y = self.wSum(x_sag, x_cor)  # calculate spatial weighted summation
        catY_sag = torch.cat([x_sag, wSum_y], 1)  # concatenation in channel dimension
        catY_cor = torch.cat([x_cor, wSum_y], 1)  # concatenation in channel dimension
        y_sag = self.convModule_sag(catY_sag)  # convolution
        y_cor = self.convModule_cor(catY_cor)  # convolution
        return y_sag, y_cor


class SLE(nn.Module):
    """
    The Spinal Landmark Estimator (SLE), which :
        1. maps input image to #nDense1 features (#nDense1 = arbitrary)
        2. maps #nDense1 features to #nDense2 features (#nDense2 = 2 * #landmarks)
    (Paper: Automated comprehensive Adolescent Idiopathic Scoliosis assessment using MVC-Net)
    """

    def __init__(self, input_dim, nDense1, nDense2):
        """
        Initialization for the building blocks of SLE.
        :param input_dim: input dimension in the form of [C,H,W]
        :param nDense1: #features of the 1st Dense Layer (served as hidden layer, #features is arbitrary)
        :param nDense2: #features of the 2nd Dense Layer (#features = 2 * #landmarks)
        """
        super(SLE, self).__init__()

        self.dim_C, self.dim_H, self.dim_W = input_dim
        self.estimator_1half = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.dim_C * self.dim_H * self.dim_W, nDense1),
            nn.Tanh()
        )

        # Note: the network structure described in original paper does not work in our implementation
        self.estimator_2half = nn.Sequential(
            nn.Linear(nDense1, nDense2),
            mul_matrixS(nDense2),  # here is the problem
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x: input image
        :return: 2 outputs of SLE
            - y1: #nDense1 landmarks features (#features is arbitrary)
            - y2: #nDense2 landmarks coordinates estimations (#features = 2 * #landmarks)
        """
        y1 = self.estimator_1half(x)
        y2 = self.estimator_2half(y1)
        return y1, y2


class CAE(nn.Module):
    """
    The Cobb Angle Estimator (CAE), which :
        1. maps #nDense1 landmark features to #nDense2 angle features
        2. adds the #nDense2 angle features (from step 1) to  #nDense2 landmarks features (from previous layer)
        3. maps summed #nDense2 angle features (from step 2) to #nDense3 angles estimations
    (Paper: Automated comprehensive Adolescent Idiopathic Scoliosis assessment using MVC-Net)
    """

    def __init__(self, nDense1, nDense2, nDense3):
        """
        Initialize the building blocks of CAE.
        :param nDense1: #features from the previous SLE (#features is arbitrary) (see 'SLE' class for details)
        :param nDense2: (#features = 2 * #landmarks)
        :param nDense3: (#features = #angles)
        """
        super(CAE, self).__init__()
        self.dense1 = nn.Linear(nDense1, nDense2)
        self.tanh = nn.Tanh()
        self.dense2 = nn.Linear(nDense2, nDense3)

    def forward(self, lm_features, lm_coordinates):
        """
        The forwarding pass of CAE, which make cobb angles estimations from two landmark features.
        :param lm_features: the output 'y1' of SLE (see 'SLE' class for details)
        :param lm_coordinates: the output 'y2' of SLE (see 'SLE' class for details)
        :return angs: #nDense3 angle estimations
        """
        out_dense1 = self.dense1(lm_features)
        ang_features = self.tanh(out_dense1)
        ang_sumFeatures = ang_features + lm_coordinates
        angs = self.dense2(ang_sumFeatures)

        return angs


class MVCNet(nn.Module):
    """
    Build MVC-Net with fixed input image size (H,W)=(256,128)
    (Paper: Automated comprehensive Adolescent Idiopathic Scoliosis assessment using MVC-Net)
    """

    def __init__(self, n_lm, n_ang):
        """
        Initialize the MVC-Net with number of landmarks and cobb angles
        :param n_lm: #landmarks
        :param n_ang: #angles
        """
        super(MVCNet, self).__init__()
        self.convM1_sag = conv_bn_prelu_dropout(1, 64, 4, 2, 1, 64, 64, 0.25)
        self.convM1_cor = conv_bn_prelu_dropout(1, 64, 4, 2, 1, 64, 64, 0.25)
        self.xModule1 = xModule([64, 128, 64], 64, 4, 2, 1, 128, 128, 0.25)
        self.xModule2 = xModule([128, 64, 32], 128, 4, 2, 1, 256, 256, 0.25)
        self.xModule3 = xModule([256, 32, 16], 256, 4, 2, 1, 512, 512, 0.25)
        self.SLE_sag = SLE([512, 16, 8], 512, n_lm)
        self.SLE_cor = SLE([512, 16, 8], 512, n_lm)
        self.CAE_sag = CAE(512, n_lm, n_ang)
        self.CAE_cor = CAE(512, n_lm, n_ang)

    def forward(self, x_sag, x_cor):
        """
        The forwarding pass of MVC-Net, which takes two input images and output 4 predictions.
        :param x_sag: input sagittal image
        :param x_cor: input coronal image
        :return:
            - out2_SLE_sag: the landmarks predictions for sagittal image
            - out2_SLE_cor: the landmarks predictions for coronal image
            - out_CAE_sag: the cobb angles predictions for sagittal image
            - out_CAE_cor: the cobb angles predictions for coronal image
        """
        sag_out_convM1 = self.convM1_sag(x_sag)  # output of convM1
        cor_out_convM1 = self.convM1_cor(x_cor)  # output of convM2
        out_xM1_sag, out_xM1_cor = self.xModule1(sag_out_convM1, cor_out_convM1)  # output of x-Module 1
        out_xM2_sag, out_xM2_cor = self.xModule2(out_xM1_sag, out_xM1_cor)  # output of x-Module 2
        out_xM3_sag, out_xM3_cor = self.xModule3(out_xM2_sag, out_xM2_cor)  # output of x-Module 3
        out1_SLE_sag, out2_SLE_sag = self.SLE_sag(out_xM3_sag)  # output of spinal landmark estimator
        out1_SLE_cor, out2_SLE_cor = self.SLE_cor(out_xM3_cor)  # output of spinal landmark estimator
        out_CAE_sag = self.CAE_sag(out1_SLE_sag, out2_SLE_sag)  # output of cobb angle estimator
        out_CAE_cor = self.CAE_cor(out1_SLE_cor, out2_SLE_cor)  # output of cobb angle estimator
        return out2_SLE_sag, out2_SLE_cor, out_CAE_sag, out_CAE_cor

    def fit_MVCNet(self, epochs, opt, lr_scheduler, train_dl, valid_dl, dev, wd, flag, ang_flag, theta,
                   global_train_steps, writer, start_epoch=1, scheme='scheme1'):
        # training for each epoch
        for epoch in range(start_epoch, epochs + 1):
            # ====================================================================================================
            # => begin training for one epoch
            # ====================================================================================================
            self.train()  # set model to training mode
            accumulated_trainSize = 0
            epoch_lm_loss_train = 0
            epoch_ang_loss_train = 0
            i_batch = 0
            # training on training set for each batch
            for [x_b, y_lm_b, y_ang_b] in train_dl:
                global_train_steps += 1
                i_batch += 1
                i_trainSize, num_C, x_H, x_W = x_b.size()
                accumulated_trainSize += i_trainSize

                # we don't have two view, so we input the same-view image to two network entries
                lm_sag_cuda, lm_cor_cuda, ang_sag_cuda, ang_cor_cuda = self.forward(x_b.to(dev), x_b.to(dev))

                # concatenate landmarks predictions or ground truths from 2 views
                lm_cat_cuda = torch.cat((lm_sag_cuda, lm_cor_cuda), 1)
                y_lm_cat_cuda = y_lm_b.repeat(1, 2).to(dev)

                # TODO: try different landmark loss
                # theta=5 in original paper, we can use theta=0.2 to cater regression loss
                lm_loss_train = loss_landmark(lm_cat_cuda, y_lm_cat_cuda, theta)
                epoch_lm_loss_train += lm_loss_train.item() * i_trainSize  # the last batch has less data

                # concatenate cobb angle predictions or ground truths from 2 views
                ang_cat_cuda = torch.cat((ang_sag_cuda, ang_cor_cuda), 1)
                y_ang_cat_cuda = y_ang_b.repeat(1, 2).to(dev)

                # TODO: try different ang_loss
                assert ang_flag == 'circular' or 'non-circular' or 'circular_absCMAE'
                if ang_flag == 'circular':
                    ang_loss_train = loss_angle(ang_cat_cuda, y_ang_cat_cuda, theta)
                if ang_flag == 'circular_absCMAE':
                    ang_loss_train = loss_angle_absCMAE(ang_cat_cuda, y_ang_cat_cuda, theta)
                if ang_flag == 'non-circular':
                    ang_loss_train = loss_angle_nonCircular(ang_cat_cuda, y_ang_cat_cuda, theta)
                epoch_ang_loss_train += ang_loss_train.item() * i_trainSize

                # train network with different schemes
                assert scheme == "scheme1" or "scheme2"
                if scheme == "scheme1":
                    # The training scheme described in original paper
                    # back-prop two losses one after another
                    if global_train_steps % 2 == 1:
                        lm_loss_train.backward()
                        plot_CAE_flag = False
                        print('back-prop using landmark loss')
                    else:
                        ang_loss_train.backward()
                        plot_CAE_flag = True
                        print('back-prop using angle loss')
                if scheme == "scheme2":
                    # try this training scheme (proposed by YC Yao)
                    if epoch <= 100:  # use lm_loss for back-prop in the first 100 epochs
                        print('back-prop using landmark loss')
                        lm_loss_train.backward()
                        plot_CAE_flag = False
                    elif global_train_steps % 2 == 1:
                        lm_loss_train.backward()
                        plot_CAE_flag = False
                        print('back-prop using landmark loss')
                    else:
                        ang_loss_train.backward()
                        plot_CAE_flag = True
                        print('back-prop using angle loss')

                # one-step back-propagation
                opt.step()
                plot_MVCNet_grad(self, writer, global_train_steps, plot_CAE_flag)
                opt.zero_grad()

                print('epoch/batch: ', epoch, '/', i_batch, ' train lm_loss: ', lm_loss_train.detach().cpu().numpy(),
                      ' train ang_loss: ', ang_loss_train.detach().cpu().numpy())

            # evaluation of training losses
            lm_meanLoss_train = epoch_lm_loss_train / accumulated_trainSize  # training loss for landmarks
            ang_meanLoss_train = epoch_ang_loss_train / accumulated_trainSize  # training loss for angles
            # ====================================================================================================
            # => finish training for one epoch
            # ====================================================================================================

            # ====================================================================================================
            # => begin evaluation on validation set for one epoch
            # ====================================================================================================
            self.eval()  # set model to evaluation mode
            accumulated_valSize = 0
            epoch_lm_loss_val = 0
            epoch_ang_loss_val = 0
            epoch_AE = 0  # accumulated absolute error
            epoch_rho = 0  # accumulated rho
            epoch_cAE = 0  # accumulated circular absolute error
            epoch_sAE = 0  # accumulated symmetric absolute error
            epoch_AE_angle = 0  # accumulated absolute error for angles in degree
            # testing on validation set for each batch
            for [x_b, y_lm_b, y_ang_b] in valid_dl:
                i_valSize, num_C, x_H, x_W = x_b.size()
                accumulated_valSize += i_valSize

                with torch.no_grad():
                    # we don't have two view, so we input the same-view image to two network entries
                    lm_sag_cuda, lm_cor_cuda, ang_sag_cuda, ang_cor_cuda = self.forward(x_b.to(dev), x_b.to(dev))

                    # concatenate landmarks predictions or ground truths from 2 views
                    lm_cat_cuda = torch.cat((lm_sag_cuda, lm_cor_cuda), 1)
                    y_lm_cat_cuda = y_lm_b.repeat(1, 2).to(dev)

                    # TODO: try different landmark loss
                    # Caveat: should be consistent with the loss used in training steps
                    lm_loss_val = loss_landmark(lm_cat_cuda, y_lm_cat_cuda, theta)  # landmark loss
                    epoch_lm_loss_val += lm_loss_val.item() * i_valSize

                    # concatenate cobb angle predictions or ground truths from 2 views
                    ang_cat_cuda = torch.cat((ang_sag_cuda, ang_cor_cuda), 1)
                    y_ang_cat_cuda = y_ang_b.repeat(1, 2).to(dev)

                    # TODO: try different ang_loss
                    if ang_flag == 'circular':
                        ang_loss_val = loss_angle(ang_cat_cuda, y_ang_cat_cuda, theta)  # angle loss
                    if ang_flag == 'circular_absCMAE':
                        ang_loss_val = loss_angle_absCMAE(ang_cat_cuda, y_ang_cat_cuda, theta)
                    if ang_flag == 'non-circular':
                        ang_loss_val = loss_angle_nonCircular(ang_cat_cuda, y_ang_cat_cuda, theta)
                    epoch_ang_loss_val += ang_loss_val.item() * i_valSize

                    # metrics for landmarks
                    # 1) AE
                    epoch_AE += lm_cat_cuda.sub(y_lm_cat_cuda).abs().mean() * i_valSize  # accumulated absolute error
                    # 2) rho
                    rho = lm_cat_cuda.mul(y_lm_cat_cuda).mean(1).sub(
                        lm_cat_cuda.mean(1).mul(y_lm_cat_cuda.mean(1))).true_divide(
                        lm_cat_cuda.std(1).mul(y_lm_cat_cuda.std(1)))  # size: [N,]
                    mean_rho = rho.mean(0)  # mean rho for a batch, size: [1, ]
                    epoch_rho += mean_rho * i_valSize  # accumulated rho

                    # metrics for cobb angles
                    # 1) cAE
                    absErr = ang_cat_cuda.sub(y_ang_cat_cuda).abs()
                    x_bar_absErr = absErr.cos().mean(1)
                    y_bar_absErr = absErr.sin().mean(1)
                    mean_cAE = y_bar_absErr.true_divide(x_bar_absErr).atan().mean(0)  # size, [1,]
                    epoch_cAE += mean_cAE * i_valSize  # accumulated circular absolute error
                    # 2) SMAE: symmetric mean absolute error
                    sum_abs_sub = ang_cat_cuda.sub(y_ang_cat_cuda).abs().sum(1)  # size: [N,]
                    sum_add = ang_cat_cuda.add(y_ang_cat_cuda).sum(1)  # size: [N,]
                    mean_sAE = sum_abs_sub.true_divide(sum_add).mean(0)  # size, [1,]
                    epoch_sAE += mean_sAE * i_valSize  # accumulated symmetric absolute error
                    # 3) AE
                    epoch_AE_angle += ang_cat_cuda.sub(y_ang_cat_cuda).abs().mean() * i_valSize  # accumulated AE

            # evaluation of validation losses
            lm_meanLoss_val = epoch_lm_loss_val / accumulated_valSize  # validation loss for landmarks
            ang_meanLoss_val = epoch_ang_loss_val / accumulated_valSize  # validation loss for angles

            # evaluation of metrics for validation set
            MAE = epoch_AE / accumulated_valSize  # mean absolute error
            Rho = epoch_rho / accumulated_valSize  # mean rho
            CMAE = epoch_cAE / accumulated_valSize  # circular mean absolute error
            SMAPE = epoch_sAE / accumulated_valSize  # symmetric mean absolute error
            MAE_angle = epoch_AE_angle / accumulated_valSize  # MAE for angle
            MAE_angle_array = MAE_angle.cpu().numpy()
            MAE_degree = np.degrees(MAE_angle_array)

            # TODO: which metric to be monitored
            # lr_scheduler.step(lm_meanLoss_val)  # optim.lr_scheduler.ReduceLROnPlateau
            lr_scheduler.step(ang_meanLoss_val)  # optim.lr_scheduler.ReduceLROnPlateau

            print('==============================================================================================')
            print('epoch: ', epoch)
            print('train loss: lm_loss= ', lm_meanLoss_train, ' ang_loss= ', ang_meanLoss_train)
            print('valid loss: lm_loss= ', lm_meanLoss_val, ' ang_loss= ', ang_meanLoss_val)
            print('metrics for validation set:')
            print('landmarks: MAE= ', MAE.cpu().numpy(), ' rho= ', Rho.cpu().numpy())
            print('angles: CMAE= ', CMAE.cpu().numpy(), ' SMAPE= ', SMAPE.cpu().numpy(), ' MAE_degree= ', MAE_degree)
            print('==============================================================================================')

            # write loss to tensorboard
            writer.add_scalar('Loss/landmark_loss_train', lm_meanLoss_train, epoch)
            writer.add_scalar('Loss/landmark_loss_val', lm_meanLoss_val, epoch)
            writer.add_scalar('Loss/ang_loss_train', ang_meanLoss_train, epoch)
            writer.add_scalar('Loss/ang_loss_val', ang_meanLoss_val, epoch)
            # write landmark metrics to tensorboard
            writer.add_scalar('Metrics/landmark_MAE', MAE, epoch)
            writer.add_scalar('Metrics/landmark_rho', Rho, epoch)
            # write angle metrics to tensorboard
            writer.add_scalar('Metrics/angle_CMAE', CMAE, epoch)
            writer.add_scalar('Metrics/angle_SMAPE', SMAPE, epoch)
            writer.add_scalar('Metrics/angle_MAE', MAE_degree, epoch)
            # ====================================================================================================
            # => finish evaluation on validation set for one epoch
            # ====================================================================================================

            # ====================================================================================================
            # TODO: choose the criterion for saving the best model
            # save the best model
            # ====================================================================================================
            checkpoint_dir = os.path.join(wd, 'model_' + flag)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            # ====================================
            # Criterion1: CMAE (validation set)
            # ====================================
            if epoch == 1:
                best_CMAE = CMAE
                best_model_CMAE_file = "best_model_CMAE_epoch" + str(epoch) + ".pt"
                best_model_CMAE_path = os.path.join(checkpoint_dir, best_model_CMAE_file)
                old_best_model_CMAE_path = best_model_CMAE_path
                # save the best model
                state = {'epoch': epoch,
                         'model_state_dict': self.cpu().state_dict(),
                         'opt_state_dict': opt.state_dict(),
                         'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                         'global_step': global_train_steps}
                torch.save(state, best_model_CMAE_path)
                # move model back to GPU
                self.to(dev)
            elif CMAE < best_CMAE:
                best_CMAE = CMAE
                os.remove(old_best_model_CMAE_path)  # remove old checkpoint
                best_model_CMAE_file = "best_model_CMAE_epoch" + str(epoch) + ".pt"
                best_model_CMAE_path = os.path.join(checkpoint_dir, best_model_CMAE_file)
                old_best_model_CMAE_path = best_model_CMAE_path
                # save the best model
                state = {'epoch': epoch,
                         'model_state_dict': self.cpu().state_dict(),
                         'opt_state_dict': opt.state_dict(),
                         'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                         'global_step': global_train_steps}
                torch.save(state, best_model_CMAE_path)
                # move model back to GPU
                self.to(dev)

            # ====================================
            # Criterion2: MAE_degree (validation set)
            # ====================================
            if epoch == 1:
                best_MAE_degree = MAE_degree
                best_model_MAEdegree_file = "best_model_MAEdegree_epoch" + str(epoch) + ".pt"
                best_model_MAEdegree_path = os.path.join(checkpoint_dir, best_model_MAEdegree_file)
                old_best_model_MAEdegree_path = best_model_MAEdegree_path
                # save the best model
                state = {'epoch': epoch,
                         'model_state_dict': self.cpu().state_dict(),
                         'opt_state_dict': opt.state_dict(),
                         'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                         'global_step': global_train_steps}
                torch.save(state, best_model_MAEdegree_path)
                # move model back to GPU
                self.to(dev)
            elif MAE_degree < best_MAE_degree:
                best_MAE_degree = MAE_degree
                os.remove(old_best_model_MAEdegree_path)  # remove old checkpoint
                best_model_MAEdegree_file = "best_model_MAEdegree_epoch" + str(epoch) + ".pt"
                best_model_MAEdegree_path = os.path.join(checkpoint_dir, best_model_MAEdegree_file)
                old_best_model_MAEdegree_path = best_model_MAEdegree_path
                # save the best model
                state = {'epoch': epoch,
                         'model_state_dict': self.cpu().state_dict(),
                         'opt_state_dict': opt.state_dict(),
                         'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                         'global_step': global_train_steps}
                torch.save(state, best_model_MAEdegree_path)
                # move model back to GPU
                self.to(dev)

            # ====================================
            # Criterion3: SMAPE (validation set)
            # ====================================
            if epoch == 1:
                best_SMAPE = SMAPE
                best_model_SMAPE_file = "best_model_SMAPE_epoch" + str(epoch) + ".pt"
                best_model_SMAPE_path = os.path.join(checkpoint_dir, best_model_SMAPE_file)
                old_best_model_SMAPE_path = best_model_SMAPE_path
                # save the best model
                state = {'epoch': epoch,
                         'model_state_dict': self.cpu().state_dict(),
                         'opt_state_dict': opt.state_dict(),
                         'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                         'global_step': global_train_steps}
                torch.save(state, best_model_SMAPE_path)
                # move model back to GPU
                self.to(dev)
            elif SMAPE < best_SMAPE:
                best_SMAPE = SMAPE
                os.remove(old_best_model_SMAPE_path)  # remove old checkpoint
                best_model_SMAPE_file = "best_model_SMAPE_epoch" + str(epoch) + ".pt"
                best_model_SMAPE_path = os.path.join(checkpoint_dir, best_model_SMAPE_file)
                old_best_model_SMAPE_path = best_model_SMAPE_path
                # save the best model
                state = {'epoch': epoch,
                         'model_state_dict': self.cpu().state_dict(),
                         'opt_state_dict': opt.state_dict(),
                         'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                         'global_step': global_train_steps}
                torch.save(state, best_model_SMAPE_path)
                # move model back to GPU
                self.to(dev)

            # ====================================
            # Criterion4: Rho (validation set)
            # ====================================
            if epoch == 1:
                best_Rho = Rho
                best_model_Rho_file = "best_model_Rho_epoch" + str(epoch) + ".pt"
                best_model_Rho_path = os.path.join(checkpoint_dir, best_model_Rho_file)
                old_best_model_Rho_path = best_model_Rho_path
                # save the best model
                state = {'epoch': epoch,
                         'model_state_dict': self.cpu().state_dict(),
                         'opt_state_dict': opt.state_dict(),
                         'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                         'global_step': global_train_steps}
                torch.save(state, best_model_Rho_path)
                # move model back to GPU
                self.to(dev)
            elif Rho > best_Rho:
                best_Rho = Rho
                os.remove(old_best_model_Rho_path)  # remove old checkpoint
                best_model_Rho_file = "best_model_Rho_epoch" + str(epoch) + ".pt"
                best_model_Rho_path = os.path.join(checkpoint_dir, best_model_Rho_file)
                old_best_model_Rho_path = best_model_Rho_path
                # save the best model
                state = {'epoch': epoch,
                         'model_state_dict': self.cpu().state_dict(),
                         'opt_state_dict': opt.state_dict(),
                         'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                         'global_step': global_train_steps}
                torch.save(state, best_model_Rho_path)
                # move model back to GPU
                self.to(dev)

            # ====================================================================================================
            # save checkpoint for each epoch
            # ====================================================================================================
            # NOTE: new checkpoint will cover the old one
            # NOTE: if you change the keys in state, you have to make changes in resume_training() defined in utilities
            saving_step = 1
            if epoch % saving_step == 0:
                # remove old checkpoint if exists
                if epoch != saving_step:
                    os.remove(old_checkpoint_path)
                now = datetime.now()
                checkpoint_file = "checkpoint_epoch" + str(epoch) + '_' + str(now) + ".pt"
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
                old_checkpoint_path = checkpoint_path
                # save checkpoint
                state = {'epoch': epoch,
                         'model_state_dict': self.cpu().state_dict(),
                         'opt_state_dict': opt.state_dict(),
                         'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                         'global_step': global_train_steps,
                         'best_CMAE': best_CMAE,
                         'old_best_model_CMAE_path': old_best_model_CMAE_path,
                         'best_MAE_degree': best_MAE_degree,
                         'old_best_model_MAEdegree_path': old_best_model_MAEdegree_path,
                         'best_SMAPE': best_SMAPE,
                         'old_best_model_SMAPE_path': old_best_model_SMAPE_path,
                         'best_Rho': best_Rho,
                         'old_best_model_Rho_path': old_best_model_Rho_path,
                         }
                torch.save(state, checkpoint_path)
                # move model back to GPU
                self.to(dev)

        # free GPU memory
        torch.cuda.empty_cache()
