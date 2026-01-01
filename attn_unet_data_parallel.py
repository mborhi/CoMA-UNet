import os 
import sys

import logging
import copy 

import numpy as np
import pandas as pd

from typing import Sequence, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader

from sklearn.metrics import r2_score

from monai.networks.blocks.convolutions import Convolution as mConvolution
from monai.networks.layers.factories import Norm
from monai.networks.nets import attentionunet
from monai.metrics.regression import SSIMMetric

import data_util 
# from VolumeDataset import VolumeDataset, CovariateVolumeDataset
import VolumeDataset
import CondConv
import create_roi_suvr_csv as crt_roi_suvr

# from torch.nn.parallel.data_parallel import DataParallel
from torch.nn import DataParallel, SyncBatchNorm

from visualization_util import loss_graph, metric_graph, plot_mae_progression_chart, boxplot_roi_value_progression

class RoiCorrMetric():

    def __init__(self, roi_indices, spatial_dims=3, win_size=7, reduction="mean"):
        self.roi_indices = roi_indices
        self.spatial_dims = spatial_dims
        self.win_size = win_size
        self.reduction = reduction
        self.num_samples = 0

        self.pred_means = [[] for i in range(len(roi_indices))]
        self.gt_means = [[] for i in range(len(roi_indices))]
        self.sample_ids = []

    def acc_roi_corr(self, pred, gt, roi):
        roi_mask = torch.zeros(roi.size(), device=roi.get_device())
        for i, idx in enumerate(self.roi_indices):
            roi_mask[:] = 0 # reset mask
            roi_mask[roi == idx] = 1
            # Take the mean in the ROI sample-wise

            pred_roi_mean = torch.sum(roi_mask * pred, dim=(-3, -2, -1)) / torch.count_nonzero(roi_mask, dim=(-3, -2, -1))
            gt_roi_mean = torch.sum(roi_mask * gt, dim=(-3, -2, -1)) / torch.count_nonzero(roi_mask, dim=(-3, -2, -1))

            self.pred_means[i].extend(pred_roi_mean.detach().cpu().numpy().flatten())
            self.gt_means[i].extend(gt_roi_mean.detach().cpu().numpy().flatten())

    def acc_sample_ids(self, unstructured_sample_ids):
        def extract_id(unstructured_sample_id):
            # /home/jagust/xnat/xnp/sshfs/xnat_data/a4/B10423472/PET_2017-01-01_FTP/analysis/suvr_cereg.nii
            tokens = unstructured_sample_id.split("/")
            if "a4" in tokens:
                ind = tokens.index("a4")
                sample_id = tokens[ind + 1] # always occurs after a4
            elif "scan" in tokens:
                ind = tokens.index("scan")
                sample_id = tokens[ind+1] + "/" + tokens[ind+2] 
            else:
                # 002-S-6009/PET_2017-05-15_FTP
                # /home/jagust/xnat/xnp/sshfs/xnat_data/adni/000-S-0059/PET_2017-12-12_FTP/analysis/rnu.nii
                ind = tokens.index("adni")
                sample_id = "/".join(tokens[ind+1:ind+3])
            return sample_id
            
        sample_ids = [extract_id(unstructured_sample_id) for i, unstructured_sample_id in enumerate(unstructured_sample_ids)]
        self.sample_ids.extend(sample_ids)

    def calc_roi_corr(self):
        roi_corrs = np.empty(len(self.roi_indices))
        for i, idx in enumerate(self.roi_indices):
            # print(self.pred_means[i])
            # print(self.gt_means[i])
            roi_corrs[i] = np.corrcoef(self.pred_means[i], self.gt_means[i])[0, 1]

        return roi_corrs
    
    def save_matrices(self, save_path, type=""):

        pred_df = pd.DataFrame(np.stack(self.pred_means))
        pred_df.to_csv(os.path.join(save_path, f"{type}pred_means.csv"), header=self.sample_ids, index=False)
        gt_df = pd.DataFrame(np.stack(self.gt_means), columns=self.sample_ids)
        gt_df.to_csv(os.path.join(save_path, f"{type}gt_means.csv"), header=self.sample_ids, index=False)


class MLP(nn.Module):

    def __init__(self, input_size, hidden_layers, num_classes, activation_fn=nn.ReLU) -> None:
        super().__init__()

        self.layers = []
        prev_layer_size = input_size
        for i in range(len(hidden_layers)):
            self.layers.append(nn.Linear(prev_layer_size, hidden_layers[i]))
            self.layers.append(activation_fn())
            prev_layer_size = hidden_layers[i]

        self.layers.append(nn.Linear(prev_layer_size, num_classes))
        self.mlp = nn.Sequential(*nn.ModuleList(self.layers))

    def forward(self, x):
        # Temporary for classification of ABeta
        class_logits = nn.functional.softmax(self.mlp(x), 1)
        return class_logits


class UpBlock(attentionunet.UpConv):

    def __init__(self, conditional, num_covars=0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if conditional:
            self.up = CondConv.CondConvolution(dropout=0.0, is_transposed=True, num_covars=num_covars, *args, **kwargs)

    def forward(self, x, covariate=None):
        # logging.info(f"covar in up block {covariate}")
        x_u: torch.Tensor = self.up(x, covariate)
        return x_u


class ObservableAttentionBlock(attentionunet.AttentionBlock):

    def __init__(self, spatial_dims: np.int, f_int: np.int, f_g: np.int, f_l: np.int, dropout=0):
        super().__init__(spatial_dims, f_int, f_g, f_l, dropout)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # print(f"g: {g.size()} | x: {x.size()}")
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi: torch.Tensor = self.relu(g1 + x1)
        psi = self.psi(psi)
        # print(f"Psi: {psi.size()}")

        if self.save_attn:
            return x*psi, psi

        return x * psi
    
class AttentionLayer(attentionunet.AttentionLayer):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        submodule: nn.Module,
        up_kernel_size=3,
        strides=2,
        dropout=0.0,
        conditional=False, 
        num_covars=0,
    ):
        super().__init__(spatial_dims, in_channels, out_channels, submodule, up_kernel_size, strides, dropout)
        self.attention = ObservableAttentionBlock(
            spatial_dims=spatial_dims, f_g=in_channels, f_l=in_channels, f_int=in_channels // 2
        )
        self.upconv = UpBlock(
            conditional=conditional,
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=in_channels,
            strides=strides,
            kernel_size=up_kernel_size,
            num_covars=num_covars
        )
        
        # if conditional:
        #     self.merge = CondConv.CondConvolution(
        #         spatial_dims=spatial_dims, in_channels=2 * in_channels, out_channels=in_channels, dropout=dropout, num_experts=10
        #     )
        # else:
        #     self.merge = mConvolution(
        #         spatial_dims=spatial_dims, in_channels=2 * in_channels, out_channels=in_channels, dropout=dropout
        #     )

    def set_save_attn(self, status):
        self.save_attn = status
        self.attention.save_attn = status

    def forward(self, x: torch.Tensor, covariate=None) -> torch.Tensor:
        """rest := skip conns (encoded)"""
        # print(f"Input to Attn Layer: {x.size()}")
        # logging.info(f"covar in atnn layer forward {covariate}")
        if isinstance(self.submodule, nn.Sequential):
            # print(f"before clone: {x.grad}")
            x_sub = x#.clone()#.to(device=x.device)
            # print(f"after clone: {x_sub.grad}")
            for idx, submodule in enumerate(self.submodule):
                # logging.info(f"{idx} submsdofp: {type(submodule)}")
                if isinstance(submodule, AttentionLayer):
                    # logging.info(f"submsdofp: {type(submodule)}")
                    # x_sub = submodule(x_sub)
                    x_sub = submodule(x_sub, covariate=covariate)
                else:
                    # x_sub = submodule(x_sub)
                    # x_sub = submodule(x_sub, covariate=covariate)
                    x_sub = submodule(x_sub, covariate=covariate[:, :, :5]) # NOTE
        else: 
            # x_sub = self.submodule(x)
            x_sub = self.submodule(x, covariate=covariate[:, :, :5]) # NOTE
        if isinstance(x_sub, tuple): # if len(x_sub) > 1 
            x_sub, rest = x_sub 
            # 2. x_sub: 256d, rest: (256e, (512e, 512e))
            # 3. x_sub: 128d, rest: (128e, (256d, (256e, (512e, 512e)))))
            # 4. (64d, (64e, (128d, (128e, (256d, (256e, (512e, 512e)))))))
            # print(f"Current xsub_out: {x_sub.size()} (with input: {x.size()})")
        else :
            # print(f"First xsub out: {x_sub.size()} with input: {x.size()}")
            rest = x_sub
            # 1. 512e
        fromlower = self.upconv(x_sub, covariate)
        att = self.attention(g=fromlower, x=x)
        if self.save_attn is not None:
            att, coeff = att
            data_util.save_attention_coeffs(self.save_attn, coeff)
        # logging.info(f"Attention coeff: {att.size()}")
        att_m: torch.Tensor = self.merge(torch.cat((att, fromlower), dim=1))
        # logging.info(f"attn_merged: {att_m.size()} | x_sub: {x_sub.size()} | rest: {len(rest)}")
        # 1. (512e, 512e)
        # 2. (256d, (256e, (512e, 512e)))
        # 3. (128d, (128e, (256d, (256e, (512e, 512e)))))
        # 4. (64d, (64e, (128d, (128e, (256d, (256e, (512e, 512e)))))))
        rest = (x_sub, rest)
        # 1. (256d, (256e, (512e, 512e)))
        # 2. (128d, (128e, (256d, (256e, (512e, 512e)))))
        # 3. (64d, (64e, (128d, (128e, (256d, (256e, (512e, 512e)))))))
        # 4. (32d, (32e, (64d, (64e, (128d, (128e, (256d, (256e, (512e, 512e)))))))))
        return att_m, (x, rest)
        # curr_dec_out, (skip, (lower_dec_out, rest))

class ObservableAttentionUnet(nn.Module):
    """
    Attention Unet based on
    Otkay et al. "Attention U-Net: Learning Where to Look for the Pancreas"
    https://arxiv.org/abs/1804.03999

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        channels (Sequence[int]): sequence of channels. Top block first. The length of `channels` should be no less than 2.
        strides (Sequence[int]): stride to use for convolutions.
        kernel_size: convolution kernel size.
        up_kernel_size: convolution kernel size for transposed convolution layers.
        dropout: dropout ratio. Defaults to no dropout.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        dropout: float = 0.0,
        conditional: bool = False,
    ):
        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.conditional = conditional
        self.with_regression = True # NOTE

        logging.info(f"Conditional when loading: {conditional} | with meta tau integ. : {self.with_regression}")
        
        ConvBlock = CondConv.CondConvBlock if conditional else attentionunet.ConvBlock
        Convolution = CondConv.CondConvolution if conditional else mConvolution

        # head = attentionunet.ConvBlock(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=channels[0], dropout=dropout)
        head = ConvBlock(spatial_dims=spatial_dims, 
                         in_channels=in_channels, 
                         out_channels=channels[0], 
                         dropout=dropout, 
                         num_covars=5#+ int(self.with_regression)
                         )

        reduce_channels = Convolution(
            spatial_dims=spatial_dims,
            in_channels=channels[0],
            out_channels=out_channels,
            kernel_size=1,
            strides=1,
            padding=0,
            conv_only=True,
            num_experts=8, # for CondConv
            num_covars=5 + int(self.with_regression)
        )
        self.up_kernel_size = up_kernel_size
        self.save_attn = None

        def _create_block(channels: Sequence[int], strides: Sequence[int]) -> nn.Module:
            if len(channels) > 2:
                subblock = _create_block(channels[1:], strides[1:])
                return AttentionLayer(
                    spatial_dims=spatial_dims,
                    in_channels=channels[0],
                    out_channels=channels[1],
                    submodule=nn.Sequential(
                        ConvBlock(
                            spatial_dims=spatial_dims,
                            in_channels=channels[0],
                            out_channels=channels[1],
                            strides=strides[0],
                            dropout=self.dropout,
                            num_covars=5 #+ int(self.with_regression), # NOTE
                        ),
                        # attentionunet.ConvBlock(
                        #     spatial_dims=spatial_dims,
                        #     in_channels=channels[0],
                        #     out_channels=channels[1],
                        #     strides=strides[0],
                        #     dropout=self.dropout,
                        # ),
                        subblock,
                    ),
                    up_kernel_size=self.up_kernel_size,
                    strides=strides[0],
                    dropout=dropout,
                    conditional=self.conditional, 
                    num_covars=5 + int(self.with_regression)
                )
            else:
                # the next layer is the bottom so stop recursion,
                # create the bottom layer as the subblock for this layer
                return self._get_bottom_layer(channels[0], channels[1], strides[0])

        encdec = _create_block(self.channels, self.strides)
        # self.model = nn.Sequential(head, encdec, reduce_channels)
        # encdec_list.insert(0, head)
        # encdec_list.append(reduce_channels)
        model_list = [head, encdec, reduce_channels]
        self.model = nn.ModuleList(model_list)

    def _get_bottom_layer(self, in_channels: int, out_channels: int, strides: int) -> nn.Module:
        ConvBlock = CondConv.CondConvBlock if self.conditional else attentionunet.ConvBlock

        return AttentionLayer(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            submodule=ConvBlock(
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                dropout=self.dropout,
                num_covars=5# + int(self.with_regression) # NOTE
            ),
            up_kernel_size=self.up_kernel_size,
            strides=strides,
            dropout=self.dropout,
            conditional=self.conditional, 
            num_covars=5 + int(self.with_regression)
        )

    def set_save_attn(self, save_attn_value):
        encdec = self.model[1]
        while isinstance(encdec, AttentionLayer):
            # encdec.save_attn = save_attn_value
            encdec.set_save_attn(save_attn_value)
            seq = encdec.submodule # sequential, attenion layer second (last) in seq
            if isinstance(seq, nn.Sequential):
                encdec = seq[-1]
                # logging.info(f"{type(encdec)}")
            else: 
                # logging.info(f"{type(seq)}")
                break


    def forward(self, x: torch.Tensor, covariate=None) -> torch.Tensor:
        x_submod_lst = []
        decoder_extractions = []
        encoder_extractions = []

        for i, layer in enumerate(self.model):
            # print(f"Input to UNet at stage ({i}): {x.size()}")
            if i == 1:
                x = layer(x, covariate)
                # (32d, (32e, (64d, (64e, (128d, (128e, (256d, (256e, (512e, 512e)))))))))
                x, x_submod = x # x := output of UNet (input for `reduce_channels`)
                decoder_extractions.append(x)
                while isinstance(x_submod, tuple):
                    # 1. (32e, (64d, (64e, (128d, (128e, (256d, (256e, (512e, 512e))))))))
                    # 2. (64e, (128d, (128e, (256d, (256e, (512e, 512e))))))))
                    # 3. (128e, (256d, (256e, (512e, 512e))))))))
                    # 4. (256e, (512e, 512e))))))))
                    e_i, d_next_i_rest = x_submod
                    encoder_extractions.append(e_i)
                    # 1. d_next_i_rest = (64d, (64e, (128d, (128e, (256d, (256e, (512e, 512e))))))))
                    # 2. d_next_i_rest = (128d, (128e, (256d, (256e, (512e, 512e))))))))
                    # 3. d_next_i_reset = (256d, (256e, (512e, 512e))))))))
                    # 4. d_next_i_reset = (512e, 512e))))))))
                    d_i_next, rest = d_next_i_rest
                    # 1. d_i_next = 64d, rest = (64e, (128d, (128e, (256d, (256e, (512e, 512e))))))))
                    # 2. d_i_next = 128d, rest = (128e, (256d, (256e, (512e, 512e))))))))
                    # 3. d_i_next = 256d, rest = (256e, (512e, 512e))))))))
                    # 4. d_i_next = 512e, rest = 512e
                    if isinstance(rest, tuple):
                        decoder_extractions.append(d_i_next)
                    else:
                        encoder_extractions.append(d_i_next)
                    x_submod = rest
            elif i == 2:
                # logging.info(f"reduce channels")
                # x = layer(x, covariate=None)
                x = layer(x, covariate=covariate)

            else:
                x = layer(x, covariate=covariate[:, :, :5])

        # print(f"encoder ex lst: {[v.size() for v in encoder_extractions]}")
        # print(f"decoder ex lst: {[v.size() for v in decoder_extractions]}")
        return x, encoder_extractions, decoder_extractions
        
        return x, x_submod_lst

class ProjectionHead(nn.Module):

    def __init__(self, in_channels, out_channels, latent_space_dim, kernel_size=3) -> None:
        super().__init__()

        # self.conv = attentionunet.ConvBlock(3, in_channels, out_channels, kernel_size=kernel_size)
        self.conv = attentionunet.ConvBlock(3, in_channels, 1, kernel_size=1)
        # self.aap = nn.AdaptiveAvgPool3d(1)
        # self.linear = nn.Linear(out_channels, latent_space_dim)
        self.act_fn = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        # x = self.aap(x).flatten(1)
        x = x.flatten(1)
        # x = self.linear(x)
        x = self.act_fn(x)

        return x

class AleatoricUncertaintyNet(nn.Module):
    def __init__(self, input_dim):
        super(AleatoricUncertaintyNet, self).__init__()
        
        # Network to estimate log(variance)
        self.fc = nn.Sequential(
            nn.Linear(input_dim + 1, 64),  # Input: x (vector) + q^ (scalar)
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output: log(sigma^2)
        )
    
    def forward(self, x, q_hat):
        # input_data = torch.cat([x, q_hat.unsqueeze(1)], dim=1)  # Concatenate x and q_hat
        if len(x.shape) == 3:
            x = x.squeeze(1)
        input_data = torch.cat([x, q_hat.unsqueeze(1)], dim=1).to(dtype=torch.float32)  # Concatenate x and q_hat
        log_sigma2 = self.fc(input_data)  # Predict log(sigma^2)
        sigma2 = torch.exp(log_sigma2)  # Ensure positive variance
        confidence = 1 / (1 + sigma2)   # Confidence score (inverse of uncertainty)
        return sigma2, confidence  # Output both uncertainty and confidence


class StackedFusionConvLayers(nn.Module):
    def __init__(self, input_feature_channels, bottleneck_feature_channel, output_feature_channels, num_convs, nonlin=nn.LeakyReLU, nonlin_kwargs=None):

        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin

        super(StackedFusionConvLayers, self).__init__()

        self.blocks = nn.Sequential(
            *([mConvolution(spatial_dims=3, in_channels=input_feature_channels, out_channels=bottleneck_feature_channel, act=(nonlin, nonlin_kwargs))] +
              [mConvolution(spatial_dims=3, in_channels=bottleneck_feature_channel, out_channels=bottleneck_feature_channel, act=(nonlin, nonlin_kwargs)) for _ in range(num_convs - 2)] +
              [mConvolution(spatial_dims=3, in_channels=bottleneck_feature_channel, out_channels=output_feature_channels,act=(self.nonlin, self.nonlin_kwargs))]
            ))

    def forward(self, x):
        return self.blocks(x)
    
class ContrastiveAttentionUNET_DP(ObservableAttentionUnet):

    def __init__(
            self,
            spatial_dims: int, 
            in_channels: int, 
            out_channels: int, 
            channels: Sequence[int], 
            strides: Sequence[int], 
            latent_spaces: Sequence[int],
            kernel_size = 3, 
            up_kernel_size = 3, 
            dropout: float = 0,
            training: bool = True, 
            embeddings_out: bool = False,
            conditional: bool = False, 
            decoder_ds: bool = False,
            **kwargs
        ):
        super().__init__(spatial_dims, in_channels, out_channels, channels, strides, kernel_size, up_kernel_size, dropout, conditional)
        self.training = training
        self.embeddings_out = embeddings_out
        self.decoder_ds = decoder_ds

        self.depth = len(channels)

        # self.projection_heads = nn.ModuleList([]) # potentially make this a dictionary
        self.projection_heads = [] # potentially make this a dictionary
        for i in range(len(channels)): # lowest layer is encoder not decoder
            proj_head = ProjectionHead(in_channels=channels[i], out_channels=int((128/(2**i))**3), latent_space_dim=latent_spaces[i])
            self.projection_heads.append(proj_head)

        self.projection_heads = nn.ModuleList(self.projection_heads)
        
        self.final_projection_head = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Linear(out_channels, latent_spaces[-1]),
                nn.ReLU(),
        )

        # One for each pos/neg quartile, 
        self.pos_dynamic_prompt = nn.Parameter(torch.randn(1, 1, 128, 128, 128), requires_grad=True)
        self.neg_dynamic_prompt = nn.Parameter(torch.randn(1, 1, 128, 128, 128), requires_grad=True)
        self.fusion_layer = StackedFusionConvLayers(input_feature_channels=2, output_feature_channels=1, bottleneck_feature_channel=8, num_convs=3)
        self.modulator = mConvolution(spatial_dims=3, in_channels=2, out_channels=1, act="ReLU")
        self.modulator_3c = mConvolution(spatial_dims=3, in_channels=3, out_channels=1, act="ReLU")
        self.reweigh = nn.Parameter(torch.ones((128, 128, 128), dtype=torch.float32), requires_grad=True)
        self.final_act = nn.ReLU()
        self.final_act = nn.ReLU()

        # NOTE new additions:
        self.pos_reweigh = nn.Parameter(torch.ones((1, 128, 128, 128), dtype=torch.float32), requires_grad=True)
        self.neg_reweigh = nn.Parameter(torch.ones((1, 128, 128, 128), dtype=torch.float32), requires_grad=True)

        self.deep_modulator_3c = StackedFusionConvLayers(input_feature_channels=3, output_feature_channels=1, bottleneck_feature_channel=16, num_convs=3)
        self.final_pred_head = mConvolution(spatial_dims=3, in_channels=2, out_channels=1, kernel_size=1)


        self.roi_indices = [
            1001, 1006, 1007, 1009, 1015, 1016, 1030, 1034, 1033, 1008, 1025, 1029, 1031, 1022, 17, 18,
            2001, 2006, 2007, 2009, 2015, 2016, 2030, 2034, 2033, 2008, 2025, 2029, 2031, 2022, 49, 50, 51, 52, 53, 54
        ]

        # Corresponding ROI names
        self.roi_names = [
            'ctx-lh-bankssts', 'ctx-lh-entorhinal', 'ctx-lh-fusiform', 'ctx-lh-inferiortemporal',
            'ctx-lh-middletemporal', 'ctx-lh-parahippocampal', 'ctx-lh-superiortemporal',
            'ctx-lh-transversetemporal', 'ctx-lh-temporalpole', 'ctx-lh-inferiorparietal',
            'ctx-lh-precuneus', 'ctx-lh-superiorparietal', 'ctx-lh-supramarginal', 'ctx-lh-postcentral',
            'Left-Hippocampus', 'Left-Amygdala', 'ctx-rh-bankssts', 'ctx-rh-entorhinal',
            'ctx-rh-fusiform', 'ctx-rh-inferiortemporal', 'ctx-rh-middletemporal',
            'ctx-rh-parahippocampal', 'ctx-rh-superiortemporal', 'ctx-rh-transversetemporal',
            'ctx-rh-temporalpole', 'ctx-rh-inferiorparietal', 'ctx-rh-precuneus',
            'ctx-rh-superiorparietal', 'ctx-rh-supramarginal', 'ctx-rh-postcentral', 'Right-Thalamus-Proper',
            'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala'
        ]

        # Dictionary mapping indices to names
        index_to_name = {
            1001: 'ctx-lh-bankssts', 1006: 'ctx-lh-entorhinal', 1007: 'ctx-lh-fusiform',
            1009: 'ctx-lh-inferiortemporal', 1015: 'ctx-lh-middletemporal',
            1016: 'ctx-lh-parahippocampal', 1030: 'ctx-lh-superiortemporal',
            1034: 'ctx-lh-transversetemporal', 1033: 'ctx-lh-temporalpole',
            1008: 'ctx-lh-inferiorparietal', 1025: 'ctx-lh-precuneus',
            1029: 'ctx-lh-superiorparietal', 1031: 'ctx-lh-supramarginal', 1022: 'ctx-lh-postcentral',
            17: 'Left-Hippocampus', 18: 'Left-Amygdala', 2001: 'ctx-rh-bankssts',
            2006: 'ctx-rh-entorhinal', 2007: 'ctx-rh-fusiform', 2009: 'ctx-rh-inferiortemporal',
            2015: 'ctx-rh-middletemporal', 2016: 'ctx-rh-parahippocampal',
            2030: 'ctx-rh-superiortemporal', 2034: 'ctx-rh-transversetemporal',
            2033: 'ctx-rh-temporalpole', 2008: 'ctx-rh-inferiorparietal',
            2025: 'ctx-rh-precuneus', 2029: 'ctx-rh-superiorparietal', 2031: 'ctx-rh-supramarginal',
            2022: 'ctx-rh-postcentral', 49: 'Right-Thalamus-Proper', 50: 'Right-Caudate',
            51: 'Right-Putamen', 52: 'Right-Pallidum', 53: 'Right-Hippocampus',
            54: 'Right-Amygdala'
        }

        self.roi_ind_vol_names_dict = {
            k: 'vol_' + '_'.join(v.split('-')) for k, v in index_to_name.items()
        }

        # Dictionary mapping names to indices
        roi_dict = {v: k for k, v in index_to_name.items()}
        self.roi_ind_names_dict = index_to_name

        print(self.roi_ind_names_dict)
        # sys.exit(0)
        # NOTE new test: 
        self.general_dynamic_prompt = nn.Parameter(torch.randn(1, 1, 128, 128, 128), requires_grad=True)

        # NOTE ROI-wise error weighing of Lambda
        self.roi_wise_reweigh = nn.ParameterList([
            nn.Parameter(torch.ones((1), dtype=torch.float32), requires_grad=True) for _ in range(len(self.roi_indices)) # native space
        ])

        self.all_stages = True
        self.only_stage_two = False

        self.with_uq = kwargs.get("with_uq", False) # for now default

    def set_training(self, mode):

        self.training = mode

    def get_depth(self):

        return self.depth

    def forward_modulator_with_uq(self, x, out, covariate=None, roi_pred_dicts=None, sample_roi_mask = None):

        dynamic_prompt = []
        neutral_dynamic_prompt = []
        suvr_volume = torch.zeros_like(out, requires_grad=False)
        saliency_volume = torch.zeros_like(out, requires_grad=False)
        # meta_tau_volume = torch.zeros_like(out, requires_grad=False)
        for b in range(x.size(0)):
            prompt_idx = covariate[b, ..., 0].item() #+ (covariate[:, -1] * 4)
            dynamic_prompt.append(self.pos_dynamic_prompt if prompt_idx == 1 else self.neg_dynamic_prompt)
            neutral_dynamic_prompt.append(self.general_dynamic_prompt)
            for i, roi_idx in enumerate(self.roi_indices):
                roi_dict_name = self.roi_ind_names_dict[roi_idx] # NOTE NOTE this is for original for getting CB pred
                suvr_volume[b][sample_roi_mask[b] == roi_idx] = np.nan_to_num(roi_pred_dicts[b][roi_dict_name]['loc']) # predicted tau value for ROI
                saliency_volume[b][sample_roi_mask[b] == roi_idx] = np.nan_to_num(roi_pred_dicts[b][roi_dict_name]['std']) # associated std for the ROI tau value

        suvr_volume = torch.where(x < 1e-04, torch.zeros_like(suvr_volume), suvr_volume)
        saliency_volume = torch.where(x < 1e-04, torch.zeros_like(saliency_volume), saliency_volume)
        dynamic_prompt = torch.vstack(dynamic_prompt).to(device=x.device)
        neutral_dynamic_prompt = torch.vstack(neutral_dynamic_prompt).to(device=x.device)
        
        modulated_prompt = neutral_dynamic_prompt + self.deep_modulator_3c(torch.cat((dynamic_prompt, saliency_volume, suvr_volume), dim=1))
        
        # version 2: genuine mod. # test this first
        final_out = self.final_pred_head(torch.cat((out, self.fusion_layer(torch.cat((modulated_prompt, out), dim=1))), dim=1))

        out = self.final_act(final_out)

        return out

    
    def forward(self, x, covariate=None, roi_pred_dicts=None, sample_roi_mask = None):
        if covariate is not None and x.device != covariate.device:
            covariate = covariate.to(device=x.device)
        out, encoder_extractions, decoder_outs = super().forward(x, covariate)

        out, encoder_extractions, decoder_outs = super().forward(x, covariate)

        # Test out saliency map method
        out = self.forward_modulator_with_uq(x, out, covariate=covariate, roi_pred_dicts=roi_pred_dicts, sample_roi_mask=sample_roi_mask)

        # NOTE
        if not self.training and not self.embeddings_out:
            return out 
        
        projected_reprs = []
        projected_decoder_outs = []
        # for i, projection_head in enumerate(self.projection_heads):
        for i in range(self.depth):
            projection_head = self.projection_heads[i]
            proj_repr = projection_head(encoder_extractions[i])
            projected_reprs.append(proj_repr)


        final_proj_repr = self.final_projection_head(out)


        if self.embeddings_out:
            return out, projected_reprs, final_proj_repr, encoder_extractions
        
        if self.decoder_ds:
            return out, projected_reprs, final_proj_repr, projected_decoder_outs

        return out, projected_reprs, final_proj_repr


def train_dp(model, criterion, train_loader, validation_loader, epochs, lr, save_path="", cuda_id=0, pred_sample_file="", from_checkpoint=False, **kwargs):


    # sample_id2roi_predictions_tr = np.load(f"{os.getcwd()}/scripts/catboost_predictions/fold_{kwargs['fold_id']}/predictions_for_train_test.npy", allow_pickle=True)[0]
    # sample_id2roi_predictions_ts = np.load(f"{os.getcwd()}/scripts/catboost_predictions/fold_{kwargs['fold_id']}/predictions_for_test_test.npy", allow_pickle=True)[0]
    
    # sample_id2roi_predictions_tr = np.load(f"{os.getcwd()}/scripts/native_space_tau_roi_predictions/fold_{kwargs['fold_id']}/predictions_for_train_test.npy", allow_pickle=True)[0]
    # sample_id2roi_predictions_ts = np.load(f"{os.getcwd()}/scripts/native_space_tau_roi_predictions/fold_{kwargs['fold_id']}/predictions_for_test_test.npy", allow_pickle=True)[0]
    


    ##### ADNI & A4 Combined cross validation dataset 
    sample_id2roi_predictions_ts = data_util.load_json_dict(f"{os.getcwd()}/training_folds/adni_a4_first_scan_combined_folds/tau_prediction_lookups/formatted_fold_{kwargs['fold_id']}_predictions_for_test.json")
    sample_id2roi_predictions_tr = data_util.load_json_dict(f"{os.getcwd()}/training_folds/adni_a4_first_scan_combined_folds/tau_prediction_lookups/formatted_fold_{kwargs['fold_id']}_predictions_for_train.json")
    sample_id2roi_predictions = {**sample_id2roi_predictions_tr, **sample_id2roi_predictions_ts}

    ##### ADNI & A4 Combined hold out validation dataset
    # sample_id2roi_predictions_ts = data_util.load_json_dict(f"{os.getcwd()}/training_folds/adni_a4_first_scan_combined_folds/hold_out_aux_prediction_lookups/formatted_hold_out_predictions_for_test.json")
    # sample_id2roi_predictions_tr = data_util.load_json_dict(f"{os.getcwd()}/training_folds/adni_a4_first_scan_combined_folds/hold_out_aux_prediction_lookups/formatted_hold_out_predictions_for_train.json")
    # sample_id2roi_predictions = {**sample_id2roi_predictions_tr, **sample_id2roi_predictions_ts}

    criterion.gen_loss.batch_reduction = None

    rnc_loss = True # NOTE # NOTE
    start_epoch = 0
    val_iter = 5 # NOTE
    overfit_val_iter = 10
    grad_acc_batch = 1
    checkpoint_iter = val_iter
    total_batch_size = len(train_loader)
    cluster_loss = isinstance(train_loader.dataset, VolumeDataset.ClusterVolumeDataset) # NOTE
    # criterion.gen_weight = 0.

    if from_checkpoint:
        optimizer = kwargs["optimizer"]
        start_epoch = kwargs["start_epoch"] #+ 1
        # scheduler = kwargs["scheduler"] if kwargs["scheduler"] is not None else StepLR(optimizer, step_size=300, gamma=0.5)
        scheduler = kwargs["scheduler"] if kwargs["scheduler"] is not None else ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.2, verbose=True)

    if not from_checkpoint:
        optimizer = torch.optim.AdamW(model.parameters(), lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

    epoch_avg_losses, epoch_total_losses= [], []
    epoch_avg_pos_losses, epoch_total_pos_losses = [], []
    epoch_avg_neg_losses, epoch_total_neg_losses = [], []
    epoch_avg_gen_losses, epoch_avg_pred_contra_losses, epoch_total_pred_contra_losses = [], [], []
    epoch_total_gen_losses, epoch_avg_tcds_losses, epoch_total_tcds_losses = [], [], []

    val_maes, val_rses, val_rrmses, val_ssim, val_mapes, val_avg_roi_corrs,  = [], [], [], [], [], []
    val_roi_maes, val_roi_mapes, val_roi_mses, val_roi_wrrmses, val_roi_corrs = [], [], [], [], []
    
    val_pos_maes, val_pos_rses, val_pos_rrmses, val_pos_ssim, val_pos_mapes, val_avg_pos_roi_corrs  = [], [], [], [], [], []
    val_pos_roi_maes, val_pos_roi_mapes, val_pos_roi_mses, val_pos_roi_wrrmses, val_pos_roi_corrs = [], [], [], [], []
    
    val_neg_maes, val_neg_rses, val_neg_rrmses, val_neg_ssim, val_neg_mapes, val_avg_neg_roi_corrs  = [], [], [], [], [], []
    val_neg_roi_maes, val_neg_roi_mapes, val_neg_roi_mses, val_neg_roi_wrrmses, val_neg_roi_corrs = [], [], [], [], []
    val_roi_rses = []

    best_avg_corr = -torch.inf
    best_mape = torch.inf

    # NOTE 
    # epochs = start_epoch + 31

    for epoch in range(start_epoch, epochs):
        model.train(True)

        epoch_loss = 0
        num_samples = 0
        epoch_gen_loss, epoch_pred_contra_loss, epoch_tcds_loss = 0, 0, 0

        epoch_pos_loss = 0
        num_pos_samples = 0
        
        epoch_neg_loss = 0
        num_neg_samples = 0

        if epoch % val_iter == 0:
            logging.info(f"\nTraining epoch {epoch + 1}\n")

        optimizer.zero_grad()

        for batch_idx, batch_data in enumerate(train_loader):

            if batch_idx % 10 == 0:
                logging.info(f"Training batch [{batch_idx}/{total_batch_size}]")
            
            anchor_data, pos_data, neg_data = batch_data
            anchor_data = data_util.filter_for_holdout(*anchor_data)
            pos_data = data_util.filter_for_holdout(*pos_data)

            if cluster_loss :
                neg_data = [data_util.filter_for_holdout(*n) for n in neg_data]
            else:
                neg_data = data_util.filter_for_holdout(*neg_data)

            if isinstance(anchor_data, int):
                continue
                
            mri, tau, roi, abeta, tau_path = anchor_data
            abeta, covars = abeta
            pos_mri, pos_tau, pos_roi, pos_abeta, pos_tau_path = pos_data
            pos_abeta, pos_covars = pos_abeta

            # mri, tau, roi, abeta, tau_path = anchor_data
        
                
            logging.info(f"Anchor MRI Abeta: {abeta} | Covars: {covars} | Anchor Path: {tau_path}")

            optimizer.zero_grad() # NOTE

            # ROI predictions
            smpl_ids = [data_util.extract_id(pth) for pth in list(tau_path)]
            smpl_roi_pred_dicts = [sample_id2roi_predictions[sid] for sid in smpl_ids]

            model_outputs = model(mri, covars.to(device=mri.device), roi_pred_dicts=smpl_roi_pred_dicts, sample_roi_mask = roi)

            # with torch.no_grad():
            if not rnc_loss:
                # model_pos_outputs = model(pos_mri, pos_covars.to(device=pos_mri.device))
                pass
            
            if cluster_loss and not rnc_loss:
                outs = [] # N-1 samples
                proj_depth = len(model_pos_outputs[1])
                neg_projected_reprs = [[] for i in range(proj_depth)]
                for n in neg_data:
                    neg_mri, neg_tau, neg_roi, neg_abeta, neg_tau_path = n
                    neg_abeta, neg_covars = neg_abeta
                    _, lst_neg_projected_reprs, neg_final_repr = model(neg_mri, neg_covars.to(device=neg_mri.device))
                    for i in range(len(model_pos_outputs[1])):
                        neg_projected_reprs[i].append(lst_neg_projected_reprs[i])

                model_neg_outputs = None, neg_projected_reprs, neg_final_repr
            
            # Treat as a batch, chunk
            elif rnc_loss :
                """
                For RNC Loss: intermediate representation is expected to include:
                Tuple[
                    entire 'batch': [anchor_repr, pos_repr, neg_reprs], 
                    corresponding labels
                ]
                """
                # pseudo_batch = [model_outputs[1][-1], model_pos_outputs[1][-1]]
                pseudo_batch = [model_outputs[1][-1]]
                y_labels = [covars[:, -1].to(device=cuda_id)]

                intermediate_extractions = (torch.vstack(pseudo_batch), torch.vstack(y_labels))
                
            else:
                neg_mri, neg_tau, neg_roi, neg_abeta, neg_tau_path = neg_data
                neg_abeta, neg_covars = neg_abeta
            
            model_decoder_ds = model.module.decoder_ds if isinstance(model, DataParallel) else model.decoder_ds
            if rnc_loss:
                anchor_pred, anchor_projected_reprs, anchor_final_repr = model_outputs
                # _, pos_projected_reprs, pos_final_repr = model_pos_outputs
                pos_final_repr = torch.zeros(anchor_final_repr.size(), dtype=torch.float16, device=cuda_id)
                neg_final_repr = torch.zeros(pos_final_repr.size(), dtype=torch.float16, device=cuda_id) # dummy, not used
            elif model_decoder_ds:
                anchor_pred, anchor_projected_reprs, anchor_final_repr, decoder_projected_reprs = model_outputs
                _, pos_projected_reprs, pos_final_repr, _ = model_pos_outputs
                _, neg_projected_reprs, neg_final_repr, _ = model_neg_outputs
            else:
                anchor_pred, anchor_projected_reprs, anchor_final_repr = model_outputs
                model_pos_outputs = (None, [torch.zeros_like(p, device=mri.device) for p in anchor_projected_reprs], torch.zeros_like(anchor_final_repr, device=mri.device))
                _, pos_projected_reprs, pos_final_repr = model_pos_outputs
                model_neg_outputs = (None, [torch.zeros_like(p, device=mri.device) for p in anchor_projected_reprs], torch.zeros_like(anchor_final_repr, device=mri.device))
                _, neg_projected_reprs, neg_final_repr = model_neg_outputs
            
            if not rnc_loss:
                intermediate_extractions = anchor_projected_reprs, zip(pos_projected_reprs, neg_projected_reprs)
            
            if model_decoder_ds:
                final_representations = (decoder_projected_reprs, abeta, tau_path)
            else:
                final_representations = (anchor_final_repr, pos_final_repr, neg_final_repr)
            

            # prediction, target, final_representations, intermediate_extractions
            loss, gen_loss, pred_space_contra_loss, total_ds_contra_loss = criterion(anchor_pred, tau, roi, final_representations, intermediate_extractions)
            # loss, gen_loss, pred_space_contra_loss, total_ds_contra_loss = criterion(anchor_pred, tau, roi, final_representations, intermediate_extractions)
                
            # loss = loss + total_hl
            logging.info(f"Total Loss (weighted): {loss}")

            loss.backward()
            optimizer.step() # NOTE
            # scheduler.step(loss)
            # if (batch_idx + 1) % grad_acc_batch == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()
            #     scheduler.step()

            batch_loss = loss.item()

            epoch_loss += batch_loss
            epoch_gen_loss += gen_loss.sum()
            epoch_pred_contra_loss += pred_space_contra_loss
            # epoch_pred_contra_loss += total_hl # NOTE
            epoch_tcds_loss += total_ds_contra_loss
            num_samples += model_outputs[0].size(0)#anchor_pred.size(0)

            for b in range(mri.size(0)):
                if abeta[b] == 1:
                    epoch_pos_loss += gen_loss[b].item() + total_ds_contra_loss#.item()
                    # epoch_pos_loss += gen_loss.item() + total_ds_contra_loss.item()
                    num_pos_samples += 1#model_outputs[0].size(0)#anchor_pred.size(0)
                
                elif abeta[b] == 0:
                    epoch_neg_loss += gen_loss[b].item() + total_ds_contra_loss#.item()
                    # epoch_neg_loss += gen_loss.item() + total_ds_contra_loss.item()
                    num_neg_samples += 1#model_outputs[0].size(0) # anchor_pred.size(0)

            # torch.cuda.empty_cache()

            # NOTE
            # if batch_idx == 9:
            #     break

        # NOTE ensure BP on the remaining accumulated batches
        # optimizer.step()
        # optimizer.zero_grad()
        scheduler.step(epoch_loss/num_samples)
        # logging.info(f"LR: {scheduler.get_last_lr()}")
        
        epoch_avg_losses.append(epoch_loss / num_samples)
        epoch_total_losses.append(epoch_loss)
        
        epoch_avg_gen_losses.append(epoch_gen_loss / num_samples)
        epoch_total_gen_losses.append(epoch_gen_loss)
        
        epoch_avg_pred_contra_losses.append(epoch_pred_contra_loss / num_samples)
        epoch_total_pred_contra_losses.append(epoch_pred_contra_loss)

        epoch_avg_tcds_losses.append(epoch_tcds_loss / num_samples)
        epoch_total_tcds_losses.append(epoch_tcds_loss)
        epoch_avg_neg_losses.append(epoch_neg_loss / num_neg_samples)
        epoch_total_neg_losses.append(epoch_neg_loss)

        loss_graph((epoch_avg_losses, epoch_avg_pos_losses, epoch_avg_neg_losses), os.path.join(save_path, "train_average_loss"), labels=["Total", "Pos", "Neg"])
        loss_graph((epoch_total_losses, epoch_total_pos_losses, epoch_total_neg_losses), os.path.join(save_path, "train_total_loss"), labels=["Total", "Pos", "Neg"])
        loss_graph((epoch_avg_gen_losses, epoch_avg_pred_contra_losses, epoch_avg_tcds_losses), os.path.join(save_path, "train_average_component_losses"), labels=["Gen.", "Pred-Space Contra (weighted)", "tCDS (weighted)"])
        loss_graph((epoch_total_gen_losses, epoch_total_pred_contra_losses, epoch_total_tcds_losses), os.path.join(save_path, "train_total_component_losses"), labels=["Gen.", "Pred-Space Contra (weighted)", "tCDS (weighted)"])

        # Checkpointing
        if not os.path.exists(os.path.join(save_path, "checkpoints")):
            os.makedirs(os.path.join(save_path, "checkpoints"))
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint, os.path.join(save_path, "checkpoints", f'checkpoint_latest_epoch.pth'))
        if epoch % checkpoint_iter == 0:
            torch.save(checkpoint, os.path.join(save_path, "checkpoints", f'checkpoint_epoch_{epoch}.pth'))

        if epoch % val_iter == 0 :
            logging.info("Starting validation...")
            with torch.no_grad():
                val_save_path = os.path.join(save_path, f"{epoch}_output_samples")
                if not os.path.exists(val_save_path):
                    os.makedirs(val_save_path)
                contrastive_test_res = contrastive_test(model, validation_loader, criterion.gen_loss.roi_indices, criterion.gen_loss.roi_weights, save_path=val_save_path, cuda_id=cuda_id, pred_sample_file=pred_sample_file, with_train_loader=isinstance(train_loader.dataset, VolumeDataset.RegressionVolumeDataset), **kwargs)
                model_embeddings_out = model.module.embeddings_out if isinstance(model, DataParallel) else model.embeddings_out
                if model_embeddings_out:
                    general, pos_res, neg_res, (X_encodings_test, X_projections_test, tau_meta_test) = contrastive_test_res
                else:
                    general, pos_res, neg_res  = contrastive_test_res
                mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations, voxel_mae = general
                pos_mae, pos_mape, pos_rse, pos_rrmse, pos_ssim_error, pos_roi_maes, pos_roi_mapes, pos_roi_rses, pos_roi_wrrmses, pos_roi_correlations = pos_res
                neg_mae, neg_mape, neg_rse, neg_rrmse, neg_ssim_error, neg_roi_maes, neg_roi_mapes, neg_roi_rses, neg_roi_wrrmses, neg_roi_correlations = neg_res
                
                record_results(criterion, save_path, val_maes, val_rses, val_rrmses, val_ssim, val_mapes, val_avg_roi_corrs, val_roi_maes, val_roi_mapes, val_roi_wrrmses, val_roi_corrs, val_roi_rses, start_epoch, epoch, mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations, val_iter)
                
                if not os.path.join(save_path, "pos_metrics"):
                    os.makedirs(os.path.join(save_path, "pos_metrics"))
                record_results(criterion, os.path.join(save_path, "pos_metrics"), val_pos_maes, val_pos_rses, val_pos_rrmses, val_pos_ssim, val_pos_mapes, val_avg_pos_roi_corrs, val_pos_roi_maes, val_pos_roi_mapes, val_pos_roi_mses, val_pos_roi_wrrmses, val_pos_roi_corrs, start_epoch, epoch, pos_mae, pos_mape, pos_rse, pos_rrmse, pos_ssim_error, pos_roi_maes, pos_roi_mapes, pos_roi_rses, pos_roi_wrrmses, pos_roi_correlations, val_iter, metric_types="pos_")
                
                if not os.path.join(save_path, "neg_metrics"):
                    os.makedirs(os.path.join(save_path, "neg_metrics"))
                record_results(criterion, os.path.join(save_path, "neg_metrics"), val_neg_maes, val_neg_rses, val_neg_rrmses, val_neg_ssim, val_neg_mapes, val_avg_neg_roi_corrs, val_neg_roi_maes, val_neg_roi_mapes, val_neg_roi_mses, val_neg_roi_wrrmses, val_neg_roi_corrs, start_epoch, epoch, neg_mae, neg_mape, neg_rse, neg_rrmse, neg_ssim_error, neg_roi_maes, neg_roi_mapes, neg_roi_rses, neg_roi_wrrmses, neg_roi_correlations, val_iter, metric_types="neg_")

                # Update weights for mse
                logging.info(f"Voxel mape: {voxel_mae} | avg: {torch.mean(voxel_mae)} | min: {torch.min(voxel_mae)} | max: {torch.max(voxel_mae)}")
                if criterion.gen_loss.voxel_wise:
                    new_weights = criterion.gen_loss.calculate_new_voxel_weights(voxel_mae/ 100, criterion.gen_loss.voxel_weights, with_update=False)
                    # criterion.gen_loss.set_voxel_wise_mask(new_weights, criterion.gen_loss.voxel_weights)
                    criterion.gen_loss.voxel_weights = new_weights
                else:
                    new_weights = criterion.gen_loss.calculate_new_weights(roi_mapes / 100, with_update=True)
                logging.info(f"Updated weights: {new_weights}, weights avg: {torch.mean(new_weights)}, weights max: {torch.max(new_weights)}")

                    
            metric_graph(np.arange(0, (epoch-start_epoch)+1, val_iter), val_maes, "Mean Absolute Error", "Epochs", "MAE", os.path.join(save_path, "val_MAE"))
            metric_graph(np.arange(0, (epoch-start_epoch)+1, val_iter), val_mapes, "Mean Absolute Percent Error", "Epochs", "MAPE", os.path.join(save_path, "val_MAPE"))
            metric_graph(np.arange(0, (epoch-start_epoch)+1, val_iter), val_rses, "Relative Squared Error", "Epochs", "RSE", os.path.join(save_path, "val_RSE"))
            metric_graph(np.arange(0, (epoch-start_epoch)+1, val_iter), val_rrmses, "RRMSE", "Epochs", "RRMSE", os.path.join(save_path, "val_RRMSE"))
            metric_graph(np.arange(0, (epoch-start_epoch)+1, val_iter), val_ssim, "SSIM", "Epochs", "SSIM", os.path.join(save_path, "val_SSIM"))
            metric_graph(np.arange(0, (epoch-start_epoch)+1, val_iter), val_avg_roi_corrs, "Averaged ROI Corr Mean", "Epochs", "Average ROI Corr Mean", os.path.join(save_path, "val_avrgd_ROI_corr_mean"))
            plot_mae_progression_chart(np.array(val_roi_maes), np.arange(0, (epoch-start_epoch)+1, val_iter), os.path.join(save_path, "val_ROI_MAEs_progression"))
            plot_mae_progression_chart(np.array(val_roi_mapes), np.arange(0, (epoch-start_epoch)+1, val_iter), os.path.join(save_path, "val_ROI_MAPEs_progression"), name="MAPE")
            # plot_mae_progression_chart(np.array(val_roi_rmses), np.arange(0, (epoch-start_epoch)+1, val_iter), os.path.join(save_path, "val_ROI_RMSEs_progression"), name="RMSE")
            plot_mae_progression_chart(np.array(val_roi_rses), np.arange(0, (epoch-start_epoch)+1, val_iter), os.path.join(save_path, "val_ROI_RSEs_progression"), name="RSE")
            plot_mae_progression_chart(np.array(val_roi_wrrmses), np.arange(0, (epoch-start_epoch)+1, val_iter), os.path.join(save_path, "val_ROI_RRMSEs_progression"), name="RRMSE")
            boxplot_roi_value_progression(np.array(val_roi_corrs), np.arange(0, (epoch-start_epoch)+1, val_iter), "Correlation", os.path.join(save_path, "val_ROI_corr"))
        
            # models_save_path = os.path.join(save_path, "models")
            # if not os.path.exists(models_save_path):
            #     os.makedirs(models_save_path)

            if mape < best_mape:
                best_mape = mape 
                logging.info(f"Lowest MAPE so far: Epoch {epoch}")

            if np.nanmean(roi_correlations).item() > best_avg_corr:
                best_avg_corr = np.nanmean(roi_correlations).item()
                logging.info(f"Highest ROI Averaged Correlations so far: Epoch {epoch}")

        if epoch !=0 and epoch > 29 and epoch % overfit_val_iter == 0:
            with torch.no_grad():
                logging.info(f"Starting In-Sample Validation...")
                contrastive_test_res = contrastive_test(model, train_loader, criterion.gen_loss.roi_indices, criterion.gen_loss.roi_weights, save_path=val_save_path, cuda_id=cuda_id, pred_sample_file=pred_sample_file, with_train_loader = not kwargs.get("with_test_loader", False), in_sample_test=True, **kwargs)
                general, pos_res, neg_res  = contrastive_test_res
                mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations, _ = general
                pos_mae, pos_mape, pos_rse, pos_rrmse, pos_ssim_error, pos_roi_maes, pos_roi_mapes, pos_roi_rses, pos_roi_wrrmses, pos_roi_correlations = pos_res
                neg_mae, neg_mape, neg_rse, neg_rrmse, neg_ssim_error, neg_roi_maes, neg_roi_mapes, neg_roi_rses, neg_roi_wrrmses, neg_roi_correlations = neg_res
                logging.info(f"In-Sample Validation Results:")
                print_metrics(criterion, mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations)
                logging.info(f"In-Sample Validation Pos. Sample Results:")
                print_metrics(criterion, pos_mae, pos_mape, pos_rse, pos_rrmse, pos_ssim_error, pos_roi_maes, pos_roi_mapes, pos_roi_rses, pos_roi_wrrmses, pos_roi_correlations, metric_types="pos_")
                logging.info(f"In-Sample Validation Neg. Sample Results:")
                print_metrics(criterion, neg_mae, neg_mape, neg_rse, neg_rrmse, neg_ssim_error, neg_roi_maes, neg_roi_mapes, neg_roi_rses, neg_roi_wrrmses, neg_roi_correlations, metric_types="neg_")



def record_results(criterion, save_path, val_maes, val_rses, val_rrmses, val_ssim, val_mapes, val_avg_roi_corrs, val_roi_maes, val_roi_mapes, val_roi_wrrmses, val_roi_corrs, val_roi_rses, start_epoch, epoch, mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations, val_iter, metric_types=""):
    print_metrics(criterion, mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations, metric_types)

    val_maes.append(mae)
    val_mapes.append(mape)
    val_rses.append(rse)
    val_rrmses.append(rrmse)
    val_ssim.append(ssim_error)
    val_roi_maes.append(roi_maes.detach().cpu().numpy())
    val_roi_mapes.append(roi_mapes.detach().cpu().numpy())
                # val_roi_mses.append(roi_mses.detach().cpu().numpy())
                # val_roi_rmses.append(roi_rmses.detach().cpu().numpy())
    val_roi_rses.append(roi_rses.detach().cpu().numpy())
    val_roi_wrrmses.append(roi_wrrmses.detach().cpu().numpy())
    val_roi_corrs.append(np.nan_to_num(roi_correlations, 0))
                # val_roi_corrs.append(torch.nan_to_num(roi_correlations, 0).detach().cpu().numpy())
    val_avg_roi_corrs.append(np.mean(np.nan_to_num(roi_correlations, 0)).item())
                # val_avg_roi_corrs.append(torch.mean(torch.nan_to_num(roi_correlations, 0)).item())

                # Save metrics
    def put_metrics(roi_corrs, roi_mapes, roi_maes, avg_corr, roi_rse, roi_rrmses, mae, mape, epoch, save_path):
        val_metrics_save_path = os.path.join(save_path, "validation_metric_results")
        if not os.path.exists(val_metrics_save_path):
            os.makedirs(val_metrics_save_path)
                        # init empty csv files for the metrics
            pd.DataFrame(index=None).to_csv(os.path.join(val_metrics_save_path, "roi_corr.csv"))
            pd.DataFrame(index=None).to_csv(os.path.join(val_metrics_save_path, "roi_mapes.csv"))
            pd.DataFrame(index=None).to_csv(os.path.join(val_metrics_save_path, "roi_maes.csv"))
            pd.DataFrame(index=None).to_csv(os.path.join(val_metrics_save_path, "roi_rse.csv"))
            pd.DataFrame(index=None).to_csv(os.path.join(val_metrics_save_path, "roi_rrmses.csv"))
            pd.DataFrame(index=None).to_csv(os.path.join(val_metrics_save_path, "mape.csv"))
            pd.DataFrame(index=None).to_csv(os.path.join(val_metrics_save_path, "mae.csv"))
            pd.DataFrame(index=None).to_csv(os.path.join(val_metrics_save_path, "avg_corr.csv"))
                    

        def append_col(df_fname, metric_val, epoch):
            if type(metric_val) == torch.Tensor:
                metric_val = [metric_val.detach().cpu().numpy()]
            elif type(metric_val) != np.ndarray:
                metric_val = np.array([metric_val])
                        
            df = pd.read_csv(df_fname)
                        
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])

            df[f"epoch_{epoch}"] = metric_val 
            df.to_csv(df_fname, index=False)

        append_col(os.path.join(val_metrics_save_path, "roi_corr.csv"), roi_corrs, epoch)
        append_col(os.path.join(val_metrics_save_path, "roi_mapes.csv"), roi_mapes, epoch)
        append_col(os.path.join(val_metrics_save_path, "roi_maes.csv"), roi_maes, epoch)
        append_col(os.path.join(val_metrics_save_path, "avg_corr.csv"), avg_corr, epoch)
        append_col(os.path.join(val_metrics_save_path, "roi_rse.csv"), roi_rse, epoch)
        append_col(os.path.join(val_metrics_save_path, "roi_rrmses.csv"), roi_rrmses, epoch)
        append_col(os.path.join(val_metrics_save_path, "mape.csv"), mape, epoch)
        append_col(os.path.join(val_metrics_save_path, "mae.csv"), mae, epoch)

    put_metrics(roi_correlations, roi_mapes.detach().cpu().numpy(), roi_maes.detach().cpu().numpy(), np.mean(np.nan_to_num(roi_correlations, 0)).item(), roi_rses.detach().cpu().numpy(), roi_wrrmses.detach().cpu().numpy(), mae, mape, epoch, save_path)

    metric_graph(np.arange(0, (epoch-start_epoch)+1, val_iter), val_maes, "Mean Absolute Error", "Epochs", "MAE", os.path.join(save_path, "val_MAE"))
    metric_graph(np.arange(0, (epoch-start_epoch)+1, val_iter), val_mapes, "Mean Absolute Percent Error", "Epochs", "MAPE", os.path.join(save_path, "val_MAPE"))
    metric_graph(np.arange(0, (epoch-start_epoch)+1, val_iter), val_rses, "Relative Squared Error", "Epochs", "RSE", os.path.join(save_path, "val_RSE"))
    metric_graph(np.arange(0, (epoch-start_epoch)+1, val_iter), val_rrmses, "RRMSE", "Epochs", "RRMSE", os.path.join(save_path, "val_RRMSE"))
    metric_graph(np.arange(0, (epoch-start_epoch)+1, val_iter), val_ssim, "SSIM", "Epochs", "SSIM", os.path.join(save_path, "val_SSIM"))
    metric_graph(np.arange(0, (epoch-start_epoch)+1, val_iter), val_avg_roi_corrs, "Averaged ROI Corr Mean", "Epochs", "Average ROI Corr Mean", os.path.join(save_path, "val_avrgd_ROI_corr_mean"))
    plot_mae_progression_chart(np.array(val_roi_maes), np.arange(0, (epoch-start_epoch)+1, val_iter), os.path.join(save_path, "val_ROI_MAEs_progression"))
    plot_mae_progression_chart(np.array(val_roi_mapes), np.arange(0, (epoch-start_epoch)+1, val_iter), os.path.join(save_path, "val_ROI_MAPEs_progression"), name="MAPE")
    # plot_mae_progression_chart(np.array(val_roi_rmses), np.arange(0, (epoch-start_epoch)+1, val_iter), os.path.join(save_path, "val_ROI_RMSEs_progression"), name="RMSE")
    plot_mae_progression_chart(np.array(val_roi_rses), np.arange(0, (epoch-start_epoch)+1, val_iter), os.path.join(save_path, "val_ROI_RSEs_progression"), name="RSE")
    plot_mae_progression_chart(np.array(val_roi_wrrmses), np.arange(0, (epoch-start_epoch)+1, val_iter), os.path.join(save_path, "val_ROI_RRMSEs_progression"), name="RRMSE")
    boxplot_roi_value_progression(np.array(val_roi_corrs), np.arange(0, (epoch-start_epoch)+1, val_iter), "Correlation", os.path.join(save_path, "val_ROI_corr"))

def print_metrics(criterion, mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations, metric_types=""):
    logging.info(f"{metric_types}Validation results:\n")
    logging.info(f"{metric_types}MAE:{mae}")
    logging.info(f"{metric_types}MAPE:{mape}")
    logging.info(f"{metric_types}RSE:{rse}")
    logging.info(f"{metric_types}RRMSE:{rrmse}")
    logging.info(f"{metric_types}SSIM Error:{ssim_error}")
    logging.info(f"{metric_types}ROI MAEs:{roi_maes}")
    logging.info(f"{metric_types}ROI MAPEs:{roi_mapes}")
                # logging.info(f"ROI MSEs:{roi_mses}")
                # logging.info(f"ROI RMSEs:{roi_rmses}")
    logging.info(f"{metric_types}ROI RSEs:{roi_rses}")
    logging.info(f"{metric_types}ROI Weighted RRMSEs: {roi_wrrmses}")
    logging.info(f"{metric_types}ROI correlations: {roi_correlations}")
    lowest_corr_idx = np.argmin(roi_correlations)
    highest_corr_idx = np.argmax(roi_correlations)
    logging.info(f"\tLowest correlation (ROI {criterion.gen_loss.roi_indices[lowest_corr_idx]}): {roi_correlations[lowest_corr_idx]}")
    logging.info(f"\tHighest correlation (ROI {criterion.gen_loss.roi_indices[highest_corr_idx]}): {roi_correlations[highest_corr_idx]}")
    logging.info(f"\t{metric_types}ROI correlations average: {np.mean(np.nan_to_num(roi_correlations, 0)).item()}")

def contrastive_test(model, test_loader, roi_indices, roi_weights, save_path="", cuda_id=0, pred_sample_file="", with_train_loader=False, **kwargs):
    ### NOTE
    #### Longitdunial ADNI predictions from ADNI & A4 cross-sectional dataset
    if kwargs.get('roi_vecs_dict', None) is not None:
        sample_id2roi_predictions = kwargs.get('roi_vecs_dict')
    else:
        sample_id2roi_predictions_ts = data_util.load_json_dict(f"{os.getcwd()}/scripts/CatBoostUQ_longitudinal_ADNI_predictions/CatBoostUQ_longitudinal_predictions/CatBoostUQ_predictions_for_unseen_longitudinal_ADNI.json")
        sample_id2roi_predictions_tr = {}#data_util.load_json_dict(f"{os.getcwd()}/training_folds/adni_a4_first_scan_combined_folds/hold_out_aux_prediction_lookups/formatted_hold_out_predictions_for_train.json")
        sample_id2roi_predictions = {**sample_id2roi_predictions_tr, **sample_id2roi_predictions_ts}
    
    model.eval()
    if isinstance(model, DataParallel):
        # model.module.set_training(False)
        model = model.module

    model.set_training(False)

    num_samples = 0
    mae, mape, rse, rrmse = 0, 0, 0, 0
    mape_smp_count = 0
    roi_maes = torch.zeros(len(roi_indices), device=cuda_id)
    roi_mapes = torch.zeros(len(roi_indices), device=cuda_id)
    # roi_rmses = torch.zeros(len(roi_indices), device=cuda_id)
    roi_rses = torch.zeros(len(roi_indices), device=cuda_id)
    roi_wrrmses = torch.zeros(len(roi_indices), device=cuda_id)
    roi_nonnan_voxels = torch.zeros(len(roi_indices), device=cuda_id)
    
    num_pos_samples = 0
    pos_mae, pos_mape, pos_rse, pos_rrmse = 0, 0, 0, 0
    mape_pos_smp_count = 0
    pos_roi_maes = torch.zeros(len(roi_indices), device=cuda_id)
    pos_roi_mapes = torch.zeros(len(roi_indices), device=cuda_id)
    # roi_rmses = torch.zeros(len(roi_indices), device=cuda_id)
    pos_roi_rses = torch.zeros(len(roi_indices), device=cuda_id)
    pos_roi_wrrmses = torch.zeros(len(roi_indices), device=cuda_id)
    pos_roi_nonnan_voxels = torch.zeros(len(roi_indices), device=cuda_id)
    
    num_neg_samples = 0
    neg_mae, neg_mape, neg_rse, neg_rrmse = 0, 0, 0, 0
    mape_neg_smp_count = 0
    neg_roi_maes = torch.zeros(len(roi_indices), device=cuda_id)
    neg_roi_mapes = torch.zeros(len(roi_indices), device=cuda_id)
    # roi_rmses = torch.zeros(len(roi_indices), device=cuda_id)
    neg_roi_rses = torch.zeros(len(roi_indices), device=cuda_id)
    neg_roi_wrrmses = torch.zeros(len(roi_indices), device=cuda_id)
    neg_roi_nonnan_voxels = torch.zeros(len(roi_indices), device=cuda_id)

    ssim_metric = SSIMMetric(spatial_dims=3, data_range=torch.tensor([1.0], device=cuda_id))
    pos_ssim_metric = SSIMMetric(spatial_dims=3, data_range=torch.tensor([1.0], device=cuda_id))
    neg_ssim_metric = SSIMMetric(spatial_dims=3, data_range=torch.tensor([1.0], device=cuda_id))
    roi_corr_metric = RoiCorrMetric(roi_indices)
    pos_roi_corr_metric = RoiCorrMetric(roi_indices)
    neg_roi_corr_metric = RoiCorrMetric(roi_indices)
    roi_correlations = torch.zeros(len(roi_indices), device=cuda_id)

    voxel_mae = torch.zeros(128, 128, 128, device=cuda_id)
    voxel_mape = torch.zeros(128, 128, 128, device=cuda_id)

    samples_saved, save_representatives = False, len(save_path) > 0 #and not with_train_loader

    model_embeddings_out = model.module.embeddings_out if isinstance(model, DataParallel) else model.embeddings_out
    if model_embeddings_out:
        # Collect the test features
        model_depth = model.get_depth()
        X_encodings_test, X_projections_test, tau_meta_test = [[] for i in range(model_depth)], [[] for i in range(model_depth)], []
        target_values_df = pd.read_csv(model.test_tau_targets_csv)

    # for batch_idx, (mri_volume, tau_volume, roi, abeta, tau_path) in enumerate(test_loader):
    for batch_idx, values in enumerate(test_loader):
        
        if with_train_loader :
            values, _, _ = values

        mri_volume, tau_volume, roi, abeta, tau_path = values
        abeta, covars = abeta

        logging.info(f"sample: {tau_path}, abeta: {abeta}, covars: {covars}")

        smpl_ids = [data_util.extract_id(pth) for pth in list(tau_path)]
        smpl_roi_pred_dicts = [sample_id2roi_predictions[sid] for sid in smpl_ids]
        pred = model(mri_volume, covars.to(device=mri_volume.device), roi_pred_dicts=smpl_roi_pred_dicts, sample_roi_mask = roi)

        if model.embeddings_out:
            pred, projected_reprs, final_proj_repr, intermediate_reprs = pred 

        diff = pred - tau_volume
        mae += torch.mean(torch.abs(diff))
        # NOTE 
        raw_mape = torch.abs(diff / tau_volume)
        nr_mape = torch.where(torch.abs(tau_volume) > 1e-08, torch.abs((tau_volume - pred) / tau_volume), torch.nan)
        mape += torch.nansum(nr_mape * 100, dim=(-3, -2, -1)).sum()
        
        # RSE
        gt_mean = torch.mean(tau_volume, dim=(-3, -2, -1))
        squared_error_num = torch.sum(torch.square(tau_volume - pred), dim=(-3, -2, -1))
        squared_error_den = torch.sum(torch.square(tau_volume - gt_mean.view(-1, 1, 1, 1, 1)), dim=(-3, -2, -1))
        nr_rse = squared_error_num / squared_error_den
        rse += torch.mean(nr_rse)
        # RRMSE
        num = torch.sum(torch.square(tau_volume - pred), dim=(-3, -2, -1))
        den = torch.sum(torch.square(tau_volume), dim=(-3, -2, -1))
        nr_rrmse = torch.sqrt(num / den)
        rrmse += torch.nanmean(nr_rrmse)

        # SSIM
        ssim_metric(y_pred=pred, y=tau_volume)
        for b in range(mri_volume.size(0)):
            if abeta[b] == 1:
                pos_ssim_metric(y_pred=pred[b][None, :], y=tau_volume[b][None, :])
            elif abeta[b] == 0:
                neg_ssim_metric(y_pred=pred[b][None, :], y=tau_volume[b][None, :])

        # ROI Correlations
        roi_corr_metric.acc_roi_corr(pred, tau_volume, roi)
        for b in range(mri_volume.size(0)):
            if abeta[0] == 1:
                pos_roi_corr_metric.acc_roi_corr(pred[b], tau_volume[b], roi[b])
                pos_roi_corr_metric.acc_sample_ids([tau_path[b]])
            else:
                neg_roi_corr_metric.acc_roi_corr(pred[b], tau_volume[b], roi[b])
                neg_roi_corr_metric.acc_sample_ids([tau_path[b]])
        # keept track of the sample ids
        roi_corr_metric.acc_sample_ids(tau_path)

        # MAE, MSE, MAPE in ROIs
        temp_roi_maes, temp_roi_mapes, temp_roi_rses, temp_roi_wrrmses, temp_roi_nonnan_voxels = calc_roi_metrics(roi_indices, roi_weights, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_nonnan_voxels, tau_volume, roi, pred, diff, raw_mape)
        
        roi_maes += temp_roi_maes
        roi_mapes += temp_roi_mapes
        roi_rses += temp_roi_rses
        roi_wrrmses += temp_roi_wrrmses
        roi_nonnan_voxels += temp_roi_nonnan_voxels
        
        num_samples += mri_volume.size(0)

        for b in range(mri_volume.size(0)):
            if abeta[b] == 1:
                pos_mae += torch.mean(torch.abs(diff))
                pos_mape += torch.nansum(nr_mape * 100, dim=(-3, -2, -1)).sum()
                mape_pos_smp_count += (torch.prod(torch.tensor(nr_mape.size())) - torch.isnan(nr_mape).sum()).item()
                pos_rse += torch.mean(nr_rse)
                pos_rrmse += torch.nanmean(nr_rrmse)
                # ROIs
                pos_roi_maes += temp_roi_maes
                pos_roi_mapes += temp_roi_mapes
                pos_roi_rses += temp_roi_rses
                pos_roi_wrrmses += temp_roi_wrrmses
                pos_roi_nonnan_voxels += temp_roi_nonnan_voxels

                num_pos_samples += 1
                
            elif abeta[b] == 0:
                neg_mae += torch.mean(torch.abs(diff))
                neg_mape += torch.nansum(nr_mape * 100, dim=(-3, -2, -1)).sum()
                mape_neg_smp_count += (torch.prod(torch.tensor(nr_mape.size())) - torch.isnan(nr_mape).sum()).item()
                neg_rse += torch.mean(nr_rse)
                neg_rrmse += torch.nanmean(nr_rrmse)
                # ROIs
                neg_roi_maes += temp_roi_maes
                neg_roi_mapes += temp_roi_mapes
                neg_roi_rses += temp_roi_rses
                neg_roi_wrrmses += temp_roi_wrrmses
                neg_roi_nonnan_voxels += temp_roi_nonnan_voxels

                num_neg_samples += 1
        


        # NOTE early stop to test


    ssim_error = ssim_metric.aggregate().item()
    ssim_metric.reset()
    mae /= num_samples
    mape = mape / mape_smp_count
    rse /= num_samples
    rrmse /= num_samples
    roi_maes /= num_samples
    roi_mapes = (roi_mapes * 100) / roi_nonnan_voxels # convert to percentage
    roi_correlations = roi_corr_metric.calc_roi_corr()
    # save the matrices
    if "in_sample_test" in kwargs and kwargs["in_sample_test"]:
        pass
    else: 
        roi_corr_metric.save_matrices(save_path)
    roi_rses /= num_samples
    roi_wrrmses /= num_samples 
    
    pos_ssim_error = pos_ssim_metric.aggregate().item()
    pos_ssim_metric.reset()
    pos_mae /= num_pos_samples
    pos_mape = pos_mape / mape_pos_smp_count
    pos_rse /= num_pos_samples
    pos_rrmse /= num_pos_samples
    pos_roi_maes /= num_pos_samples
    pos_roi_mapes = (pos_roi_mapes * 100) / pos_roi_nonnan_voxels # convert to percentage
    pos_roi_rses /= num_pos_samples
    pos_roi_wrrmses /= num_pos_samples 
    pos_roi_correlations = pos_roi_corr_metric.calc_roi_corr()
    if "in_sample_test" in kwargs and kwargs["in_sample_test"]:
        pass
    else: 
        pos_roi_corr_metric.save_matrices(save_path, type="pos_")
    
    neg_ssim_error = neg_ssim_metric.aggregate().item()
    neg_ssim_metric.reset()
    neg_mae /= num_neg_samples
    neg_mape = neg_mape / mape_neg_smp_count
    neg_rse /= num_neg_samples
    neg_rrmse /= num_neg_samples
    neg_roi_maes /= num_neg_samples
    neg_roi_mapes = (neg_roi_mapes * 100) / neg_roi_nonnan_voxels # convert to percentage
    neg_roi_rses /= num_neg_samples
    neg_roi_wrrmses /= num_neg_samples 
    neg_roi_correlations = neg_roi_corr_metric.calc_roi_corr()
    if "in_sample_test" in kwargs and kwargs["in_sample_test"]:
        pass
    else: 
        neg_roi_corr_metric.save_matrices(save_path, type="neg_")

    model_embeddings_out = model.module.embeddings_out if isinstance(model, DataParallel) else model.embeddings_out
    if model_embeddings_out:
        return (mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations), \
            (pos_mae, pos_mape, pos_rse, pos_rrmse, pos_ssim_error, pos_roi_maes, pos_roi_mapes, pos_roi_rses, pos_roi_wrrmses, pos_roi_correlations), \
            (neg_mae, neg_mape, neg_rse, neg_rrmse, neg_ssim_error, neg_roi_maes, neg_roi_mapes, neg_roi_rses, neg_roi_wrrmses, neg_roi_correlations), \
            (X_encodings_test, X_projections_test, tau_meta_test)


    return (mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations, 100* voxel_mape / num_samples), \
            (pos_mae, pos_mape, pos_rse, pos_rrmse, pos_ssim_error, pos_roi_maes, pos_roi_mapes, pos_roi_rses, pos_roi_wrrmses, pos_roi_correlations), \
            (neg_mae, neg_mape, neg_rse, neg_rrmse, neg_ssim_error, neg_roi_maes, neg_roi_mapes, neg_roi_rses, neg_roi_wrrmses, neg_roi_correlations)

def calc_roi_metrics(roi_indices, roi_weights, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_nonnan_voxels, tau_volume, roi, pred, diff, raw_mape):
    roi_mask = torch.zeros(roi.size(), device=roi.get_device())
    roi_bool_mask = torch.zeros(roi.size(), device=roi.get_device(), dtype=torch.bool)
    temp_roi_maes = torch.zeros(len(roi_indices), device=roi.get_device())
    temp_roi_mapes = torch.zeros(len(roi_indices), device=roi.get_device())
    temp_roi_rses = torch.zeros(len(roi_indices), device=roi.get_device())
    temp_roi_wrrmses = torch.zeros(len(roi_indices), device=roi.get_device())
    temp_roi_nonnan_voxels = torch.zeros(len(roi_indices), device=roi.get_device())
    for i, idx in enumerate(roi_indices):
        roi_mask[:] = 0 # reset mask
        roi_bool_mask[:] = False
        roi_mask[roi == idx] = 1 # roi_weights[i]
        roi_bool_mask[roi == idx] = True
        roi_mask_size = torch.count_nonzero(roi_mask, dim=(-3, -2, -1))

            # logging.info(f"ROI {idx} mask size: {torch.count_nonzero(roi_mask)}")
            # calculate average MAE in ROI
        nr_roi_mae = torch.sum(torch.abs(diff) * roi_mask, dim=(-3, -2, -1)) / roi_mask_size # batched
        temp_roi_maes[i] += torch.sum(nr_roi_mae)
            # calculate average MAPE in ROI
        roi_raw_mape = raw_mape[roi_bool_mask]
        temp_roi_mapes[i] = torch.nansum(roi_raw_mape)
        temp_roi_nonnan_voxels[i] = (torch.count_nonzero(roi_bool_mask) - torch.count_nonzero(torch.isnan(roi_raw_mape))).item()

            # calculate average RRMSE in ROI
        num = torch.sum(roi_mask * torch.square(diff), dim=(-3, -2, -1))
        den = torch.sum(roi_mask * torch.square(tau_volume), dim=(-3, -2, -1))
        weighted_roi_squared_err = num / den 
        temp_roi_wrrmses[i] = torch.sum(torch.sqrt(weighted_roi_squared_err))
        # calculate average RSE in ROI 
        gt_mean = torch.sum(roi_mask * tau_volume, dim=(-3, -2, -1)) / roi_mask_size
        roi_se_num = torch.sum(roi_mask * torch.square(tau_volume - pred), dim=(-3, -2, -1))
        roi_se_den = torch.sum(roi_mask * torch.square(tau_volume - gt_mean.view(-1, 1, 1, 1, 1)), dim=(-3, -2, -1))
        nr_roi_rse = roi_se_num / roi_se_den
        temp_roi_rses[i] += torch.sum(nr_roi_rse)
    # wrrmses
    return temp_roi_maes, temp_roi_mapes, temp_roi_rses, temp_roi_wrrmses, temp_roi_nonnan_voxels


def test(model, test_loader, roi_indices, roi_weights, save_path="", cuda_id=0, pred_sample_file=""):

    model.eval()
    model.set_training(False)

    num_samples = 0
    mae, mape, rse, rrmse = 0, 0, 0, 0
    mape_smp_count = 0
    roi_maes = torch.zeros(len(roi_indices), device=cuda_id)
    roi_mapes = torch.zeros(len(roi_indices), device=cuda_id)
    # roi_rmses = torch.zeros(len(roi_indices), device=cuda_id)
    roi_rses = torch.zeros(len(roi_indices), device=cuda_id)
    roi_wrrmses = torch.zeros(len(roi_indices), device=cuda_id)
    roi_nonnan_voxels = torch.zeros(len(roi_indices), device=cuda_id)

    ssim_metric = SSIMMetric(spatial_dims=3, data_range=torch.tensor([1.0], device=cuda_id))
    roi_corr_metric = RoiCorrMetric(roi_indices)
    roi_correlations = torch.zeros(len(roi_indices), device=cuda_id)

    samples_saved, save_representatives = False, True

    for batch_idx, (mri_volume, tau_volume, roi, tau_path) in enumerate(test_loader):

        smpl_ids = ["/".join(pth.split("/")[8:10]) for pth in list(tau_path)]
        smpl_roi_pred_dicts = None #[sample_id2roi_predictions[sid] for sid in smpl_ids]
        pred = model(mri_volume, roi_pred_dicts=smpl_roi_pred_dicts)

        diff = pred - tau_volume
        mae += torch.mean(torch.abs(diff))


        raw_mape = torch.abs(diff / tau_volume)
        nr_mape = torch.where(tau_volume != 0, torch.abs((tau_volume - pred) / tau_volume), torch.nan)
        mape += torch.nansum(nr_mape * 100, dim=(-3, -2, -1)).sum()
        mape_smp_count += (torch.prod(torch.tensor(nr_mape.size())) - torch.isnan(nr_mape).sum()).item()
        
        # RSE
        gt_mean = torch.mean(tau_volume, dim=(-3, -2, -1))
        squared_error_num = torch.sum(torch.square(tau_volume - pred), dim=(-3, -2, -1))
        squared_error_den = torch.sum(torch.square(tau_volume - gt_mean.view(-1, 1, 1, 1, 1)), dim=(-3, -2, -1))
        nr_rse = squared_error_num / squared_error_den
        rse += torch.mean(nr_rse)
        # RRMSE
        num = torch.sum(torch.square(tau_volume - pred), dim=(-3, -2, -1))
        den = torch.sum(torch.square(tau_volume), dim=(-3, -2, -1))
        nr_rrmse = torch.sqrt(num / den)
        rrmse += torch.nanmean(nr_rrmse)
        
        # SSIM
        ssim_metric(y_pred=pred, y=tau_volume)

        # ROI Correlations
        roi_corr_metric.acc_roi_corr(pred, tau_volume, roi)
        # keept track of the sample ids
        # logging.info(tau_path)
        roi_corr_metric.acc_sample_ids(tau_path)
        # batch_correlations = roi_corr_metric.comp_roi_corr(pred, tau_volume, roi)
        # roi_correlations += batch_correlations

        # MAE, MSE, MAPE in ROIs
        roi_mask = torch.zeros(roi.size(), device=roi.get_device())
        roi_bool_mask = torch.zeros(roi.size(), device=roi.get_device(), dtype=torch.bool)
        for i, idx in enumerate(roi_indices):


            roi_mask[:] = 0 # reset mask
            roi_bool_mask[:] = False
            roi_mask[roi == idx] = roi_weights[i]
            roi_bool_mask[roi == idx] = True
            roi_mask_size = torch.count_nonzero(roi_mask, dim=(-3, -2, -1))

            # if any(roi_mask_size == 0):
            #     logging.info(f"ROI [{idx}] Mask Size: {roi_mask_size}")
            #     logging.info(f"Roi mask nonzero shape: {roi_mask_size.size()}")
            #     logging.info(f"\tmask size: {roi_mask_size.t()}")#: {tau_path.t()}")
            #     faulty_inds = (roi_mask_size == 0).nonzero()[:, 0]
            #     for j, faulty_idx in enumerate(faulty_inds):
            #         logging.info(f"\tFaulty sample: {tau_path[faulty_idx]} in ROI {idx}")
            #         assert roi_mask_size[faulty_idx, 0] == 0, "ROI marked as faulty but size isn't 0"

            # logging.info(f"ROI {idx} mask size: {torch.count_nonzero(roi_mask)}")
            # calculate average MAE in ROI
            nr_roi_mae = torch.sum(torch.abs(diff) * roi_mask, dim=(-3, -2, -1)) / roi_mask_size # batched
            roi_maes[i] += torch.sum(nr_roi_mae)
            # calculate average MAPE in ROI
            roi_raw_mape = raw_mape[roi_bool_mask]
            roi_mapes[i] = torch.nansum(roi_raw_mape)
            roi_nonnan_voxels[i] = (torch.count_nonzero(roi_bool_mask) - torch.count_nonzero(torch.isnan(roi_raw_mape))).item()
            # calculate average RRMSE in ROI
            num = torch.sum(roi_mask * torch.square(diff), dim=(-3, -2, -1))
            den = torch.sum(roi_mask * torch.square(tau_volume), dim=(-3, -2, -1))
            weighted_roi_squared_err = num / den 
            roi_wrrmses[i] = torch.sum(torch.sqrt(weighted_roi_squared_err))
            # calculate average RSE in ROI 
            gt_mean = torch.sum(roi_mask * tau_volume, dim=(-3, -2, -1)) / roi_mask_size
            roi_se_num = torch.sum(roi_mask * torch.square(tau_volume - pred), dim=(-3, -2, -1))
            roi_se_den = torch.sum(roi_mask * torch.square(tau_volume - gt_mean.view(-1, 1, 1, 1, 1)), dim=(-3, -2, -1))
            nr_roi_rse = roi_se_num / roi_se_den
            roi_rses[i] += torch.sum(nr_roi_rse)

        num_samples += mri_volume.size(0)

        # Representatitive sample outputs
        if not samples_saved and save_representatives:
            rep_samples_paths = [
                ("7029_2022-03-30", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/067-S-7029/PET_2022-03-30_FTP/analysis/rnu.nii", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/067-S-7029/PET_2022-03-30_FTP/analysis/raparc+aseg.nii"),
                ("7032_2022-03-01", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/131-S-7032/PET_2022-03-01_FTP/analysis/rnu.nii", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/131-S-7032/PET_2022-03-01_FTP/analysis/raparc+aseg.nii"),
                ("6005_2017-04-27", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/024-S-6005/PET_2017-04-27_FTP/analysis/rnu.nii", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/024-S-6005/PET_2017-04-27_FTP/analysis/raparc+aseg.nii"),
                ("6005_2021-07-20", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/024-S-6005/PET_2021-07-20_FTP/analysis/rnu.nii", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/024-S-6005/PET_2021-07-20_FTP/analysis/raparc+aseg.nii")
            ]

            for sample_id, mri_path, roi_path in rep_samples_paths:
                # Load MRI
                vd = VolumeDataset(f"{os.getcwd()}/training_folds/testfold1.csv", cuda_id=cuda_id)
                mri = vd.load_volume_file(mri_path).unsqueeze(0)
                roi_mask = vd.load_volume_file(roi_path, is_mask=True).unsqueeze(0)
                mri[roi_mask == 0] = 0
                # Forward pass on model
                pred = model(mri)
                # Save output
                data_util.write_tensor_to_nii(pred, os.path.join(save_path, f"{sample_id}_prediction.nii"))
                # data_util.write_tensor_to_nii(roi_mask, os.path.join(save_path, f"{sample_id}_roi_mask.nii"))
                samples_saved = True

        # NOTE early stop to test
        # if batch_idx == 9:
        #     break

    ssim_error = ssim_metric.aggregate().item()
    ssim_metric.reset()
    # roi_corrcoef = roi_corr_metric.compute_roi_corr(tau_volumes.detach().clone(), tau_volumes, rois)#roi_corr(preds, tau_volumes, rois, roi_indices)
    mae /= num_samples
    # mape = mape / num_samples
    mape = mape / mape_smp_count
    rse /= num_samples
    rrmse /= num_samples
    roi_maes /= num_samples
    # roi_mapes = (roi_mapes * 100) / num_samples # convert to percentage
    roi_mapes = (roi_mapes * 100) / roi_nonnan_voxels # convert to percentage
    # roi_mses /= num_samples
    # roi_correlations /= num_samples
    roi_correlations = roi_corr_metric.calc_roi_corr()
    # save the matrices
    roi_corr_metric.save_matrices(save_path)
    # roi_rmses /= num_samples
    roi_rses /= num_samples
    roi_wrrmses /= num_samples 


    return mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations
    

def run_iteration(model, model_outputs, covars, neg_data, pos_data, criterion, tau, roi, cluster_loss=False, rnc_loss=False):

    # model = DataParallel(model, device_ids=[0, 1])
    model = model.module
    pos_mri, pos_covars = pos_data

    # Transfer to other device
    original_device = pos_mri.device
    new_device = return_next_device(original_device)
    
    model_pos_outputs = model(pos_mri, pos_covars.to(device=original_device))
    model = model.to(device=new_device)
    if cluster_loss and not rnc_loss:
        outs = [] # N-1 samples
        proj_depth = len(model_pos_outputs[1])
        neg_projected_reprs = [[] for i in range(proj_depth)]
        for n in neg_data:
            neg_mri, neg_tau, neg_roi, neg_abeta, neg_tau_path = n
            neg_abeta, neg_covars = neg_abeta
            _, lst_neg_projected_reprs, neg_final_repr = model(neg_mri.to(device=new_device), neg_covars.to(device=new_device))
            for i in range(len(model_pos_outputs[1])):
                neg_projected_reprs[i].append(lst_neg_projected_reprs[i].to(device=original_device))

        model_neg_outputs = None, neg_projected_reprs, neg_final_repr.to(device=original_device)
    
    # Treat as a batch, chunk
    elif rnc_loss :
        """
        For RNC Loss: intermediate representation is expected to include:
        Tuple[
            entire 'batch': [anchor_repr, pos_repr, neg_reprs], 
            corresponding labels
        ]
        """
        # Get pseudo batch
        pseudo_batch = [model_outputs[1][-1], model_pos_outputs[1][-1]]
        y_labels = [covars[:, -1].to(device=original_device), pos_covars[:, -1].to(device=original_device)]
        # choose random indices
        rand_inds = np.random.randint(0, len(neg_data), size=(2,))
        pseudo_batch = [model_outputs[1][-1]]
        # y_labels = [covars[:, -1].to(device=cuda_id), pos_covars[:, -1].to(device=cuda_id)]
        y_labels = [covars[:, -1].to(device=original_device)]
        # choose random indices
        rand_inds = np.random.randint(0, len(neg_data), size=(1,))
        neg_mri_mini_batch = torch.vstack([neg_data[ind][0] for ind in rand_inds])
        neg_covars_mini_batch = torch.vstack([neg_data[ind][3][-1] for ind in rand_inds])
        _, lst_neg_projected_reprs, neg_final_repr = model(neg_mri_mini_batch.to(device=original_device), neg_covars_mini_batch.to(device=original_device))
        pseudo_batch.append(lst_neg_projected_reprs[-1])
        y_labels.extend([neg_data[ind][3][-1][:, -1].to(device=original_device) for ind in rand_inds])
        
        intermediate_extractions = (torch.vstack(pseudo_batch), torch.vstack(y_labels))
        
    else:
        neg_mri, neg_tau, neg_roi, neg_abeta, neg_tau_path = neg_data
        neg_abeta, neg_covars = neg_abeta
        model_neg_outputs = model(neg_mri.to(device=new_device), neg_covars.to(device=new_device))

    # move model back
    model = model.to(device=original_device)
    # model = model.module

    if rnc_loss:
        anchor_pred, anchor_projected_reprs, anchor_final_repr = model_outputs
        _, pos_projected_reprs, pos_final_repr = model_pos_outputs
        neg_final_repr = torch.zeros_like(pos_final_repr, device=original_device) # dummy, not used

    elif model.decoder_ds:
        anchor_pred, anchor_projected_reprs, anchor_final_repr, decoder_projected_reprs = model_outputs
        _, pos_projected_reprs, pos_final_repr, _ = model_pos_outputs
        _, neg_projected_reprs, neg_final_repr, _ = model_neg_outputs
    else:
        anchor_pred, anchor_projected_reprs, anchor_final_repr = model_outputs
        _, pos_projected_reprs, pos_final_repr = model_pos_outputs
        _, neg_projected_reprs, neg_final_repr = model_neg_outputs
    
    if not rnc_loss:
        intermediate_extractions = anchor_projected_reprs, zip(pos_projected_reprs, neg_projected_reprs)
    
    if model.decoder_ds:
        final_representations = (decoder_projected_reprs, None, None)
    else:
        final_representations = (anchor_final_repr, pos_final_repr, neg_final_repr)

    loss, gen_loss, pred_space_contra_loss, total_ds_contra_loss = criterion(anchor_pred, tau, roi, final_representations, intermediate_extractions)

    return loss, gen_loss, pred_space_contra_loss, total_ds_contra_loss
            

def return_next_device(curr_device):
    if int(str(curr_device)[-1]) == 0: return 1
    return 0

if __name__ == "__main__":
    base_path = f"{os.getcwd()}"
    CUDA_ID = 0
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", 
                        level=logging.INFO,
                        handlers=[logging.FileHandler(filename=os.path.join(base_path, "saved_model_analysis", "test.log"), mode='w'),
                                  stream_handler]
                        )
    
    test_lookup_file = f"{os.getcwd()}/training_folds/outlier_removed_splits/test_lookup_1.csv"
    train_lookup_file = f"{os.getcwd()}/training_folds/outlier_removed_splits/training_lookup_1.csv"
    adni_covariate_lookup_file = f"{os.getcwd()}/scripts/covariate.csv"
    a4_covariate_lookup_file = ""

    holdout_ids = ["7029", "6005", "7032"]

    batch_size = 2
    train_dataset = VolumeDataset.CovariateVolumeDataset(adni_covariate_lookup_file, train_lookup_file, cuda_id=CUDA_ID)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = VolumeDataset.CovariateVolumeDataset(a4_covariate_lookup_file, test_lookup_file, cuda_id=CUDA_ID)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    roi_indices = [1001,1006,1007,1009,1015,1016,1030,1034,1033,1008,1025,1029,1031,1022,17,18,2001,2006,2007,2009,2015,2016,2030,2034,2033,2008,2025,2029,2031,2022,49,50,51,52,53,54]
    model_params = (3, 1, 1, [32, 64, 128, 256, 512], [2]*5)
    params_file = f"{os.getcwd()}/results/2024-01-10_13-39-42/fold_1/models/epoch_99_model.pt"
    # params_file = f"{os.getcwd()}/results/2024-01-21_10-47-13/fold_2/models/epoch_100_model.pt"
    with torch.no_grad():
        data_util.load_model(ObservableAttentionUnet, train_loader, test_dataloader, roi_indices, params_file, f"{os.getcwd()}/saved_model_analysis", CUDA_ID, *model_params, **{"training": True})

    sys.exit(0)

