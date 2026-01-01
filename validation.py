import os
import sys
base_path = f"{os.getcwd()}"
sys.path.append(f'{os.getcwd()}')
import pickle 
import glob 
import logging
import argparse
import random

import torch
# torch.backends.cudnn.deterministic = True

# random.seed(0)
# torch.manual_seed(0)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(0)

import torch.nn as nn
# import torch.optim as optim
from torch.nn import DataParallel#, SyncBatchNorm

import numpy as np
# np.random.seed(0)
import matplotlib.pyplot as plt 

from monai.networks.nets.unet import UNet
# from data_util import compute_mean_std, load_split_datasets, get_PCA, create_dataloader
import data_util
import criterions
from model import *
import unetr
import attn_unet_analysis
import attn_unet_data_parallel
from unetr import GenUNETR, GenAttnUnet, AttnUNETR, SwinUnetr, AttnSwinUnetr
# from criterions import RoiRRMSE, RoiRSE, RoiMSE
from ImageDataset import ImageDataset
from VolumeDataset import VolumeDataset, ContrastiveVolumeDataset, CovariateVolumeDataset, ClusterVolumeDataset, RegressionVolumeDataset

# Templte = prototype
POS_TEMPLATE_PATHS = [
    f"{os.getcwd()}/scripts/templates_tau_quart/abpos_quart1.nii", 
    f"{os.getcwd()}/scripts/templates_tau_quart/abpos_quart2.nii", 
    f"{os.getcwd()}/scripts/templates_tau_quart/abpos_quart3.nii", 
    f"{os.getcwd()}/scripts/templates_tau_quart/abpos_quart4.nii", 
]
NEG_TEMPLATE_PATHS = [
    f"{os.getcwd()}/scripts/templates_tau_quart/abneg_quart1.nii", 
    f"{os.getcwd()}/scripts/templates_tau_quart/abneg_quart2.nii", 
    f"{os.getcwd()}/scripts/templates_tau_quart/abneg_quart3.nii", 
    f"{os.getcwd()}/scripts/templates_tau_quart/abneg_quart4.nii", 
]

def volume_validation(splits_dir, batch_size, num_epochs, lr, model_params, model_name = "UNETR", save_path = "", cuda_id=-1, **kwargs):
    # 7029: /home/jagust/Generative-PET-models/7029
    # /home/jagust/xnat/xnp/sshfs/xnat_data/adni/067-S-7029/PET_2022-03-30_FTP/analysis/rnu.nii,/home/jagust/xnat/xnp/sshfs/xnat_data/adni/067-S-7029/PET_2022-03-30_FTP/analysis/suvr_cereg.nii,/home/jagust/xnat/xnp/sshfs/xnat_data/adni/067-S-7029/PET_2022-03-30_FTP/analysis/raparc+aseg.nii
    
    # 7032: /home/jagust/Generative-PET-models/7032
    # /home/jagust/xnat/xnp/sshfs/xnat_data/adni/131-S-7032/PET_2022-03-01_FTP/analysis/rnu.nii,/home/jagust/xnat/xnp/sshfs/xnat_data/adni/131-S-7032/PET_2022-03-01_FTP/analysis/suvr_cereg.nii,/home/jagust/xnat/xnp/sshfs/xnat_data/adni/131-S-7032/PET_2022-03-01_FTP/analysis/raparc+aseg.nii
    
    # 6005: /home/jagust/Generative-PET-models/6005
    # /home/jagust/xnat/xnp/sshfs/xnat_data/adni/024-S-6005/PET_2017-04-27_FTP/analysis/rnu.nii,/home/jagust/xnat/xnp/sshfs/xnat_data/adni/024-S-6005/PET_2017-04-27_FTP/analysis/suvr_cereg.nii,/home/jagust/xnat/xnp/sshfs/xnat_data/adni/024-S-6005/PET_2017-04-27_FTP/analysis/raparc+aseg.nii
    # /home/jagust/xnat/xnp/sshfs/xnat_data/adni/024-S-6005/PET_2021-07-20_FTP/analysis/rnu.nii,/home/jagust/xnat/xnp/sshfs/xnat_data/adni/024-S-6005/PET_2021-07-20_FTP/analysis/suvr_cereg.nii,/home/jagust/xnat/xnp/sshfs/xnat_data/adni/024-S-6005/PET_2021-07-20_FTP/analysis/raparc+aseg.nii

    indices = [1001,1006,1007,1009,1015,1016,1030,1034,1033,1008,1025,1029,1031,1022,17,18,2001,2006,2007,2009,2015,2016,2030,2034,2033,2008,2025,2029,2031,2022,49,50,51,52,53,54]
    if kwargs["template"]:
        # indices = [1, 2, 3, 4, 5, 6, 7]
        indices = [1, 2, 3, 4, 5, 6, 7, 8]
    avg_mae, avg_mape, avg_ssim = 0, 0, 0
    # avg_roi_corrcoefs, avg_roi_maes, avg_roi_mapes = torch.zeros(len(indices), device=cuda_id), torch.zeros(len(indices), device=cuda_id), torch.zeros(len(indices), device=cuda_id)
    avg_roi_corrcoefs, avg_roi_maes, avg_roi_mapes = np.zeros(len(indices)), torch.zeros(len(indices), device=cuda_id), torch.zeros(len(indices), device=cuda_id)
    for k in range(3, 4):
    # for k in range(2, 4):
    # for k in range(4, 5):
        logging.info(f"Starting fold {k+1}...")
        # Set up save dir
        fold_save_path = os.path.join(save_path, f"fold_{k+1}")
        if not os.path.exists(fold_save_path):
            os.makedirs(fold_save_path)
        
        if model_name == "UNETR":
            model = GenUNETR(*model_params)
        elif model_name == "AttnUNET":
            model = GenAttnUnet(*model_params)
        elif model_name == "AttnUNETR":
            model = AttnUNETR(*model_params)
        elif model_name == "SwinUNETR":
            model = SwinUnetr(**model_params)
        elif model_name == "AttnSwinUNETR":
            model = AttnSwinUnetr(**model_params)
        elif model_name == "UNET":
            model = UNet(**model_params)
        elif model_name == "ContraAttnUNET":
            # latent_space = 64^3
            # model = attn_unet_analysis.ContrastiveAttenionUNET(*model_params, latent_spaces=[262144]*len(model_params[-1]))
            # model = attn_unet_analysis.ContrastiveAttentionUNET(*model_params, latent_spaces=[262144]*len(model_params[-1]), conditional=True, decoder_ds=kwargs["decoder_ds"])
            # model = attn_unet_analysis.ContrastiveAttentionUNET(*model_params, latent_spaces=[2048]*len(model_params[-1]), conditional=True, decoder_ds=kwargs["decoder_ds"])
            model = attn_unet_data_parallel.ContrastiveAttentionUNET_DP(*model_params, latent_spaces=[2048]*len(model_params[-1]), conditional=True, decoder_ds=kwargs["decoder_ds"])
            # if kwargs:
            #     model.embeddings_out = kwargs["embeddings_out"]
            #     model.train_tau_targets_csv = kwargs["train_tau_target_csv"]
            #     model.test_tau_targets_csv = kwargs["test_tau_target_csv"]
        
        if cuda_id != -1:
            model = model.cuda(cuda_id)
        
        logging.info(model)
        # Load dataset
        logging.info(f"Getting split dataset...")
        train_dataset, test_dataset = data_util.load_split_datasets(splits_dir, VolumeDataset, k+1, cuda_id, contra=("Contra" in model_name), template=kwargs["template"], resize=kwargs["resize"], with_covars=kwargs["with_covars"], smoothing=kwargs["smoothing"])

        # train_loader = data_util.create_dataloader(train_dataset, batch_size, shuffle=True)
        # test_loader = data_util.create_dataloader(test_dataset, batch_size)
        train_loader = data_util.create_dataloader(train_dataset, batch_size, shuffle=True, contra=(model_name == "ContraAttnUNET"))
        if model_name != "ContraAttnUNET":
            test_loader = data_util.create_dataloader(test_dataset, 4, contra=False)
        else :
            test_loader = data_util.create_dataloader(test_dataset, batch_size, contra=False)

        # criterion_weights = torch.ones(len(indices), dtype=torch.float)
        # bias some ROIs
        # criterion_weights[:] = torch.tensor([225.0] * len(indices), dtype=torch.float)
        if "template" in kwargs and kwargs["template"]:
            # criterion_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 3.0], dtype=torch.float)
            # criterion_weights = torch.tensor([1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=torch.float)
            # criterion_weights[:] = 360 * (criterion_weights / torch.linalg.norm(criterion_weights))
            # NOTE
            criterion_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float)
        else:
            criterion_weights = torch.tensor([225.0] * len(indices), dtype=torch.float)
        
        # loss_criterion = RoiRRMSE(criterion_weights, indices)
        # loss_criterion = RoiRSE(criterion_weights, indices)
        logging.info(f"Starting model training...")
        if model_name == "ContraAttnUNET":
            # contra_loss = nn.TripletMarginWithDistanceLoss
            contra_loss = nn.TripletMarginLoss
            # tcds_comp_weights = torch.tensor([1.]*len(model_params[-1])) # torch.linspace(1, 0, len(model_params[-1]))
            tcds_comp_weights = torch.square(torch.arange(0, len(model_params[-1]))).to(dtype=torch.float)
            tcds_comp_weights = 5 * (tcds_comp_weights / torch.norm(tcds_comp_weights))
            # NOTE make sure to change the data loader / data set from cluster to contrastive in data_util 
            cds_loss = criterions.TruncatedCDS(contra_loss, tcds_comp_weights)
            if "rnc" in kwargs and kwargs["rnc"]:
                cds_loss = criterions.RnCLoss()
            # cds_loss = criterions.ClusterNPairLoss(tcds_comp_weights)
            gen_loss = criterions.RoiMSE(criterion_weights, indices, voxel_wise=False)
            if "decoder_ds" in kwargs and kwargs["decoder_ds"]:
                decoder_loss = criterions.NPairLoss(POS_TEMPLATE_PATHS, NEG_TEMPLATE_PATHS)
                decoder_loss.load_templates(train_dataset)
            else: 
                decoder_loss = contra_loss(1)
            # loss_criterion = criterions.GenerativeContrastiveLoss(cds_loss, gen_loss, contra_loss(margin=1e-4), regulatory_weight=0.00, ds_regulatory_weight=0.0001)
            # loss_criterion = criterions.GenerativeContrastiveLoss(cds_loss, gen_loss, decoder_loss, regulatory_weight=0.1, ds_regulatory_weight=1.)
            loss_criterion = criterions.GenerativeContrastiveLoss(cds_loss, gen_loss, decoder_loss, regulatory_weight=0., ds_regulatory_weight=1.) #NOTE
            logging.info(f"{loss_criterion}")
            model.set_save_attn(None)
            # attn_unet_analysis.train(model, loss_criterion, train_loader, test_loader, num_epochs, lr, save_path=fold_save_path, cuda_id=cuda_id)
            attn_unet_data_parallel.train_dp(model, loss_criterion, train_loader, test_loader, num_epochs, lr, save_path=fold_save_path, cuda_id=cuda_id, **{"fold_id": k})
        else:
            # loss_criterion = RoiRSE(criterion_weights, indices)
            loss_criterion = criterions.RoiMSE(criterion_weights, indices, voxel_wise=False)
            logging.info(f"{loss_criterion}")
            unetr.train(model, loss_criterion, train_loader, test_loader, num_epochs, lr, save_path=fold_save_path, cuda_id=cuda_id)
    
        with torch.no_grad():
            if model_name == "ContraAttnUNET":
                # (mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations, voxel_mapes), _, _ = attn_unet_analysis.contrastive_test(model, test_loader, indices, criterion_weights, save_path=fold_save_path, cuda_id=cuda_id, with_train_loader=isinstance(train_dataset, RegressionVolumeDataset))
                (mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations, voxel_mapes), _, _ = attn_unet_data_parallel.contrastive_test(model, test_loader, indices, criterion_weights, save_path=fold_save_path, cuda_id=cuda_id, with_train_loader=isinstance(train_dataset, RegressionVolumeDataset), **{"fold_id": k})
            else :
                mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations = unetr.test(model, test_loader, indices, criterion_weights, save_path=fold_save_path, cuda_id=cuda_id)
            
        avg_mae += mae 
        avg_mape += mape
        avg_ssim += ssim_error
        avg_roi_corrcoefs += roi_correlations
        avg_roi_maes += roi_maes 
        avg_roi_mapes += roi_mapes

        logging.info("Test Results:")
        logging.info(f"MAE:{mae}")
        logging.info(f"MAPE:{mape}")
        logging.info(f"SSIM Error:{ssim_error}")
        logging.info(f"ROI MAES:{roi_maes}")
        logging.info(f"ROI MAPES:{roi_mapes}")
        logging.info(f"ROI correlations: {roi_correlations}")
        lowest_corr_idx = np.argmin(roi_correlations)
        highest_corr_idx = np.argmax(roi_correlations)
        logging.info(f"\tLowest correlation (ROI {indices[lowest_corr_idx]}): {roi_correlations[lowest_corr_idx]}")
        logging.info(f"\tHighest correlation (ROI {indices[highest_corr_idx]}): {roi_correlations[highest_corr_idx]}")
        logging.info(f"\tAverage correlation: {np.mean(roi_correlations)}")

    
    logging.info(f"Cross validation results:")
    logging.info(f"Cross validation Average MAE: {avg_mae / 5}")
    logging.info(f"Cross validation Average MAPE: {avg_mape / 5}")
    logging.info(f"Cross validation Average SSIM: {avg_ssim / 5}")
    logging.info(f"Cross validation Average ROI MAEs: {avg_roi_maes / 5}")
    logging.info(f"Cross validation Average ROI MAPEs: {avg_roi_mapes / 5}")
    logging.info(f"Cross validation Average ROI Corrs: {avg_roi_corrcoefs / 5}")

def from_checkpoint_volume_validation(splits_dir, batch_size, num_epochs, lr, model_params, optimizer_params, optimizer, scheduler_params, scheduler, checkpoint_path, model_name = "UNETR", save_path = "", cuda_id=-1, **kwargs):
    
    save_path = os.path.join("/".join(checkpoint_path.split("/")[:6]))
    from_checkpoint = True

    indices = [1001,1006,1007,1009,1015,1016,1030,1034,1033,1008,1025,1029,1031,1022,17,18,2001,2006,2007,2009,2015,2016,2030,2034,2033,2008,2025,2029,2031,2022,49,50,51,52,53,54]
    if kwargs["template"]:
        # indices = [1, 2, 3, 4, 5, 6, 7]
        indices = [1, 2, 3, 4, 5, 6, 7, 8]

    avg_mae, avg_mape, avg_ssim = 0, 0, 0
    # avg_roi_corrcoefs, avg_roi_maes, avg_roi_mapes = torch.zeros(len(indices), device=cuda_id), torch.zeros(len(indices), device=cuda_id), torch.zeros(len(indices), device=cuda_id)
    avg_roi_corrcoefs, avg_roi_maes, avg_roi_mapes = np.zeros(len(indices)), torch.zeros(len(indices), device=cuda_id), torch.zeros(len(indices), device=cuda_id)
    start_fold = int(checkpoint_path.split("/")[6].split("_")[-1]) - 1
    # for k in range(start_fold, 5):
    for k in range(start_fold, start_fold+1):
        # /home/jagust/Generative-PET-models/results/2024-03-16_04-44-29/fold_1/checkpoints/checkpoint_epoch_0.pth
        # if f"fold_{k+1}" == checkpoint_path.split("/")[6]:
        if k+1 == int(checkpoint_path.split("/")[6].split("_")[-1]):
            # checkpoint_state_dicts = torch.load(checkpoint_path, map_location=torch.device(f'cuda:{cuda_id}'))
            checkpoint_state_dicts = torch.load(checkpoint_path, map_location=torch.device(f'cpu'))
            # start_epoch = int(checkpoint_path.split(".")[0].split("_")[-1])+1
            start_epoch = int(checkpoint_state_dicts['epoch'])+1
            # new_resume_path = "fork_resumed_" + checkpoint_path.split("/")[6]
            new_resume_path = "native_target_finetune_" + checkpoint_path.split("/")[6]
            # new_resume_path = "native_target_finetune_frozen_UNET_" + checkpoint_path.split("/")[6]
            # new_resume_path = "transfer_learning_fork_resumed_" + checkpoint_path.split("/")[6]
            # new_resume_path = "safe_transfer_learning_fork_resumed_" + checkpoint_path.split("/")[6]
            # new_resume_path = "native_target_transfer_learning_fork_resumed_" + checkpoint_path.split("/")[6]
            # new_resume_path = "base_transfer_learning_fork_resumed_" + checkpoint_path.split("/")[6]
            # fold_save_path = os.path.join(save_path, f"resumed_fold_{k+1}") # must change dir to avoid overwriting graphs
            fold_save_path = os.path.join(save_path, new_resume_path) # must change dir to avoid overwriting graphs
            if not os.path.exists(fold_save_path):
                os.makedirs(fold_save_path)
        else :
            from_checkpoint = False
            fold_save_path = os.path.join(save_path, f"fold_{k+1}") # must change dir to avoid overwriting graphs
            if not os.path.exists(fold_save_path):
                os.makedirs(fold_save_path)

        logging.info(f"Starting fold {k+1}...")

        if model_name == "UNETR":
            model = GenUNETR(*model_params)
        elif model_name == "AttnUNET":
            model = GenAttnUnet(*model_params)
        elif model_name == "AttnUNETR":
            model = AttnUNETR(*model_params)
        elif model_name == "SwinUNETR":
            model = SwinUnetr(**model_params)
        elif model_name == "AttnSwinUNETR":
            model = AttnSwinUnetr(**model_params)
        elif model_name == "UNET":
            model = UNet(**model_params)
        elif model_name == "ContraAttnUNET":
            # latent_space = 64^3
            # model = attn_unet_analysis.ContrastiveAttentionUNET(*model_params, latent_spaces=[262144]*len(model_params[-1]), conditional=True, decoder_ds=kwargs["decoder_ds"])
            model = attn_unet_data_parallel.ContrastiveAttentionUNET_DP(*model_params, latent_spaces=[2048]*len(model_params[-1]), conditional=True, decoder_ds=kwargs["decoder_ds"])
            # attn_unet_data_parallel.freeze_unet_params(model)
            # if kwargs:
            #     model.embeddings_out = kwargs["embeddings_out"]
            #     model.train_tau_targets_csv = kwargs["train_tau_target_csv"]
            #     model.test_tau_targets_csv = kwargs["test_tau_target_csv"]

        if cuda_id != -1:
            model = model.to(device=f'cuda:{cuda_id}')
        
        # model = DataParallel(model, device_ids=[0, 1])
        # model = DataParallel(model, device_ids=[1, 0])

        if from_checkpoint:
            
            optim = optimizer(model.parameters(), **optimizer_params)
            # optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)

            model.load_state_dict(checkpoint_state_dicts['model_state_dict'])
            optim.load_state_dict(checkpoint_state_dicts['optimizer_state_dict'])
            sched = None
            if 'scheduler_state_dict' in checkpoint_state_dicts.keys():
                sched = scheduler(optim, **scheduler_params)
                sched.load_state_dict(checkpoint_state_dicts['scheduler_state_dict'])
        
        if isinstance(model, DataParallel):
            model = model.module
        
        if cuda_id != -1:
            model = model.cuda(cuda_id)
        logging.info(model)
        # Load dataset
        logging.info(f"Getting split dataset...")
        train_dataset, test_dataset = data_util.load_split_datasets(splits_dir, VolumeDataset, k+1, cuda_id, contra=("Contra" in model_name), template=kwargs["template"], resize=kwargs["resize"], with_covars=kwargs["with_covars"], smoothing=kwargs["smoothing"])

        # train_loader = data_util.create_dataloader(train_dataset, batch_size, shuffle=True)
        # test_loader = data_util.create_dataloader(test_dataset, batch_size)
        train_loader = data_util.create_dataloader(train_dataset, batch_size, shuffle=True, contra=(model_name == "ContraAttnUNET"))
        if model_name != "ContraAttnUNET":
            test_loader = data_util.create_dataloader(test_dataset, 4, contra=(model_name == "ContraAttnUNET"))
        else :
            test_loader = data_util.create_dataloader(test_dataset, batch_size, contra=False)

        # criterion_weights = torch.ones(len(indices), dtype=torch.float)
        # bias some ROIs
        # criterion_weights[:] = torch.tensor([225.0] * len(indices), dtype=torch.float)
        if "template" in kwargs and kwargs["template"]:
            # criterion_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 3.0], dtype=torch.float)
            # NOTE old
            # criterion_weights = torch.tensor([1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=torch.float)
            # criterion_weights[:] = 360 * (criterion_weights / torch.linalg.norm(criterion_weights))
            # end old
            criterion_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float)
        else:
            criterion_weights = torch.tensor([225.0] * len(indices), dtype=torch.float)
        # loss_criterion = RoiRRMSE(criterion_weights, indices)
        # loss_criterion = RoiRSE(criterion_weights, indices)
        # Resume evaluation
        kwargs_ = {}
        if from_checkpoint:
            kwargs_ = {
                "optimizer": optim, 
                "scheduler": sched, 
                "start_epoch": start_epoch, 
                "fold_id": k
            }
        logging.info(f"Starting model training...")
        if model_name == "ContraAttnUNET":
            # contra_loss = nn.TripletMarginWithDistanceLoss
            contra_loss = nn.TripletMarginLoss
            # tcds_comp_weights = torch.tensor([1.]*len(model_params[-1])) # torch.linspace(1, 0, len(model_params[-1]))
            tcds_comp_weights = torch.square(torch.arange(0, len(model_params[-1]))).to(dtype=torch.float)
            tcds_comp_weights = 5 * (tcds_comp_weights / torch.norm(tcds_comp_weights))
            # NOTE make sure to change the data loader / data set from cluster to contrastive in data_util 
            cds_loss = criterions.TruncatedCDS(contra_loss, tcds_comp_weights)
            if "rnc" in kwargs and kwargs["rnc"]:
                cds_loss = criterions.RnCLoss()
            # cds_loss = criterions.ClusterNPairLoss(tcds_comp_weights)
            gen_loss = criterions.RoiMSE(criterion_weights, indices, voxel_wise=False)
            if "decoder_ds" in kwargs and kwargs["decoder_ds"]:
                decoder_loss = criterions.NPairLoss(POS_TEMPLATE_PATHS, NEG_TEMPLATE_PATHS)
                decoder_loss.load_templates(train_dataset)
            else: 
                decoder_loss = contra_loss(1)
            # loss_criterion = criterions.GenerativeContrastiveLoss(cds_loss, gen_loss, contra_loss(margin=1e-4), regulatory_weight=0.00, ds_regulatory_weight=0.0001)
            # loss_criterion = criterions.GenerativeContrastiveLoss(cds_loss, gen_loss, decoder_loss, regulatory_weight=0.1, ds_regulatory_weight=1.)
            loss_criterion = criterions.GenerativeContrastiveLoss(cds_loss, gen_loss, decoder_loss, regulatory_weight=0., ds_regulatory_weight=1.)
            logging.info(f"{loss_criterion}")
            model.set_save_attn(None)
            # attn_unet_analysis.train(model, loss_criterion, train_loader, test_loader, num_epochs, lr, save_path=fold_save_path, cuda_id=cuda_id, from_checkpoint=from_checkpoint, **kwargs_)
            attn_unet_data_parallel.train_dp(model, loss_criterion, train_loader, test_loader, num_epochs, lr, save_path=fold_save_path, cuda_id=cuda_id, from_checkpoint=from_checkpoint, **kwargs_)
        else:
            # loss_criterion = RoiRSE(criterion_weights, indices)
            loss_criterion = criterions.RoiMSE(criterion_weights, indices, voxel_wise=False)
            logging.info(f"{loss_criterion}")
            unetr.train(model, loss_criterion, train_loader, test_loader, num_epochs, lr, save_path=fold_save_path, cuda_id=cuda_id, from_checkpoint=from_checkpoint, **kwargs_)
    
        with torch.no_grad():
            if model_name == "ContraAttnUNET":
                # (mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations), _, _ = attn_unet_analysis.contrastive_test(model, test_loader, indices, criterion_weights, save_path=fold_save_path, with_train_loader=isinstance(train_loader.dataset, RegressionVolumeDataset), cuda_id=cuda_id, **{"fold_id": k})
                (mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations), _, _ = attn_unet_data_parallel.contrastive_test(model, test_loader, indices, criterion_weights, save_path=fold_save_path, with_train_loader=isinstance(train_loader.dataset, RegressionVolumeDataset), cuda_id=cuda_id, **{"fold_id": k})
            else :
                mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations = unetr.test(model, test_loader, indices, criterion_weights, save_path=fold_save_path, cuda_id=cuda_id)
            
        avg_mae += mae 
        avg_mape += mape
        avg_ssim += ssim_error
        avg_roi_corrcoefs += roi_correlations
        avg_roi_maes += roi_maes 
        avg_roi_mapes += roi_mapes

        logging.info("Test Results:")
        logging.info(f"MAE:{mae}")
        logging.info(f"MAPE:{mape}")
        logging.info(f"SSIM Error:{ssim_error}")
        logging.info(f"ROI MAES:{roi_maes}")
        logging.info(f"ROI MAPES:{roi_mapes}")
        logging.info(f"ROI correlations: {roi_correlations}")
        lowest_corr_idx = np.argmin(roi_correlations)
        highest_corr_idx = np.argmax(roi_correlations)
        logging.info(f"\tLowest correlation (ROI {indices[lowest_corr_idx]}): {roi_correlations[lowest_corr_idx]}")
        logging.info(f"\tHighest correlation (ROI {indices[highest_corr_idx]}): {roi_correlations[highest_corr_idx]}")
        logging.info(f"\tAverage correlation: {np.mean(roi_correlations)}")

    
    logging.info(f"Cross validation results:")
    logging.info(f"Cross validation Average MAE: {avg_mae / 5}")
    logging.info(f"Cross validation Average MAPE: {avg_mape / 5}")
    logging.info(f"Cross validation Average SSIM: {avg_ssim / 5}")
    logging.info(f"Cross validation Average ROI MAEs: {avg_roi_maes / 5}")
    logging.info(f"Cross validation Average ROI MAPEs: {avg_roi_mapes / 5}")
    logging.info(f"Cross validation Average ROI Corrs: {avg_roi_corrcoefs / 5}")


def single_split_validation(data_dir, batch_size, num_epochs, lr, model_params, model_name = "UNETR", save_path = "", cuda_id=-1, **kwargs):
    indices = [1001,1006,1007,1009,1015,1016,1030,1034,1033,1008,1025,1029,1031,1022,17,18,2001,2006,2007,2009,2015,2016,2030,2034,2033,2008,2025,2029,2031,2022,49,50,51,52,53,54]
    # Set up save dir
    training_save_path = os.path.join(save_path, f"training")
    if not os.path.exists(training_save_path):
        os.makedirs(training_save_path)
    
    if model_name == "UNETR":
        model = GenUNETR(*model_params)
    elif model_name == "AttnUNET":
        model = GenAttnUnet(*model_params)
    elif model_name == "AttnUNETR":
        model = AttnUNETR(*model_params)
    elif model_name == "SwinUNETR":
        model = SwinUnetr(**model_params)
    elif model_name == "AttnSwinUNETR":
        model = AttnSwinUnetr(**model_params)
    elif model_name == "UNET":
        model = UNet(**model_params)
    elif model_name == "ContraAttnUNET":
        model = attn_unet_analysis.ContrastiveAttentionUNET(*model_params, latent_spaces=[262144]*len(model_params[-1]))
        model.embeddings_out = kwargs["embeddings_out"]
        model.train_tau_targets_csv = kwargs["train_tau_target_csv"]
        model.test_tau_targets_csv = kwargs["test_tau_target_csv"]
    
    if cuda_id != -1:
        model = model.cuda(cuda_id)
    logging.info(model)
    # Load dataset
    logging.info(f"Getting split dataset...")
    # Need to revise this so that one data loader can be of different type
    if model_name == "ContraAttnUNET":
        train_dataset, test_dataset = data_util.load_single_split_datasets(data_dir, ContrastiveVolumeDataset, CovariateVolumeDataset, cuda_id, contra=True)
    else:
        train_dataset, test_dataset = data_util.load_single_split_datasets(data_dir, VolumeDataset, VolumeDataset,  cuda_id)

    train_loader = data_util.create_dataloader(train_dataset, batch_size, shuffle=True, contra=(model_name == "ContraAttnUNET"))
    if model_name != "ContraAttnUNET":
        test_loader = data_util.create_dataloader(test_dataset, 4, contra=(model_name == "ContraAttnUNET"))
    else :
        test_loader = data_util.create_dataloader(test_dataset, batch_size, contra=(model_name == "ContraAttnUNET"))

    criterion_weights = torch.ones(len(indices), dtype=torch.float)
    # bias some ROIs
    criterion_weights[:] = torch.tensor([10.0] * len(indices), dtype=torch.float)
    # loss_criterion = RoiRRMSE(criterion_weights, indices)
    # loss_criterion = RoiRSE(criterion_weights, indices)
    logging.info(f"Starting model training...")
    if model_name == "ContraAttnUNET":
        contra_loss = nn.TripletMarginWithDistanceLoss
        tcds_comp_weights = torch.tensor([1.]*len(model_params[-1])) # torch.linspace(1, 0, len(model_params[-1]))
        cds_loss = criterions.TruncatedCDS(contra_loss, tcds_comp_weights)
        gen_loss = criterions.RoiMSE(torch.tensor([15.0] * len(indices), dtype=torch.float), indices)
        loss_criterion = criterions.GenerativeContrastiveLoss(cds_loss, gen_loss, contra_loss, 0.1, 0.05)
        logging.info(f"{loss_criterion}")
        attn_unet_analysis.train(model, loss_criterion, train_loader, test_loader, num_epochs, lr, save_path=training_save_path, cuda_id=cuda_id)
        model.embeddings_out = False
    else:
        loss_criterion = criterions.RoiMSE(criterion_weights, indices, voxel_wise=False)
        logging.info(f"{loss_criterion}")
        unetr.train(model, loss_criterion, train_loader, test_loader, num_epochs, lr, save_path=training_save_path, cuda_id=cuda_id)
    
    # evaluate 
    testing_save_path = os.path.join(save_path, "testing")
    if not os.path.exists(testing_save_path):
        os.makedirs(testing_save_path)
    with torch.no_grad():
        if model_name == "ContraAttnUNET":
            (mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations), _, _ = attn_unet_analysis.contrastive_test(model, test_loader, indices, criterion_weights, save_path=testing_save_path, cuda_id=cuda_id)
        else :
            mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations = unetr.test(model, test_loader, indices, criterion_weights, save_path=testing_save_path, cuda_id=cuda_id)
        # mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations = attn_unet_analysis.test(model, test_loader, indices, criterion_weights, save_path=testing_save_path, cuda_id=cuda_id)
        # mae, mape, ssim_error, roi_maes, roi_mapes, roi_correlations = unetr.test(model, test_loader, indices, cuda_id=cuda_id)

    logging.info("Test Results:")
    logging.info(f"MAE:{mae}")
    logging.info(f"MAPE:{mape}")
    logging.info(f"SSIM Error:{ssim_error}")
    logging.info(f"ROI MAES:{roi_maes}")
    logging.info(f"ROI MAPES:{roi_mapes}")
    logging.info(f"ROI correlations: {roi_correlations}")
    lowest_corr_idx = np.argmin(roi_correlations)
    highest_corr_idx = np.argmax(roi_correlations)
    logging.info(f"\tLowest correlation (ROI {indices[lowest_corr_idx]}): {roi_correlations[lowest_corr_idx]}")
    logging.info(f"\tHighest correlation (ROI {indices[highest_corr_idx]}): {roi_correlations[highest_corr_idx]}")
    logging.info(f"\tAverage correlation: {np.mean(roi_correlations)}")


def five_fold_cross_validation(splits_dir, batch_size, num_epochs, lr, col_list="", saved_normalizations=[], pca_models=[], cuda_id=-1, save_path=""):

    mapes, comp_mapes = [], []
    for k in range(5):
        logging.info(f"Starting fold {k+1}...")
        # Get datasets
        logging.info(f"Getting split dataset...")
        # train_dataset, test_dataset = get_split_datasets(splits_dir, k, cuda_id, col_list=col_list)
        train_dataset, test_dataset = data_util.load_split_datasets(splits_dir, ImageDataset, k+1, cuda_id, file_base_name="fold")
        train_dataset.set_col_list(col_list)
        test_dataset.set_col_list(col_list)
        # Standardize
        logging.info(f"Standardizing split...")
        # train_x_mean, train_x_std, train_y_mean, train_y_std = quick_mean_std(datasets[:k] + datasets[k+1:])
        if 0 <= k < len(saved_normalizations):
            train_x_mean, train_x_std, train_y_mean, train_y_std = tuple(np.loadtxt(saved_normalizations[k], dtype=np.float))
        else: 
            # train_x_mean, train_x_std, train_y_mean, train_y_std = compute_mean_std(datasets[:k] + datasets[k+1:], 216, cuda_id)
            train_x_mean, train_x_std, train_y_mean, train_y_std = compute_mean_std([train_dataset], 16, cuda_id)
            # ym, ys = train_y_mean.unsqueeze(0), train_y_std.unsqueeze(0)
            ym, ys = train_y_mean, train_y_std
            norm_vals_save_path = os.path.join(save_path, "split_normalization_vals")
            if not os.path.exists(norm_vals_save_path):
                os.makedirs(norm_vals_save_path)
            np.savetxt(os.path.join(norm_vals_save_path, f"split-{k}-mean-std"), [train_x_mean, train_x_std, ym, ys])
        logging.info(f"Mean and std vals: {train_x_mean}, {train_x_std}, {train_y_mean}, {train_y_std}")
        train_x_mean = 0
        train_x_std = 1 
        logging.info(f"Skipping z-normalization")
        train_dataset.set_mean_std(train_x_mean, train_x_std, 0, 1)
        test_dataset.set_mean_std(train_x_mean, train_x_std, 0, 1) # Don't center targets
        # for i, dataset in enumerate(datasets):
        #     dataset.set_mean_std(train_x_mean, train_x_std, train_y_mean, train_y_std)
        # Save for reuse
        # PCA on suvr
        logging.info(f"Starting PCA...")
        if 0 <= k < len(pca_models):
            with open(pca_models[k], 'rb') as f:
                pca = pickle.load(f)
        else:
            # pca = get_PCA(datasets[:k] + datasets[k+1:])
            pca = get_PCA([train_dataset], center=False)
            pca_save_dir = os.path.join(save_path, 'saved_pcas')
            if not os.path.exists(pca_save_dir):
                os.makedirs(pca_save_dir)
            with open(os.path.join(pca_save_dir, f'pca_model_{k}.pkl'), 'wb') as f:
                pickle.dump(pca, f)

        num_components = pca.n_components_
        components = pca.components_
        explained_var = pca.explained_variance_

        logging.info(f"PCA explained variance values: {explained_var}")
        logging.info(f"PCA explained variance percentages: {pca.explained_variance_ratio_}")
        # train_loader, test_loader = create_fold_dataloader(k, datasets, batch_size)
        train_loader = data_util.create_dataloader(train_dataset, batch_size, shuffle=True).cuda(cuda_id)
        test_loader = data_util.create_dataloader(test_dataset, batch_size).cuda(cuda_id)
        # input_shape=(128, 128, 128), in_channels=1, out_channels=64, num_heads=2, output_size=114, cuda_id=-1
        # NOTE testing no PCA:
        pca = None
        model = ConvAttn(input_shape=(128, 128, 128), in_channels=1, first_out_channels=16, out_channels=32, num_heads=4, output_size=114) # num ROIs
        # model = ConvAttn(input_shape=(128, 128, 128), in_channels=1, first_out_channels=16, out_channels=32, num_heads=4, output_size=num_components)
        if cuda_id != -1:
            model = model.cuda(cuda_id)
        # model = ConvAttn(pca=pca)
        logging.info(model)
        graph_save_path = os.path.join(save_path, f"fold_{k}")
        if not os.path.exists(graph_save_path):
            os.makedirs(graph_save_path)
        avg_train_loss, train_losses = train_model(model, train_loader, test_loader, num_epochs, lr, pca=pca, device_id=cuda_id, graph_save_path=graph_save_path)
        logging.info(f"Avg train loss: {avg_train_loss}")
        
        with torch.no_grad():
            mape, p, comp_mape, full_mape, comp_full_mape, original_spc_corrs = test_model(model, test_loader, pca=pca)
        logging.info(f"Mean Absolute Percentage Error: {mape}")
        logging.info(f"Component-wise Mean Absolute Error: {comp_mape}")
        logging.info(f"First Principle Component Correlation: {p}")
        logging.info(f"Original Space MAPE: {full_mape}")
        logging.info(f"Original Space Component Wise MAPE:\t{comp_full_mape}")
        logging.info(f"Original Space Component Wise Correlations:\t{original_spc_corrs}")
        mapes.append(mape)
        comp_mapes.append(comp_mape)
        # all_losses.append(train_losses)

        graph_loss(train_losses, os.path.join(save_path, f"./weighted_mse_train_losses_fold_{k+1}"))

    return mapes, comp_mapes

def graph_loss(train_losses, save_path=""):
    plt.plot(np.arange(len(train_losses)), train_losses)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")

    if save_path != "":
        plt.savefig(save_path)
    plt.close()

def train_from_checkpoint(data_dir, batch_size, num_epochs, lr, model_params, model_name, optimizer_params, optimizer, scheduler_params, scheduler, checkpoint_path, cuda_id):
    # /home/jagust/Generative-PET-models/results/2024-02-24_20-10-11/training/checkpoints/
    save_path = os.path.join("/".join(checkpoint_path.split("/")[:6]))
    checkpoint_state_dicts = torch.load(checkpoint_path)
    start_epoch = int(checkpoint_path.split(".")[0].split("_")[-1])
    # Initialize
    if model_name == "UNETR":
        model = GenUNETR(*model_params)
    elif model_name == "AttnUNET":
        model = GenAttnUnet(*model_params)
    elif model_name == "AttnUNETR":
        model = AttnUNETR(*model_params)
    elif model_name == "SwinUNETR":
        model = SwinUnetr(**model_params)
    elif model_name == "AttnSwinUNETR":
        model = AttnSwinUnetr(**model_params)
    elif model_name == "UNET":
        model = UNet(**model_params)
    elif model_name == "ContraAttnUNET":
        model = attn_unet_analysis.ContrastiveAttentionUNET(*model_params, latent_spaces=[768]*3)

    if cuda_id != -1:
        model = model.cuda(cuda_id)
        
    optim = optimizer(model.parameters(), **optimizer_params)

    model.load_state_dict(checkpoint_state_dicts['model_state_dict'])
    optim.load_state_dict(checkpoint_state_dicts['optimizer_state_dict'])
    sched = None
    if 'scheduler_state_dict' in checkpoint_state_dicts.keys():
        sched = scheduler(optim, **scheduler_params)
        sched.load_state_dict(checkpoint_state_dicts['scheduler_state_dict'])


    indices = [1001,1006,1007,1009,1015,1016,1030,1034,1033,1008,1025,1029,1031,1022,17,18,2001,2006,2007,2009,2015,2016,2030,2034,2033,2008,2025,2029,2031,2022,49,50,51,52,53,54]
    # Set up save dir
    training_save_path = os.path.join(save_path, f"resumed_training") # must change dir to avoid overwriting graphs
    if not os.path.exists(training_save_path):
        os.makedirs(training_save_path)
    logging.info(model)
    # Load dataset
    logging.info(f"Getting split dataset...")
    # Need to revise this so that one data loader can be of different type
    if model_name == "ContraAttnUNET":
        train_dataset, test_dataset = data_util.load_single_split_datasets(data_dir, ContrastiveVolumeDataset, CovariateVolumeDataset, cuda_id, contra=True)
    else:
        train_dataset, test_dataset = data_util.load_single_split_datasets(data_dir, VolumeDataset, VolumeDataset,  cuda_id)

    train_loader = data_util.create_dataloader(train_dataset, batch_size, shuffle=True, contra=(model_name == "ContraAttnUNET"))
    if model_name != "ContraAttnUNET":
        test_loader = data_util.create_dataloader(test_dataset, 4, contra=(model_name == "ContraAttnUNET"))
    else :
        test_loader = data_util.create_dataloader(test_dataset, batch_size, contra=(model_name == "ContraAttnUNET"))

    # Resume evaluation
    kwargs = {
        "optimizer": optim, 
        "scheduler": sched, 
        "start_epoch": start_epoch
    }
    logging.info(f"Starting model training...")
    if model_name == "ContraAttnUNET":
        contra_loss = nn.TripletMarginWithDistanceLoss
        cds_loss = criterions.TruncatedCDS(contra_loss, [1., 1., 1.])
        gen_loss = criterions.RoiMSE(torch.tensor([15.0] * len(indices), dtype=torch.float), indices)
        loss_criterion = criterions.GenerativeContrastiveLoss(cds_loss, gen_loss, contra_loss, 0.0001, 0.0001)
        attn_unet_analysis.train(model, loss_criterion, train_loader, test_loader, num_epochs, lr, save_path=training_save_path, cuda_id=cuda_id)
    else:
        criterion_weights = torch.ones(len(indices), dtype=torch.float)
        # bias some ROIs
        criterion_weights[:] = torch.tensor([10.0] * len(indices), dtype=torch.float)
        # loss_criterion = RoiRRMSE(criterion_weights, indices)
        # loss_criterion = RoiRSE(criterion_weights, indices)
        loss_criterion = criterions.RoiMSE(criterion_weights, indices)
        unetr.train(model, loss_criterion, train_loader, test_loader, num_epochs, lr, save_path=training_save_path, cuda_id=cuda_id, from_checkpoint=True, **kwargs)
    
    testing_save_path = os.path.join(save_path, "testing")
    if not os.path.exists(testing_save_path):
        os.makedirs(testing_save_path)
    with torch.no_grad():
        if model_name == "ContraAttnUNET":
            (mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations), _, _ = attn_unet_analysis.contrastive_test(model, test_loader, indices, criterion_weights, save_path=testing_save_path, cuda_id=cuda_id)
        else :
            mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations = unetr.test(model, test_loader, indices, criterion_weights, save_path=testing_save_path, cuda_id=cuda_id)
        # mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations = attn_unet_analysis.test(model, test_loader, indices, criterion_weights, save_path=testing_save_path, cuda_id=cuda_id)
        # mae, mape, ssim_error, roi_maes, roi_mapes, roi_correlations = unetr.test(model, test_loader, indices, cuda_id=cuda_id)

    logging.info("Test Results:")
    logging.info(f"MAE:{mae}")
    logging.info(f"MAPE:{mape}")
    logging.info(f"SSIM Error:{ssim_error}")
    logging.info(f"ROI MAES:{roi_maes}")
    logging.info(f"ROI MAPES:{roi_mapes}")
    logging.info(f"ROI correlations: {roi_correlations}")
    lowest_corr_idx = np.argmin(roi_correlations)
    highest_corr_idx = np.argmax(roi_correlations)
    logging.info(f"\tLowest correlation (ROI {indices[lowest_corr_idx]}): {roi_correlations[lowest_corr_idx]}")
    logging.info(f"\tHighest correlation (ROI {indices[highest_corr_idx]}): {roi_correlations[highest_corr_idx]}")
    logging.info(f"\tAverage correlation: {np.mean(roi_correlations)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-save_path', help='Path to save output to', type=str, default=base_path)
    parser.add_argument('-model_type', help='The type of model to train', type=str, default="UNETR")
    parser.add_argument('-cuda_id', help='The ID of the CUDA device to train model on', type=int, default=-1)
    parser.add_argument('-batch_size', help='The dataloader batch size', type=int, default=1)
    parser.add_argument('-resume_training', action="store_true")
    parser.add_argument('-checkpoint_path', required=False, type=str, default=False)
    parser.add_argument('-description', required=False, type=str, default="")
    parser.add_argument('-template_space', action="store_true")
    parser.add_argument('-covariates', action="store_true")
    parser.add_argument('-smoothing', action="store_true")
    parser.add_argument('-rnc', action="store_true")
    opt = parser.parse_args()
    save_path = opt.save_path

    log_file_path = os.path.join(save_path, "mse_train.log")
    logging.basicConfig(filename=log_file_path, filemode='w', format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, stream=None, force=True)
    # saved_pca_file = lambda x: f"{os.getcwd()}/saved_pcas_exp/pca_model_{x}.pkl"
    # saved_pcas = [] # [saved_pca_file(i) for i in range(5)]
    # # saved_pcas = glob.glob(f"{os.getcwd()}/saved_pcas_exp/*")
    # saved_split_norm_file = lambda x: f"{os.getcwd()}/split_normalization_vals/split-{x}-mean-std"
    # saved_normalization_vals = [saved_split_norm_file(i) for i in range(5)]
    # saved_normalization_vals = glob.glob(f"{os.getcwd()}/split_normalization_vals/*")
    
    # Random seeds
    # random.seed(0)
    # torch.manual_seed(0)
    # np.random.seed(0)
    logging.info(f"Experiment description: {opt.description}")

    CUDA_ID = opt.cuda_id
    splits_dir = os.path.join(base_path, "training_folds")
    col_list = '' #os.path.join(base_path, "suvr-cols.txt")
    batch_size = opt.batch_size
    num_epochs = 61 # NOTE
    # num_epochs = 51 # NOTE
    # lr = 0.0001
    # lr = 0.0001
    lr = 0.001 # NOTE
    logging.info(f"Learning Rate: {lr}")
    # print(f"Starting 5-Fold Cross Validation\n")
    with_linear_preds = False
    decoder_ds = False

    model_type = opt.model_type

    if model_type == "AttnUNET" or model_type == "ContraAttnUNET":
        model_params = (3, 1, 1, [32, 64, 128, 256, 512], [2]*5) # AttnUNET
        # model_params = (3, 1, 1, [16, 32, 64, 128, 256], [2]*5) # AttnUNET
        # model_params = (3, 1, 1, [16, 32, 64, 128], [2]*4) # AttnUNET NOTE
        # model_params = (3, 1, 1, [32, 64, 128], [2]*3) # AttnUNET
        # model_params = (3, 1, 1, [8, 16, 32], [2]*3) # AttnUNET
    elif model_type == "UNETR" or model_type == "AttnUNETR":
        model_params = (1, 1, (128, 128, 128)) # AttnUnetr, UNETR
    elif model_type == "SwinUNETR" or model_type == "AttnSwinUNETR":
        # model_params = {"in_channels": 1, "out_channels": 1, "img_size": (128, 128, 128), "depths": (2, 2, 6, 2), "feature_size": 24} # SwinUNETR, AttnSwinUNETR
        # model_params = {"in_channels": 1, "out_channels": 1, "img_size": (128, 128, 128), "depths": (2, 2, 2, 2), "feature_size": 12} # SwinUNETR, AttnSwinUNETR
        model_params = {"in_channels": 1, "out_channels": 1, "img_size": (128, 128, 128), "depths": (1, 1, 1, 1), "feature_size": 12} # SwinUNETR, AttnSwinUNETR
    elif model_type == "UNET":
        # model_params = {"spatial_dims": 3, "in_channels": 1, "out_channels": 1, "channels": (4, 8, 16, 32, 64), "strides": (2, 2, 2, 2), "num_res_units": 2}
        model_params = {"spatial_dims": 3, "in_channels": 1, "out_channels": 1, "channels": (32, 64, 128, 256, 512), "strides": (2, 2, 2, 2), "num_res_units": 2}

    # model_type = "AttnUNET" # "UNETR"
    # model_type = "AttnUNETR" # "UNETR"
    # model_type = "SwinUNETR"
    # model_type = "UNETR"
    # model_params = (1, 1, (128, 128, 128)) # AttnUnetr, UNETR
    # model_params = {"in_channels": 1, "out_channels": 1, "img_size": (128, 128, 128), "depths": (2, 2, 2, 2)} # SwinUNETR, AttnSwinUNETR
    # model_params = (3, 1, 1, [32, 64, 128, 256, 512], [2]*5) # AttnUNET
    logging.info(f"Model params: {model_params}")
    
    logging.info("Starting 5-Fold Cross Validation...")
    # volume_validation(os.path.join(splits_dir, "volume_splits"), batch_size, num_epochs, lr, model_params, model_type, cuda_id=CUDA_ID, save_path=save_path)
    if opt.resume_training:
        res_kwargs = {
            "template": opt.template_space,
            "decoder_ds": decoder_ds, 
            "resize": True, 
            "with_covars": opt.covariates, 
            "rnc" : opt.rnc, 
            "smoothing": opt.smoothing
        }
        # from_checkpoint_volume_validation(os.path.join(splits_dir, "outlier_removed_splits"), batch_size, num_epochs, lr, model_params, {"lr": lr}, torch.optim.Adam, {"step_size":300, "gamma":1.}, torch.optim.lr_scheduler.StepLR, opt.checkpoint_path, model_name=model_type, cuda_id=CUDA_ID, **res_kwargs)
        # from_checkpoint_volume_validation(os.path.join(splits_dir, "new_folds"), batch_size, num_epochs, lr, model_params, {"lr": lr}, torch.optim.Adam, {"step_size":300, "gamma":1.}, torch.optim.lr_scheduler.StepLR, opt.checkpoint_path, model_name=model_type, cuda_id=CUDA_ID, **res_kwargs)
        from_checkpoint_volume_validation(os.path.join(splits_dir, "adni_a4_first_scan_combined_folds"), batch_size, num_epochs, lr, model_params, {"lr": lr}, torch.optim.Adam, {"step_size":300, "gamma":1.}, torch.optim.lr_scheduler.StepLR, opt.checkpoint_path, model_name=model_type, cuda_id=CUDA_ID, **res_kwargs)
    else :
        volume_validation_kwargs = {
            "template": opt.template_space, 
            "decoder_ds": decoder_ds, 
            "resize": True, 
            "with_covars": opt.covariates, 
            "rnc" : opt.rnc, 
            "smoothing": opt.smoothing
        }
        # volume_validation(os.path.join(splits_dir, "outlier_removed_splits"), batch_size, num_epochs, lr, model_params, model_type, cuda_id=CUDA_ID, save_path=save_path, **volume_validation_kwargs)
        # volume_validation(os.path.join(splits_dir, "new_folds"), batch_size, num_epochs, lr, model_params, model_type, cuda_id=CUDA_ID, save_path=save_path, **volume_validation_kwargs)
        volume_validation(os.path.join(splits_dir, "adni_a4_first_scan_combined_folds"), batch_size, num_epochs, lr, model_params, model_type, cuda_id=CUDA_ID, save_path=save_path, **volume_validation_kwargs)
    sys.exit(0)
    if opt.resume_training:
        train_from_checkpoint(os.path.join(base_path, "big_training_removed_outliers"), batch_size, num_epochs, lr, model_params, model_type, {"lr": lr}, torch.optim.Adam, {"step_size":20, "gamma":0.5}, torch.optim.lr_scheduler.StepLR, opt.checkpoint_path, cuda_id=CUDA_ID)
    else:
        kwargs = {
            "embeddings_out": with_linear_preds, 
            "train_tau_target_csv": f"{os.getcwd()}/scripts/ADNI_ID_ABETA_TAU.csv", 
            "test_tau_target_csv": f"{os.getcwd()}/scripts/A4_ID_ABETA_TAU.csv"
        }
        single_split_validation(os.path.join(base_path, "big_training_removed_outliers"), batch_size, num_epochs, lr, model_params, model_type, cuda_id=CUDA_ID, save_path=save_path, **kwargs)
    
    # mapes, comp_mapes = five_fold_cross_validation(splits_dir, batch_size, num_epochs, lr, col_list=col_list, saved_normalizations=saved_normalization_vals, pca_models=saved_pcas, cuda_id=CUDA_ID, save_path=save_path)
    # # print(f"Mean Absolute Percentage Errors: {mapes}\tAverage: {sum(mapes)/5}")
    # logging.info(f"Mean Absolute Percentage Errors: {mapes}\tAverage: {sum(mapes)/5}")
    # logging.info(f"Component-wise Mean Absolute Percentage (Decimal) Errors: {comp_mapes}\tAverage: {torch.mean(torch.vstack(comp_mapes), dim=1)}")
    # mapes_dir = os.path.join(save_path, "weighted_mse_mapes")
    # if not os.path.exists(mapes_dir):
    #     os.makedirs(mapes_dir)
    # for i in range(len(mapes)):
    #     np.savez(os.path.join(mapes_dir, f"weighted_mse_train_losses_{i}.npz"), mapes[i].detach().cpu().numpy())