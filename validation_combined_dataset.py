import sys
base_path = f"{os.getcwd()}"
sys.path.append(f'{os.getcwd()}')
import os
import pickle 
import glob 
import logging
import argparse
import random

import torch
from torch.utils.data import DataLoader
# torch.backends.cudnn.deterministic = True

# random.seed(0)
# torch.manual_seed(0)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(0)

import torch.nn as nn

import numpy as np

import data_util
import criterions
import attn_unet_data_parallel

from VolumeDataset_ADNI_A4_combined import CombinedVolumeDataset

def hold_out_training_and_val(splits_dir, batch_size, num_epochs, lr, model_params, save_path = "", cuda_id=-1, **kwargs):
    indices = [1001,1006,1007,1009,1015,1016,1030,1034,1033,1008,1025,1029,1031,1022,17,18,2001,2006,2007,2009,2015,2016,2030,2034,2033,2008,2025,2029,2031,2022,49,50,51,52,53,54]

    # ho_save_path = os.path.join(save_path, f"fold_{k+1}")
    ho_save_path = os.path.join(save_path, f"hold_out")
    if not os.path.exists(ho_save_path):
        os.makedirs(ho_save_path)

    model = attn_unet_data_parallel.ContrastiveAttentionUNET_DP(*model_params, latent_spaces=[2048]*len(model_params[-1]), conditional=True, decoder_ds=kwargs["decoder_ds"])
    
    if cuda_id != -1:
        model = model.cuda(cuda_id)
    
    logging.info(model)
    # Load dataset
    logging.info(f"Getting split dataset...")
    ho_train_lookup_file = os.path.join(splits_dir, f"hold_out_training_lookup.csv")
    stacked_train_meta_tau_lookup_dict = {}
    stacked_train_cog_lookup_dict = {}
    # for k in range(5):
    # /home/jagust/Generative-PET-models/training_folds/adni_a4_first_scan_combined_folds/hold_out_aux_prediction_lookups/formatted_hold_out_Tau_Meta_predictions_for_test.json
    train_tau_meta_lookup_file = os.path.join(splits_dir, "hold_out_aux_prediction_lookups", f"formatted_hold_out_Tau_Meta_predictions_for_train.json") 
    train_tau_meta_lookup_dict = data_util.load_json_dict(train_tau_meta_lookup_file)
        # stacked_train_meta_tau_lookup_dict = {**stacked_train_meta_tau_lookup_dict, **train_tau_meta_lookup_dict}
        
    train_cog_lookup_file = os.path.join(splits_dir, "hold_out_aux_prediction_lookups", f"hold_out_MMSCORE_train_predictions.json") 
    train_cog_lookup_dict = data_util.load_json_dict(train_cog_lookup_file)
        # stacked_train_cog_lookup_dict = {**stacked_train_cog_lookup_dict, **train_cog_lookup_dict}
    
    train_dataloader = DataLoader(
            CombinedVolumeDataset(ho_train_lookup_file, train_tau_meta_lookup_dict, train_cog_lookup_dict, resize=True, cuda_id=cuda_id), 
            shuffle=True, batch_size=batch_size
    )
    
    ho_test_lookup_file = os.path.join(splits_dir, f"hold_out_test_lookup.csv")
    stacked_test_meta_tau_lookup_dict = {}
    stacked_test_cog_lookup_dict = {}
    # for k in range(5):
    test_tau_meta_lookup_file = os.path.join(splits_dir, "hold_out_aux_prediction_lookups", f"formatted_hold_out_Tau_Meta_predictions_for_test.json") 
    test_tau_meta_lookup_dict = data_util.load_json_dict(test_tau_meta_lookup_file)
        # stacked_test_meta_tau_lookup_dict = {**stacked_test_meta_tau_lookup_dict, **test_tau_meta_lookup_dict}

    test_cog_lookup_file = os.path.join(splits_dir, "hold_out_aux_prediction_lookups", f"hold_out_MMSCORE_test_predictions.json") 
    test_cog_lookup_dict = data_util.load_json_dict(test_cog_lookup_file)
        # stacked_test_cog_lookup_dict = {**stacked_test_cog_lookup_dict, **test_cog_lookup_dict}

    test_dataloader = DataLoader(
        CombinedVolumeDataset(ho_test_lookup_file, test_tau_meta_lookup_dict, test_cog_lookup_dict, resize=True, cuda_id=cuda_id), 
        shuffle=False, batch_size=batch_size
    )

    criterion_weights = torch.tensor([225.0] * len(indices), dtype=torch.float)

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
    decoder_loss = contra_loss(1)

    loss_criterion = criterions.GenerativeContrastiveLoss(cds_loss, gen_loss, decoder_loss, regulatory_weight=0., ds_regulatory_weight=1.) #NOTE
    logging.info(f"{loss_criterion}")
    model.set_save_attn(None)
    # attn_unet_analysis.train(model, loss_criterion, train_loader, test_loader, num_epochs, lr, save_path=fold_save_path, cuda_id=cuda_id)
    attn_unet_data_parallel.train_dp(model, loss_criterion, train_dataloader, test_dataloader, num_epochs, lr, save_path=ho_save_path, cuda_id=cuda_id, **{"fold_id": None, "with_test_loader": True})


    with torch.no_grad():
        (mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations, voxel_mapes), _, _ = attn_unet_data_parallel.contrastive_test(model, test_dataloader, indices, criterion_weights, save_path=ho_save_path, cuda_id=cuda_id, with_train_loader=False, **{"fold_id": None})
        
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



def cross_validation(splits_dir, batch_size, num_epochs, lr, model_params, save_path = "", cuda_id=-1, **kwargs):
    indices = [1001,1006,1007,1009,1015,1016,1030,1034,1033,1008,1025,1029,1031,1022,17,18,2001,2006,2007,2009,2015,2016,2030,2034,2033,2008,2025,2029,2031,2022,49,50,51,52,53,54]
    avg_mae, avg_mape, avg_ssim = 0, 0, 0
    avg_roi_corrcoefs, avg_roi_maes, avg_roi_mapes = np.zeros(len(indices)), torch.zeros(len(indices), device=cuda_id), torch.zeros(len(indices), device=cuda_id)
    
    # for k in range(3, 5):
    # for k in range(2, 5):
    for k in range(3, 1, -1):
        logging.info(f"Starting fold {k+1}...")
        # Set up save dir
        fold_save_path = os.path.join(save_path, f"fold_{k+1}")
        if not os.path.exists(fold_save_path):
            os.makedirs(fold_save_path)
    
        model = attn_unet_data_parallel.ContrastiveAttentionUNET_DP(*model_params, latent_spaces=[2048]*len(model_params[-1]), conditional=True, decoder_ds=kwargs["decoder_ds"])

        if cuda_id != -1:
            model = model.cuda(cuda_id)
        
        logging.info(model)
        # Load dataset
        logging.info(f"Getting split dataset...")
        train_lookup_file = os.path.join(splits_dir, f"training_lookup_{k+1}.csv")
        train_tau_meta_lookup_file = os.path.join(splits_dir, "meta_tau_lookups", f"formatted_fold_{k}_Tau_Meta_predictions_for_train.json") 
        train_tau_meta_lookup_dict = data_util.load_json_dict(train_tau_meta_lookup_file)
        train_cog_lookup_file = os.path.join(splits_dir, "cognition_lookups", f"fold_{k}_MMSCORE_train_predictions.json") 
        train_cog_lookup_dict = data_util.load_json_dict(train_cog_lookup_file)
        train_dataloader = DataLoader(
                CombinedVolumeDataset(train_lookup_file, train_tau_meta_lookup_dict, train_cog_lookup_dict, resize=True, cuda_id=cuda_id), 
                shuffle=True, batch_size=batch_size
        )

        test_lookup_file = os.path.join(splits_dir, f"test_lookup_{k+1}.csv")
        test_tau_meta_lookup_file = os.path.join(splits_dir, "meta_tau_lookups", f"formatted_fold_{k}_Tau_Meta_predictions_for_test.json") 
        test_tau_meta_lookup_dict = data_util.load_json_dict(test_tau_meta_lookup_file)
        test_cog_lookup_file = os.path.join(splits_dir, "cognition_lookups", f"fold_{k}_MMSCORE_test_predictions.json") 
        test_cog_lookup_dict = data_util.load_json_dict(test_cog_lookup_file)
        test_dataloader = DataLoader(
            CombinedVolumeDataset(test_lookup_file, test_tau_meta_lookup_dict, test_cog_lookup_dict, resize=True, cuda_id=cuda_id), 
            shuffle=False, batch_size=batch_size
        )

        
        criterion_weights = torch.tensor([225.0] * len(indices), dtype=torch.float)

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
        decoder_loss = contra_loss(1)

        loss_criterion = criterions.GenerativeContrastiveLoss(cds_loss, gen_loss, decoder_loss, regulatory_weight=0., ds_regulatory_weight=1.) #NOTE
        logging.info(f"{loss_criterion}")
        model.set_save_attn(None)
        # attn_unet_analysis.train(model, loss_criterion, train_loader, test_loader, num_epochs, lr, save_path=fold_save_path, cuda_id=cuda_id)
        attn_unet_data_parallel.train_dp(model, loss_criterion, train_dataloader, test_dataloader, num_epochs, lr, save_path=fold_save_path, cuda_id=cuda_id, **{"fold_id": k, "with_test_loader": True})
    
    
        with torch.no_grad():
            (mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations, voxel_mapes), _, _ = attn_unet_data_parallel.contrastive_test(model, test_dataloader, indices, criterion_weights, save_path=fold_save_path, cuda_id=cuda_id, with_train_loader=False, **{"fold_id": k})
            
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

def from_checkpoint_cross_validation(splits_dir, batch_size, num_epochs, lr, model_params, optimizer_params, optimizer, scheduler_params, scheduler, checkpoint_path, save_path = "", cuda_id=-1, **kwargs):
    save_path = os.path.join("/".join(checkpoint_path.split("/")[:6]))
    from_checkpoint = True

    indices = [1001,1006,1007,1009,1015,1016,1030,1034,1033,1008,1025,1029,1031,1022,17,18,2001,2006,2007,2009,2015,2016,2030,2034,2033,2008,2025,2029,2031,2022,49,50,51,52,53,54]
    avg_mae, avg_mape, avg_ssim = 0, 0, 0
    avg_roi_corrcoefs, avg_roi_maes, avg_roi_mapes = np.zeros(len(indices)), torch.zeros(len(indices), device=cuda_id), torch.zeros(len(indices), device=cuda_id)
    
    start_fold = int(checkpoint_path.split("/")[6].split("_")[-1]) - 1

    for k in range(start_fold, 5):
        logging.info(f"Continuing fold {k+1}...")
        # Set up save dir
        if k+1 == int(checkpoint_path.split("/")[6].split("_")[-1]):
            checkpoint_state_dicts = torch.load(checkpoint_path, map_location=torch.device(f'cpu'))
            start_epoch = int(checkpoint_state_dicts['epoch'])+1
            new_resume_path = "native_target_finetune_" + checkpoint_path.split("/")[6]
            fold_save_path = os.path.join(save_path, new_resume_path) # must change dir to avoid overwriting graphs
            if not os.path.exists(fold_save_path):
                os.makedirs(fold_save_path)
        else:
            from_checkpoint = False
            fold_save_path = os.path.join(save_path, f"fold_{k+1}")
            if not os.path.exists(fold_save_path):
                os.makedirs(fold_save_path)
    
        model = attn_unet_data_parallel.ContrastiveAttentionUNET_DP(*model_params, latent_spaces=[2048]*len(model_params[-1]), conditional=True, decoder_ds=kwargs["decoder_ds"])

        if cuda_id != -1:
            model = model.cuda(cuda_id)

        if from_checkpoint:
            optim = optimizer(model.parameters(), **optimizer_params)
            # optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
            model.load_state_dict(checkpoint_state_dicts['model_state_dict'])
            optim.load_state_dict(checkpoint_state_dicts['optimizer_state_dict'])
            sched = None
            if 'scheduler_state_dict' in checkpoint_state_dicts.keys():
                sched = scheduler(optim, **scheduler_params)
                sched.load_state_dict(checkpoint_state_dicts['scheduler_state_dict'])
        
        logging.info(model)
        # Load dataset
        logging.info(f"Getting split dataset...")
        ### NOTE this should be the "testing" if we consider using a nested training split to capture the eh relatoinship between the predictions and uncertainty
        train_lookup_file = os.path.join(splits_dir, f"training_lookup_{k+1}.csv")
        train_tau_meta_lookup_file = os.path.join(splits_dir, "meta_tau_lookups", f"formatted_fold_{k}_Tau_Meta_predictions_for_train.json") 
        train_tau_meta_lookup_dict = data_util.load_json_dict(train_tau_meta_lookup_file)
        train_cog_lookup_file = os.path.join(splits_dir, "cognition_lookups", f"fold_{k}_MMSCORE_train_predictions.json") 
        train_cog_lookup_dict = data_util.load_json_dict(train_cog_lookup_file)
        train_dataloader = DataLoader(
                CombinedVolumeDataset(train_lookup_file, train_tau_meta_lookup_dict, train_cog_lookup_dict, resize=True, cuda_id=cuda_id), 
                shuffle=True, batch_size=batch_size
        )
        ## NOTE this should be the out-of-sample prediction from training the CatBoost model with Uncertainty quantification on the full "cross -val " dataset and prediction the entiret hold out,
        # urrently none of the of the predictions n these files are for the cross -val samples --- at least they shouldn't be
        test_lookup_file = os.path.join(splits_dir, f"test_lookup_{k+1}.csv")
        test_tau_meta_lookup_file = os.path.join(splits_dir, "meta_tau_lookups", f"formatted_fold_{k}_Tau_Meta_predictions_for_test.json") 
        test_tau_meta_lookup_dict = data_util.load_json_dict(test_tau_meta_lookup_file)
        test_cog_lookup_file = os.path.join(splits_dir, "cognition_lookups", f"fold_{k}_MMSCORE_test_predictions.json") 
        test_cog_lookup_dict = data_util.load_json_dict(test_cog_lookup_file)
        test_dataloader = DataLoader(
            CombinedVolumeDataset(test_lookup_file, test_tau_meta_lookup_dict, test_cog_lookup_dict, resize=True, cuda_id=cuda_id), 
            shuffle=False, batch_size=batch_size
        )

        
        criterion_weights = torch.tensor([225.0] * len(indices), dtype=torch.float)

        kwargs_ = {}
        if from_checkpoint:
            kwargs_ = {
                "optimizer": optim, 
                "scheduler": sched, 
                "start_epoch": start_epoch, 
            }

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
        decoder_loss = contra_loss(1)

        loss_criterion = criterions.GenerativeContrastiveLoss(cds_loss, gen_loss, decoder_loss, regulatory_weight=0., ds_regulatory_weight=1.) #NOTE
        logging.info(f"{loss_criterion}")
        model.set_save_attn(None)
        # attn_unet_analysis.train(model, loss_criterion, train_loader, test_loader, num_epochs, lr, save_path=fold_save_path, cuda_id=cuda_id)
        attn_unet_data_parallel.train_dp(model, loss_criterion, train_dataloader, test_dataloader, num_epochs, lr, save_path=fold_save_path, cuda_id=cuda_id, **{**kwargs_, **{"fold_id": k, "with_test_loader": True}})
    
    
        with torch.no_grad():
            (mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations, voxel_mapes), _, _ = attn_unet_data_parallel.contrastive_test(model, test_dataloader, indices, criterion_weights, save_path=fold_save_path, cuda_id=cuda_id, with_train_loader=False, **{"fold_id": k})
            
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
    parser.add_argument('-cross_val', action="store_true")
    opt = parser.parse_args()
    save_path = opt.save_path

    log_file_path = os.path.join(save_path, "mse_train.log")
    # logging.basicConfig(filename=log_file_path, filemode='w', format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, stream=None, force=True)
    logging.basicConfig(filename=log_file_path, filemode='w', format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, force=True)

    logging.info(f"Experiment description: {opt.description}")

    CUDA_ID = opt.cuda_id
    splits_dir = os.path.join(base_path, "training_folds")
    col_list = '' #os.path.join(base_path, "suvr-cols.txt")
    batch_size = opt.batch_size
    num_epochs = 61 # NOTE
    # lr = 0.0001
    lr = 0.0001
    # lr = 0.001 # NOTE
    logging.info(f"Learning Rate: {lr}")
    # print(f"Starting 5-Fold Cross Validation\n")
    with_linear_preds = False
    decoder_ds = False

    model_type = opt.model_type

    model_params = (3, 1, 1, [32, 64, 128, 256, 512], [2]*5)
    logging.info(f"Model params: {model_params}")
    
    logging.info("Starting 5-Fold Cross Validation...")
    val_kwargs = {
        "template": opt.template_space, 
        "decoder_ds": decoder_ds, 
        "resize": True, 
        "with_covars": opt.covariates, 
        "rnc" : opt.rnc, 
        "smoothing": opt.smoothing
    }

    if opt.cross_val:
        if opt.resume_training:
            res_kwargs = {
                "template": opt.template_space,
                "decoder_ds": decoder_ds, 
                "resize": True, 
                "with_covars": opt.covariates, 
                "rnc" : opt.rnc, 
                "smoothing": opt.smoothing
            }
            from_checkpoint_cross_validation(os.path.join(splits_dir, "adni_a4_first_scan_combined_folds"), batch_size, num_epochs, lr, model_params, optimizer_params={'lr': lr}, optimizer=torch.optim.AdamW, scheduler_params={"mode":'min', "patience":5, 'factor': 0.2}, scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau, checkpoint_path=opt.checkpoint_path, cuda_id=CUDA_ID, save_path=save_path, **val_kwargs)
        else:
            cross_validation(os.path.join(splits_dir, "adni_a4_first_scan_combined_folds"), batch_size, num_epochs, lr, model_params, cuda_id=CUDA_ID, save_path=save_path, **val_kwargs)
    else:
        hold_out_training_and_val(os.path.join(splits_dir, "adni_a4_first_scan_combined_folds"), batch_size, num_epochs, lr, model_params, cuda_id=CUDA_ID, save_path=save_path, **val_kwargs)
    