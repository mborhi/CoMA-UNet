import os
import sys

sys.path.append(f'{os.getcwd()}')

import json

import torch 
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import SimpleITK as sitk
import nibabel as nib

import pandas as pd
import numpy as np
np.random.seed(0)

import data_util

class CombinedVolumeDataset(Dataset):
    def __init__(self, 
                 lookup_df_file,
                 tau_meta_dict,
                 cog_dict,
                 resize=True,
                 ab_covar_lookup_dict=None,
                 covariate_lookup_file=f"{os.getcwd()}/scripts/A4_ADNI_combined_W_Covars.csv",
                 cuda_id=-1):
        
        self.resize = resize
        self.cuda_id = cuda_id
        self.ab_covar_lookup_dict = ab_covar_lookup_dict
        self.lookup_df = pd.read_csv(lookup_df_file)
        print(self.lookup_df.head())
        # ensure this is aligned
        covariate_lookup = pd.read_csv(covariate_lookup_file)
        print(covariate_lookup.head())
        self.covariate_lookup = covariate_lookup[covariate_lookup["SAMPLE_ID"].isin(self.lookup_df["id"].values)]

        # sex_codes, _ = pd.factorize(self.covariate_lookup["Sex"])
        # self.covariate_lookup["Sex"] = sex_codes
        self.covariate_lookup["Sex"] = self.covariate_lookup["Sex"].map({"Male": 0, "Female": 1, "M": 0, "F": 1})

        cols = ['Age', 'Cognition', 'Education']
        self.covariate_lookup[cols] = (self.covariate_lookup[cols] - self.covariate_lookup[cols].min()) / (self.covariate_lookup[cols].max() - self.covariate_lookup[cols].min())

        self.tau_meta_lookup = tau_meta_dict
        self.cog_lookup = cog_dict

    def __len__(self):
        return self.lookup_df.shape[0]

    def __getitem__(self, idx):
        mri_path = self.lookup_df.iloc[idx, self.lookup_df.columns.get_loc("MRI")]
        tau_path =  self.lookup_df.iloc[idx, self.lookup_df.columns.get_loc("tau")]
        roi_path =  self.lookup_df.iloc[idx, self.lookup_df.columns.get_loc("roi")]

        # Load MRI, tau PET, and corresponding ROI mask
        mri_tensor = self.load_volume_file(mri_path, is_mask=False)
        tau_tensor = self.load_volume_file(tau_path)
        roi_tensor = self.load_volume_file(roi_path, is_mask=False)
        # Mask the background in the MRI
        mri_tensor[roi_tensor == 0] = 0 

        ### Load covariates
        sample_id = self.lookup_df.iloc[idx, self.lookup_df.columns.get_loc("id")]
        ## 1. Abeta
        ### Abeta_Covar Age	Sex	Education Cognition
        abeta = self.covariate_lookup.loc[self.covariate_lookup["SAMPLE_ID"] == sample_id, "Abeta_Covar"].iloc[0]
        age = self.covariate_lookup.loc[self.covariate_lookup["SAMPLE_ID"] == sample_id, "Age"].iloc[0]
        sex = self.covariate_lookup.loc[self.covariate_lookup["SAMPLE_ID"] == sample_id, "Sex"].iloc[0]
        edu = self.covariate_lookup.loc[self.covariate_lookup["SAMPLE_ID"] == sample_id, "Education"].iloc[0] / 30 # div by 30 to scale to be between 0 and 1
        # cog = self.covariate_lookup.loc[self.covariate_lookup["SAMPLE_ID"] == sample_id, "Cognition"].iloc[0]
        cog = self.cog_lookup[sample_id]
        # meta = self.tau_meta_lookup.loc[self.covariate_lookup["SAMPLE_ID"] == sample_id, "Tau_Meta_loc"].iloc[0]
        meta = self.tau_meta_lookup[sample_id]["Tau_Meta"]["loc"]

        if self.ab_covar_lookup_dict is not None and np.isnan(abeta):
            abeta = self.ab_covar_lookup_dict[sample_id]

        covars = torch.from_numpy(np.array([[abeta, age, sex, edu, cog, meta]]))

        if self.cuda_id != -1:
            covars = covars.cuda(self.cuda_id)
        
        return mri_tensor, tau_tensor, roi_tensor, (abeta, covars), tau_path

    def load_volume_file(self, volume_path, is_mask=False):
        # Load volume
        # volume = sitk.ReadImage(volume_path)
        volume = data_util.read_image_with_retry(volume_path, max_retries=10, retry_delay=10) # NOTE 
        # Resize voxels
        if self.resize:
            new_spacing = [2.0, 2.0, 2.0]
            volume = self.resize_volume(volume, new_spacing)
        # volume_tensor = torch.from_numpy(sitk.GetArrayFromImage(resampled_volume)).to(dtype=torch.float32).unsqueeze(dim=0) # add channel dim
        volume_tensor = torch.from_numpy(sitk.GetArrayFromImage(volume)).to(dtype=torch.float32).unsqueeze(dim=0) # add channel dim
        volume_tensor = torch.nan_to_num(volume_tensor)
        if self.cuda_id != -1:
            volume_tensor = volume_tensor.cuda(self.cuda_id)

        return volume_tensor

    def resize_volume(self, volume, new_spacing):
        # Resample images to 2mm spacing with SimpleITK
        original_spacing = volume.GetSpacing()
        original_size = volume.GetSize()

        out_size = [
            int(np.round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
            int(np.round(original_size[2] * (original_spacing[2] / new_spacing[2])))
        ]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(new_spacing)
        resample.SetSize(out_size)
        resample.SetOutputDirection(volume.GetDirection())
        resample.SetOutputOrigin(volume.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(volume.GetPixelIDValue())

        # resample.SetInterpolator(sitk.sitkBSpline)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)

        resampled_volume = resample.Execute(volume)

        return resampled_volume 

if __name__ == "__main__":
    CUDA_ID = 0
    # cross_val_lookup_folder = f"{os.getcwd()}/training_folds/adni_a4_first_scan_combined_folds"
    # for fold_idx in range(5):
    #     train_lookup_file = os.path.join(cross_val_lookup_folder, f"training_lookup_{fold_idx+1}.csv")
    #     train_tau_meta_lookup_file = os.path.join(cross_val_lookup_folder, "meta_tau_lookups", f"formatted_fold_{fold_idx}_Tau_Meta_predictions_for_train.json") 
    #     train_tau_meta_lookup_dict = data_util.load_json_dict(train_tau_meta_lookup_file)
    #     train_cog_lookup_file = os.path.join(cross_val_lookup_folder, "cognition_lookups", f"fold_{fold_idx}_MMSCORE_train_predictions.json") 
    #     train_cog_lookup_dict = data_util.load_json_dict(train_cog_lookup_file)
    #     train_dataloader = DataLoader(
    #             CombinedVolumeDataset(train_lookup_file, train_tau_meta_lookup_dict, train_cog_lookup_dict, resize=True, cuda_id=CUDA_ID), 
    #             shuffle=True, batch_size=1
    #     )
        
    #     test_lookup_file = os.path.join(cross_val_lookup_folder, f"test_lookup_{fold_idx+1}.csv")
    #     test_tau_meta_lookup_file = os.path.join(cross_val_lookup_folder, "meta_tau_lookups", f"formatted_fold_{fold_idx}_Tau_Meta_predictions_for_test.json") 
    #     test_tau_meta_lookup_dict = data_util.load_json_dict(test_tau_meta_lookup_file)
    #     test_cog_lookup_file = os.path.join(cross_val_lookup_folder, "cognition_lookups", f"fold_{fold_idx}_MMSCORE_test_predictions.json") 
    #     test_cog_lookup_dict = data_util.load_json_dict(test_cog_lookup_file)
    #     test_dataloader = DataLoader(
    #         CombinedVolumeDataset(test_lookup_file, test_tau_meta_lookup_dict, test_cog_lookup_dict, resize=True, cuda_id=CUDA_ID), 
    #         shuffle=True, batch_size=1
    #     )

    #     for batch_idx, (mri_tensor, tau_tensor, roi_tensor, (abeta, covars), tau_path) in enumerate(train_dataloader):
    #         print(f"[{batch_idx}]: {abeta, covars} ({tau_path})")
        
    #     for batch_idx, (mri_tensor, tau_tensor, roi_tensor, (abeta, covars), tau_path) in enumerate(test_dataloader):
    #         print(f"[{batch_idx}]: {abeta, covars} ({tau_path})")

    test_lookup_file = f"{os.getcwd()}/ADNI_completely_unseen_samples_paths.csv"
    base = f"{os.getcwd()}/scripts/CatBoostUQ_all_combined_cross_val_data_trained_demographic_biofluid_native_space_roi_tau_unseen_ADNI_predictions"
    test_tau_meta_lookup_file = os.path.join(base, "CatBoostUQ_TauMeta_predictions_for_unseen_ADNI.json") 
    test_tau_meta_lookup_dict = data_util.load_json_dict(test_tau_meta_lookup_file)
    test_cog_lookup_file = os.path.join(base, f"KNN_MMSCORE_predictions_for_unseen_ADNI.json") 
    test_cog_lookup_dict = data_util.load_json_dict(test_cog_lookup_file)
    ab_covar_lookup_file = os.path.join(base, "CatBoostUQ_Abeta_Covar_predictions_for_unseen_ADNI.json")
    ab_covar_lookup_dict = data_util.load_json_dict(ab_covar_lookup_file)
    test_dataloader = DataLoader(
        CombinedVolumeDataset(
            test_lookup_file, 
            test_tau_meta_lookup_dict, 
            test_cog_lookup_dict,
            ab_covar_lookup_dict=ab_covar_lookup_dict, 
            covariate_lookup_file="ADNI_W_Covars_formatted.csv",
            resize=True, 
            cuda_id=CUDA_ID), 
        shuffle=True, batch_size=1
    )

    # for batch_idx, (mri_tensor, tau_tensor, roi_tensor, (abeta, covars), tau_path) in enumerate(train_dataloader):
    #     print(f"[{batch_idx}]: {abeta, covars} ({tau_path})")
    
    for batch_idx, (mri_tensor, tau_tensor, roi_tensor, (abeta, covars), tau_path) in enumerate(test_dataloader):
        print(f"[{batch_idx}]: {abeta, covars} ({tau_path})")