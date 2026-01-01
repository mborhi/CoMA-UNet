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

def check_nan(val, default):
    if np.isnan(val):
        return default 
    return val


def resize_volume(volume, new_spacing):
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

class InferenceVolumeDataset(Dataset):
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
        self.covariate_lookup = covariate_lookup[covariate_lookup["SAMPLE_ID"].isin(self.lookup_df["SAMPLE_ID"].values)]

        # cleanup covar df 
        if not ("Sex" in self.covariate_lookup.columns) and "PTGENDER" in self.covariate_lookup.columns:
            self.covariate_lookup = self.covariate_lookup.rename(columns={"PTGENDER": "Sex"})
        if not ("Cognition" in self.covariate_lookup.columns) and "MMSCORE" in self.covariate_lookup.columns:
            self.covariate_lookup = self.covariate_lookup.rename(columns={"MMSCORE": "Cognition"})
        # sex_codes, _ = pd.factorize(self.covariate_lookup["Sex"])
        # self.covariate_lookup["Sex"] = sex_codes
        self.covariate_lookup["Sex"] = self.covariate_lookup["Sex"].map({"Male": 0, "Female": 1, "M": 0, "F": 1})

        # cols = ['Age', 'Cognition', 'Education']
        cols = ['Age', 'Education']
        for col in cols:
            self.covariate_lookup[col] = (self.covariate_lookup[col] - self.covariate_lookup[col].min()) / (self.covariate_lookup[col].max() - self.covariate_lookup[col].min())

        self.tau_meta_lookup = tau_meta_dict
        self.cog_lookup = cog_dict

        self.age_mean = self.covariate_lookup['Age'].mean()
        self.edu_mean = self.covariate_lookup['Education'].mean()

    def __len__(self):
        return self.lookup_df.shape[0]

    def __getitem__(self, idx):
        mri_path = self.lookup_df.iloc[idx, self.lookup_df.columns.get_loc("MRI")]
        # tau_path =  self.lookup_df.iloc[idx, self.lookup_df.columns.get_loc("MRI")]
        # tau_path =  self.lookup_df.iloc[idx, self.lookup_df.columns.get_loc("tau")]
        roi_path =  self.lookup_df.iloc[idx, self.lookup_df.columns.get_loc("roi")]

        # Load MRI, tau PET, and corresponding ROI mask
        mri_tensor = self.load_volume_file(mri_path)
        # mri_tensor = data_util.load_nifti_vol(mri_path, cuda_id=self.cuda_id)
        # tau_tensor = self.load_volume_file(tau_path)
        roi_tensor = self.load_volume_file(roi_path)
        # roi_tensor = data_util.load_nifti_vol(roi_path, cuda_id=self.cuda_id)
        # Mask the background in the MRI
        mri_tensor[roi_tensor == 0] = 0 

        ### Load covariates
        sample_id = self.lookup_df.iloc[idx, self.lookup_df.columns.get_loc("SAMPLE_ID")]
        ## 1. Abeta
        ### Abeta_Covar Age	Sex	Education Cognition
        abeta = self.covariate_lookup.loc[self.covariate_lookup["SAMPLE_ID"] == sample_id, "Abeta_Covar"].iloc[0]
        age = self.covariate_lookup.loc[self.covariate_lookup["SAMPLE_ID"] == sample_id, "Age"].iloc[0]
        sex = self.covariate_lookup.loc[self.covariate_lookup["SAMPLE_ID"] == sample_id, "Sex"].iloc[0]
        edu = self.covariate_lookup.loc[self.covariate_lookup["SAMPLE_ID"] == sample_id, "Education"].iloc[0]

        abeta = check_nan(abeta, 0)
        # age = check_nan(age, self.age_mean)
        age = check_nan(abeta, self.age_mean)
        sex = check_nan(sex, 0)
        # edu = check_nan(edu, self.edu_mean)
        edu = check_nan(sex, self.edu_mean)

        # cog = self.covariate_lookup.loc[self.covariate_lookup["SAMPLE_ID"] == sample_id, "Cognition"].iloc[0]
        cog = self.cog_lookup[sample_id] / 30 # to scale to be between 0 and 1 
        # meta = self.tau_meta_lookup.loc[self.covariate_lookup["SAMPLE_ID"] == sample_id, "Tau_Meta_loc"].iloc[0]
        meta = self.tau_meta_lookup[sample_id]["Tau_Meta"]["loc"]

        if self.ab_covar_lookup_dict is not None and np.isnan(abeta):
            abeta = self.ab_covar_lookup_dict[sample_id]

        covars = torch.from_numpy(np.array([[abeta, age, sex, edu, cog, meta]]))

        if self.cuda_id != -1:
            covars = covars.cuda(self.cuda_id)
        
        # return mri_tensor, tau_tensor, roi_tensor, (abeta, covars), tau_path
        return mri_tensor, mri_tensor, roi_tensor, (abeta, covars), mri_path

    def load_volume_file(self, volume_path):
        # Load volume
        # volume = sitk.ReadImage(volume_path)
        volume = data_util.read_image_with_retry(volume_path, max_retries=10, retry_delay=10) # NOTE 
        # Resize voxels
        if self.resize:
            new_spacing = [2.0, 2.0, 2.0]
            volume = resize_volume(volume, new_spacing)
        # volume_tensor = torch.from_numpy(sitk.GetArrayFromImage(resampled_volume)).to(dtype=torch.float32).unsqueeze(dim=0) # add channel dim
        volume_tensor = torch.from_numpy(sitk.GetArrayFromImage(volume)).to(dtype=torch.float32).unsqueeze(dim=0) # add channel dim
        volume_tensor = torch.nan_to_num(volume_tensor)
        if self.cuda_id != -1:
            volume_tensor = volume_tensor.cuda(self.cuda_id)

        return volume_tensor

def UCSF_test():
    CUDA_ID = 0
    test_lookup_file = f"{os.getcwd()}/scripts/UCSF_data_CatBoost_UQ_predictions/UCSF_paths.csv"
    base = f"{os.getcwd()}/scripts/UCSF_data_CatBoost_UQ_predictions"
    test_tau_meta_lookup_file = os.path.join(base, "CatBoostUQ_Tau_Meta_predictions_for_UCSF_data.json") 
    test_tau_meta_lookup_dict = data_util.load_json_dict(test_tau_meta_lookup_file)
    test_cog_lookup_file = os.path.join(base, f"KNN_MMSCORE_predictions_for_UCSF_data.json") 
    test_cog_lookup_dict = data_util.load_json_dict(test_cog_lookup_file)
    ab_covar_lookup_file = os.path.join(base, "CatBoostUQ_Abeta_Covar_predictions_for_UCSF_data.json")
    ab_covar_lookup_dict = data_util.load_json_dict(ab_covar_lookup_file)
    test_dataloader = DataLoader(
        InferenceVolumeDataset(
            test_lookup_file, 
            test_tau_meta_lookup_dict, 
            test_cog_lookup_dict,
            ab_covar_lookup_dict=ab_covar_lookup_dict, 
            covariate_lookup_file=os.path.join(base, "UCSF_data_Covar_lookup.csv"),
            resize=True, 
            cuda_id=CUDA_ID), 
        shuffle=True, batch_size=1
    )

    # for batch_idx, (mri_tensor, tau_tensor, roi_tensor, (abeta, covars), tau_path) in enumerate(train_dataloader):
    #     print(f"[{batch_idx}]: {abeta, covars} ({tau_path})")
    
    for batch_idx, (mri_tensor, tau_tensor, roi_tensor, (abeta, covars), tau_path) in enumerate(test_dataloader):
        print(f"[{batch_idx}]: {abeta, covars} ({tau_path}) | {mri_tensor.shape}, {roi_tensor.shape}")

def A4_test():
    CUDA_ID = 0
    test_lookup_file = f"{os.getcwd()}/scripts/Unseen_A4_data/unseen_A4_sample_path_lookup.csv"
    base = f"{os.getcwd()}/scripts/Unseen_A4_data"
    test_tau_meta_lookup_file = os.path.join(base, "CatBoostUQ_Tau_Meta_predictions_for_Additional_A4_data.json") 
    test_tau_meta_lookup_dict = data_util.load_json_dict(test_tau_meta_lookup_file)
    test_cog_lookup_file = os.path.join(base, f"KNN_MMSCORE_predictions_for_unseen_A4_data.json") 
    test_cog_lookup_dict = data_util.load_json_dict(test_cog_lookup_file)
    test_dataloader = DataLoader(
        InferenceVolumeDataset(
            test_lookup_file, 
            test_tau_meta_lookup_dict, 
            test_cog_lookup_dict,
            ab_covar_lookup_dict=None, 
            covariate_lookup_file=os.path.join(base, "unseen_A4_Covar_lookup.csv"),
            resize=True, 
            cuda_id=CUDA_ID), 
        shuffle=True, batch_size=1
    )

    # for batch_idx, (mri_tensor, tau_tensor, roi_tensor, (abeta, covars), tau_path) in enumerate(train_dataloader):
    #     print(f"[{batch_idx}]: {abeta, covars} ({tau_path})")
    
    for batch_idx, (mri_tensor, tau_tensor, roi_tensor, (abeta, covars), tau_path) in enumerate(test_dataloader):
        print(f"[{batch_idx}]: {abeta, covars} ({tau_path}) | {mri_tensor.shape}, {roi_tensor.shape}")

def NACC_test():
    CUDA_ID = 0
    test_lookup_file = f"{os.getcwd()}/scripts/NACC_data/NACC_paths.csv"
    base = f"{os.getcwd()}/scripts/NACC_data"
    test_tau_meta_lookup_file = os.path.join(base, "CatBoostUQ_Tau_Meta_predictions_for_NACC.json") 
    test_tau_meta_lookup_dict = data_util.load_json_dict(test_tau_meta_lookup_file)
    test_cog_lookup_file = os.path.join(base, f"KNN_MMSCORE_predictions_for_NACC_data.json") 
    test_cog_lookup_dict = data_util.load_json_dict(test_cog_lookup_file)
    test_dataloader = DataLoader(
        InferenceVolumeDataset(
            test_lookup_file, 
            test_tau_meta_lookup_dict, 
            test_cog_lookup_dict,
            ab_covar_lookup_dict=os.path.join(base, "CatBoostUQ_Abeta_Covar_predictions_for_NACC.json"), 
            covariate_lookup_file=os.path.join(base, "NACC_Covar_lookup.csv"),
            resize=True, 
            cuda_id=CUDA_ID), 
        shuffle=True, batch_size=1
    )

    # for batch_idx, (mri_tensor, tau_tensor, roi_tensor, (abeta, covars), tau_path) in enumerate(train_dataloader):
    #     print(f"[{batch_idx}]: {abeta, covars} ({tau_path})")
    
    for batch_idx, (mri_tensor, tau_tensor, roi_tensor, (abeta, covars), tau_path) in enumerate(test_dataloader):
        print(f"[{batch_idx}]: {abeta, covars} ({tau_path}) | {mri_tensor.shape}, {roi_tensor.shape}")

def NACC_nonSCAN_test():
    CUDA_ID = 0
    test_lookup_file = f"{os.getcwd()}/scripts/NACC_nonSCAN/all_paths.csv"
    base = f"{os.getcwd()}/scripts/NACC_nonSCAN"
    test_tau_meta_lookup_file = os.path.join(base, "CatBoostUQ_Tau_Meta_predictions_for_nonSCAN_NACC.json") 
    test_tau_meta_lookup_dict = data_util.load_json_dict(test_tau_meta_lookup_file)
    test_cog_lookup_file = os.path.join(base, f"KNN_MMSCORE_predictions_for_nonSCAN_NACC.json") 
    test_cog_lookup_dict = data_util.load_json_dict(test_cog_lookup_file)
    test_dataloader = DataLoader(
        InferenceVolumeDataset(
            test_lookup_file, 
            test_tau_meta_lookup_dict, 
            test_cog_lookup_dict,
            ab_covar_lookup_dict=os.path.join(base, "CatBoostUQ_Abeta_Covar_predictions_for_nonSCAN_NACC.json"), 
            covariate_lookup_file=os.path.join(base, "NACC_nonSCAN_Covar_lookup.csv"),
            resize=True, 
            cuda_id=CUDA_ID), 
        shuffle=True, batch_size=1
    )

    # for batch_idx, (mri_tensor, tau_tensor, roi_tensor, (abeta, covars), tau_path) in enumerate(train_dataloader):
    #     print(f"[{batch_idx}]: {abeta, covars} ({tau_path})")
    
    for batch_idx, (mri_tensor, tau_tensor, roi_tensor, (abeta, covars), tau_path) in enumerate(test_dataloader):
        print(f"[{batch_idx}]: {abeta, covars} ({tau_path}) | {mri_tensor.shape}, {roi_tensor.shape}")

def ADNI_wAutopsy_test():
    CUDA_ID = 0
    test_lookup_file = f"{os.getcwd()}/scripts/ADNI_wAutopsy_CatBoostUQ_predictions/ADNI_wAutopsy_paths.csv"
    base = f"{os.getcwd()}/scripts/ADNI_wAutopsy_CatBoostUQ_predictions"
    test_tau_meta_lookup_file = os.path.join(base, "CatBoostUQ_Tau_Meta_predictions_for_ADNI_wAutopsy.json") 
    test_tau_meta_lookup_dict = data_util.load_json_dict(test_tau_meta_lookup_file)
    test_cog_lookup_file = os.path.join(base, f"KNN_MMSCORE_predictions_for_ADNI_wAutopsy.json") 
    test_cog_lookup_dict = data_util.load_json_dict(test_cog_lookup_file)
    test_dataloader = DataLoader(
        InferenceVolumeDataset(
            test_lookup_file, 
            test_tau_meta_lookup_dict, 
            test_cog_lookup_dict,
            ab_covar_lookup_dict=None, 
            covariate_lookup_file=os.path.join(base, "ADNI_wAutopsy_Covar_lookup.csv"),
            resize=True, 
            cuda_id=CUDA_ID), 
        shuffle=True, batch_size=1
    )

    # for batch_idx, (mri_tensor, tau_tensor, roi_tensor, (abeta, covars), tau_path) in enumerate(train_dataloader):
    #     print(f"[{batch_idx}]: {abeta, covars} ({tau_path})")
    
    for batch_idx, (mri_tensor, tau_tensor, roi_tensor, (abeta, covars), tau_path) in enumerate(test_dataloader):
        print(f"[{batch_idx}]: {abeta, covars} ({tau_path}) | {mri_tensor.shape}, {roi_tensor.shape}")

if __name__ == "__main__":
    # A4_test()
    # ADNI_wAutopsy_test()
    # NACC_test()
    NACC_nonSCAN_test()
    pass