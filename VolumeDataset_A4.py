import os
import sys

sys.path.append(f'{os.getcwd()}')

import torch 
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    
from torch.utils.data import Dataset

import SimpleITK as sitk
import nibabel as nib

import pandas as pd
import numpy as np
np.random.seed(0)

import data_util

class A4VolumeDataset(Dataset):
    def __init__(self, 
                 tau_meta_lookup,
                 resize=True,
                 cuda_id=-1):
        
        self.resize = resize
        self.cuda_id = cuda_id
        self.lookup_df = pd.read_csv(f"{os.getcwd()}/a4_lookup_w_ids.csv")
        # ensure this is aligned
        covariate_lookup = pd.read_csv(f"{os.getcwd()}/scripts/A4_W_Covars_for_DL_inference.csv")
        self.covariate_lookup = covariate_lookup[covariate_lookup["BID"].isin(self.lookup_df["id"].values)]

        sex_codes, _ = pd.factorize(self.covariate_lookup["Sex"])
        self.covariate_lookup["Sex"] = sex_codes

        cols = ['Age', 'Cognition', 'Education']
        self.covariate_lookup[cols] = (self.covariate_lookup[cols] - self.covariate_lookup[cols].min()) / (self.covariate_lookup[cols].max() - self.covariate_lookup[cols].min())

        self.tau_meta_lookup = pd.read_csv(tau_meta_lookup)

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
        ### ABETA	Age	Sex	Education	Cognition
        abeta = self.covariate_lookup.loc[self.covariate_lookup["BID"] == sample_id, "ABETA"].iloc[0]
        age = self.covariate_lookup.loc[self.covariate_lookup["BID"] == sample_id, "Age"].iloc[0]
        sex = self.covariate_lookup.loc[self.covariate_lookup["BID"] == sample_id, "Sex"].iloc[0]
        edu = self.covariate_lookup.loc[self.covariate_lookup["BID"] == sample_id, "Education"].iloc[0]
        cog = self.covariate_lookup.loc[self.covariate_lookup["BID"] == sample_id, "Cognition"].iloc[0]
        meta = self.tau_meta_lookup.loc[self.covariate_lookup["BID"] == sample_id, "Tau_Meta_loc"].iloc[0]

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