import os
import sys

# from data_util import np
sys.path.append(f'{os.getcwd()}')

import random
import logging 

import torch 
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset, Sampler
import torch.nn.functional as F

from monai.transforms import LoadImage, ResizeWithPadOrCrop, GaussianSmooth, Compose

import SimpleITK as sitk
import nibabel as nib

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import glob
import pandas as pd
import numpy as np
np.random.seed(0)

# from data_util import * 
import data_util
import attn_unet_analysis

class VolumeDataset(Dataset):
    def __init__(self, 
                 lookup_file,
                 resize=True,
                 transform=None, 
                 mri_mean=0, 
                 mri_std=1, 
                 suvr_mean=np.float(0), 
                 suvr_std=np.float(1), 
                 mri_file_type=None,
                 tau_file_type=None,
                 smoothing=False,
                #  cache_rate=1.0,
                #  num_workers=1, 
                 cuda_id=-1):
        
        # super(VolumeDataset, self).__init__(cache_rate=cache_rate, num_workers=num_workers)
        # "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/014-S-6988/PET_2022-06-30_FTP/analysis/rnu.nii"
        
        self.transform = transform
        self.resize = resize
        # self.col_list = col_list
        self.mri_mean = mri_mean
        self.mri_std = mri_std
        self.suvr_mean = suvr_mean
        self.suvr_std = suvr_std
        self.mri_file_type = mri_file_type
        self.tau_file_type = tau_file_type
        self.cuda_id = cuda_id
        self.lookup_df = pd.read_csv(lookup_file)
        # Image loading
        self.smoothing = smoothing
        self.template_space = False
        # self.TEMPLATE_SPACE_ROI_PATH = f"{os.getcwd()}/scripts/pet_dimYeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii"
        self.TEMPLATE_SPACE_ROI_PATH = f"{os.getcwd()}/scripts/AD_ROI_MASK.nii"
        self.TEMPLATE_SPACE_MRI_MASK_PATH = f"{os.getcwd()}/scripts/rmni152_2009_256.nii"
        self.TAU_MASK_MNI = f"{os.getcwd()}/scripts/tau_maskmni152_2009_256.nii"
        logging.info(f"ROI Path: {self.TEMPLATE_SPACE_ROI_PATH}")
        # (padding_left,padding_right, padding_top, padding_bottom, padding_top,padding_bottom, padding_front,padding_back, padding_front, padding_back)
        mri_mask_tensor_original_size = self.load_volume_file_with_mask(self.TEMPLATE_SPACE_MRI_MASK_PATH, lambda x: x, resize=False)
        def mri_masking_fn(v):
            v[mri_mask_tensor_original_size == 0] = 0
            return v
        self.mri_masking_fn = mri_masking_fn
        tau_mask_tensor_original_size = self.load_volume_file_with_mask(self.TEMPLATE_SPACE_ROI_PATH, lambda x: x, resize=False)
        def tau_masking_fn(v):
            v[tau_mask_tensor_original_size == 0] = 0
            return v
        self.tau_masking_fn = tau_masking_fn

        self.valid_indices = [
            i for i in range(self.lookup_df.shape[0]) 
            if os.path.exists(self.lookup_df.iloc[i, self.lookup_df.columns.get_loc("MRI")])
        ]

        invalid_ids = [
            self.lookup_df.iloc[i, self.lookup_df.columns.get_loc("MRI")] 
            for i in range(self.lookup_df.shape[0]) 
            if not os.path.exists(self.lookup_df.iloc[i, self.lookup_df.columns.get_loc("MRI")])
        ]
        print("invalid sids", len(invalid_ids), invalid_ids)
        
        # NOTE new way to filter invalid file paths
        self.lookup_df = self.lookup_df.iloc[self.valid_indices]

    def __len__(self):
        # return self.lookup_df.shape[0]
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # logging.info(idx)
        # idx = self.valid_indices[idx]
        mri_path = self.lookup_df.iloc[idx, self.lookup_df.columns.get_loc("MRI")]
        tau_path =  self.lookup_df.iloc[idx, self.lookup_df.columns.get_loc("tau")]
        roi_path =  self.lookup_df.iloc[idx, self.lookup_df.columns.get_loc("roi")]

        if isinstance(self.mri_file_type, str):
            mri_dir_path = "/".join(mri_path.split("/")[:-1])
            mri_path = os.path.join(mri_dir_path, self.mri_file_type)
            if self.mri_file_type[0] == "w":
                self.template_space = True

        if isinstance(self.tau_file_type, str):
            tau_dir_path = "/".join(tau_path.split("/")[:-1])
            tau_path = os.path.join(tau_dir_path, self.tau_file_type)

        if self.template_space:
            # (padding_left,padding_right, padding_top, padding_bottom, padding_top,padding_bottom, padding_front,padding_back, padding_front, padding_back)
            self.pad_dims = (128, 128, 128) if self.resize else (216, 216, 216)
            self.transform = data_util.pad_volume(self.pad_dims) 
            # roi_path = self.TEMPLATE_SPACE_ROI_PATH # NOTE, currently used; removed for analysis
            # mri_mask_path = self.TEMPLATE_SPACE_MRI_MASK_PATH

        # Load MRI, tau PET, and corresponding ROI mask
        # mri_tensor = self.load_volume_file(mri_path, nonzeros=False)
        mri_tensor = self.load_volume_file(mri_path, is_mask=False)
        # mri_tensor = self.load_volume_file_with_mask(mri_path, self.mri_masking_fn, resize=self.resize)

        tau_tensor = self.load_volume_file(tau_path)
        # tau_tensor = self.load_volume_file_with_mask(tau_path, self.tau_masking_fn, resize=self.resize)
        
        roi_tensor = self.load_volume_file(roi_path, is_mask=False)

        if self.smoothing:
            smoothing = GaussianSmooth()
            tau_tensor = smoothing(tau_tensor.squeeze(0)).unsqueeze(0)

        # NOTE changed for full image
        if not self.template_space:
            mri_tensor[roi_tensor == 0] = 0 
            # tau_tensor[roi_tensor == 0] = 0 
            pass
        else:
            # tau_tensor[roi_tensor == 0] = 0
            pass
            # mri_mask_tensor = self.load_volume_file(self.TEMPLATE_SPACE_MRI_MASK_PATH, nonzeros=False)
            # msk = mri_mask_tensor > 0 
            # mri_tensor[mri_mask_tensor == 0] = 0

        if self.template_space:
            # NOTE new for skull stripping
            tau_masking_tensor = self.load_volume_file(self.TAU_MASK_MNI, is_mask=True)
            tau_tensor[tau_masking_tensor == 0] = 0
        else:
            # tau_tensor[roi_tensor == 0] = 0 # NOTE
            pass


        """
        % load the 3 images
        i1=load(*subject roi image*) %% load in the parcellated brain
        mri=load(*subject mri*) %% load in the mri scan
        tau=load(*subject tau*) %% load in the tau scan

        % mask out the skull etc from the tau and mri images
        mri(i1==0)=0; %% assign all voxels in the mri scan as 0 where voxels in the parcellated brain (i1) are 0
        tau(i1==0)=0; %% assign all voxels in the tau scan as 0 where voxels in the parcellated brain (i1) are 0

        % generate the informative regions mask
        mask=zeros(size(i1)); %% define a new empty matrix 

        % informative indicies are (1001,1006,1007,1009,1015,1016,1030,1034,1033,1008,1025,1029,1031,1022,17,18,2001,2006,2007,2009,2015,2016,2030,2034,2033,2008,2025,2029,2031,2022,49,50,51,52,53,54; 

        mask(i1==1001|i1==1006|i1==1007|i1==1009|i1==1015|i1==1016|i1==1030|i1==1034|i1==1033|i1==1008|i1==1025|i1==1029|i1==1031|i1==1022|i1==17|i1==18|i1==2001|i1==2006|i1==2007|i1==2009|i1==2015|i1==2016|i1==2030|i1==2034|i1==2033|i1==2008|i1==2025|i1==2029|i1==2031|i1==2022|i1==49|i1==50|i1==51|i1==52|i1==53|i1==54)=1; %% assign all voxels of interest in the mask as 1
        """
        
        return mri_tensor, tau_tensor, roi_tensor, tau_path 

    def load_volume_file_with_mask(self, volume_path, apply_mask_fn, resize=False, is_mask=False):
        # volume = sitk.ReadImage(volume_path)
        volume = data_util.read_image_with_retry(volume_path, max_retries=10, retry_delay=10) # NOTE 
        volume_tensor = torch.from_numpy(sitk.GetArrayFromImage(volume)).to(dtype=torch.float32).unsqueeze(dim=0) # add channel dim
        # volume_matrix = np.transpose(nib.load(volume_path).get_fdata(), (2, 1, 0))
        # volume_tensor = torch.from_numpy(volume_matrix).to(dtype=torch.float32).unsqueeze(dim=0) # add channel dim
        # volume_tensor = data_util.read_image_nib_with_retry(volume_path)
        # Apply mask
        volume_tensor = apply_mask_fn(volume_tensor)
        # Resize
        if resize:
            new_spacing = [2.0, 2.0, 2.0]
            volume = sitk.GetImageFromArray(volume_tensor.squeeze(0).detach().cpu().numpy())
            volume = self.resize_volume(volume, new_spacing)
            volume_tensor = torch.from_numpy(sitk.GetArrayFromImage(volume)).to(dtype=torch.float32).unsqueeze(0)
        
        if not is_mask:
            # volume_tensor = torch.div(torch.sub(volume_tensor, self.mri_mean), self.mri_std)
            volume_tensor = volume_tensor / 255.
            
        if self.cuda_id != -1:
            volume_tensor = volume_tensor.cuda(self.cuda_id)

        if self.transform:
            volume_tensor = self.apply_transforms(volume_tensor)

        return volume_tensor

    def load_volume_file_nib(self, volume_path, smooth=False):
        volume = data_util.read_image_nib_with_retry(volume_path, max_retries=10, retry_delay=10)

    def load_volume_file(self, volume_path, is_mask=False):
        # Load volume
        # volume = sitk.ReadImage(volume_path)
        volume = data_util.read_image_with_retry(volume_path, max_retries=10, retry_delay=10) # NOTE 
        # Resize voxels
        if self.resize:
            new_spacing = [2.0, 2.0, 2.0]
            # resampled_volume = self.resize_volume(volume, new_spacing)
            volume = self.resize_volume(volume, new_spacing)
        # volume_tensor = torch.from_numpy(sitk.GetArrayFromImage(resampled_volume)).to(dtype=torch.float32).unsqueeze(dim=0) # add channel dim
        volume_tensor = torch.from_numpy(sitk.GetArrayFromImage(volume)).to(dtype=torch.float32).unsqueeze(dim=0) # add channel dim
        volume_tensor = torch.nan_to_num(volume_tensor) # NOTE new, shouldn't make a difference
        if self.cuda_id != -1:
            volume_tensor = volume_tensor.cuda(self.cuda_id)

        if self.transform:
            volume_tensor = self.apply_transforms(volume_tensor)
        
        # if nonzeros:
        #     volume_tensor = torch.where(volume_tensor != 0, torch.div(torch.sub(volume_tensor, self.mri_mean), self.mri_std), 0)
        if not is_mask:
            # volume_tensor = torch.div(torch.sub(volume_tensor, self.mri_mean), self.mri_std)
            volume_tensor = volume_tensor #255.

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
    
    def apply_transforms(self, volume_tensor):
        # Drop zeros, resize, normalize
        if volume_tensor.size(dim=-3) != self.pad_dims[-3]:
            volume_tensor = self.transform(volume_tensor)
        return volume_tensor

    def set_mean_std(self, mri_mean, mri_std, suvr_mean, suvr_std):
        self.mri_mean = mri_mean
        self.mri_std = mri_std
        self.suvr_mean = suvr_mean
        self.suvr_std = suvr_std

    def cast_to_type(self, value, cast_type):
        if type(value) == cast_type :
            return value, cast_type
        elif type(value) == torch.Tensor :
            value = value.detach().cpu()
            value = cast_type(value)
            return value, torch.Tensor
        else :
            value_type = type(value)
            return cast_type(value), value_type

    def get_mris(self):
        return torch.vstack([
            self.load_mri_file(mri_file) for mri_file in self.mri_files
        ])

    def resize_tensor(self, t, new_spacing):
        volume = sitk.GetImageFromArray(t.squeeze(0).detach().cpu().numpy())
        volume = self.resize_volume(volume, new_spacing)
        t = torch.from_numpy(sitk.GetArrayFromImage(volume)).to(dtype=torch.float32, device=self.cuda_id).unsqueeze(0)
        return t


    def monai_spatial_transforms(self, volume_path):
        # crop and pad
        # gaussian smoothing
        loader = LoadImage(dtype=torch.float32, image_only=True)
        volume = loader(volume_path)
        transform_comp = Compose([ResizeWithPadOrCrop((1, 128, 128, 128)), GaussianSmooth()])
        volume_tensor = transform_comp(volume)
        

class CustomSampler(Sampler):
    def __init__(self, data_source, skip_ids, shuffle=False, rnd_seed=0):
        self.data_source = data_source
        data_source["ids_temp"] = data_source["tau"].apply(lambda x: self.get_id_from_path(x))
        self.indices = np.array(data_source[~data_source["ids_temp"].isin(skip_ids)].index)
        self.indices = np.array([
            i for i in range(data_source.shape[0]) 
            if os.path.isfile(data_source.iloc[i, data_source.columns.get_loc("MRI")]) and i in self.indices
        ])
        logging.info(f"Original length: {len(data_source)}, new length: {len(self.indices)}")
        if shuffle:
            # random.seed(rnd_seed)
            random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def get_id_from_path(self, file_path:str) -> int:

        chunks = file_path.split("/")
        id_chunk = chunks[-4]
        if "-" in id_chunk:
            id_chunk = os.path.join(id_chunk, chunks[-3])

        return id_chunk
    
class CovariateVolumeDataset(VolumeDataset):

    # def __init__(self, lookup_file, covariate_lookup_file, transform=None, mri_mean=0, mri_std=1, suvr_mean=np.float(0), suvr_std=np.float(1), cuda_id=-1):
    def __init__(self, covariate_lookup_file, *args, with_all_covars=False, **kwargs):
        # super().__init__(lookup_file, transform, mri_mean, mri_std, suvr_mean, suvr_std, cuda_id=cuda_id)
        logging.info(f"Covariate VD cluf: {covariate_lookup_file} | args: {args}, kwargs: {kwargs}")
        super().__init__(*args, **kwargs)
        self.covariate_lookup = pd.read_csv(covariate_lookup_file)
        self.abeta_quart_lookup = pd.read_csv(f"{os.getcwd()}/scripts/ADNI_ID_ABETA_TAU_QUARTS.csv")
        self.abeta_col_name = "Abeta_Covar"
        self.id_col_name = "ADNI_ID"
        # Covariates
        self.with_all_covars = with_all_covars
        self.covar_col_names = ["Abeta_Covar", "Age", "Sex", "Education", "Cognition"]
        if with_all_covars:
            self.covariate_lookup["Sex"] = self.covariate_lookup["Sex"].map({"M": 0, "F": 1})
            covar_cols_to_scale = ["Age", "Education", "Cognition"]
            self.covariate_lookup[covar_cols_to_scale] = MinMaxScaler().fit_transform(
                self.covariate_lookup[covar_cols_to_scale]
            )


            # scaler = MinMaxScaler()
            # # Fit into 0 to 1 range
            # covars_df = self.covariate_lookup[self.covar_col_names]
            # scaler.fit(covars_df)
            # covar_df_scaled = scaler.transform(covars_df)
            # # Convert the scaled subset back to a DataFrame
            # covars_df_scaled = pd.DataFrame(covar_df_scaled, columns=self.covar_col_names)

            # # Replace the original columns in the original DataFrame with the scaled values
            # self.covariate_lookup[self.covar_col_names] = covars_df_scaled

    def __getitem__(self, idx):
        mri_tensor, tau_tensor, roi_tensor, tau_path = super().__getitem__(idx)

        volume_id = self.get_id_from_path(tau_path)
        abeta = self.covariate_lookup[self.covariate_lookup[self.id_col_name] == volume_id][self.abeta_col_name].values 
        abeta = -1 if len(abeta) == 0 else abeta.item()
        
        if self.with_all_covars:
            tau_path_for_covars = self.lookup_df.iloc[idx, self.lookup_df.columns.get_loc("tau")]
            covars = self.get_all_covariates(tau_path_for_covars)
            return mri_tensor, tau_tensor, roi_tensor, (abeta, covars), tau_path
        
        # logging.info(volume_id)
        # if np.isnan(abeta) or (not isinstance(abeta, np.float) and not isinstance(abeta, int)):
        #     abeta = -1
        # logging.info(f"{volume_id}, {type(abeta)}, {abeta}")

        return mri_tensor, tau_tensor, roi_tensor, abeta, tau_path

    def get_id_from_path(self, file_path:str) -> int:

        chunks = file_path.split("/")
        id_chunk = chunks[-4]
        if "-" in id_chunk:
            id_chunk = os.path.join(id_chunk, chunks[-3])

        return id_chunk
    
    def find_nan_ids(self):

        invalid_rows = self.covariate_lookup[pd.isna(self.covariate_lookup[self.abeta_col_name])]
        invalid_ids = invalid_rows[self.id_col_name].values

        return invalid_ids

    def lookup_covar(self, vol_id, covar_col_name):
        covar = self.covariate_lookup[self.covariate_lookup[self.id_col_name] == vol_id][covar_col_name].values
        return covar

    def get_all_covariates(self, tau_path, meta_tau=False):
        # tau_path = self.lookup_df.iloc[idx, self.lookup_df.columns.get_loc("tau")]
        volume_id = self.get_id_from_path(tau_path)
         
        # logging.info(f"Contra check id: {volume_id}")
        abeta = self.covariate_lookup[self.covariate_lookup[self.id_col_name] == volume_id][self.abeta_col_name].values

        if len(abeta) != 1:
            logging.info(abeta)

        abeta = abeta.item()
        
        if np.isnan(abeta) or (not isinstance(abeta, np.float) and not isinstance(abeta, int)):
            abeta = -1

        covars = self.covariate_lookup[self.covariate_lookup[self.id_col_name] == volume_id][self.covar_col_names].values
        covars = torch.from_numpy(covars).to(dtype=torch.float)

        if meta_tau:
            meta_tau = self.covariate_lookup[self.covariate_lookup[self.id_col_name] == volume_id]["Tau_Meta"].values
            covars = torch.cat((covars, torch.from_numpy(np.array([meta_tau]))), -1)

        return covars

class ContrastiveVolumeDataset(CovariateVolumeDataset):

    def __init__(self, lookup_file, covariate_lookup_file, holdout_ids=[], *args, **kwargs):
        # super().__init__(lookup_file, covariate_lookup_file, transform, mri_mean, mri_std, suvr_mean, suvr_std, cuda_id=cuda_id)
        logging.info(f"ContrastiveVD luf: {lookup_file} | cluf: {covariate_lookup_file} | hoi: {holdout_ids} | args: {args}, kwargs: {kwargs}")
        super().__init__(covariate_lookup_file, lookup_file, *args, **kwargs)
        self.holdout_ids = holdout_ids

    def __getitem__(self, idx):
        anchor_data = super().__getitem__(idx)
        anchor_covariate, _ = anchor_data[3]
        
        # idx2 = idx 
        # pos_covariate = 1 if anchor_covariate == 0 else 0
        # while idx2 == idx or anchor_covariate != pos_covariate:

        #     idx2 = torch.randint(0, len(self.lookup_df), (1,)).item()
        #     pos_covariate = self.get_covariate(idx2)
        #     # logging.info(f"random sample covar: {pos_covariate}")

        #     # if pos_covariate == anchor_covariate :
        # pos_data = super().__getitem__(idx2)

        anchor_quartile = self.abeta_quart_lookup[self.abeta_quart_lookup[self.id_col_name] == data_util.get_id_from_path(anchor_data[-1])]["quartile_lub"].values.item()
        
        pos_data_choices_ids = self.abeta_quart_lookup[self.abeta_quart_lookup[self.abeta_col_name] == anchor_covariate][self.abeta_quart_lookup["quartile_lub"] == anchor_quartile][self.id_col_name].values
        pos_data_choices_ids = data_util.remove_invalid("/home/jagust/xnat/xnp/sshfs/xnat_data/adni/-id-", pos_data_choices_ids)
        id2 = np.random.choice(pos_data_choices_ids)
        while len(self.lookup_df[self.lookup_df["ids_temp"] == id2].index) != 1:
            id2 = np.random.choice(pos_data_choices_ids)
        idx2 = self.lookup_df[self.lookup_df["ids_temp"] == id2].index.item()
        pos_data = super().__getitem__(idx2)
        
        neg_data_choices_ids = self.abeta_quart_lookup[self.abeta_quart_lookup[self.abeta_col_name] == anchor_covariate][self.abeta_quart_lookup["quartile_lub"] == anchor_quartile][self.id_col_name].values
        neg_data_choices_ids = data_util.remove_invalid("/home/jagust/xnat/xnp/sshfs/xnat_data/adni/-id-", neg_data_choices_ids)
        # neg_data_choices_ids = self.abeta_quart_lookup[self.abeta_quart_lookup[self.abeta_col_name] != anchor_covariate][self.abeta_quart_lookup["quartile_lub"] == anchor_quartile][self.id_col_name].values
        id3 = np.random.choice(neg_data_choices_ids)
        while len(self.lookup_df[self.lookup_df["ids_temp"] == id3].index) != 1:
            id3 = np.random.choice(neg_data_choices_ids)
        idx3 = self.lookup_df[self.lookup_df["ids_temp"] == id3].index.item()
        neg_data = super().__getitem__(idx3)

        # pos_covariate = self.get_covariate(idx2)

        # idx3 = idx2
        # neg_covariate = pos_covariate
        # while idx3 == idx2 or neg_covariate == pos_covariate or neg_covariate == -1:

        #     idx3 = torch.randint(0, len(self.lookup_df), (1,)).item()
        #     neg_covariate = self.get_covariate(idx3)
        #     # logging.info(f"random sample covar: {pos_covariate}")

        #     # if neg_covariate != anchor_covariate:
        # neg_data = super().__getitem__(idx3)

        return anchor_data, pos_data, neg_data

    def get_covariate(self, idx):
        tau_path = self.lookup_df.iloc[idx, self.lookup_df.columns.get_loc("tau")]
        volume_id = super().get_id_from_path(tau_path)

        if volume_id in self.holdout_ids:
            return -1
         
        # logging.info(f"Contra check id: {volume_id}")
        abeta = self.covariate_lookup[self.covariate_lookup[self.id_col_name] == volume_id][self.abeta_col_name].values

        if len(abeta) != 1:
            logging.info(abeta)

        abeta = abeta.item()
        

        if np.isnan(abeta) or (not isinstance(abeta, np.float) and not isinstance(abeta, int)):
            abeta = -1

        return abeta

class ClusterVolumeDataset(CovariateVolumeDataset):

    def __init__(self, lookup_file, covariate_lookup_file, holdout_ids=[], *args, **kwargs):
        # super().__init__(lookup_file, covariate_lookup_file, transform, mri_mean, mri_std, suvr_mean, suvr_std, cuda_id=cuda_id)
        logging.info(f"ClusterVD luf: {lookup_file} | cluf: {covariate_lookup_file} | hoi: {holdout_ids} | args: {args}, kwargs: {kwargs}")
        super().__init__(covariate_lookup_file, lookup_file, *args, **kwargs)
        self.holdout_ids = holdout_ids

    def __getitem__(self, idx):
        anchor_data = super().__getitem__(idx)
        anchor_covariate, _ = anchor_data[3]

        anchor_quartile = self.abeta_quart_lookup[self.abeta_quart_lookup[self.id_col_name] == data_util.get_id_from_path(anchor_data[-1])]["quartile_lub"].values.item()
        
        pos_data_choices_ids = self.abeta_quart_lookup[self.abeta_quart_lookup[self.abeta_col_name] == anchor_covariate][self.abeta_quart_lookup["quartile_lub"] == anchor_quartile][self.id_col_name].values
        pos_data_choices_ids = data_util.remove_invalid("/home/jagust/xnat/xnp/sshfs/xnat_data/adni/-id-", pos_data_choices_ids)
        id2 = np.random.choice(pos_data_choices_ids)
        while len(self.lookup_df[self.lookup_df["ids_temp"] == id2].index) != 1:
            id2 = np.random.choice(pos_data_choices_ids)
        idx2 = self.lookup_df[self.lookup_df["ids_temp"] == id2].index.item()
        pos_data = super().__getitem__(idx2)
        
        neg_covariate = 1 if anchor_covariate == 0 else 0
        neg_samples = []
        for quart in range(1, 5):
            neg_sample = self.get_data(neg_covariate, quart)
            if neg_sample is not None:
                neg_samples.append(neg_sample)
            if quart != anchor_quartile:
                neg_sample = self.get_data(anchor_covariate, quart)
                if neg_sample is not None:
                    neg_samples.append(neg_sample)

        return anchor_data, pos_data, neg_samples

    def get_covariate(self, idx):
        tau_path = self.lookup_df.iloc[idx, self.lookup_df.columns.get_loc("tau")]
        volume_id = super().get_id_from_path(tau_path)

        if volume_id in self.holdout_ids:
            return -1
         
        # logging.info(f"Contra check id: {volume_id}")
        abeta = self.covariate_lookup[self.covariate_lookup[self.id_col_name] == volume_id][self.abeta_col_name].values

        if len(abeta) != 1:
            logging.info(abeta)

        abeta = abeta.item()
        

        if np.isnan(abeta) or (not isinstance(abeta, np.float) and not isinstance(abeta, int)):
            abeta = -1

        return abeta

    def get_data(self, abeta, quart):
        neg_data_choices_ids = self.abeta_quart_lookup[self.abeta_quart_lookup[self.abeta_col_name] == abeta][self.abeta_quart_lookup["quartile_lub"] == quart][self.id_col_name].values
        neg_data_choices_ids = self.lookup_df[self.lookup_df['ids_temp'].isin(neg_data_choices_ids)]["ids_temp"].values
        neg_data_choices_ids = data_util.remove_invalid("/home/jagust/xnat/xnp/sshfs/xnat_data/adni/-id-", neg_data_choices_ids)
        # neg_data_choices_ids = self.abeta_quart_lookup[self.abeta_quart_lookup[self.abeta_col_name] != anchor_covariate][self.abeta_quart_lookup["quartile_lub"] == anchor_quartile][self.id_col_name].values
        # logging.info(f"abeta: {abeta} | quart: {quart} | reduced_choices: {reduced_choices}")
        if len(neg_data_choices_ids) == 0:
            return None
            logging.info(f"all abeta: {abeta} | quart: {quart} choices: {neg_data_choices_ids}")
        id3_reduced = np.random.choice(neg_data_choices_ids)
        # id3 = np.random.choice(neg_data_choices_ids)
        # logging.info(f"id3_redced: {id3_reduced} | {len(self.lookup_df[self.lookup_df['ids_temp'] == id3_reduced].index) == 1}")
        # while len(self.lookup_df[self.lookup_df["ids_temp"] == id3_reduced].index) != 1:
        # # while len(self.lookup_df[self.lookup_df["ids_temp"] == id3].index) != 1:
        #     id3 = np.random.choice(neg_data_choices_ids)
        idx3 = self.lookup_df[self.lookup_df["ids_temp"] == id3_reduced].index.item()
        # idx3 = self.lookup_df[self.lookup_df["ids_temp"] == id3].index.item()
        neg_data = super().__getitem__(idx3)

        return neg_data

# class RegressionVolumeDataset(ContrastiveVolumeDataset, ClusterVolumeDataset):
class RegressionVolumeDataset(ClusterVolumeDataset):
# class RegressionVolumeDataset(ContrastiveVolumeDataset):


    # def __init__(self, lookup_file, covariate_lookup_file, holdout_ids=[], *args, **kwargs):

    def __init__(self, lookup_file, covariate_lookup_file, holdout_ids=[], mode='contrastive', *args, **kwargs):
        logging.info(f"RegressionVD luf: {lookup_file} | cluf: {covariate_lookup_file} | hoi: {holdout_ids} | args: {args}, kwargs: {kwargs}")
        if mode == 'contrastive':
            super().__init__(lookup_file, covariate_lookup_file, holdout_ids, *args, **kwargs)
        elif mode == 'cluster':
            # ClusterVolumeDataset.__init__(self, lookup_file, covariate_lookup_file, holdout_ids, *args, **kwargs)
            super().__init__(lookup_file, covariate_lookup_file, holdout_ids, *args, **kwargs)
        # else:
        #     raise ValueError("Invalid mode. Use 'contrastive' or 'cluster'.")

        self.meta_tau_col_name = "Tau_Meta"
        self.mode = mode
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # idx = self.valid_indices[idx]
        anchor_data, pos_data, neg_data = super().__getitem__(idx)

        meta_tau_associated_batch = []
        meta_tau_associated_batch.append(self.assign_meta_tau(anchor_data))
        meta_tau_associated_batch.append(self.assign_meta_tau(pos_data))
        
        # for b in neg_data:
        #     meta_tau_assoc_neg = self.assign_meta_tau(b[0])
        #     mata_tau_assoc_neg.append(meta_tau_assoc_neg)
        # for b in neg_data:
        
        if self.mode == "contrastive":
            meta_tau_assoc_neg = self.assign_meta_tau(neg_data)
            meta_tau_associated_batch.append(meta_tau_assoc_neg)
        elif self.mode == "cluster":
            meta_tau_assoc_neg = []
            for n in neg_data:
                meta_tau_assoc_neg_sample = self.assign_meta_tau(n)
                meta_tau_assoc_neg.append(meta_tau_assoc_neg_sample)
            meta_tau_associated_batch.append(meta_tau_assoc_neg)
            
        return tuple(meta_tau_associated_batch)


    def assign_meta_tau(self, data):
        # data_batch_with_meta_tau = []
        # for data in data_batch:
        tau_path = data[-1]
        all_covars = data[3]
        meta_tau = self.get_meta_tau(tau_path)
        # all_covars = (all_covars[0], all_covars[1], meta_tau)
        all_covars = (all_covars[0], torch.cat((all_covars[1], torch.tensor([meta_tau]).unsqueeze(0)), -1))
        data_with_meta_tau = data[0], data[1], data[2], all_covars, tau_path
        # data_batch_with_meta_tau.append(data_with_meta_tau)

        # return tuple(data_batch_with_meta_tau)
        return data_with_meta_tau

    def get_meta_tau(self, tau_path):
        volume_id = data_util.get_id_from_path(tau_path)
        meta_tau = self.covariate_lookup[self.covariate_lookup[self.id_col_name] == volume_id][self.meta_tau_col_name].values

        if len(meta_tau) != 1:
            logging.info(f"debug - meta_tau: {meta_tau}")

        meta_tau = meta_tau.item()
        if np.isnan(meta_tau) or (not isinstance(meta_tau, np.float) and not isinstance(meta_tau, int)):
            meta_tau = 0

        return meta_tau
    
class PredictedMetaTauDataset(RegressionVolumeDataset):

    def __init__(self, predicted_lookup_table, lookup_file, covariate_lookup_file, holdout_ids=[], mode='contrastive', *args, **kwargs):
        super().__init__(lookup_file, covariate_lookup_file, holdout_ids, mode, *args, **kwargs)

        self.predicted_lookup_table = predicted_lookup_table


    def get_meta_tau(self, tau_path):
        volume_id = data_util.get_id_from_path(tau_path)
        # meta_tau = self.predicted_lookup_table[volume_id] # NOTE original 
        meta_tau = self.predicted_lookup_table[volume_id]['pred'].item() # NOTE with UQ

        # if len(meta_tau) != 1:
        #     logging.info(f"debug - meta_tau: {meta_tau}")

        # meta_tau = meta_tau.item()
        if isinstance(meta_tau, np.ndarray):
            meta_tau = meta_tau[0]
        elif np.isnan(meta_tau) or (not isinstance(meta_tau, np.float) and not isinstance(meta_tau, int)):
            meta_tau = 0

        return meta_tau
    

def format_volume_path(pth: str):
    # /home/jagust/xnat/xnp/sshfs/xnat_data/adni/000-S-0059/PET_2017-12-12_FTP/analysis/suvr_cereg.nii
    return pth.split("/")[:-3]

def generate_downsampled_volumes(out_dir, template_space=False, resize=True, pad_dims=(128, 128, 128), smoothing=False):
    rep_samples_paths = [
        ("7029_2022-03-30", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/067-S-7029/PET_2022-03-30_FTP/analysis/rnu.nii", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/067-S-7029/PET_2022-03-30_FTP/analysis/raparc+aseg.nii", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/067-S-7029/PET_2022-03-30_FTP/analysis/suvr_cereg.nii"),
        ("7032_2022-03-01", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/131-S-7032/PET_2022-03-01_FTP/analysis/rnu.nii", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/131-S-7032/PET_2022-03-01_FTP/analysis/raparc+aseg.nii", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/131-S-7032/PET_2022-03-01_FTP/analysis/suvr_cereg.nii"),
        ("6005_2017-04-27", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/024-S-6005/PET_2017-04-27_FTP/analysis/rnu.nii", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/024-S-6005/PET_2017-04-27_FTP/analysis/raparc+aseg.nii", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/024-S-6005/PET_2017-04-27_FTP/analysis/suvr_cereg.nii"),
        ("6005_2021-07-20", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/024-S-6005/PET_2021-07-20_FTP/analysis/rnu.nii", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/024-S-6005/PET_2021-07-20_FTP/analysis/raparc+aseg.nii", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/024-S-6005/PET_2021-07-20_FTP/analysis/suvr_cereg.nii")
    ]
    # vd = VolumeDataset(f"{os.getcwd()}/training_folds/testfold1.csv", cuda_id=0)
    vd = VolumeDataset(f"{os.getcwd()}/training_folds/outlier_removed_splits/training_lookup_1.csv", cuda_id=0)
    if template_space:
        vd.resize=resize
        vd.pad_dims = pad_dims
        vd.transform = data_util.pad_volume(pad_dims)
        # roi_path = f"{os.getcwd()}/scripts/pet_dimYeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii"
        roi_path =  f"{os.getcwd()}/scripts/AD_ROI_MASK.nii"
    for sample_id, mri_path, roi_path, tau_path in rep_samples_paths:
        # Load MRI
        if template_space :
            roi_path = vd.TEMPLATE_SPACE_ROI_PATH
            mri_dir_path = "/".join(mri_path.split("/")[:-1])
            mri_path = os.path.join(mri_dir_path, "wrnu.nii")
            tau_dir_path = "/".join(tau_path.split("/")[:-1])
            tau_path = os.path.join(tau_dir_path, "wsuvr_cereg.nii")

        roi_mask = vd.load_volume_file(roi_path, is_mask=True).unsqueeze(0)
        tau = vd.load_volume_file(tau_path, is_mask=False).unsqueeze(0)
        if vd.template_space :
            mri_dir_path = "/".join(mri_path.split("/")[:-1])
            mri_path = os.path.join(mri_dir_path, "wrnu.nii")
            mri = vd.load_volume_file_with_mask(mri_path, vd.mri_masking_fn, resize=vd.resize).unsqueeze(0)
            

        if not vd.template_space:
            mri = vd.load_volume_file(mri_path).unsqueeze(0)
            # roi_mask = vd.load_volume_file(roi_path, is_mask=True).unsqueeze(0)
            # mri[roi_mask == 0] = 0
        
        # Apply smoothin g
        if smoothing:
            smoothing = GaussianSmooth()
            tau = smoothing(tau.squeeze(0)).unsqueeze(0)
            print(tau.shape)
        
        print(mri.shape)
        # if not vd.template_space:
        #     mri[roi_mask == 0] = 0
        #     tau[roi_mask == 0] = 0
        # else:
        #     tau[roi_mask == 0] = 0

        # tau_masking_tensor = vd.load_volume_file(f"{os.getcwd()}/scripts/tau_maskmni152_2009_256.nii", is_mask=True)
        # tau[tau_masking_tensor == 0] = 0

        # Save output
        # data_util.write_tensor_to_nii(mri, os.path.join(out_dir, f"{sample_id}_mri.nii"))
        data_util.write_tensor_to_nii(tau, os.path.join(out_dir, f"{sample_id}_tau.nii"))
        # data_util.write_tensor_to_nii(roi_mask, os.path.join(out_dir, f"{sample_id}_mask.nii"))

def write_downsampled_volumes(out_dir, template_space=True, resize=True, pad_dims=(128, 128, 128)):
    rep_samples_paths = [
        ("7029_2022-03-30", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/067-S-7029/PET_2022-03-30_FTP/analysis/rnu.nii", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/067-S-7029/PET_2022-03-30_FTP/analysis/raparc+aseg.nii", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/067-S-7029/PET_2022-03-30_FTP/analysis/suvr_cereg.nii"),
        ("7032_2022-03-01", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/131-S-7032/PET_2022-03-01_FTP/analysis/rnu.nii", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/131-S-7032/PET_2022-03-01_FTP/analysis/raparc+aseg.nii", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/131-S-7032/PET_2022-03-01_FTP/analysis/suvr_cereg.nii"),
        ("6005_2017-04-27", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/024-S-6005/PET_2017-04-27_FTP/analysis/rnu.nii", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/024-S-6005/PET_2017-04-27_FTP/analysis/raparc+aseg.nii", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/024-S-6005/PET_2017-04-27_FTP/analysis/suvr_cereg.nii"),
        ("6005_2021-07-20", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/024-S-6005/PET_2021-07-20_FTP/analysis/rnu.nii", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/024-S-6005/PET_2021-07-20_FTP/analysis/raparc+aseg.nii", "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/024-S-6005/PET_2021-07-20_FTP/analysis/suvr_cereg.nii")
    ]
    # vd = VolumeDataset.VolumeDataset(f"{os.getcwd()}/training_folds/testfold1.csv", cuda_id=cuda_id)
    covar_lookup = f"{os.getcwd()}/scripts/ADNI_W_Covars.csv"
    temp_fold = f"{os.getcwd()}/training_folds/testfold1.csv"
    vd = CovariateVolumeDataset(covar_lookup, temp_fold, with_all_covars=True)
    if template_space:
        vd.template_space = template_space
        vd.resize=resize
        vd.pad_dims=pad_dims

    vd.transform = data_util.pad_volume(pad_dims) 
    for sample_id, mri_path, roi_path, tau_path in rep_samples_paths:

        if vd.template_space:
            # (padding_left,padding_right, padding_top, padding_bottom, padding_top,padding_bottom, padding_front,padding_back, padding_front, padding_back)
            vd.pad_dims = (128, 128, 128) if vd.resize else (216, 216, 216)
            vd.transform = data_util.pad_volume(vd.pad_dims) 
            roi_path = vd.TEMPLATE_SPACE_ROI_PATH
            # mri_mask_path = vd.TEMPLATE_SPACE_MRI_MASK_PATH

        # Load MRI, tau PET, and corresponding ROI mask
        # mri_tensor = vd.load_volume_file(mri_path, nonzeros=False)
        mri_tensor = vd.load_volume_file_with_mask(mri_path, vd.mri_masking_fn, resize=vd.resize)
        # mri_tensor = vd.load_volume_file(tau_path, nonzeros=False) # NOTE this line is ONLY for testing if net can learn id
        tau_tensor = vd.load_volume_file(tau_path)
        # tau_tensor = vd.load_volume_file_with_mask(tau_path, vd.tau_masking_fn, resize=vd.resize)
        roi_tensor = vd.load_volume_file(roi_path, is_mask=True)

        if not vd.template_space:
            mri_tensor[roi_tensor == 0] = 0
            tau_tensor[roi_tensor == 0] = 0
        else:
            tau_tensor[roi_tensor == 0] = 0


        data_util.write_tensor_to_nii(mri_tensor, os.path.join(out_dir, f"{sample_id}_mri.nii"))
        # data_util.write_tensor_to_nii(tau_tensor, os.path.join(out_dir, f"{sample_id}_tau.nii"))
        # data_util.write_tensor_to_nii(roi_tensor, os.path.join(out_dir, f"{sample_id}_mask.nii"))


if __name__ == "__main__":
    CUDA_ID = -1
    from sklearn.decomposition import PCA

    # dataset = VolumeDataset(f"{os.getcwd()}/scripts/directory_lookup.csv", cuda_id=CUDA_ID)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # test_lookup_file = f"{os.getcwd()}/big_training/a4_testing.csv"
    # test_dataset = VolumeDataset(test_lookup_file, cuda_id=CUDA_ID)
    # test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=False)
    
    train_lookup_file = f"{os.getcwd()}/big_training/adni_training.csv"
    train_dataset = VolumeDataset(train_lookup_file, resize=True, mri_file_type="wrnu.nii", tau_file_type="wsuvr_cereg.nii", cuda_id=CUDA_ID)
    # train_dataset = VolumeDataset(train_lookup_file, resize=True, cuda_id=CUDA_ID)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # covariate_lookup_file = f"{os.getcwd()}/scripts/covariate.csv"
    # train_dataset = ContrastiveVolumeDataset(train_lookup_file, covariate_lookup_file, cuda_id=CUDA_ID)
    # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    def comp_roi_corr(pred, gt, roi, roi_indicies):
        correlations = torch.empty(len(roi_indices))
        
        def nanvar(tensor, dim=None, keepdim=False):
            tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
            output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
            return output

        def nanstd(tensor, dim=None, keepdim=False):
            output = nanvar(tensor, dim=dim, keepdim=keepdim)
            output = output.sqrt()
            return output

        for i, idx in enumerate(roi_indices):
            # mask and reshape
            x = pred.detach().clone()
            y = gt.detach().clone()
            x[roi != idx] = torch.nan
            y[roi != idx] = torch.nan
            x_flat = x.view(x.size(0), -1)
            y_flat = y.view(y.size(0), -1)

            # Calculate the mean along the spatial dimensions
            x_mean = torch.nanmean(x_flat, dim=1, keepdim=True)
            y_mean = torch.nanmean(y_flat, dim=1, keepdim=True)

            # Calculate the covariance
            cov_xy = torch.nanmean((x_flat - x_mean) * (y_flat - y_mean), dim=1, keepdim=True)

            # Calculate the standard deviations
            std_x = nanstd(x_flat, dim=1, keepdim=True)
            std_y = nanstd(y_flat, dim=1, keepdim=True)
            # std_x = torch.std(x_flat, dim=1, keepdim=True)
            # std_y = torch.std(y_flat, dim=1, keepdim=True)

            # Calculate the correlation coefficient
            corr_coeff = cov_xy / (std_x * std_y)

            print(corr_coeff)
            correlations[i] = torch.mean(corr_coeff).item()

        # move back to device 
        # return torch.nan_to_num(correlations, nan=0, posinf=0, neginf=0)
        return correlations

    def sample_wise_roi_mean(batched_input, roi, roi_indicies):
        roi_mask = torch.zeros(roi.size(), device=roi.get_device())
        for i, idx in enumerate(roi_indices):
            roi_mask[:] = 0 # reset mask
            roi_mask[roi == idx] = 1
            mu = torch.sum(roi_mask * batched_input, dim=(-3, -2, -1)) / torch.count_nonzero(roi_mask, dim=(-3, -2, -1))
            flat_mu = torch.flatten(mu)
            print(mu)
            print(mu.size())

    roi_indices = [1001,1006,1007,1009,1015,1016,1030,1034,1033,1008,1025,1029,1031,1022,17,18,2001,2006,2007,2009,2015,2016,2030,2034,2033,2008,2025,2029,2031,2022,49,50,51,52,53,54]
    # roi_indices = [1006,1016]
    # roi_sizes = torch.zeros(len(roi_indices)).cuda(1)

    # covar_lookup_df = pd.read_csv(f"{os.getcwd()}/scripts/ADNI_ID_ABETA_TAU_QUARTS.csv")
    # def get_quart_abeta(tau_path):
    #     sample_id = data_util.get_id_from_path(tau_path)
    #     quart = covar_lookup_df[covar_lookup_df["ADNI_ID"] == sample_id]["quartile_lub"].values
    #     if len(quart) == 0:
    #         return -1 
    #     quart = quart[0]
    #     abeta = covar_lookup_df[covar_lookup_df["ADNI_ID"] == sample_id]["Abeta_Covar"].values
    #     if len(abeta) == 0:
    #         return -1
    #     abeta = abeta[0]
    #     return quart, abeta, sample_id
    # for batch_idx, (mri, tau, roi, tau_path) in enumerate(train_dataloader):
    #     print(f"[{batch_idx}/{len(train_dataloader)}]")
    #     break
    # base = f"{os.getcwd()}/scripts/templates_tau_quart"
    # abneg_fnames = [os.path.join(base, f"abneg_quart{i+1}") for i in range(4)]
    # abneg_templates = [train_dataset.load_volume_file(f) for f in abneg_fnames]
    # abpos_fnames = [os.path.join(base, f"abpos_quart{i+1}") for i in range(4)]
    # abpos_templates = [train_dataset.load_volume_file(f) for f in abpos_fnames]
    # abneg_templates = torch.cat(abneg_templates, 0)
    # abpos_templates = torch.cat(abpos_templates, 0)
    # X = torch.cat((abneg_templates, abpos_templates), 0).detach().cpu().numpy()
    # print(f"X shape before reshape: {X.shape}")
    # X = X.reshape(X.shape[0], -1)
    # pca = PCA(n_components=3)
    # X_transformed = pca.fit_transform(X)
    # def cosine_sim_np(x, y):
    #     v = np.dot(x, y)
    #     d = np.linalg.norm(x) * np.linalg.norm(y)
    #     return v / d
    # similarities = [[] for i in range(X.shape[0])] 
    # for batch_idx, (mri, tau, roi, tau_path) in enumerate(train_dataloader):
    #     print(f"[{batch_idx}/{len(train_dataloader)}]")
    #     # sample_wise_roi_mean(mri, roi, roi_indices)
    #     tau_np = tau.squeeze(0).detach().cpu().numpy()

    #     tau_transformed = pca.transform(tau_np.reshape(1, -1))
    #     vals = get_quart_abeta(tau_path[0])
    #     if vals == -1:
    #         continue
    #     sample_quart, sample_abeta, sid = vals
    #     template_idx = sample_quart + (sample_abeta * 4) - 1
    #     corresponding_transformed_template = X_transformed[template_idx, :]
    #     # measure the similarity
    #     similarity = cosine_sim_np(corresponding_transformed_template, tau_transformed[0, :])
    #     print(f'{sid} - quart {sample_quart} ({sample_abeta}) similarity: {similarity}')
    #     # bin
    #     similarities[template_idx].append(similarity)

    #     # if batch_idx == 9:
    #     #     break
        
    #     if all([len(b) >= 200 for b in similarities]):
    #         break

        
        # print(f"pad to size: {train_dataloader.dataset.pad_dims}")
        # roi_mask = torch.zeros(roi.size(), device=roi.get_device())
        # for i, idx in enumerate(roi_indices):
        #     roi_mask[:] = 0 # reset mask
        #     roi_mask[roi == idx] = 1
        #     roi_mask_size = torch.count_nonzero(roi_mask, dim=(-3, -2, -1))
        #     print(roi_mask_size)


        # roi_mask = torch.zeros(roi.size(), device=roi.get_device())
        # num_samples += roi_mask.size(0)
        # for i, idx in enumerate(roi_indices):
        #     roi_mask[:] = 0 # reset mask
        #     roi_mask[roi == idx] = 1
        #     roi_mask_size = torch.count_nonzero(roi_mask, dim=(-3, -2, -1))
        #     # roi_sizes[i] += torch.sum(roi_mask_size)
        #     if any(roi_mask_size == 0) :
        #         print(f"Roi mask nonzero shape: {roi_mask_size.size()}")
        #         print(f"\tmask size: {roi_mask_size.t()}")#: {tau_path.t()}")
        #         faulty_inds = (roi_mask_size == 0).nonzero()[:, 0]
        #         print(f"\tFauly inds: {faulty_inds}")
        #         for j, faulty_idx in enumerate(faulty_inds):
        #             print(f"\tFaulty sample: {tau_path[faulty_idx]} in ROI {idx}")
        #             assert roi_mask_size[faulty_idx, 0] == 0, "ROI marked as faulty but size isn't 0"
        #             bad_samples.append(tau_path[faulty_idx])

            # tau[roi_mask == 1] = 
    model_params = (3, 1, 1, [32, 64, 128, 256, 512], [2]*5) # AttnUNET
    model = attn_unet_analysis.ContrastiveAttentionUNET(*model_params, latent_spaces=[262144]*len(model_params[-1]), conditional=True)
    model.set_save_attn(None)

    # for batch_idx, (mri, tau, roi, tau_path) in enumerate(train_dataloader):
    #     print(f"[{batch_idx}/{len(train_dataloader)}]")
    #     pred = model(mri)
        # sample_wise_roi_mean(mri, roi, roi_indices)
    #     roi_mask = torch.zeros(roi.size(), device=roi.get_device())
    #     num_samples += roi_mask.size(0)
    #     for i, idx in enumerate(roi_indices):
    #         roi_mask[:] = 0 # reset mask
    #         roi_mask[roi == idx] = 1
    #         roi_mask_size = torch.count_nonzero(roi_mask, dim=(-3, -2, -1))
    #         # print(f"\tmask size: {roi_mask_size.t()}")#: {tau_path.t()}")
    #         # roi_sizes[i] += torch.sum(roi_mask_size)
    #         if any(roi_mask_size == 0) :
    #             print(f"Roi mask nonzero shape: {roi_mask_size.size()}")
    #             faulty_inds = (roi_mask_size == 0).nonzero()[:, 0]
    #             for j, faulty_idx in enumerate(faulty_inds):
    #                 print(f"\tFaulty sample: {tau_path[faulty_idx]} in ROI {idx}")
    #                 assert roi_mask_size[faulty_idx, 0] == 0, "ROI marked as faulty but size isn't 0"
    #                 bad_samples.append(tau_path[faulty_idx])
        
    #     #     if torch.count_nonzero(roi_mask_size) != tau.size(0):
    #     #         logging.info(f"ROI [{idx}]\n: {roi_mask_size}\n\n")
    #     #         logging.info(f"Errenous ROI, {idx}, in volume: {tau_path}")

    # print(f"Bad Samples:\n{bad_samples}")
    
    # roi_sizes /= num_samples
    # logging.info(torch.sort(roi_sizes))
    # logging.info(f"Ordered ROI sizes (small to large): {torch.tensor(roi_indices, device=roi_sizes.get_device())[torch.argsort(roi_sizes)]}")



