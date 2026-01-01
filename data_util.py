import os
import sys

sys.path.append(f'{os.getcwd()}')
import json
import logging 
import pickle

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
# from torchvision import transforms
import torch.nn.functional as F

from monai.data import DataLoader as MDataLoader

import ImageDataset
# from VolumeDataset import VolumeDataset
import VolumeDataset as vd
import visualization_util

# from sklearn.decomposition import PCA
from PCA import PCA
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import RFE

import glob
import pandas as pd
import numpy as np

import SimpleITK as sitk
import nibabel as nib
from nibabel.processing import smooth_image as nib_smooth_image
import time


SELECTED_SAMPLES = [
    "067-S-7029/PET_2022-03-30_FTP", 
    "131-S-7032/PET_2022-03-01_FTP",
    "024-S-6005/PET_2017-04-27_FTP",
    "024-S-6005/PET_2021-07-20_FTP"
]

def load_json_dict(json_file):
    with open(json_file, 'r') as f:
        d = json.load(f)
    return d 

def remove_invalid(base_path, ids):
    """
    Missing: /home/jagust/xnat/xnp/sshfs/xnat_data/adni/041-S-4200/PET_2017-10-17_FTP/analysis/rnu.nii
    Missing: /home/jagust/xnat/xnp/sshfs/xnat_data/adni/135-S-4598/PET_2021-07-28_FTP/analysis/rnu.nii
    Missing: /home/jagust/xnat/xnp/sshfs/xnat_data/adni/137-S-4351/PET_2017-11-09_FTP/analysis/rnu.nii
    Broken symbolic link: /home/jagust/xnat/xnp/sshfs/xnat_data/adni/035-S-7000/PET_2021-10-13_FTP/analysis/rnu.nii
    """
    faulty_ids = ["041-S-4200/PET_2017-10-17_FTP", "135-S-4598/PET_2021-07-28_FTP", "137-S-4351/PET_2017-11-09_FTP", "116-S-4483/PET_2018-04-03_FTP"] # "035-S-7000/PET_2021-10-13_FTP"
    for faulty_id in faulty_ids:
        if faulty_id in ids:
        # if "116-S-4483/PET_2018-04-03_FTP" in ids:
            ids = np.delete(ids, np.argwhere("116-S-4483/PET_2018-04-03_FTP" == ids)) # NOTE
            # ids = np.delete(ids, np.argwhere(faulty_id == ids))

    return np.array([i for i in ids if os.path.exists(base_path.replace("-id-", i))])

def reduce_image_size(image):
    binary_mask = (image > 0).float()

    # Find the bounding box coordinates of the object
    non_zero_indices = torch.nonzero(binary_mask)
    min_coords = non_zero_indices.min(dim=0)[0]
    max_coords = non_zero_indices.max(dim=0)[0]

    # Crop the object using the bounding box coordinates
    cropped_image = image[min_coords[0]:max_coords[0]+1, min_coords[1]:max_coords[1]+1, min_coords[2]:max_coords[2]+1]

    return cropped_image

def read_image_with_retry(volume_path, max_retries=20, retry_delay=30): 
    volume = None
    for attempt in range(1, max_retries + 1):
        try:
            volume = sitk.ReadImage(volume_path)
            # print(f"Volume successfully read on attempt {attempt}.")
            break  # Exit loop if successful
        except RuntimeError as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print("Retrying...")
                time.sleep(retry_delay)
            else:
                print("Maximum retries reached. Unable to read the volume.")
                raise  # Re-raise the exception if all attempts fail

    return volume

def read_image_nib_with_retry(volume_path, max_retries=20, retry_delay=60, smooth=False): 

    volume_matrix = None
    for attempt in range(1, max_retries + 1):
        try:
            # volume_matrix = np.transpose(nib.load(volume_path).get_fdata(), (2, 1, 0))
            volume = nib.load(volume_path)
            sitk_volume = sitk.ReadImage(volume_path)
            # sitk_mat_rsa = np.transpose(sitk.GetArrayFromImage(sitk_volume), (2, 1, 0))
            # nib_volume = nib.Nifti1Image(sitk_mat_rsa, volume.affine.copy(), volume.header)
            nib_volume = nib.Nifti1Image(sitk.GetArrayFromImage(sitk_volume), volume.affine.copy(), volume.header)
            if smooth:
                nib_volume = nib_smooth_image(nib_volume, fwhm=2, mode='nearest')
            volume_matrix = nib_volume.get_fdata()
            break  # Exit loop if successful
        except FileNotFoundError as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print("Retrying...")
                time.sleep(retry_delay)
            else:
                print("Maximum retries reached. Unable to read the volume.")
                raise  # Re-raise the exception if all attempts fail

    volume_tensor = torch.from_numpy(volume_matrix).to(dtype=torch.float32).unsqueeze(dim=0) # add channel dim
    return volume_tensor

def load_nifti_vol(volume_path, cuda_id=-1, add_channel_dim=True, **kwargs):
    
    def resize_volume(vol, new_spc):
        # Resample images to 2mm spacing with SimpleITK
        original_spacing = vol.GetSpacing()
        original_size = vol.GetSize()

        out_size = [
            int(np.round(original_size[0] * (original_spacing[0] / new_spc[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / new_spc[1]))),
            int(np.round(original_size[2] * (original_spacing[2] / new_spc[2])))
        ]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(new_spc)
        resample.SetSize(out_size)
        resample.SetOutputDirection(vol.GetDirection())
        resample.SetOutputOrigin(vol.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(vol.GetPixelIDValue())

        # resample.SetInterpolator(sitk.sitkBSpline)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)

        resampled_vol = resample.Execute(vol)

        return resampled_vol 
    
    volume = read_image_with_retry(volume_path, max_retries=10, retry_delay=10)
    new_spacing = [2.0, 2.0, 2.0]
    volume = resize_volume(volume, new_spacing)
    # volume_tensor = torch.from_numpy(sitk.GetArrayFromImage(resampled_volume)).to(dtype=torch.float32).unsqueeze(dim=0) # add channel dim
    volume_tensor = torch.from_numpy(sitk.GetArrayFromImage(volume)).to(dtype=torch.float32)
    volume_tensor = torch.nan_to_num(volume_tensor)
    if add_channel_dim:
        volume_tensor = volume_tensor.unsqueeze(dim=0)
    if cuda_id != -1:
        volume_tensor = volume_tensor.cuda(cuda_id)

    return volume_tensor

def quick_mean_std(datasets):
    x = torch.vstack([dataset.get_mris() for dataset in datasets])
    y = torch.vstack([torch.from_numpy(dataset.get_targets() for dataset in datasets)])

    x_mean = torch.mean(x, dim=2)
    x_std = torch.std(x, dim=2)
    y_mean = torch.mean(y, dim=1)
    y_std = torch.std(y, dim=1)

    return x_mean, x_std, y_mean, y_std


def compute_mean_std(datasets, batch_size, cuda_id):
    dataloader = DataLoader(ConcatDataset(datasets), batch_size=batch_size, shuffle=False) #, num_workers=2)
    num_samples = 0
    x_mean = 0
    y_mean = 0
    x_var = 0
    y_var = 0
    num_samples = 0

    nz_x_mean = 0
    nz_x_mean_squares = 0
    
    for i, (batch, y) in enumerate(dataloader):
        print(f"mean-std calc batch: {i}/{len(dataloader)}")
        batch_size, num_channels, depth, height, width = batch.size()
        # batch = batch.view(batch_size, num_channels, -1)  # Flatten the spatial dimensions
        batch = batch.view(batch_size, -1)  # Flatten the spatial dimensions

        nonzero_batch_mean = torch.sum(batch, dim=1) / torch.count_nonzero(batch)
        nonzero_batch_mean_squares = torch.sum(torch.square(batch), 1) / torch.count_nonzero(batch)
        nz_x_mean += torch.sum(nonzero_batch_mean, dim=0)
        nz_x_mean_squares += torch.sum(nonzero_batch_mean_squares, dim=0)

        # Compute mean and variance for the current batch
        # batch_mean = torch.mean(batch, dim=1)
        # batch_var = torch.var(batch, dim=1)

        # Update the running mean and variance
        # x_mean += torch.sum(batch_mean, dim=0)
        # x_var += torch.sum(batch_var, dim=0)
        # y_mean += torch.sum(torch.mean(y, dim=1), dim=0)
        # y_var += torch.sum(torch.var(y, dim=1), dim=0)
        
        num_samples += batch_size

    # Compute the overall mean and variance
    # overall_x_mean = x_mean / num_samples
    # overall_x_var = x_var / num_samples
    overall_y_mean = y_mean / num_samples
    overall_y_var= y_var / num_samples
    # Compute standard deviation from variance
    # overall_x_std = torch.sqrt(overall_x_var)
    overall_y_std = torch.sqrt(torch.tensor(overall_y_var))

    """
    mean_intensity = running_sum / count
    std_intensity = np.sqrt(running_sum_squares / count - (mean_intensity ** 2))
    """
    overall_x_mean = nz_x_mean / num_samples 
    overall_x_std = torch.sqrt(torch.tensor(nz_x_mean_squares / num_samples - torch.square(overall_x_mean)))

    return overall_x_mean, overall_x_std, overall_y_mean, overall_y_std

def get_PCA(datasets:list, center=False):
    # suvr_vectors = torch.vstack([suvr for i, (_, suvr) in enumerate(dataloader)]).detach().cpu().numpy()
    # suvr_vectors = dataset_tensor.detach().cpu().numpy()
    suvr_vectors = np.vstack([dataset.get_targets() for dataset in datasets])
    # suvr_vectors -= np.mean(suvr_vectors, axis=0)

    pca = PCA(n_components=3, whiten=False, center=center, svd_solver='full')
    pca.fit(suvr_vectors)

    num_components = pca.n_components_
    components = pca.components_
    explained_var = pca.explained_variance_

    return pca


def test_PCA(pca, datasets:list):
    suvr_vectors = np.vstack([dataset.get_targets() for dataset in datasets])

    print(f"Data mean: {np.mean(suvr_vectors)}")
    reduced = pca.transform(suvr_vectors) 
    reconstructed = pca.inverse_transform(reduced)
    # scores = pca.score_samples(suvr_vectors)


    print(f"Transformed: {reduced}")
    print(f"Mean transformed: {np.mean(reduced)}")
    # print(f"Scores: {len(pca.score_samples(suvr_vectors))}")

    # error = np.mean(np.power(suvr_vectors - reconstructed, 2))
    error = mean_squared_error(suvr_vectors, reconstructed)
    print(f"Reconstruction error: {error}")

    transformed, back, evals, evecs = do_pca(suvr_vectors)
    print(f"{transformed}")
    print(np.mean(transformed))
    print(f"err: {mean_squared_error(suvr_vectors, back)}")
    

    return error

def get_split_datasets(all_splits_files, index, cuda_id, col_list=""):
    all_samples = set()
    for root, _, _ in os.walk(all_splits_files):
        for i, split_file in enumerate(glob.glob(os.path.join(root, "*.csv"))):
            samples_df = pd.read_csv(split_file, header=None, names=['samples'])
            # splits.append(list(samples_df['samples'].values))
            samples = set(samples_df['samples'].values)
            all_samples = all_samples.union(samples)
            if i == index:
                train_samples = samples

    test_samples = all_samples - train_samples 

    test_dataset = ImageDataset.ImageDataset(list(test_samples), col_list=col_list, transform=None, cuda_id=cuda_id)
    train_dataset = ImageDataset.ImageDataset(list(train_samples), col_list=col_list, transform=None, cuda_id=cuda_id)

    return train_dataset, test_dataset


def load_split_datasets(splits_dir:str, dataset:Dataset, index:int, cuda_id:int, file_base_name = "_lookup_", contra=False, template=False, resize=True, with_covars=True, smoothing=False):
    
    # train_samples_df = pd.read_csv(os.path.join(splits_dir, f"training{file_base_name}{index}.csv"), header=None, names=['samples'])
    # train_samples = train_samples_df['samples'].values
    # test_samples_df = pd.read_csv(os.path.join(splits_dir, f"test{file_base_name}{index}.csv"), header=None, names=['samples'])
    # test_samples = test_samples_df['samples'].values
    
    base_path = f"{os.getcwd()}"
    # with open(os.path.join(base_path,"fold1_test_xgb_pred_table.pkl"), "rb") as f:
    #     test_predict_meta_tau_table = pickle.load(f)
    
    # with open(os.path.join(base_path,"fold1_train_xgb_pred_table.pkl"), "rb") as f:
    #     train_predict_meta_tau_table = pickle.load(f)

    ## NOTE for synthetic noise testing
    # meta_tau_df = pd.read_csv(f"{os.getcwd()}/scripts/ADNI_W_Covars.csv")
    # adni_ids, meta_taus = meta_tau_df["ADNI_ID"].values, meta_tau_df["Tau_Meta"].values
    # # pcts = meta_taus * np.random.uniform(0.05, 0.06, size=meta_taus.shape) # 2024-10-08_07-38-20
    # # pcts = meta_taus * np.random.uniform(0.04, 0.05, size=meta_taus.shape) # 
    # # pcts = meta_taus * np.random.uniform(0.04, 0.06, size=meta_taus.shape) # 
    # # meta_taus_noised = meta_taus + pcts * np.random.uniform(-1, 1, pcts.shape)
    # np.random.seed(50)
    # sd_noise_level = 0.2
    # meta_taus_noised = meta_taus + np.random.normal(0, 0.2, size=meta_taus.shape) # run: 2024-10-17_21-34-06 | revised ErrW: 
    # # meta_taus_noised = meta_taus + np.random.normal(0, 0.25, size=meta_taus.shape) # sd=0.25, run: 2024-10-17_22-47-59 | revised ErrW: 2024-10-21_01-42-11
    # # meta_taus_noised = meta_taus + np.random.normal(0, 0.15, size=meta_taus.shape) # run: 2024-10-18_23-36-34
    # # meta_taus_noised = meta_taus + np.random.normal(0, 0.275, size=meta_taus.shape) # run: 2024-10-19_00-21-39
    # msk = np.isnan(meta_taus)
    # logging.info(f"SD: {sd_noise_level} | R2: {r2_score(meta_taus[~msk], meta_taus_noised[~msk])}\tMAPE:{100*np.mean(np.abs(meta_taus[~msk] - meta_taus_noised[~msk]) / meta_taus[~msk])}")

    # noised_meta_tau_table = dict(zip(adni_ids, meta_taus_noised))

    # # meta_tau_pred_table_train_test = np.load(f"{os.getcwd()}/predictions_for_meta_tau.npy", allow_pickle=True)[0]
    # meta_tau_pred_table_train_test = np.load(f"{os.getcwd()}/with_volume_preds/predictions_for_meta_tau.npy", allow_pickle=True)[0]
    # # meta_tau_pred_table_train_test = np.load(f"{os.getcwd()}/trn_gt_tst_pred_for_meta_tau.npy", allow_pickle=True)[0]
    # # meta_tau_pred_table_hold_out = np.load(f"{os.getcwd()}/predictions_for_meta_tau_hold_outs.npy", allow_pickle=True)[0]
    # meta_tau_pred_table_hold_out = np.load(f"{os.getcwd()}/with_volume_preds/predictions_for_meta_tau_hold_outs.npy", allow_pickle=True)[0]
    # meta_tau_pred_table = {**meta_tau_pred_table_train_test, **meta_tau_pred_table_hold_out}
    # # meta_tau_pred_table = np.load(f"{os.getcwd()}/all_samples_gt_for_meta_tau.npy", allow_pickle=True)[0]

    # zero-indexed
    # meta_tau_pred_table_train_test = np.load(f"{os.getcwd()}/scripts/catboost_predictions/fold_{index-1}/predictions_for_meta_tau.npy", allow_pickle=True)[0]
    # # meta_tau_pred_table_train_test = np.load(f"{os.getcwd()}/trn_gt_tst_pred_for_meta_tau.npy", allow_pickle=True)[0]
    # # meta_tau_pred_table_hold_out = np.load(f"{os.getcwd()}/predictions_for_meta_tau_hold_outs.npy", allow_pickle=True)[0]
    # meta_tau_pred_table_hold_out = np.load(f"{os.getcwd()}/scripts/catboost_predictions/fold_{index-1}/predictions_for_meta_tau_hold_outs.npy", allow_pickle=True)[0]
    
    # meta_tau_pred_table_train_test = np.load(f"{os.getcwd()}/scripts/native_space_tau_roi_predictions/fold_{index-1}/predictions_for_meta_tau.npy", allow_pickle=True)[0]
    # meta_tau_pred_table_train_test = np.load(f"{os.getcwd()}/scripts/brr_native_space_roi_tau_predictions/fold_{index-1}/predictions_for_meta_tau.npy", allow_pickle=True)[0]
    
    meta_tau_pred_table_train_test = np.load(f"{os.getcwd()}/scripts/ngboost_native_space_roi_tau_predictions/fold_{index-1}/predictions_for_meta_tau.npy", allow_pickle=True)[0]
    
    # meta_tau_pred_table_train_test = np.load(f"{os.getcwd()}/scripts/NGBoost_base_native_space_roi_tau_predictions/fold_{index-1}/predictions_for_meta_tau.npy", allow_pickle=True)[0]
    
    # meta_tau_pred_table_train_test = np.load(f"{os.getcwd()}/trn_gt_tst_pred_for_meta_tau.npy", allow_pickle=True)[0]
    
    # meta_tau_pred_table_hold_out = np.load(f"{os.getcwd()}/predictions_for_meta_tau_hold_outs.npy", allow_pickle=True)[0]
    # meta_tau_pred_table_hold_out = np.load(f"{os.getcwd()}/scripts/native_space_tau_roi_predictions/fold_{index-1}/predictions_for_meta_tau_hold_outs.npy", allow_pickle=True)[0]
    # meta_tau_pred_table_hold_out = np.load(f"{os.getcwd()}/scripts/brr_native_space_roi_tau_predictions/fold_{index-1}/predictions_for_meta_tau_hold_outs.npy", allow_pickle=True)[0]
    # meta_tau_pred_table_hold_out = np.load(f"{os.getcwd()}/scripts/ngboost_native_space_roi_tau_predictions/fold_{index-1}/predictions_for_meta_tau_hold_outs.npy", allow_pickle=True)[0]
    # meta_tau_pred_table_hold_out = np.load(f"{os.getcwd()}/scripts/NGBoost_base_native_space_roi_tau_predictions/fold_{index-1}/predictions_for_meta_tau_hold_outs.npy", allow_pickle=True)[0]
    
    # meta_tau_pred_table = {**meta_tau_pred_table_train_test, **meta_tau_pred_table_hold_out}
    meta_tau_pred_table = meta_tau_pred_table_train_test

    # NOTE below is for MRI -> PET
    # meta_tau_df = pd.read_csv(f"{os.getcwd()}/scripts/ADNI_W_Covars.csv")
    # # meta_tau_pred_table = meta_tau_df.set_index("ADNI_ID")[["ADNI_ID", "Tau_Meta"]].to_dict(orient='index')
    # adni_ids, meta_taus = meta_tau_df["ADNI_ID"].values, meta_tau_df["Tau_Meta"].values
    # meta_tau_pred_table = dict(zip(adni_ids, meta_taus))

    if contra:
        covar_lookup = f"{os.getcwd()}/scripts/ADNI_W_Covars.csv" # f"{os.getcwd()}/scripts/ADNI_covariate.csv"
        # train_dataset = vd.ContrastiveVolumeDataset(os.path.join(splits_dir, f"training{file_base_name}{index}.csv"), covar_lookup, holdout_ids=SELECTED_SAMPLES, with_all_covars=with_covars, transform=None, resize=resize, smoothing=smoothing, cuda_id=cuda_id)
        # train_dataset = vd.ClusterVolumeDataset(os.path.join(splits_dir, f"training{file_base_name}{index}.csv"), covar_lookup, holdout_ids=SELECTED_SAMPLES, with_all_covars=with_covars, transform=None, resize=resize, cuda_id=cuda_id)
        # test_dataset = vd.CovariateVolumeDataset(covar_lookup, os.path.join(splits_dir, f"test{file_base_name}{index}.csv"), with_all_covars=with_covars, transform=None, resize=resize,smoothing=smoothing, cuda_id=cuda_id)
        
        # mode='cluster'
        # train_dataset = vd.RegressionVolumeDataset(os.path.join(splits_dir, f"training{file_base_name}{index}.csv"), covar_lookup, holdout_ids=SELECTED_SAMPLES, mode=mode, with_all_covars=with_covars, transform=None, resize=resize, smoothing=smoothing, cuda_id=cuda_id)
        # test_dataset = vd.RegressionVolumeDataset(os.path.join(splits_dir, f"test{file_base_name}{index}.csv"), covar_lookup, with_all_covars=with_covars, transform=None, mode=mode, resize=resize, smoothing=smoothing, cuda_id=cuda_id)
        
        mode='cluster'
        train_lookup_file = os.path.join(splits_dir, f"training{file_base_name}{index}.csv")
        test_lookup_file = os.path.join(splits_dir, f"test{file_base_name}{index}.csv")


        # train_dataset = vd.PredictedMetaTauDataset(noised_meta_tau_table, train_lookup_file, covar_lookup, mode=mode, holdout_ids=SELECTED_SAMPLES, with_all_covars=with_covars, transform=None, resize=True, smoothing=True, cuda_id=cuda_id)
        # test_dataset = vd.PredictedMetaTauDataset(noised_meta_tau_table, test_lookup_file, covar_lookup, with_all_covars=with_covars, transform=None, mode=mode, resize=True, smoothing=True, cuda_id=cuda_id)
        train_dataset = vd.PredictedMetaTauDataset(meta_tau_pred_table, train_lookup_file, covar_lookup, mode=mode, holdout_ids=SELECTED_SAMPLES, with_all_covars=with_covars, transform=None, resize=True, smoothing=smoothing, cuda_id=cuda_id)
        test_dataset = vd.PredictedMetaTauDataset(meta_tau_pred_table, test_lookup_file, covar_lookup, with_all_covars=with_covars, transform=None, mode=mode, resize=True, smoothing=smoothing, cuda_id=cuda_id)
    

        # return train_dataset, test_dataset
    else:
        train_dataset = dataset(os.path.join(splits_dir, f"training{file_base_name}{index}.csv"), transform=None, resize=resize, smoothing=smoothing, cuda_id=cuda_id)
        test_dataset = dataset(os.path.join(splits_dir, f"test{file_base_name}{index}.csv"), transform=None, resize=resize, smoothing=smoothing, cuda_id=cuda_id)
    # if template:
    #     train_dataset = dataset(os.path.join(splits_dir, f"training{file_base_name}{index}.csv"), resize=resize, mri_file_type="wrnu.nii", tau_file_type="wsuvr_cereg.nii", cuda_id=cuda_id)
    #     test_dataset = dataset(os.path.join(splits_dir, f"test{file_base_name}{index}.csv"), resize=resize, mri_file_type="wrnu.nii", tau_file_type="wsuvr_cereg.nii", cuda_id=cuda_id)
    #     return train_dataset, test_dataset
    if template:
        train_dataset.mri_file_type="wrnu.nii"
        train_dataset.tau_file_type="wsuvr_cereg.nii"
        test_dataset.mri_file_type="wrnu.nii"
        test_dataset.tau_file_type="wsuvr_cereg.nii"
    # test_dataset = ImageDataset.ImageDataset(list(test_samples), col_list=col_list, transform=None, cuda_id=cuda_id)
    # train_dataset = ImageDataset.ImageDataset(list(train_samples), col_list=col_list, transform=None, cuda_id=cuda_id)

    return train_dataset, test_dataset

def load_single_split_datasets(dir: str, train_dataset_constructor: Dataset, test_dataset_constructor:Dataset, cuda_id: int, contra=False, ):

    if contra:
        train_covar_lookup = f"{os.getcwd()}/scripts/ADNI_covariate.csv"
        test_covar_lookup = f"{os.getcwd()}/scripts/A4_covariate.csv"
        train_dataset = train_dataset_constructor(os.path.join(dir, "adni_training.csv"), train_covar_lookup, holdout_ids=SELECTED_SAMPLES, transform=None, cuda_id=cuda_id)
        test_dataset = test_dataset_constructor(os.path.join(dir, "a4_testing.csv"), test_covar_lookup, transform=None, cuda_id=cuda_id)

        return train_dataset, test_dataset

    train_dataset = train_dataset_constructor(os.path.join(dir, "adni_training.csv"), transform=None, cuda_id=cuda_id, mri_file_type="c1rnu.nii")
    test_dataset = test_dataset_constructor(os.path.join(dir, "a4_testing.csv"), transform=None, cuda_id=cuda_id, mri_file_type="c1rnu.nii")

    assert len(train_dataset) == 1695, f"Expected training dataset to have 1695 samples, but instead has {len(train_dataset)}"
    assert len(test_dataset) == 444, f"Expected training dataset to have 444 samples, but instead has {len(test_dataset)}"

    return train_dataset, test_dataset


def create_splits_lookup_tables(splits_dir, lookup_file, out_dir):
    lookup_df = pd.read_csv(lookup_file)
    for index in range(1, 6):
        train_df = pd.read_csv(os.path.join(splits_dir, f"trainingfold{index}.csv"), header=None, names=['samples'])
        train_ids = ["/".join(sample.split("/")[4:]) for sample in train_df['samples'].values]
        test_df = pd.read_csv(os.path.join(splits_dir, f"testfold{index}.csv"), header=None, names=['samples'])
        test_ids = ["/".join(sample.split("/")[4:]) for sample in test_df['samples'].values]

        # Locate these in the look up table
        train_lookup_df = lookup_df.loc[lookup_df['MRI'].str.contains('|'.join(train_ids))]
        test_lookup_df = lookup_df.loc[lookup_df['MRI'].str.contains('|'.join(test_ids))]

        train_lookup_df.to_csv(os.path.join(out_dir, f"training_lookup_{index}.csv"), index=False)
        test_lookup_df.to_csv(os.path.join(out_dir, f"test_lookup_{index}.csv"), index=False)


def create_dataloader(dataset, batch_size, shuffle=False, contra=False):
    if contra:
        skip_ids = dataset.find_nan_ids()
        sampler = vd.CustomSampler(dataset.lookup_df, skip_ids=skip_ids, shuffle=shuffle)
        return DataLoader(dataset, batch_size, sampler=sampler)
    else :
        # skip_ids = dataset.find_nan_ids()
        # sampler = vd.CustomSampler(dataset.lookup_df, skip_ids=skip_ids, shuffle=shuffle)
        # return DataLoader(dataset, batch_size, sampler=sampler)
        return DataLoader(dataset, batch_size, shuffle=shuffle) # NOTE original
    

def get_splits(splits_dir, cuda_id, col_list=""):
    splits = []
    for root, _, _ in os.walk(splits_dir):
        for split_file in glob.glob(os.path.join(root, "*.csv")):
            samples_df = pd.read_csv(split_file, header=None, names=['samples'])
            # splits.append(list(samples_df['samples'].values))
            splits.append(set(samples_df['samples'].values))

    datasets = []
    for i, split_files in enumerate(splits):
        # Create dataset
        datasets.append(ImageDataset.ImageDataset(split_files, transform=None, cuda_id=cuda_id, col_list=col_list))
    
    return datasets
    
def create_fold_dataloader(split_idx: int, datasets: list, batch_size: int) -> tuple:
    test_dataset = datasets[split_idx]
    train_datasets = datasets[:split_idx] + datasets[split_idx+1:]
    train_dataset = ConcatDataset(train_datasets)

    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)

    return train_loader, test_loader

def dataset_info(dataloader):
    avg_size = torch.zeros(5)
    for batch_idx, (mri, suvr) in enumerate(dataloader):
        print(mri.size())
        b, c, w, h, d = mri.shape
        avg_size = torch.add(avg_size, torch.tensor([b, c, w, h, d]))

    print(f"avg size: {avg_size / batch_idx}")

def do_pca(data,nRedDim=3,normalise=1):
    # Centre data
    # mu = np.mean(data, axis=0)
    # data -= mu
    # Covariance matrix
    C = np.cov(np.transpose(data))
    # Compute eigenvalues and sort into descending order
    evals,evecs = np.linalg.eig(C) 
    indices = np.argsort(evals) 
    indices = indices[::-1]
    evecs = evecs[:,indices]
    evals = evals[indices] 
    if nRedDim>0:
        evecs = evecs[:,:nRedDim]
    if normalise:
        for i in range(np.shape(evecs)[1]):
            evecs[:,i] / np.linalg.norm(evecs[:,i]) * np.sqrt(evals[i])
    # Produce the new data matrix
    x = np.dot(np.transpose(evecs),np.transpose(data)) # Compute the original data again 
    y = np.transpose(np.dot(evecs,x))# + mu
    return x,y,evals,evecs

def analysis(dataset, pca):
    suvr_vectors = dataset.get_targets()
    mean = np.mean(suvr_vectors)
    print(f"SUVR mean: {mean}")

    transformed = pca.transform(suvr_vectors)
    transformed_mean = np.mean(transformed[:, 0])
    print(f"SUVR PC mean: {transformed_mean}")

def write_tensor_to_nii(tensor, save_path, add_channel=False):
    if add_channel:
        tensor = tensor.unsqueeze(0)
    np_mat = tensor.detach().cpu().numpy()
    sitk_img = sitk.GetImageFromArray(np_mat)
    sitk.WriteImage(sitk_img, save_path)

def analyze_region(pred_file, tau_file, roi_file, roi_ids):
    # 2022, 1022
    pred_volume = sitk.ReadImage(pred_file)
    tau_volume = sitk.ReadImage(tau_file)
    roi_volume = sitk.ReadImage(roi_file)

    pred = torch.from_numpy(sitk.GetArrayFromImage(pred_volume))
    tau = torch.from_numpy(sitk.GetArrayFromImage(tau_volume))
    roi = torch.from_numpy(sitk.GetArrayFromImage(roi_volume))

    mask = torch.zeros(roi.size())
    for roi_id in roi_ids: 
        mask[:] = 0 # reset 
        mask[roi == roi_id] = 1

        pred_region = pred[mask == 1]
        tau_region = tau[mask == 1]

        # Compare 
        mae = torch.abs(pred_region - tau_region).sum() / torch.count_nonzero(mask, )
        mape = torch.abs((tau_region - pred_region) / tau_region).sum() / torch.count_nonzero(mask, )
        pred_roi_mean = pred_region.sum() / torch.count_nonzero(mask, )
        tau_roi_mean = tau_region.sum() / torch.count_nonzero(mask, )

        print(f"\nREGION: {roi_id}:")
        print(f"Regions size: {torch.count_nonzero(mask)}")
        print(f"MAE: {mae}")
        print(f"MAPE: {mape}")
        print(f"Pred mean: {pred_roi_mean}")
        print(f"Tau mean: {tau_roi_mean}")
        print(f"Pred Variance: {torch.var(pred_region)}")
        print(f"Tau Variance: {torch.var(tau_region)}")

def analyze_sample(tau_file, roi_file, roi_ids=[]):

    tau_volume = sitk.ReadImage(tau_file)
    roi_volume = sitk.ReadImage(roi_file)

    tau = torch.from_numpy(sitk.GetArrayFromImage(tau_volume))
    roi = torch.from_numpy(sitk.GetArrayFromImage(roi_volume))

    mask = torch.zeros(roi.size())
    for roi_id in roi_ids: 
        mask[:] = 0 # reset 
        mask[roi == roi_id] = 1
        tau_region = tau[mask == 1]

        print(f"Region {roi_id} mean: {torch.mean(tau_region)}")
        print(f"Region {roi_id} var: {torch.mean(tau_region)}")
        print(f"Region {roi_id} min: {torch.min(tau_region)}")
        print(f"Region {roi_id} max: {torch.max(tau_region)}")


def run_analyze_sample():
    predf = f"{os.getcwd()}/results/2023-11-28_00-07-37/fold_1/80_output_samples/prediction.nii"
    tauf = f"{os.getcwd()}/results/2023-11-28_00-07-37/fold_1/80_output_samples/gt_tau_pet.nii"
    roif = f"{os.getcwd()}/results/2023-11-28_00-07-37/fold_1/80_output_samples/roi_mask.nii"
    roinds = [1001,1006,1007,1009,1015,1016,1030,1034,1033,1008,1025,1029,1031,1022,17,18,2001,2006,2007,2009,2015,2016,2030,2034,2033,2008,2025,2029,2031,2022,49,50,51,52,53,54]
    analyze_region(predf, tauf, roif, roinds)
    
    tauf = "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/037-S-4214/PET_2021-03-10_FTP/analysis/suvr_cereg.nii"
    roif = "/home/jagust/xnat/xnp/sshfs/xnat_data/adni/037-S-4214/PET_2021-03-10_FTP/analysis/raparc+aseg.nii"

    analyze_sample(tauf, roif, [1022, 1029, 2022, 2029])


def load_model(ModelClass, train_dataloader, test_dataloader, roi_indices, params_file, save_path, cuda_id, *args, **kwargs):
    logging.info(f"Loading Model...")
    model = ModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(params_file))
    # model.load_state_dict(torch.load(params_file, map_location=torch.device(f'cuda:{CUDA_ID}')))
    model = model.to(device=torch.device(f'cuda:{cuda_id}'))
    model.eval()
    logging.info(f"Starting Encoding Output Extraction for Train Dataset...")
    train_loader_size = len(train_dataloader)
    train_encoded_outputs, train_covariates = [], []
    for batch_idx, (mri_volume, tau_volume, roi_mask, abeta, tau_path) in enumerate(train_dataloader):

        if batch_idx % 10 :
            logging.info(f"[{batch_idx}/{train_loader_size}]")

        gen_tau, encoded_lst = model(mri_volume) # Encoded: 512 x 8 x 8 x 8 

        train_encoded_outputs.append(encoded_lst[-1])
        train_covariates.append(abeta)

        break

        # nr_mape = torch.where(tau_volume != 0, torch.abs((tau_volume - gen_tau) / tau_volume), torch.nan)
        # mape = torch.nanmean(nr_mape * 100, dim=(-3, -2, -1)).sum()

        # logging.info(f"[{batch_idx}] MAPE: {mape}%")

    # train_encoded_outputs = torch.mean(torch.cat(train_encoded_outputs, 0), 1)
    train_encoded_outputs = torch.cat(train_encoded_outputs, 0)
    train_encoded_outputs = train_encoded_outputs.view(train_encoded_outputs.size(0), -1)
    logging.info(f"Encoded Outputs Aggreagte Shape: {train_encoded_outputs.size()}")
    train_covariates = torch.cat(train_covariates, 0)
    logging.info(f"Covars. Aggreagte Shape: {train_covariates.size()}")

    train_encoded_outputs = train_encoded_outputs.detach().cpu().numpy()
    train_encoded_outputs = train_encoded_outputs.reshape(train_encoded_outputs.shape[0], -1)
    train_covariates = train_covariates.detach().cpu().numpy()
    # Save
    # np.save(os.path.join(save_path, "fold1_train_encoded.npy"), train_encoded_outputs)
    # np.save(os.path.join(save_path, "fold1_train_covariates.npy"), train_covariates)
    
    logging.info(f"Starting Encoding Output Extraction for Test Dataset...")
    test_loader_size = len(test_dataloader)
    test_encoded_outputs, test_covariates = [], []
    for batch_idx, (mri_volume, tau_volume, roi_mask, abeta, tau_path) in enumerate(test_dataloader):

        if batch_idx % 10 :
            logging.info(f"[{batch_idx}/{test_loader_size}]")

        gen_tau, encoded_lst = model(mri_volume) # Encoded: 512 x 8 x 8 x 8 

        test_encoded_outputs.append(encoded_lst[-1])
        test_covariates.append(abeta)

        break

    # test_encoded_outputs = torch.mean(torch.cat(test_encoded_outputs, 0), 1)
    test_encoded_outputs = torch.cat(test_encoded_outputs, 0)
    test_encoded_outputs = test_encoded_outputs.view(test_encoded_outputs.size(0), -1)
    logging.info(f"Encoded Outputs Aggreagte Shape: {test_encoded_outputs.size()}")
    test_covariates = torch.cat(test_covariates, 0)
    logging.info(f"Covars. Aggreagte Shape: {test_covariates.size()}")

    test_encoded_outputs = test_encoded_outputs.detach().cpu().numpy()
    test_encoded_outputs = test_encoded_outputs.reshape(test_encoded_outputs.shape[0], -1)
    test_covariates = test_covariates.detach().cpu().numpy()
    # Save
    # np.save(os.path.join(save_path, "fold1_test_encoded.npy"), test_encoded_outputs)
    # np.save(os.path.join(save_path, "fold1_test_covariates.npy"), test_covariates)

    pls = PLSRegression(n_components=512)
    rfe = RFE(estimator=pls, n_features_to_select=512)
    
    X_rfe = rfe.fit_transform(train_encoded_outputs, train_covariates)

    logging.info(f"RFE: {X_rfe}")

    X_test_rfe = rfe.transform(test_encoded_outputs)
    y_pred = pls.predict(X_test_rfe)

    mse = np.mean(np.square(test_covariates - y_pred))

    logging.info(f"MSE: {mse}")
    
    # results = test(model, dataloader, roi_indices, torch.fill(torch.empty(len(roi_indices)), 5.), save_path, CUDA_ID)
    # mae, mape, rse, rrmse, ssim_error, roi_maes, roi_mapes, roi_rses, roi_wrrmses, roi_correlations = results

def x(f1, f2, ts, model):
    m = np.load(f"{os.getcwd()}/corrected_means/mask1.npy")[:, 1:]
    # mask has False on overlaps (i.e. places to remove)
    v1 = pd.read_csv(f1).values[:, 1:]
    v2 = pd.read_csv(f2).values[:, 1:]

    v1c = np.where(m, v1, np.nan)
    v1c = v1c[~np.isnan(v1c)].reshape(35, -1)
    v2c = np.where(m, v2, np.nan)
    v2c = v2c[~np.isnan(v2c)].reshape(35, -1)

    base_save = os.path.join(f"{os.getcwd()}/corrected_means", model, ts)
    if not os.path.exists(base_save):
        os.makedirs(base_save)
    df1c = pd.DataFrame(v1c)
    df1c.to_csv(os.path.join(base_save, f"{f1.split('/')[-1]}"), header=False)
    df2c = pd.DataFrame(v2c)

    df2c.to_csv(os.path.join(base_save, f"{f2.split('/')[-1]}"), header=False)

    visualization_util.scatter_corr(v1c[0, :], v2c[0, :], save_path=os.path.join(base_save, 'sctr.png'))


    rs = [np.corrcoef(v1[i, :], v2[i, :])[0, 1] for i in range(35)]
    rsc = [np.corrcoef(v1c[i, :], v2c[i, :])[0, 1] for i in range(35)]
    logging.info(rs == rsc)
    logging.info(np.mean(rsc))
    logging.info(f"{np.mean(rs)}\n")
    
def filter_for_holdout(mri_volume, tau_volume, rois, abeta, tau_paths, selected_samples=SELECTED_SAMPLES):
    hold_out_idxs = [i for i, sample in enumerate(tau_paths) if sample in selected_samples]
    if len(hold_out_idxs) > 0:
        holdout_mask = torch.ones(len(tau_paths), dtype=bool)[hold_out_idxs] = False
        mri_volume = mri_volume[holdout_mask, :]
        tau_volume = tau_volume[holdout_mask, :]
        rois = rois[holdout_mask, :]

        tau_paths = [sample for i, sample in enumerate(tau_paths) if sample not in selected_samples]
    
    if len(tau_paths) == 0:
        return -1 

    return (mri_volume, tau_volume, rois, abeta, tau_paths)

def extract_id(unstructured_sample_id):
    # /home/jagust/xnat/xnp/sshfs/xnat_data/a4/B10423472/PET_2017-01-01_FTP/analysis/suvr_cereg.nii
    tokens = unstructured_sample_id.split("/")
    if "A4_processing" in tokens:
        ind = tokens.index("A4_processing")
        sample_id = tokens[ind + 2]
    elif "a4" in tokens:
        ind = tokens.index("a4")
        sample_id = tokens[ind + 1] # always occurs after a4
    elif "ucsf" in tokens:
        ind = tokens.index("ucsf")
        sample_id = tokens[ind+1] + "/" + tokens[ind+2] 
    elif "scan" in tokens:
        ind = tokens.index("scan")
        sample_id = tokens[ind+1] + "/" + tokens[ind+2] 
    elif "processed" in tokens:
        ind = tokens.index("processed")
        sample_id = tokens[ind+1]
    elif "outputs" in tokens:
        ind = tokens.index("outputs")
        sample_id = tokens[ind+1]
    else:
        # for t in tokens:
        #     if "NACC" in t:
        #         return t
        # 002-S-6009/PET_2017-05-15_FTP
        # /home/jagust/xnat/xnp/sshfs/xnat_data/adni/000-S-0059/PET_2017-12-12_FTP/analysis/rnu.nii
        ind = tokens.index("adni")
        sample_id = "/".join(tokens[ind+1:ind+3])
    return sample_id

def get_id_from_path(file_path:str) -> int:

        chunks = file_path.split("/")
        id_chunk = chunks[-4]
        if "-" in id_chunk:
            id_chunk = os.path.join(id_chunk, chunks[-3])

        return id_chunk

def get_volume_paths_from_id(sample_id, df):
    pass

def check_for_longitudinal():
    splits_dir = f"{os.getcwd()}/training_folds/outlier_removed_splits"

    def get_sample_id(p):
        chunks = p.split("/")
        id_chunk = chunks[-4]
        id = id_chunk.split("-")[-1]
        return id

    for i in range(1, 6):
        train_df = pd.read_csv(os.path.join(splits_dir, f"training_lookup_{i}.csv"))
        test_df = pd.read_csv(os.path.join(splits_dir, f"test_lookup_{i}.csv"))

        train_ids = set(map(lambda x: get_sample_id(x), train_df["MRI"].values))
        test_ids = set(map(lambda x: get_sample_id(x), test_df["MRI"].values))

        intersect = train_ids.intersection(test_ids)

        print(intersect)
        print(len(intersect))


def convert_npy_to_nii(npy_path, save_path=""):

    volume = np.load(npy_path)

    # print(volume.shape)

    if len(volume.shape) != 3:
        # raise RuntimeError(f"Volume must have 3 dimensions, instead has {len(volume.shape)}")
        volume = np.squeeze(volume)
    
    if save_path == "":
        save_path = npy_path[:-4] + ".nii"
    sitk_img = sitk.GetImageFromArray(volume)
    sitk.WriteImage(sitk_img, save_path)

def form_attn_save_path(path: str, vdim: int):
    parts = path.split(".")
    parts[-1] = str(vdim)
    parts = "_vdim".join(parts)
    return os.path.join(parts)

def save_attention_coeffs(path:str, torch_tensor):
    volume = torch_tensor.detach().cpu().numpy()

    if len(volume.shape) != 3:
        # raise RuntimeError(f"Volume must have 3 dimensions, instead has {len(volume.shape)}")
        volume = np.squeeze(volume)

    save_path = form_attn_save_path(path, volume.shape[-1])
    sitk_img = sitk.GetImageFromArray(volume)
    sitk.WriteImage(sitk_img, save_path + ".nii")


def pad_volume(target_size):
    def pad(volume):
        num_dims = volume.dim()
        crop_or_pad_dims = []
        for i in range(1, 4):
            pad_before = max(0, (target_size[i-1] - volume.size(dim=num_dims-i)) // 2)
            pad_after = max(0, target_size[i-1] - volume.size(dim=num_dims-i) - pad_before)
            crop_or_pad_dims.extend([pad_before, pad_after])
        # Apply padding
        padded_volume = torch.nn.functional.pad(volume, tuple(crop_or_pad_dims), 'constant', 0)
        if padded_volume.size(dim=-2) != target_size[1]:
            padded_volume = padded_volume[:, :, :target_size[1], :]
        return padded_volume

    return pad

def load_template(resize=True, pad_dims=(128, 128, 128)):
    vdd = vd.VolumeDataset(f"{os.getcwd()}/training_folds/testfold1.csv", cuda_id=0)
    vdd.resize=resize
    vdd.pad_dims = pad_dims
    vdd.transform = pad_volume(pad_dims)
    roi_path = f"{os.getcwd()}/scripts/pet_dimYeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii"

    roi_mask = vdd.load_volume_file(roi_path, is_mask=True).squeeze(0)

    return roi_mask

def find_renamed():

    train = pd.read_csv(f"{os.getcwd()}/training_folds/outlier_removed_splits/training_lookup_1.csv")
    test = pd.read_csv(f"{os.getcwd()}/training_folds/outlier_removed_splits/test_lookup_1.csv")

    paths = np.concatenate((train["MRI"].values, test["MRI"].values))
    
    missing = []
    for path in paths:
        if not os.path.isfile(path):
            missing.append(path)
            print(f"Missing: {path}")
    
    np.save("missing_samples", paths)


if __name__ == "__main__":
    base_path = f"{os.getcwd()}"
    CUDA_ID = 1
    logging.basicConfig(filename=os.path.join(base_path, "saved_model_analysis", "test.log"), filemode='w', format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, stream=None, force=True)
    
    test_lookup_file = f"{os.getcwd()}/training_folds/volume_splits/test_lookup_1.csv"
    test_dataset = vd.VolumeDataset(test_lookup_file, cuda_id=CUDA_ID)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    train_lookup_file = f"{os.getcwd()}/training_folds/volume_splits/training_lookup_1.csv"
    train_dataset = vd.VolumeDataset(train_lookup_file, cuda_id=CUDA_ID)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=False)

    roi_indices = [1001,1006,1007,1009,1015,1016,1030,1034,1033,1008,1025,1029,1031,1022,17,18,2001,2006,2007,2009,2015,2016,2030,2034,2033,2008,2025,2029,2031,2022,49,50,51,52,53,54]

    
    # model_params = (3, 1, 1, [32, 64, 128, 256, 512], [2]*5)
    # params_file = f"{os.getcwd()}/results/2024-01-18_08-55-43/fold_2/models/epoch_99_model.pt"
    # # params_file = f"{os.getcwd()}/results/2024-01-21_10-47-13/fold_2/models/epoch_100_model.pt"
    # load_model(GenAttnUnet, dataloader, roi_indices, params_file, f"{os.getcwd()}/saved_model_analysis", *model_params)
    # model_timestamps_fold1 = [
    #     "2024-01-08_17-36-11", # UNETR
    #     "2024-01-08_17-46-03", # AttnUNETR
    #     "2024-01-10_12-46-08", # SwinUNETR
    #     "2024-01-10_13-39-42", # AttnUNET
    #     # "2024-01-14_15-25-21" # AttnSwinUnetr
    # ]
    # model_timestamps_fold2 = [ # fold 2
    #     "2024-01-18_08-55-31", # UNETR
    #     "2024-01-21_10-47-13", # AttnUNETR
    #     "2024-01-21_10-44-15", # SwinUNETR
    #     "2024-01-18_08-55-43" # AttnUNET
    # ]
    # models = ["UNETR", "AttnUNETR", "SwinUNETR", "AttnUNET"]
    # for i, m in enumerate(model_timestamps_fold1):
    #     logging.info(f"{models[i]}")
    #     p1 = f"{os.getcwd()}/results/{m}/fold_1/95_output_samples/gt_means.csv"
    #     p2 = f"{os.getcwd()}/results/{m}/fold_1/95_output_samples/pred_means.csv"
    #     x(p1, p2, m, models[i])
    # sys.exit(0)
    # /home/jagust/xnat/xnp/sshfs/xnat_data/adni/014-S-6988/PET_2022-06-30_FTP/analysis/rnu.nii


    # splits_dir = os.path.join(base_path, "training_folds")
    # new_splits_dir = os.path.join(splits_dir, "volume_splits_clean")
    # if not os.path.exists(new_splits_dir):
    #     os.makedirs(new_splits_dir)
    # create_splits_lookup_tables(splits_dir, f"{os.getcwd()}/scripts/directory_lookup.csv", new_splits_dir)

    # mri_dataset, test_dataset = load_split_datasets(splits_dir, 1, cuda_id=1, col_list=col_list)
    # train_x_mean, train_x_std, train_y_mean, train_y_std = compute_mean_std([mri_dataset], 216, 1)
    # mri_dataset.set_mean_std(train_x_mean, train_x_std, train_y_mean, train_y_std)
    # print(f"Mean and std vals: {train_x_mean}, {train_x_std}, {train_y_mean}, {train_y_std}")
    # dataloader = DataLoader(mri_dataset, batch_size=10, shuffle=False)
    # pca = get_PCA([mri_dataset], center=True)
    # num_components = pca.n_components_
    # components = pca.components_
    # explained_var = pca.explained_variance_ratio_
    # print(explained_var)
    # analysis(mri_dataset, pca)
    # suvrs = mri_dataset.get_targets()
    # roi_mean = np.mean(suvrs, axis=0)[1]
    # roi_min = np.min(suvrs, 0)[1]
    # roi_max = np.max(suvrs, 0)[1]
    # print(f"ROI {4} mean: {roi_mean}\trange: {roi_max - roi_min}\tmin: {roi_min}\tmax: {roi_max}")

    # corrs = np.array([-0.12185512, 0.5962195, 0.13045024, -0.104546, 0.09704352, -0.10274239, 0.08535595, 0.06127429, -0.00792209, 0.50680255, -0.09883042, -0.06698694, -0.01192661, -0.13435925, 0.44959556, 0.08049222, -0.09248748, -0.03858992, 0.06617693, -0.04124083, 0.49528436, 0.10999164, -0.03240087, 0.00996894, -0.15445169, 0.10965347, -0.02319586, -0.1307594, -0.02267317, -0.02325068, 0.0406019, -0.12527345, -0.02264107, 0.13671615, 0.05459768, 0.1839616, 0.09675928, 0.2083873, 0.26051385, 0.2466155, 0.10955304, 0.26260741, 0.23125657, 0.0, -0.14128762, 0.12830093, -0.0259128, -0.0224791, -0.07513035, -0.14752467, -0.15034866, -0.14797227, 0.00080193, -0.10987038, -0.03899369, -0.10810825, 0.02394566, -0.07940742, -0.10934205, 0.02955912, 0.06647944, 0.05666482, 0.04794537, 0.03202874, 0.08398924, -0.07393333, -0.0099705, 0.0079673, 0.07114747, 0.00798259, -0.02508499, -0.08553386, -0.01776437, -0.10464768, 0.11383347, -0.08475204, 0.11495435, -0.034387, 0.0, -0.06263724, 0.00600505, -0.01932981, -0.1252492, -0.15144022, -0.03379844, -0.09718493, -0.13689618, 0.01651256, -0.00777494, -0.01961858, -0.03400184, -0.04012769, -0.07302953, -0.05434298, -0.0321995, 0.03624677, -0.02331894, -0.0197673, -0.04467866, -0.01262048, -0.13500058, 0.08337743, -0.08206532, 0.18886274, 0.04129098, -0.07762937, -0.11633164, -0.00394868, -0.02217065, -0.09129512, -0.14959769, 0.2561367, -0.01010054, 0.0])
    # hi_corr_idxs = np.argwhere(np.abs(corrs) > 0.5)
    # print(hi_corr_idxs)
    # rois = pd.read_csv(f"{os.getcwd()}/7045/PET_2022-03-09_FTP/roi_info_suvr.csv").columns[hi_corr_idxs]
    # print(rois)


    # test_PCA(pca, [mri_dataset, test_dataset])

    # dataset_info(dataloader)

    # for batch_idx, (mri, suvr) in enumerate(dataloader):
    #     print(f"{batch_idx}. {mri.size()} => {suvr.size()}")
