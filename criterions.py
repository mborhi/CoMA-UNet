
import logging
import torch 
import torch.nn as nn

# import monai.losses
# from torch.nn.modules.loss import _Loss
import numpy as np

import pandas as pd
import data_util
import os
import VolumeDataset

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

class RoiRRMSE(nn.Module):
    """
    Relative Root Mean Squared Error weighted by ROI
    """

    def __init__(self, roi_weights, roi_indices, reduction="mean") -> None:
        super().__init__()

        self.roi_weights = roi_weights
        self.roi_indices = roi_indices
        self.reduction = reduction

    def forward(self, pred, gt, roi):
        """
         % generate the informative regions mask
        mask=zeros(size(i1)); %% define a new empty matrix 

        % informative indicies are (1001,1006,1007,1009,1015,1016,1030,1034,1033,1008,1025,1029,1031,1022,17,18,2001,2006,2007,2009,2015,2016,2030,2034,2033,2008,2025,2029,2031,2022,49,50,51,52,53,54; 

        mask(i1==1001|i1==1006|i1==1007|i1==1009|i1==1015|i1==1016|i1==1030|i1==1034|i1==1033|i1==1008|i1==1025|i1==1029|i1==1031|i1==1022|i1==17|i1==18|i1==2001|i1==2006|i1==2007|i1==2009|i1==2015|i1==2016|i1==2030|i1==2034|i1==2033|i1==2008|i1==2025|i1==2029|i1==2031|i1==2022|i1==49|i1==50|i1==51|i1==52|i1==53|i1==54)=1; %% assign all voxels of interest in the mask as 1
        """
        # mask = torch.zeros(roi.size(), device=roi.get_device())
        mask = torch.ones(roi.size())
        if roi.get_device() != -1:
            mask = mask.to(device=roi.get_device())
            
        for i, idx in enumerate(self.roi_indices):
            mask[roi == idx] = self.roi_weights[i]

        # Computed ROI weighted Relative Root Mean Squared Error
        num = torch.sum(mask * torch.square(gt - pred), dim=(-3, -2, -1))
        den = torch.sum(mask * torch.square(gt), dim=(-3, -2, -1))
        weighted_squared_err = num / den 
        wrrmse_loss = torch.sqrt(weighted_squared_err)

        if self.reduction == "mean":
            return torch.mean(wrrmse_loss)
        
        else:
            return torch.sum(wrrmse_loss)

        wmse = torch.sum(mask * torch.square(gt - pred), dim=(-3, -2, -1)) / torch.count_nonzero(mask, dim=(-3, -2, -1))
        # Since the mask has 0 values, it produces nan values in wmse
        # wmse = torch.nan_to_num(wmse, nan=0.)
        wrmse = torch.sqrt(wmse)

        if self.reduction == "mean":
            # mean over nonzero mask values
            return torch.mean(wrmse)
            # return torch.sum(wrmse) / torch.count_nonzero(mask)
        
        else: 
            return torch.sum(wrmse)

class RoiRSE(nn.Module):
    """
    Relative Squared Error weighted by ROI
    """

    def __init__(self, roi_weights, roi_indices, reduction="mean") -> None:
        super().__init__()

        self.roi_weights = roi_weights
        self.roi_indices = roi_indices
        self.reduction = reduction

    def __str__(self):
        return f"RoiRSE(\n" \
               f"  roi_weights={self.roi_weights}\n" \
               f"  roi_indices={self.roi_indices}\n" \
               f"  reduction={self.reduction}\n" \
               f")"

    def forward(self, pred, gt, roi):
        # mask = torch.zeros(roi.size(), device=roi.get_device())
        mask = torch.ones(roi.size(), device=roi.get_device())
        for i, idx in enumerate(self.roi_indices):
            mask[roi == idx] = self.roi_weights[i]
        
        # Compte ROI weighted Relative Squared Error 
        # gt_mean = torch.sum(mask * gt, dim=(-3, -2, -1)) / torch.count_nonzero(mask, dim=(-3, -2, -1))
        gt_mean = torch.mean(mask * gt, dim=(-3, -2, -1)) # weighted mean
        squared_error_num = torch.sum(mask * torch.square(gt - pred), dim=(-3, -2, -1))
        # squared_error_den = torch.sum(mask * torch.square(gt - gt_mean.view(-1, 1, 1, 1, 1)), dim=(-3, -2, -1))
        squared_error_den = torch.sum(torch.square(gt - gt_mean.view(-1, 1, 1, 1, 1)), dim=(-3, -2, -1))
        wrse = squared_error_num / squared_error_den

        if self.reduction == "mean":
            # mean over nonzero mask values
            return torch.mean(wrse)
            # return torch.sum(wrse) / torch.count_nonzero(mask)
        
        else: 
            return torch.sum(wrse)


class RoiMSE(nn.Module):

    def __init__(self, roi_weights, roi_indices, reduction="mean", scale_factor=360, voxel_wise=True) -> None:
        super().__init__()

        self.roi_weights = roi_weights
        self.roi_indices = roi_indices
        self.batch_reduction = reduction
        self.scale_factor = scale_factor
        self.voxel_wise = voxel_wise

        if voxel_wise:
            # voxel_weights = torch.zeros(128, 128, 128, device=roi_weights.device)
            voxel_weights = torch.ones(128, 128, 128, device=roi_weights.device)
            roi_mask = data_util.load_template()
            for i, idx in enumerate(self.roi_indices):
                voxel_weights[roi_mask == idx]  = self.roi_weights[i]

            norm_voxel_weights = voxel_weights / torch.norm(voxel_weights)
            nscaling_factor = 5. / torch.mean(norm_voxel_weights)
            self.voxel_weights = nscaling_factor * norm_voxel_weights
        else:
            self.voxel_weights = None

    def __str__(self):
        return f"RoiMSE(\n" \
               f"  (roi_indices, roi_weights)={list(zip(self.roi_indices, self.roi_weights))}\n" \
               f"  batch_reduciton={self.batch_reduction}\n" \
               f")"

    def calculate_new_weights(self, errors, with_update=False):
        new_weights = self.roi_weights * (1/2) * errors.to(device=self.roi_weights.device) # taken as percentage error
        new_weights = self.scale_factor * (new_weights / torch.norm(new_weights))
        if with_update:
            self.update_weights(new_weights)
        return new_weights
    
    def calculate_new_voxel_weights(self, errors, voxel_weights, with_update=False):
        new_weights = voxel_weights * (1 + errors.to(device=self.voxel_weights.device)) # taken as percentage error
        new_weights = new_weights / torch.norm(new_weights)
        scaling_factor = torch.mean(voxel_weights) / torch.mean(new_weights)
        new_weights *= scaling_factor
        if with_update:
            self.update_weights(new_weights)
        return new_weights

    def update_weights(self, weights):
        # self.roi_weights = weights # NOTE
        return

    def forward_with_weights_by_class(self, prediction, targets, weights):
        original_weights = self.roi_weights
        self.update_weights(weights)
        loss = self.forward(prediction, targets)
        self.update_weights(original_weights)
        return loss
    
    def forward(self, pred, gt, roi):
        mask = torch.ones(roi.size(), device=roi.get_device()) if self.voxel_wise else torch.zeros(roi.size(), device=roi.get_device()) 
        # mask = torch.ones(roi.size(), device=roi.get_device()) #if self.voxel_wise else torch.zeros(roi.size(), device=roi.get_device()) 
        for i, idx in enumerate(self.roi_indices):
            mask[roi == idx] = self.roi_weights[i]
        
        # Compte ROI weighted MSE 
        # loss = torch.square(mask * (pred - gt))
        if self.voxel_weights is not None:
            mask = self.voxel_weights.to(device=pred.device).unsqueeze(0).expand_as(pred)
            # mask[roi == 0] = 0
        # loss = mask * torch.mean(torch.square(pred - gt), dim=(-3, -2, -1)) # change: March 12
        # logging.info(f"mask norm:{mask.norm()}, mask:avg:{mask.mean()}, mask sum: {mask.sum()}")

        # loss = torch.mean(mask) * torch.mean(torch.square(pred - gt), dim=(-3, -2, -1)) # change: March 12

        l = torch.mean(torch.square(pred - gt), dim=(-3, -2, -1))
        loss = torch.zeros(l.size(), device=pred.device)
        for b in range(pred.size(0)):
            loss[b] = torch.mean(mask[b] * l[b])
            # loss[b] = l[b]

        logging.info(f"real loss norm: {torch.square(pred - gt).norm()}, mean: {torch.square(pred - gt).mean(dim=(-3, -2, -1))}, sum: {torch.square(pred - gt).sum()}")
        logging.info(f"loss norm: {loss.norm()}, mean: {loss.mean()}, sum: {loss.sum()}")
        # loss = mask * torch.mean(torch.abs(pred - gt), dim=(-3, -2, -1))

        if self.batch_reduction == "mean":
            # mean over nonzero mask values
            return torch.mean(loss)
        else :
            return loss


class WeightedCCCL(nn.Module):
    def __init__(self, weights):
        super(WeightedCCCL, self).__init__()
        self.weights = weights
    
    def forward(self, prediction, target):
        total_loss = 0
        for p in range(prediction.size(1)):
            x = prediction[:, p]#.unsqueeze(0)
            y = target[:, p]#.unsqueeze(0)
            xstd = torch.std(x)
            ystd = torch.std(y)
            xv = x - torch.mean(x)
            yv = y - torch.mean(y)
            rxy = torch.sum(xv * yv) / (torch.norm(xv) * torch.norm(yv))

            ccc = (2 * rxy * xstd * ystd) / (torch.var(x) + torch.var(y) + torch.square(torch.mean(x)- torch.mean(y)))
            # ccc = 2*torch.cov(torch.cat((x, y), dim=0)) / (x.var() + y.var() + (x.mean() - y.mean())**2)

            if torch.isnan(ccc):
                # ccc = torch.tensor(0, dtype=torch.float32, requires_grad=True)

                batch_cccl = self.weights[p] * (1 - (0 * x))
            else :
                batch_cccl = self.weights[p] * (1 - ccc)
            total_loss += batch_cccl

        return total_loss

class WeightedCC(nn.Module):
    def __init__(self, weights):
        super(WeightedCC, self).__init__()
        self.weights = weights
    
    def forward(self, prediction, target):
        total_loss = 0
        for p in range(prediction.size(1)):
            x = prediction[:, p]#.unsqueeze(0)
            y = target[:, p]

            vx = x - torch.mean(x)
            vy = y - torch.mean(y)

            cost = torch.sum(vx * vy) / (torch.norm(vx) * torch.norm(vy))

            total_loss = self.weights[p] * (1 - cost)
        
        return total_loss

class WeightedMSE(nn.Module):
    """
    For vector prediction of tau levels in ROIs
    """
    def __init__(self, weights):
        super(WeightedMSE, self).__init__()
        self.weights = weights

    def forward(self, prediction, targets):
        squared_errors = torch.square(prediction - targets)
        weighted_squared_errors = squared_errors * self.weights.unsqueeze(0)
        loss = torch.mean(weighted_squared_errors)
        return loss
    
class WeightedLoss(nn.Module):
    def __init__(self, weights):
        super(WeightedLoss, self).__init__()
        self.weights = weights

    def forward(self, prediction, targets):
        """
        Average of the weighted component sum of absolute differences (column sum for batched inputs)
        X, Y MxN matrix, w is M vector
        L(X, Y) = mean(weight_column * |X_column - Y_column|)
                = (1/M) Sum_m w_m(Sum_n |X_m,n - Y_m,n|)

        Essentially weighted average of L1Loss with reduction='sum'
        """
        loss = 0
        for i, weight in enumerate(self.weights):
            loss += weight * torch.sum(torch.abs(prediction[:, i] - targets[:, i]))
        loss /= len(self.weights)
        
        return loss

class TripletContrastiveLoss(nn.TripletMarginWithDistanceLoss):
    
    def __init__(self, temperature: float = 0.5, batch_size: int = -1, reduction="sum") -> None:
        super().__init__(temperature, batch_size, reduction)


class TruncatedCDS(nn.Module):
    """
    Contrastive Deep Supervsion, Zhang et. al. (2022). https://arxiv.org/abs/2207.05306
    """

    def __init__(self, contra_loss: nn.TripletMarginWithDistanceLoss, intermediate_weights) -> None:
        super(TruncatedCDS, self).__init__()
        self.contra_loss = contra_loss(margin=1)
        self.intermediate_weights = intermediate_weights
    
    def __str__(self):
        return f"TruncatedCDS(\n" \
               f"  contra_loss={self.contra_loss}\n" \
               f"  intermediate_weights={self.intermediate_weights}\n" \
               f")"

    def get_contra_loss(self, input, positive, negative):

        return self.contra_loss(input, positive, negative)

    def forward(self, intermediate_lst, repr_intermediate_lst):
        """"
        Parameters
        -----------
        intermediate_lst: list[Tensor]
            List of Tensor of length K, where K is the depth (number of feature extractors to downsampling)
            of the model

        repr_intermediate_lst: list[tuple[Tensor]]
            List of Tensor tuples, where in the ith tuple the first position is the ith-intermediate representation
            of a positive sample, and the second position is the ith-intermediate representation of a negative sample
        """
        contra_loss = torch.tensor([0], dtype=torch.float, device=intermediate_lst[0].device, requires_grad=True)
        for i, (pos_intermediate, neg_intermediate) in enumerate(repr_intermediate_lst) :
            intermediate = intermediate_lst[i]
            # pos_intermediate, neg_intermediate = repr_intermediate_lst[i]
            intermediate_repr_contra_loss = self.get_contra_loss(intermediate, pos_intermediate, neg_intermediate)
            logging.info(f"tCDS Intermediate Repr {i} Loss (unweighted): {intermediate_repr_contra_loss}")
            contra_loss = contra_loss + self.intermediate_weights[i] * intermediate_repr_contra_loss

        return contra_loss

class ManifoldSimilarity(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.proxies = None

    def forward(self, x):
        pass

class NPairLoss(nn.Module):
    
    def __init__(self, pos_quartile_template_paths, neg_quartile_template_paths) -> None:
        super().__init__()
        # NOTE maybe just directly load these in and pass the tensors as args
        self.pos_quartile_templates_paths = pos_quartile_template_paths # [self.load_(p) for p in pos_quartile_template_paths]
        self.neg_quartile_templates_paths = neg_quartile_template_paths # [self.load_(p) for p in neg_quartile_template_paths]
        self.pos_quartile_templates = pos_quartile_template_paths # [self.load_(p) for p in pos_quartile_template_paths]
        self.neg_quartile_templates = neg_quartile_template_paths # [self.load_(p) for p in neg_quartile_template_paths]
        # self.sim = lambda x, t : torch.exp(x.mT @ t)
        self.covar_lookup_df = pd.read_csv(f"{os.getcwd()}/scripts/ADNI_ID_ABETA_TAU_QUARTS.csv")
        self.similarity = nn.functional.cosine_similarity
        self.betas = [1.] * 10
    
    def get_npair_loss(self, anchor, abeta, quartile):
        # anchor dim = [batch, E]
        # quartile_template dim = [batch, E]
        # negative_templates dim = [batch, N-1, E]
        if abeta == 1:
            quartile_template = self.pos_quartile_templates[quartile-1] # 0-indexed
            neg_templates = self.pos_quartile_templates[:quartile-1] + \
                            self.pos_quartile_templates[quartile:] + \
                            self.neg_quartile_templates
        else :
            quartile_template = self.neg_quartile_templates[quartile-1]
            neg_templates = self.neg_quartile_templates[:quartile-1] + \
                            self.neg_quartile_templates[quartile:] + \
                            self.pos_quartile_templates
        
        neg_templates = torch.vstack(neg_templates)
        # neg_templates = np.vstack(neg_templates)
        # NOTE add different weighting for further away templates?
        if anchor.size(-1) != quartile_template.size(-1):
            return 0 
        pos_sim = self.similarity(anchor, quartile_template, dim=-1)
        numerator = torch.exp(pos_sim)
        neg_sim = self.similarity(anchor.unsqueeze(1), neg_templates, dim=-1)
        denominator = numerator + torch.sum(torch.exp(neg_sim), dim=-1)

        loss = -torch.log(numerator / denominator)

        return loss

    def lookup_quartile_from_path(self, path):
        if isinstance(path, tuple):
            path = path[0]
        sample_id = data_util.get_id_from_path(path)
        quart = self.covar_lookup_df[self.covar_lookup_df["ADNI_ID"] == sample_id]["quartile_lub"].values
        if len(quart) == 0:
            return -1 
        quart = quart[0]
        return quart, sample_id

    def load_templates(self, dataset_inst):
        _ = dataset_inst[0] # ensures that Dataset is properly initialized
        self.neg_quartile_templates = []
        for i, f in enumerate(self.neg_quartile_templates_paths):
            template_tensor = dataset_inst.load_volume_file(f)
            new_spacing = [2.0, 2.0, 2.0]
            downsampled_template = dataset_inst.resize_tensor(template_tensor, new_spacing)
            self.neg_quartile_templates.append(downsampled_template.flatten(1).detach().cpu())
            # self.neg_quartile_templates.append(template_tensor.flatten(1).detach().cpu())
        
        self.pos_quartile_templates = []
        for i, f in enumerate(self.pos_quartile_templates_paths):
            template_tensor = dataset_inst.load_volume_file(f)
            new_spacing = [2.0, 2.0, 2.0]
            downsampled_template = dataset_inst.resize_tensor(template_tensor, new_spacing)
            self.pos_quartile_templates.append(downsampled_template.flatten(1).detach().cpu())
            # self.pos_quartile_templates.append(template_tensor.flatten(1).detach().cpu())
    

    def forward(self, anchor_lst, abeta, path):
        quartile, sid = self.lookup_quartile_from_path(path)
        total_loss = 0
        for i, anchor in enumerate(anchor_lst):
            l = self.get_npair_loss(anchor, abeta, quartile)
            logging.info(f"DS on Decoder loss at {i}: {l}")
            total_loss += self.betas[i] * l
        
        return total_loss

class ClusterNPairLoss(nn.Module):

    def __init__(self, intermediate_weights, temp = 1.) -> None:
        super().__init__()
        self.intermediate_weights = intermediate_weights

        self.similarity = nn.functional.cosine_similarity
        self.temp = temp # TODO 
        # self.similarity = nn.CosineSimilarity(dim=-1)
    
    def get_npair(self, anchor, pos, negs):
        if not isinstance(negs, torch.Tensor):
            negs = torch.cat(negs, dim=0)

        # logging.info(f"anchor proj num unique: {torch.unique(anchor).size()}, mean: {torch.mean(anchor)}")
        # logging.info(f"pos proj num unique: {torch.unique(pos).size()}, mean: {torch.mean(pos)}")
        # logging.info(f"negs proj num unique: {torch.unique(negs, dim=1).size()}, mean: {torch.mean(negs, dim=1)}")

        pos_sim = self.similarity(anchor, pos, dim=-1)
        # pos_sim = self.similarity(anchor, pos)
        numerator = torch.exp(pos_sim/self.temp)
        neg_sim = self.similarity(anchor, negs, dim=-1)
        # neg_sim = self.similarity(anchor, negs)
        denominator = numerator + torch.sum(torch.exp(neg_sim/self.temp), dim=-1)
    
        loss = -torch.log(numerator / denominator)

        # diffs = 0
        # for i in range(int(negs.size(0))):
        #     diffs += torch.exp(self.similarity(anchor, negs[i]) - pos_sim)
        # loss = torch.log(1 + diffs)

        return loss

    def forward(self, intermediate_lst, repr_intermediate_lst):
        contra_loss = torch.tensor([1e-8], dtype=torch.float, device=intermediate_lst[0].device, requires_grad=True)
        contra_loss.retain_grad()
        # contra_loss = 0
        for i, (pos_intermediate, neg_intermediates) in enumerate(repr_intermediate_lst) :
            intermediate = intermediate_lst[i]
            # pos_intermediate, neg_intermediate = repr_intermediate_lst[i]
            intermediate_repr_contra_loss = self.get_npair(intermediate, pos_intermediate, neg_intermediates)
            logging.info(f"tCDS Intermediate Repr {i} Loss (unweighted): {intermediate_repr_contra_loss}")
            contra_loss = contra_loss + intermediate_repr_contra_loss
        
        return contra_loss


class GenerativeContrastiveLoss(nn.Module):

    def __init__(self, ds_contra_loss, gen_loss, pred_space_contra_loss, regulatory_weight, ds_regulatory_weight) -> None:
        """
        L =  L_Gen(X, Y) + lambda_1 * L_tCDS (X; g) + lambda_2 * L_Contra(X; h_k)

        Parameters
        ----------
        ds_contra_loss: ContrastiveDeepSupervisionLoss
            Applied to the intermediate extractions
            
        gen_loss: nn.Module
            The generative loss function applied to the final output
        
        pred_space_contra_loss: nn.Module
            The contrastive loss applied to the final output; applied in the latent space
            derived from the approximated target manifold
        
        regulatory_weight: float
            The weight to multiply the `pred_space_contra_loss` by to account for different
            scale of the loss components [lambda_2]
        
        ds_regulatory_weight: float 
            The regulatory term for the (truncated) Contrastive Deep Supervised Loss [lambda_1]
        """
        super().__init__()

        self.ds_contra_loss = ds_contra_loss
        self.gen_loss = gen_loss
        self.pred_space_contra_loss = pred_space_contra_loss
        self.reg_weight = regulatory_weight
        self.ds_reg_weight = ds_regulatory_weight
        self.gen_weight = 1.
        # uq
        # self.heteroscedastic_loss = HeteroscedasticLoss()

    def __str__(self):
        return f"GenerativeContrastiveLoss(\n" \
               f"  tCDS Loss (ds_contra_loss)={self.ds_contra_loss}\n" \
               f"  Generative Loss (gen_loss)={self.gen_loss}\n" \
               f"  Prediction Space Contrastive Loss (pred_space_contra_loss)={self.pred_space_contra_loss}\n" \
               f"  Pred. Space Contra. Loss Reg. Weight [lambda_2] (reg_weight)={self.reg_weight}\n" \
               f"  tCDS Loss Reg. Weight [lambda_1] (ds_reg_weight)={self.ds_reg_weight}\n" \
               f")"

    # def forward_heteroscedastic_loss(self, *args): return self.heteroscedastic_loss(*args)

    def get_pred_space_contra_loss(self, representations):
        # if isinstance(self.pred_space_contra_loss, NPairLoss):
        #     decoder_proj_lst, abeta = representations
        #     self.pred_space_contra_loss(decoder_proj_lst, abeta)
        # anchor, pos, neg = representations
        return self.pred_space_contra_loss(*representations)

    def get_ds_contra_loss(self, intermediate_extractions):
        # anchor_intermediate_reprs, contra_intermediate_reprs = intermediate_extractions
        # return self.ds_contra_loss(anchor_intermediate_reprs, contra_intermediate_reprs)
        return self.ds_contra_loss(*intermediate_extractions)

    def forward(self, prediction, target, roi, final_representations, intermediate_extractions):
        """
        L = [ L_gen (X, Y) + a * L_PSContra (X; X~) ] + b * L_CDS (Z; Z~)

        where:
            X is the final prediction of the model,
            Y is the ground truth for the final prediction
            X~ is the set of generated for positive / negative samples 
            Z is the set of intermediate representations of X
            Z~ is the set of intermediate representations for X~
            a, b are regulartory weights

        Note that the Contrastive Deep Supervision Loss is truncated, and only caluclates
        based on the intermediate representations
        """ 
        gen_loss = self.gen_loss(prediction, target, roi) 
        reduced_gen_loss = torch.sum(gen_loss) if self.gen_loss.batch_reduction is None else gen_loss
        pred_space_contra_loss = self.get_pred_space_contra_loss(final_representations)
        total_pred_space_contra_loss = self.reg_weight * pred_space_contra_loss
        ds_contra_loss = self.get_ds_contra_loss(intermediate_extractions)
        total_ds_contra_loss = self.ds_reg_weight * ds_contra_loss

        if total_pred_space_contra_loss.device != gen_loss.device:
            total_pred_space_contra_loss = total_pred_space_contra_loss.to(device=gen_loss.device)

        total_loss = self.gen_weight * reduced_gen_loss + total_pred_space_contra_loss + total_ds_contra_loss

        logging.info(f"\tLoss from gen crit: {gen_loss}\t\
                Loss from pred space (unweighted): {pred_space_contra_loss}\t\
                Loss from tCDS (unweighted): {ds_contra_loss}")

        return total_loss, gen_loss, total_pred_space_contra_loss, total_ds_contra_loss
        # return total_gen_loss + total_ds_contra_loss, gen_loss, 0, total_ds_contra_loss

### Rank-N-Contrast Loss https://github.com/kaiwenzha/Rank-N-Contrast/tree/main 
class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        else:
            raise ValueError(self.similarity_type)


class RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        # features: [bs, 2, feat_dim]
        # labels: [bs, label_dim]

        if len(features.shape) == 2 * len(labels.shape):

            features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
            labels = labels.repeat(2, 1)  # [2bs, label_dim]

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss

class HeteroscedasticLoss(nn.Module):
    def forward(self, q, q_hat, sigma2):
        return torch.mean((q - q_hat) ** 2 / (2 * sigma2) + torch.log(sigma2))

        # # Initialize loss and optimizer
        # criterion = HeteroscedasticLoss()
        # optimizer = optim.Adam(model.parameters(), lr=0.001)

        # # Dummy training loop
        # q = torch.randn(5)  # Ground truth q
        # for epoch in range(100):
        #     optimizer.zero_grad()
        #     sigma2, _ = model(x, q_hat)
        #     loss = criterion(q, q_hat, sigma2)
        #     loss.backward()
        #     optimizer.step()
        #     if epoch % 10 == 0:
        #         print(f"Epoch {epoch}, Loss: {loss.item()}")


if __name__ == "__main__":
    base_path = f"{os.getcwd()}"
    splits_dir = os.path.join(base_path, "training_folds", "outlier_removed_splits")
    cuda_id = 0
    k = 0
    train_dataset, test_dataset = data_util.load_split_datasets(splits_dir, VolumeDataset, k+1, cuda_id, contra=True, template=True, resize=True)
    
    decoder_loss = NPairLoss(POS_TEMPLATE_PATHS, NEG_TEMPLATE_PATHS)
    decoder_loss.load_templates(train_dataset)

    print(f"Means of the down sampled templates:")
    print(f"pos: {torch.stack(decoder_loss.pos_quartile_templates).size()}")
    print(f"neg: {torch.stack(decoder_loss.neg_quartile_templates).size()}")


    
