import os
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import sys
import pandas as pd
import seaborn as sns
from torchinfo import summary

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from metaSR_utils import d_vector_dict_lbls, extract_label, read_MFB,\
    read_MFB_train, read_feats2
from Stg2_models import background_resnet, metaGenerator, metaGenerator_test
from Stg2_losses import Prototypical

from pipeline_utilities import log_print

n_epochs = 180 
# n_epochs = 30 

n_shot = 3
n_query = 2
aug_percent = 0.8  # Percentage of batch to augment with SpecAugment

patience_early_stop = 30  # Early stopping patience

# Learning rate configuration
new_lr = 1e-4
layer4_lr_ratio = 0.01  # Layer4 will use layer4_lr_ratio * new_lr
wd = 1e-4

cp_num=100
# Episode setting
n_shot_test=3
n_query_test=2
nb_class_test=0

max_iter_test = 500  # number of episodes to run

# Test setting
enroll_length=400
test_length=100

dataset_name = 'bal_all_18K'  # dataset name
params_name = f'Ffc1bnb4-1fc-s{n_shot}q{n_query}-rms_aol_lossSepW4'  # parameters name for logging


root_path = Path.home().joinpath('Dropbox','DATASETS_AUDIO')                                          # recommend SSD
TRAIN_FEAT_LUIS = root_path / f'Dvectors/{dataset_name}/input_feats'

TEST_DATA_FOLDER = root_path / 'Dvectors/wavs_test_pairs'
TEST_FEAT_AOLME = TEST_DATA_FOLDER / 'input_feats_groups'
TEST_WAV_AOLME = TEST_DATA_FOLDER / 'input_wavs_groups'
veri_test_dir = TEST_FEAT_AOLME / 'test_pairs_groups_aolme.txt'

# TEST_FEAT_AOLME = TEST_DATA_FOLDER / 'tts3_fix_feats'
# TEST_WAV_AOLME = TEST_DATA_FOLDER / 'tts3_fix_input_wavs'

# TEST_DATA_FOLDER = root_path / 'Dvectors/tts3_test_pairs'
# TEST_FEAT_AOLME = TEST_DATA_FOLDER / 'tts3_var_input_feats'
# TEST_WAV_AOLME = TEST_DATA_FOLDER / 'tts3_var_input_wavs'
# veri_test_dir = TEST_FEAT_AOLME / 'test_pairs_tts3.txt'

SHORT_SIZE = 100   # 100ms == 1 seconds

USE_NORM = True  # Normalize sliced input
USE_SCALE = False

loss_type = 'prototypical'
use_GC = True
use_cuda = True
use_variable = True
use_checkpoint = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

seed = 100
lbl_th = 95
run_id = f'{dataset_name}_{params_name}'  # run id for logging
# run_id = f'voxceleb2_{params_name}'  # run id for logging
log_dir = 'saved_model/' + run_id

rms_prop_flag = True  # Use RMSprop optimizer

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_path = log_dir + f'/{run_id}_log.txt'
results_query_path = log_dir + f'/{run_id}_extraDetails_query.txt'

pretrained_path = 'saved_model/checkpoint_100_original.pth'  # path to pre-trained model

norm_flag = True
samples_flag = False


def verify_bn_stats_unchanged(model, log_path):
    """
    Verify that BatchNorm running statistics haven't changed during training.
    """
    if not hasattr(model, '_initial_bn_stats'):
        log_print("Warning: No initial BatchNorm statistics found. Cannot verify.", log_path=log_path)
        return False
    
    log_print("\n" + "*"*80, log_path=log_path)
    log_print("VERIFYING BATCHNORM RUNNING STATISTICS HAVEN'T CHANGED", log_path=log_path)
    log_print("*"*80, log_path=log_path)
    
    all_unchanged = True
    changed_modules = []
    unchanged_modules = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)) and name in model._initial_bn_stats:
            initial_stats = model._initial_bn_stats[name]
            
            # Check running_mean
            if torch.allclose(initial_stats['running_mean'], module.running_mean, atol=1e-8, rtol=1e-8):
                mean_status = "✓ UNCHANGED"
            else:
                mean_status = "✗ CHANGED"
                all_unchanged = False
                changed_modules.append(f"{name}.running_mean")
            
            # Check running_var
            if torch.allclose(initial_stats['running_var'], module.running_var, atol=1e-8, rtol=1e-8):
                var_status = "✓ UNCHANGED"
            else:
                var_status = "✗ CHANGED"
                all_unchanged = False
                changed_modules.append(f"{name}.running_var")
            
            # Check num_batches_tracked
            if initial_stats['num_batches_tracked'] == module.num_batches_tracked:
                track_status = "✓ UNCHANGED"
            else:
                track_status = "✗ CHANGED"
                all_unchanged = False
                changed_modules.append(f"{name}.num_batches_tracked")
            
            log_print(f"{name:<40} | mean: {mean_status} | var: {var_status} | tracked: {track_status}", log_path=log_path)
            
            if mean_status == "✓ UNCHANGED" and var_status == "✓ UNCHANGED" and track_status == "✓ UNCHANGED":
                unchanged_modules.append(name)
    
    log_print("-" * 80, log_path=log_path)
    log_print(f"SUMMARY:", log_path=log_path)
    log_print(f"  Unchanged BatchNorm modules: {len(unchanged_modules)}", log_path=log_path)
    log_print(f"  Changed BatchNorm statistics: {len(changed_modules)}", log_path=log_path)
    
    if all_unchanged:
        log_print("✓ SUCCESS: All BatchNorm running statistics remain unchanged!", log_path=log_path)
    else:
        log_print("✗ WARNING: Some BatchNorm running statistics have changed!", log_path=log_path)
        log_print("  Statistics that changed:", log_path=log_path)
        for stat_name in changed_modules:
            log_print(f"    - {stat_name}", log_path=log_path)
    
    log_print("="*80, log_path=log_path)
    
    return all_unchanged

def verify_frozen_parameters(model, log_path):
    """
    Verify that frozen parameters haven't changed during training.
    
    Args:
        model: The model to verify
        log_path: Path to log file for output
    
    Returns:
        bool: True if all frozen parameters are unchanged, False otherwise
    """
    if not hasattr(model, '_frozen_params_snapshot'):
        log_print("Warning: No frozen parameter snapshots found. Cannot verify parameter freeze.", log_path=log_path)
        return False
    
    log_print("\n" + "*"*80, log_path=log_path)
    log_print("VERIFYING FROZEN PARAMETERS HAVEN'T CHANGED", log_path=log_path)
    log_print("*"*80, log_path=log_path)
    
    all_frozen = True
    changed_params = []
    unchanged_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad and name in model._frozen_params_snapshot:
            original_param = model._frozen_params_snapshot[name]
            current_param = param.data
            
            # Check if parameters are equal (within floating point tolerance)
            if torch.allclose(original_param, current_param, atol=1e-8, rtol=1e-8):
                unchanged_params.append(name)
                status = "✓ UNCHANGED"
            else:
                changed_params.append(name)
                all_frozen = False
                status = "✗ CHANGED"
                
                # Calculate difference statistics
                diff = torch.abs(original_param - current_param)
                max_diff = torch.max(diff).item()
                mean_diff = torch.mean(diff).item()
                
                log_print(f"{name:<60} | {status} | Max diff: {max_diff:.2e} | Mean diff: {mean_diff:.2e}", log_path=log_path)
            
            if status == "✓ UNCHANGED":
                log_print(f"{name:<60} | {status}", log_path=log_path)
    
    log_print("-" * 80, log_path=log_path)
    log_print(f"SUMMARY:", log_path=log_path)
    log_print(f"  Unchanged frozen parameters: {len(unchanged_params)}", log_path=log_path)
    log_print(f"  Changed frozen parameters: {len(changed_params)}", log_path=log_path)
    
    if all_frozen:
        log_print("✓ SUCCESS: All frozen parameters remain unchanged!", log_path=log_path)
    else:
        log_print("✗ WARNING: Some frozen parameters have changed!", log_path=log_path)
        log_print("  Parameters that changed:", log_path=log_path)
        for param_name in changed_params:
            log_print(f"    - {param_name}", log_path=log_path)
    
    log_print("="*80, log_path=log_path)
    
    return all_frozen


def generate_extra_details(df, output_folder):

    output_folder = Path(output_folder)
    
    # Extract speaker ID from filename
    df['speaker_id'] = df['filename'].apply(lambda x: x.split('_')[1])

    # Compute total predictions and accuracy (%)
    df['total'] = df['misslabeled_count'] + df['correct_count']
    df['accuracy'] = 100 * df['correct_count'] / df['total']

    # --- Speaker-level stats ---
    speaker_stats = df.groupby('speaker_id').agg(
        total_files=('filename', 'count'),
        total_correct=('correct_count', 'sum'),
        total_mislabeled=('misslabeled_count', 'sum'),
        total_preds=('total', 'sum')
    )
    speaker_stats['overall_accuracy'] = 100 * speaker_stats['total_correct'] / speaker_stats['total_preds']
    speaker_stats = speaker_stats.sort_values('overall_accuracy', ascending=False)

    # Round values
    speaker_stats_rounded = speaker_stats.copy()
    speaker_stats_rounded['overall_accuracy'] = speaker_stats_rounded['overall_accuracy'].round(1)
    speaker_stats_rounded[['total_correct', 'total_mislabeled', 'total_preds']] = speaker_stats_rounded[['total_correct', 'total_mislabeled', 'total_preds']].round(2)

    # --- Save speaker stats ---
    speaker_stats_path = output_folder / "speaker_stats_summary.tsv"
    speaker_stats_rounded.to_csv(speaker_stats_path, sep='\t')

    # --- Plot 1: Bar plot of overall accuracy ---


    plt.figure(figsize=(10, 6))

    # Add the number of filesper speaker to the speaker name in X-axis. Trim name to first 6 characters for better readability
    speaker_labels = []
    for speaker, row in speaker_stats_rounded.iterrows():
        if len(speaker) > 6:
            # If speaker ID is longer than 5 characters, trim it
            trimmed_speaker = speaker[:5]  # Trim speaker ID to first 6 characters
        else:
            trimmed_speaker = speaker

        speaker_labels.append(f"{trimmed_speaker}({int(row['total_files'])})")

    speaker_stats_rounded.index = speaker_labels

    sns.barplot(x=speaker_stats_rounded.index, y=speaker_stats_rounded['overall_accuracy'], palette='viridis')
    plt.ylabel("Overall Accuracy (%)")
    plt.xlabel("Speaker ID")
    plt.title("Speaker Recognition Accuracy per Speaker")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.axhline(y=50, color='r', linestyle='--', label='50% Threshold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_folder / "accuracy_per_speaker_barplot.png")
    plt.close()

    # Save full file-level info (rounded)
    df_export = df.copy()
    df_export['accuracy'] = df_export['accuracy'].round(1)
    df_export[['correct_count', 'misslabeled_count', 'total']] = df_export[['correct_count', 'misslabeled_count', 'total']].round(2)

    # Sort by accuracy
    df_export = df_export.sort_values('accuracy', ascending=False)

    df_export_path = output_folder / "full_accuracy_per_file.tsv"
    df_export.to_csv(df_export_path, sep='\t', index=False)

def evaluation(test_generator, model, use_cuda, n_shot_test, n_query_test, nb_class_test):

    total_acc = []
    ans_episode, accum_samples_epi = 0, 0
    log_interval = 100
    total_idx = 0
    misslabeled_files = []
    correct_files = []
    episodes_accuracies_list = []

    # switch to test mode
    model.eval()
    debug_idx = 0
    debug_total = 0


    # Lists to store info for printing
    support_len_list = []
    query_len_list = []
    correct_query_list = []



    with torch.no_grad():
        # for batch_idx, (data) in enumerate(test_loader):
        for t, (data) in test_generator:
            inputs, targets_g, filenames_batch = data  # target size:(batch size), input size:(batch size, 1, n_filter, T)
            support, query = inputs

            # Separate support and query filenames
            support_filenames = filenames_batch[:n_shot_test * nb_class_test]
            query_filenames = filenames_batch[n_shot_test * nb_class_test:]

            len_supp = support.size(0)  # number of support samples
            len_query = query.size(0)  # number of query samples
            debug_total += len_supp + len_query
            # log_print(f'{debug_idx}, len_supp: {len_supp}, len_query: {len_query}| Accumulated: {debug_total}', log_path=log_path)
            support_len_list.append(len_supp)
            query_len_list.append(len_query)
            debug_idx += 1

            #normalize sliced input
            if USE_NORM:
                support = support - torch.mean(support, dim=3, keepdim=True)
                query = query - torch.mean(query, dim=3, keepdim=True)
            n_samples = query.size(0)  # batch size

            if use_cuda:
                support = support.cuda(non_blocking=True)
                query = query.cuda(non_blocking=True)

            targets_e = tuple([i for i in range(nb_class_test)]) * (n_query_test)
            targets_e = torch.tensor(targets_e, dtype=torch.long).cuda()

            support = model(support)  # out size:(n_support * n_class, dim_embed)
            query = model(query)      # out size:(n_query   * n_class, dim_embed)

            support = support.reshape(n_shot_test, nb_class_test, -1)
            prototype = support.mean(dim=0)
            angle_e = F.linear(query, F.normalize(prototype))

            # calculate accuracy of predictions in the current episode
            temp_ans = (torch.max(angle_e, 1)[1].long().view(targets_e.size()) == targets_e).sum().item()
            total_acc.append(temp_ans/angle_e.size(0) * 100)

            log_print(f'\n\tBatch {t+1}/{max_iter_test}', log_path=results_query_path, print=False)

            # Calculate again the accuracy for each query sample and log it with the corresponding filenames
            query_preds = torch.max(angle_e, 1)[1].long().view(targets_e.size())
            for i in range(len(query_preds)):
                pred_label = query_preds[i].item()
                true_label = targets_e[i].item()
                query_filename = query_filenames[i]

                log_print(f'Query {i+1}/{len(query_preds)} Query: {query_filename}, '
                          f'Predicted: {pred_label}, True: {true_label}', log_path=results_query_path, print=False)
                
                if pred_label != true_label:
                    misslabeled_files.append(query_filename)
                
                if pred_label == true_label:
                    correct_files.append(query_filename)

            log_print(f'\tSupport filenames: {support_filenames}', log_path=results_query_path, print=False)

            ans_episode += temp_ans
            accum_samples_epi += n_samples
            acc_episode = 100. * ans_episode / accum_samples_epi

            # if t % log_interval == 0:
            stds = np.std(total_acc, axis=0)
            ci95 = 1.96 * stds / np.sqrt(len(total_acc))
            current_episode_accuracy = temp_ans / n_samples * 100 
            episodes_accuracies_list.append(current_episode_accuracy)
            correct_query_list.append(temp_ans)

            total_idx = total_idx + 1
    
    # Calculate the final accuracy
    final_accuracy = np.mean(episodes_accuracies_list)
    log_print(f'\n\nFinal Overall Accuracy {n_shot_test}-shot = {final_accuracy:.2f}%', log_path=log_path)

    log_print(f'Avg correct query samples: {np.mean(correct_query_list):.2f} | Avg query length {np.mean(query_len_list):.2f}', log_path=log_path)
    log_print(f'Accumulated correct query: {ans_episode} | Accumulated query samples: {accum_samples_epi}\n\n', log_path=log_path)

    
    # Create a dictionary with misslabeled files and the count of occurrences
    misslabeled_count = {}
    for file in misslabeled_files:
        if file in misslabeled_count:
            misslabeled_count[file] += 1
        else:
            misslabeled_count[file] = 1
    
    # Create a dictionary with correct files and the count of occurrences
    correct_count = {}
    for file in correct_files:
        if file in correct_count:
            correct_count[file] += 1
        else:
            correct_count[file] = 1

    # Merge the dictionaries into a dataframe with 3 columns: filename, misslabeled_count, correct_count
    misslabeled_df = pd.DataFrame(list(misslabeled_count.items()), columns=['filename', 'misslabeled_count'])
    correct_df = pd.DataFrame(list(correct_count.items()), columns=['filename', 'correct_count'])
    results_df = pd.merge(misslabeled_df, correct_df, on='filename', how='outer').fillna(0)
    results_df['misslabeled_count'] = results_df['misslabeled_count'].astype(int)
    results_df['correct_count'] = results_df['correct_count'].astype(int)

    generate_extra_details(results_df, log_dir)




def get_eer(score_list, label_list):
    fpr, tpr, threshold = roc_curve(label_list, score_list, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    intersection = abs(1 - tpr - fpr)
    DCF2 = 100 * (0.01 * (1 - tpr) + 0.99 * fpr)
    DCF3 = 1000 * (0.001 * (1 - tpr) + 0.999 * fpr)
    log_print("Epoch=%d  EER= %.2f  Thres= %0.5f  DCF0.01= %.3f  DCF0.001= %.3f" % (
    cp_num, 100 * fpr[np.argmin(intersection)], eer_threshold, np.min(DCF2), np.min(DCF3)), log_path=log_path)


    return eer, eer_threshold


def perform_verification(veri_test_dir, dict_embeddings):
    # Perform speaker verification using veri_test.txt
    f = open(veri_test_dir)
    score_list = []
    label_list = []
    num = 0

    while True:
        line = f.readline()
        if not line: break

        label = int(line.split(" ")[0])
        enroll_filename = line.split(" ")[1]
        test_filename = line.split(" ")[2].replace("\n", "")

        # Remove the extension from the filename
        enroll_filename = enroll_filename.split('.')[0]
        test_filename = test_filename.split('.')[0]

        with torch.no_grad():
            # Get embeddings from dictionary
            enroll_speaker_id = extract_label(Path(enroll_filename), samples_flag=samples_flag)
            enroll_data_list = dict_embeddings[enroll_speaker_id]
            # Find the enroll embedding in the list of tuples (feature, filename)
            enroll_embedding = None
            for feat, filename in enroll_data_list:
                if filename.stem == enroll_filename:
                    enroll_embedding = feat
                    break

            test_speaker_id = extract_label(Path(test_filename), samples_flag=samples_flag)
            test_data_list = dict_embeddings[test_speaker_id]
            # Find the enroll embedding in the list of tuples (feature, filename)
            test_embedding = None
            for feat, filename in test_data_list:
                if filename.stem == test_filename:
                    test_embedding = feat
                    break
            
            # Verify if embeddings were found
            if enroll_embedding is None or test_embedding is None:
                sys.exit(f"Warning: Embedding not found for {enroll_filename} or {test_filename}. Skipping this pair.")
            

            score = F.cosine_similarity(enroll_embedding, test_embedding)
            score = score.data.cpu().numpy()[0]
            del enroll_embedding
            del test_embedding

        score_list.append(score)
        label_list.append(label)
        num += 1
        # log_print("%d) Score:%0.4f, Label:%s" % (num, score, bool(label)), log_path=log_path)

    f.close()
    eer, eer_threshold = get_eer(score_list, label_list)
    return eer


class SpecAugment(nn.Module):
    """
    SpecAugment implementation for PyTorch.
    
    Args:
        freq_mask_param (int): Maximum frequency mask size
        time_mask_param (int): Maximum time mask size  
        num_freq_masks (int): Number of frequency masks to apply
        num_time_masks (int): Number of time masks to apply
        p (float): Probability of applying SpecAugment to selected samples
        time_warp_param (int): Maximum time warp parameter (0 to disable)
        mask_value (float): Value to use for masking
        batch_ratio (float): Percentage of batch samples to randomly select for augmentation (0.0-1.0)
    """
    
    def __init__(self, freq_mask_param=27, time_mask_param=100, 
                 num_freq_masks=1, num_time_masks=1,
                 mask_value=0.0, batch_ratio=1.0):
        super(SpecAugment, self).__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.mask_value = mask_value
        self.batch_ratio = batch_ratio  # Percentage of batch to augment
        
    def forward(self, x):
        """
        Apply SpecAugment to input spectrograms.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, freq_bins, time_steps)
                             or (batch_size, freq_bins, time_steps)
        
        Returns:
            torch.Tensor: Augmented spectrograms with same shape as input
        """
        if not self.training:
            return x
            
        # (batch_size, channels, freq_bins, time_steps)
        batch_size, channels, freq_bins, time_steps = x.shape         

        # Apply augmentation to each sample in the batch
        augmented = x.clone()
        
        # Randomly select which samples in the batch to augment
        num_samples_to_augment = int(batch_size * self.batch_ratio)
        if num_samples_to_augment > 0:
            # Randomly select indices
            selected_indices = random.sample(range(batch_size), num_samples_to_augment)
        else:
            selected_indices = []

        for i in selected_indices:
            # # Check individual sample probability
            # if random.random() > self.p:
            #     continue
                
            # Frequency masking
            for _ in range(self.num_freq_masks):
                augmented[i, 0] = self._freq_mask(augmented[i, 0])
            
            # Time masking
            for _ in range(self.num_time_masks):
                augmented[i, 0] = self._time_mask(augmented[i, 0])
        

        return augmented
    
    def _freq_mask(self, spec):
        """Apply frequency masking to a single spectrogram."""
        freq_bins, time_steps = spec.shape
        
        if freq_bins == 0:
            return spec
        
        # Random mask size
        mask_size = random.randint(0, min(self.freq_mask_param, freq_bins))
        
        if mask_size == 0:
            return spec
        
        # Random mask position
        mask_start = random.randint(0, freq_bins - mask_size)
        mask_end = mask_start + mask_size
        
        # Apply mask
        masked_spec = spec.clone()
        masked_spec[mask_start:mask_end, :] = self.mask_value

        return masked_spec
    
    def _time_mask(self, spec):
        """Apply time masking to a single spectrogram."""
        freq_bins, time_steps = spec.shape
        
        if time_steps == 0:
            return spec
        
        # Random mask size
        mask_size = random.randint(0, min(self.time_mask_param, time_steps))
        
        if mask_size == 0:
            return spec
        
        # Random mask position
        mask_start = random.randint(0, time_steps - mask_size)
        mask_end = mask_start + mask_size
        
        # Apply mask
        masked_spec = spec.clone()
        masked_spec[:, mask_start:mask_end] = self.mask_value

        return masked_spec


# Standalone SpecAugment function for easy integration
def apply_specaugment(spectrograms, freq_mask_param=5, time_mask_param=40,
                     num_freq_masks=2, num_time_masks=2,
                     mask_value=0.0, batch_ratio=1.0):
    """
    Standalone function to apply SpecAugment to spectrograms.
    
    Args:
        spectrograms (torch.Tensor): Input spectrograms
        batch_ratio (float): Percentage of batch to augment (0.0-1.0)
        Other args: SpecAugment parameters
    
    Returns:
        torch.Tensor: Augmented spectrograms
    """
    augmenter = SpecAugment(
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
        num_freq_masks=num_freq_masks,
        num_time_masks=num_time_masks,
        mask_value=mask_value,
        batch_ratio=batch_ratio
    )
    
    # Set to training mode to enable augmentation
    augmenter.train()
    
    with torch.no_grad():
        return augmenter(spectrograms)


def plot_loss_accuracy(train_losses, train_losses_e, train_losses_g, train_acc_e, train_acc_g, log_dir, run_id):
    # Plot episode loss and accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_e, label='Episode Loss (loss_e)')
    plt.plot(train_losses_g, label='Global Loss (loss_g)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_e, label='Episode Accuracy (acc_e)')
    plt.plot(train_acc_g, label='Global Accuracy (acc_g)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{run_id}_plot.png'))
    plt.close()

def train(train_generator, model, objective, optimizer, n_episode, log_dir, scheduler, aug_percent=0.8, early_stop_patience=5):

    # switch to train mode
    model.train()
    
    # Keep frozen BatchNorm layers in eval mode
    def keep_frozen_bn_in_eval(module):
        """Keep frozen BatchNorm layers in eval mode during training"""
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()
        for child in module.children():
            keep_frozen_bn_in_eval(child)
    
    keep_frozen_bn_in_eval(model)

    early_stop = False
    best_loss = float('inf')
    epochs_no_improve = 0


    # for batch_idx, (data) in enumerate(train_loader):
    log_interval = int(n_episode / 2)
    avg_train_losses = []
    avg_train_losses_e = []
    avg_train_losses_g = []
    avg_train_acc_e = []
    avg_train_acc_g = []
    for t, (data) in train_generator:
        epoch = int(t / n_episode)


        if t % n_episode == 0:
            losses = AverageMeter()
            losses_e = AverageMeter()
            losses_g = AverageMeter()
            accuracy_e = AverageMeter()
            accuracy_g = AverageMeter()

        inputs, targets_g = data  # target size:(batch size), input size:(batch size, 1, n_filter, T)

        if aug_percent > 0:
            inputs = apply_specaugment(inputs, batch_ratio=aug_percent)

        targets_e = tuple([i for i in range(nb_class_train)]) * (n_query)
        targets_e = torch.tensor(targets_e, dtype=torch.long).cuda(non_blocking=True)
        support, query = split_support_query(inputs)

        loss, loss_e, loss_g, acc_e, acc_g =\
            objective(support, query, targets_g, targets_e, model, use_GC)
        losses.update(loss.item(), query.size(0))
        losses_e.update(loss_e.item(), query.size(0))
        losses_g.update(loss_g.item(), inputs.size(0))
        accuracy_e.update(acc_e * 100, query.size(0))
        accuracy_g.update(acc_g * 100, inputs.size(0))

        # episode number in epoch
        ith_episode = t % n_episode

        # Print t and ith_episode
        # print(f'Epoch: {epoch}, Episode: {t % n_episode}/{n_episode}, Ith Episode: {ith_episode}')

        if ith_episode % log_interval == 0:
            print(
                'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\t'
                'Loss {loss.avg:.4f} (loss_e: {loss_e.avg:.4f} / loss_g: {loss_g.avg:.4f})\t'
                'Acc e / g {acc_episode.avg:.4f} / {acc_global.avg:.4f}'.format(
                epoch, ith_episode, n_episode, 100. * ith_episode / n_episode,
                loss=losses, loss_e=losses_e, loss_g=losses_g, acc_episode=accuracy_e, acc_global=accuracy_g))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % n_episode == 0 and t != 0: #epoch interval
            scheduler.step(losses.avg, epoch)

            # calculate average loss and accuracy over an epoch
            avg_train_losses.append(losses.avg)
            avg_train_losses_e.append(losses_e.avg)
            avg_train_losses_g.append(losses_g.avg)
            avg_train_acc_e.append(accuracy_e.avg)
            avg_train_acc_g.append(accuracy_g.avg)

            # Verify frozen parameters every 5 epochs
            # if epoch % 5 == 0:
            # verify_frozen_parameters(model, os.path.join(log_dir, f'{run_id}_log.txt'))

            # Early stopping check
            if losses.avg < best_loss:
                best_loss = losses.avg
                epochs_no_improve = 0
                # Save best model
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                           '{}/bestCP_{}.pth'.format(log_dir, str(epoch).zfill(3)))
            else:
                epochs_no_improve += 1
                print(f'No improvement in loss for {epochs_no_improve} epoch(s).')
                if epochs_no_improve >= early_stop_patience:
                    print(f'Early stopping at epoch {epoch+1}. Best loss: {best_loss:.4f}')
                    early_stop = True
                    break

    # Save last checkpoint
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                '{}/lastCP_{}.pth'.format(log_dir, str(epoch).zfill(3)))

    # find position of lowest training loss
    minposs = avg_train_losses.index(min(avg_train_losses)) + 1
    print('Lowest training loss at epoch %d' % minposs)

    # Plot and save loss/accuracy
    plot_loss_accuracy(avg_train_losses, avg_train_losses_e, avg_train_losses_g, avg_train_acc_e, avg_train_acc_g, log_dir, run_id)

    last_model_path = '{}/lastCP_{}.pth'.format(log_dir, str(epoch).zfill(3))

    # Verify that frozen parameters haven't changed during training
    verify_frozen_parameters(model, os.path.join(log_dir, f'{run_id}_log.txt'))
    
    # Verify that BatchNorm running statistics haven't changed during training
    verify_bn_stats_unchanged(model, os.path.join(log_dir, f'{run_id}_log.txt'))

    return early_stop, last_model_path

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_model(pretrained_path, n_classes, log_path, print_model_architecture=False):
    log_print('=> loading checkpoint', log_path=log_path)

    # Initialize the model
    model = background_resnet(num_classes=n_classes, inter_size=256, backbone='resnet34')
    model = model.cuda()

    # Capture initial parameter snapshots for verification
    initial_params = {}
    for name, param in model.named_parameters():
        initial_params[name] = param.data.clone()

    # Load the checkpoint
    checkpoint = torch.load(pretrained_path)

    # Extract the nested state_dict if needed
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Create a clean copy of the state dict
    filtered_state_dict = {}
    
    # Get model's state dict for comparison
    model_dict = model.state_dict()
    
    # Only include parameters with matching shapes
    for k, v in state_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered_state_dict[k] = v
        else:
            if k in model_dict:
                print(f"Skipping {k}: checkpoint shape {v.shape} vs model shape {model_dict[k].shape}")
            else:
                print(f"Skipping {k}: not found in model")
    
    # Load the filtered state dict
    model.load_state_dict(filtered_state_dict, strict=False)
    print("Successfully loaded filtered state dict")
    
    # # Debug: Print BatchNorm running statistics after loading
    # print("\n" + "="*80)
    # print("BATCHNORM RUNNING STATISTICS AFTER LOADING PRETRAINED MODEL")
    # print("="*80)
    # for name, module in model.named_modules():
    #     if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
    #         print(f"{name}:")
    #         print(f"  running_mean: {module.running_mean[:5]}...")  # Show first 5 values
    #         print(f"  running_var:  {module.running_var[:5]}...")   # Show first 5 values
    #         print(f"  num_batches_tracked: {module.num_batches_tracked}")
    # print("="*80)

    # Freeze specified layers and their BatchNorm statistics
    def freeze_bn_stats(module):
        """Recursively freeze BatchNorm running statistics"""
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.eval()  # Keep in eval mode to prevent running stats updates
            # Optionally, you can also freeze the parameters
            for param in module.parameters():
                param.requires_grad = False
        for child in module.children():
            freeze_bn_stats(child)
    
    for name, module in [
        ('conv1', model.pretrained.conv1),
        ('bn1', model.pretrained.bn1),
        ('avg_bn', model.pretrained.avg_bn),
        ('relu', model.pretrained.relu),
        ('layer1', model.pretrained.layer1),
        ('layer2', model.pretrained.layer2),
        ('layer3', model.pretrained.layer3)
        # ('layer4', model.pretrained.layer4)
    ]:
        for param in module.parameters():
            param.requires_grad = False
        # freeze_bn_stats(module)  # Freeze BatchNorm running statistics
    
    model.fc1.requires_grad_(True)  # unfreeze fc1 layer
    model.pretrained.layer4.requires_grad_(True)  # unfreeze layer4

    # Capture post-freeze parameter snapshots for verification
    post_freeze_params = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:  # Only capture frozen parameters
            post_freeze_params[name] = param.data.clone()

    # Store parameter snapshots in the model for later verification
    model._frozen_params_snapshot = post_freeze_params
    
    # Store BatchNorm running statistics for verification
    model._initial_bn_stats = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            model._initial_bn_stats[name] = {
                'running_mean': module.running_mean.clone(),
                'running_var': module.running_var.clone(),
                'num_batches_tracked': module.num_batches_tracked.clone()
            }

    # The new layers are already created in the model, just ensure they're trainable
    model.fc_inter1.requires_grad_(True)  # Train the intermediate layer
    # model.fc_inter2.requires_grad_(True)  # Train the intermediate layer
    # model.fc_inter3.requires_grad_(True)  # Train the intermediate layer
    model.fc_final.requires_grad_(True)        # Train the final layer

    # Create parameter groups with different learning rates
    layer4_lr = new_lr * layer4_lr_ratio  # Use configurable ratio for layer4
    other_layers_lr = new_lr  # Use standard learning rate for other layers
    
    # Separate parameters into groups
    layer4_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'pretrained.layer4' in name or 'fc1' in name:
                layer4_params.append(param)
            else:
                other_params.append(param)
    
    # Log the parameter groups
    log_print(f"\nParameter groups for different learning rates:", log_path=log_path)
    log_print(f"Layer4 parameters: {len(layer4_params)} groups, LR: {layer4_lr}", log_path=log_path)
    log_print(f"Other parameters: {len(other_params)} groups, LR: {other_layers_lr}", log_path=log_path)

    # setup optimizer with different learning rates
    if rms_prop_flag:
        optimizer = optim.RMSprop([
            {'params': layer4_params, 'lr': layer4_lr},
            {'params': other_params, 'lr': other_layers_lr}
        ], alpha=0.99, eps=1e-8)
    else:
        optimizer = optim.SGD([
            {'params': layer4_params, 'lr': layer4_lr, 'weight_decay': wd},
            {'params': other_params, 'lr': other_layers_lr, 'weight_decay': wd}
        ], momentum=0.9, nesterov=True, dampening=0)


    # Move optimizer tensors to GPU if necessary
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()


    if print_model_architecture:

        # Print detailed trainable status for each layer with better formatting
        log_print("\n" + "="*100, log_path=log_path)
        log_print("DETAILED TRAINABLE STATUS BY LAYER", log_path=log_path)
        log_print("="*100, log_path=log_path)
        log_print(f"{'Layer Name':<60} | {'Parameters':>12} | {'Trainable':>12} | {'Status'}", log_path=log_path)
        log_print("-" * 100, log_path=log_path)

        total_params = 0
        trainable_params = 0
        
        for name, param in model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
                status = "✓ TRAINABLE"
                trainable_str = "YES"
            else:
                status = "✗ FROZEN"
                trainable_str = "NO"

            log_print(f"{name:<60} | {num_params:>12,} | {trainable_str:>12} | {status}", log_path=log_path)

        log_print("-" * 100, log_path=log_path)
        log_print(f"{'SUMMARY':<60} | {total_params:>12,} | {trainable_params:>12,} |", log_path=log_path)
        log_print("="*100, log_path=log_path)
        log_print(f"Total parameters: {total_params:,}", log_path=log_path)
        log_print(f"Trainable parameters: {trainable_params:,}", log_path=log_path)
        log_print(f"Frozen parameters: {total_params - trainable_params:,}", log_path=log_path)
        log_print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%", log_path=log_path)
        log_print(f"Model size (MB): {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024:.2f}", log_path=log_path)
        log_print("="*100, log_path=log_path)

    return model, optimizer



## lr from checkpoint = 1.0e-5
def create_optimizer(model, new_lr=1e-1, wd=1e-4):
    # setup optimizer
    optimizer = optim.SGD([
        {'params': model.parameters(), 'weight_decay': wd}
    ], lr=new_lr, momentum=0.9, nesterov=True, dampening=0)

    # RMSprop
    # optimizer = optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99, eps=1e-8)

    return optimizer


def split_support_query(inputs):
    B, C, Fr, T = inputs.size()
    inputs = inputs.reshape(n_shot + n_query, nb_class_train, C, Fr, T)
    support = inputs[:n_shot].reshape(-1, C, Fr, T)
    query = inputs[n_shot:].reshape(-1, C, Fr, T)

    if use_variable:
        min_win, max_win = SHORT_SIZE, T
        win_size = random.randrange(min_win, max_win)
        query = query[:, :, :, :win_size].contiguous()

    return support, query


def complete_inference(finetrained_path, log_path, store_features=False):
    # Load dataset
    test_DB, length_db, num_speakers = read_feats2(TEST_FEAT_AOLME, n_shot_test, n_query_test, dataset_id='aolme_tst', log_path=log_path)

    n_classes = num_speakers  # Number of classes in the test set

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model_test, _ = load_model(finetrained_path, n_classes, log_path, print_model_architecture=False)
    # model_test = load_model_test(finetrained_path)
    if use_cuda:
        model_test.cuda()

    nb_class_test=num_speakers
    # nb_class_test=15

    # Print number of speakers and length of the database
    log_print(f'Number of speakers in test set: {num_speakers}', log_path=log_path)
    log_print(f'Length of test set: {length_db}', log_path=log_path)


    # make generator for unseen speaker identification
    test_generator = metaGenerator_test(test_DB, read_MFB, enroll_length=enroll_length, test_length=test_length,
                                   nb_classes=nb_class_test, n_support=n_shot_test, n_query=n_query_test,
                                   max_iter=max_iter_test, xp=np)
    # evaluate
    evaluation(test_generator, model_test, use_cuda, n_shot_test, n_query_test, nb_class_test)

    list_of_feats = sorted(list(TEST_FEAT_AOLME.glob('*.pkl')))
    list_of_wavs = sorted(list(TEST_WAV_AOLME.glob('*.wav')))

    # Print separator
    log_print("\n" + "=" * 50, log_path=log_path)
    log_print("Starting enrollment and verification...", log_path=log_path)

    # Enroll and test
    dict_embeddings = d_vector_dict_lbls(list_of_feats, model_test,
                                         list_of_wavs,
                                         norm_flag=norm_flag, samples_flag=samples_flag)

    log_print(f"\nKeys in dict_embeddings: {dict_embeddings.keys()}\n", log_path=log_path)

    # Perform verification
    _ = perform_verification(veri_test_dir, dict_embeddings)


if __name__ == '__main__':

    all_train_DB, n_data, n_classes = read_feats2(TRAIN_FEAT_LUIS, n_shot, n_query, log_path=log_path)
    print('Training set size: %d' % n_data)
    print('Number of classes: %d' % n_classes)


    nb_class_train = n_classes  # Use all classes for training
    train_DB = all_train_DB  # Use the entire dataset for training

    nb_samples_per_class = n_shot + n_query
    n_episode = int(n_data / ((nb_samples_per_class) * nb_class_train))
    print(f'Number of episodes per epoch: {n_episode}/n')

    if use_checkpoint:
        model, optimizer = load_model(pretrained_path, n_classes, log_path, print_model_architecture=True)
    else:
        sys.error("Pretrained model path is not provided. Please set use_checkpoint to True and provide a valid path.")


    objective = Prototypical()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, min_lr=1e-7, threshold=1e-5, verbose=1)

    model.cuda()

    train_generator = metaGenerator(train_DB, read_MFB_train,
                                    nb_classes=nb_class_train, nb_samples_per_class=nb_samples_per_class,
                                    max_iter=n_episode * n_epochs, xp=np)

    early_stopped, trained_model_path = train(train_generator, model, objective, optimizer, n_episode, log_dir, scheduler, aug_percent=aug_percent, early_stop_patience=patience_early_stop)

    if early_stopped:
        print("Training stopped early due to no improvement in loss (early stopping).")
    
    ##trained_model_path = "/home/luis/Dropbox/clustering/meta-SR/saved_model/checkpoint_100_original.pth"

    complete_inference(trained_model_path, log_path, store_features=True)



