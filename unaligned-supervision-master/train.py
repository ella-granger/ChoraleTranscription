import os
from datetime import datetime
import numpy as np
from sacred import Experiment
from sacred.commands import print_config, save_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
# from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

from onsets_and_frames import *
from onsets_and_frames.dataset import EMDATASET
from torch.nn import DataParallel
from onsets_and_frames.transcriber import load_weights
from soft_dtw_cuda import SoftDTW


def set_diff(model, diff=True):
    for layer in model.children():
        for p in layer.parameters():
            p.requires_grad = diff

ex = Experiment('train_transcriber')

@ex.config
def config():
    multi_ckpt = False # Flag if the ckpt was trained on pitch only or instrument-sensitive. The provided checkpoints were trained on pitch only.

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sequence_length = SEQ_LEN #if HOP_LENGTH == 512 else 3 * SEQ_LEN // 4


@ex.automain
def train(logdir, device, iterations, checkpoint_interval, batch_size,
          sequence_length, learning_rate, learning_rate_decay_steps,
          clip_gradient_norm, epochs, transcriber_ckpt, multi_ckpt,
          train_data_path, labels_path, tsv_path, train_groups, train_mode):
    
    ex.observers.append(FileStorageObserver.create(logdir))
    sw = SummaryWriter(logdir)

    print_config(ex.current_run)
    save_config(ex.current_run.config, logdir + "/config.json")

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    conversion_map = None
    instrument_map = None
    dataset = EMDATASET(audio_path=train_data_path,
                        labels_path=labels_path,
                        tsv_path=tsv_path,
                        groups=train_groups,
                        sequence_length=sequence_length,
                        seed=42,
                        device=device,
                        instrument_map=instrument_map,
                        conversion_map=conversion_map
                        )
    print('len dataset', len(dataset), len(dataset.data))
    print('instruments', dataset.instruments, len(dataset.instruments))
    train_data, test_data = random_split(dataset, [len(dataset) - 5, 5])

    if not multi_ckpt:
        model_complexity = 64 if '512' in transcriber_ckpt else 48
        saved_transcriber = torch.load(transcriber_ckpt, device) # .cpu()
        # We create a new transcriber with N_KEYS classes for each instrument:
        transcriber = OnsetsAndFrames(N_MELS,
                (MAX_MIDI - MIN_MIDI + 1),
                model_complexity,
                onset_complexity=1.,
                n_instruments=len(dataset.instruments) + 1,
                train_mode=train_mode).to(device)
        # We load weights from the saved pitch-only checkkpoint and duplicate the final layer as an initialization:
        load_weights(transcriber, saved_transcriber, n_instruments=len(dataset.instruments) + 1)
    else:
        # The checkpoint is already instrument-sensitive
        transcriber = torch.load(transcriber_ckpt).to(device)

    # We recommend to train first only onset detection. This will already give good note durations because the combined stack receives
    # information from the onset stack

    set_diff(transcriber.frame_stack, False)
    # set_diff(transcriber.offset_stack, False)
    # set_diff(transcriber.combined_stack, False)
    set_diff(transcriber.velocity_stack, False)

    parallel_transcriber = DataParallel(transcriber)
    optimizer = torch.optim.Adam(list(transcriber.parameters()), lr=learning_rate, weight_decay=1e-5)
    transcriber.zero_grad()
    optimizer.zero_grad()
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    def cross_entropy(x, y):
        result = F.binary_cross_entropy(x, y, reduction='none')
        factor = x.view(-1, x.size(-1)).size(0)
        # print(factor)
        return result / factor

    def mse_loss(x, y):
        result = F.mse_loss(x, y, reduction='none')
        factor = x.view(-1, x.size(-1)).size(0)
        return result / factor

    loss_function = SoftDTW(use_cuda=True, dist_func=cross_entropy, normalize=False)
    # loss_function = SoftDTW(use_cuda=True, gamma=0.1, dist_func=mse_loss, normalize=False)

    train_loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
    loader_cycle = cycle(train_loader)
    eval_loader = DataLoader(test_data, 1, shuffle=False, drop_last=False)
    step = 0
    for epoch in range(1, epochs + 1):
        print('epoch', epoch)
        if device != "cpu":
            torch.cuda.empty_cache()

        POS = 1.1 # Pseudo-label positive threshold (value > 1 means no pseudo label).
        NEG = -0.1 # Pseudo-label negative threshold (value < 0 means no pseudo label).
        # """
        with torch.no_grad():
            if epoch % 1 == 0:
                dataset.update_pts(parallel_transcriber,
                                   POS=POS,
                                   NEG=NEG,
                                   to_save=logdir + '/alignments', # MIDI alignments and predictions will be saved here
                                   first=epoch == 1,
                                   update=True,
                                   BEST_BON=epoch > 5 # after 5 epochs, update label only if bag of notes distance improved
                               )
        # """

        transcriber.train()

        onset_total_tp = 0.
        onset_total_pp = 0.
        onset_total_p = 0.

        # itr = tqdm(loader)
        itr = tqdm(range(iterations))
        for _ in itr:
            curr_loader = loader_cycle
            batch = next(curr_loader)
            optimizer.zero_grad()
            loss_function.gamma = max(0.1, 3 - 0.0001 * step)
            # print(batch)

            n_weight = 1 if HOP_LENGTH == 512 else 2
            transcription, transcription_losses = transcriber.run_on_batch(batch, parallel_transcriber,
                                                                           positive_weight=n_weight,
                                                                           inv_positive_weight=n_weight,
                                                                           loss_function=loss_function)

            onset_pred = transcription['onset'].detach() > 0.5
            onset_total_pp += onset_pred
            onset_tp = onset_pred * batch['onset'].detach()
            onset_total_tp += onset_tp
            onset_total_p += batch['onset'].detach()

            onset_recall = (onset_total_tp.sum() / onset_total_p.sum()).item()
            onset_precision = (onset_total_tp.sum() / onset_total_pp.sum()).item()

            pitch_onset_recall = (onset_total_tp[..., -N_KEYS:].sum() / onset_total_p[..., -N_KEYS:].sum()).item()
            pitch_onset_precision = (onset_total_tp[..., -N_KEYS:].sum() / onset_total_pp[..., -N_KEYS:].sum()).item()

            transcription_loss = sum(transcription_losses.values())
            loss = transcription_loss
            loss.backward()

            if clip_gradient_norm:
                clip_grad_norm_(transcriber.parameters(), clip_gradient_norm)

            optimizer.step()
            itr.set_description(f"loss: {loss.item()}, Onset Precision: {onset_precision}, Onset Recall: {onset_recall}, Pitch Onset Precision: {pitch_onset_precision}, Pitch Onset Recall: {pitch_onset_recall}")
            if step % 20 == 0:
                sw.add_scalar("Train/Loss", loss.item(), step)
                sw.add_scalar("Train/gamma", loss_function.gamma, step)

                sw.add_scalar("Train/Onset Precision", onset_precision, step)
                sw.add_scalar("Train/Onset Recall", onset_recall, step)
                sw.add_scalar("Train/Pitch Precision", pitch_onset_precision, step)
                sw.add_scalar("Train/Pitch Recall", pitch_onset_recall, step)

            if step % 200 == 0:
                transcriber.eval()
                with torch.no_grad():
                    eval_cycle = cycle(eval_loader)
                    total_prec = 0
                    total_recall = 0
                    total_p_prec = 0
                    total_p_recall = 0
                    for k in tqdm(range(100)):
                        b = next(eval_cycle)
                        trans = transcriber.eval_on_batch(b)

                        real_label = b['real_label']
                        # print(real_label)
                        # print(real_label.size())

                        onset = (real_label == 3).float()
                        offset = (real_label == 1).float()
                        frame = (real_label > 1).float()

                        onset_pred = trans['onset'].detach() > 0.5
                        onset_total_pp += onset_pred
                        onset_tp = onset_pred * onset.detach()
                        onset_total_tp += onset_tp
                        onset_total_p += onset.detach()

                        onset_recall = (onset_total_tp.sum() / onset_total_p.sum()).item()
                        onset_precision = (onset_total_tp.sum() / onset_total_pp.sum()).item()
                        
                        pitch_onset_recall = (onset_total_tp[..., -N_KEYS:].sum() / onset_total_p[..., -N_KEYS:].sum()).item()
                        pitch_onset_precision = (onset_total_tp[..., -N_KEYS:].sum() / onset_total_pp[..., -N_KEYS:].sum()).item()

                        total_prec += onset_precision
                        total_recall += onset_recall

                        total_p_prec += pitch_onset_precision
                        total_p_recall += pitch_onset_recall

                        if k < 5:
                            sw.add_figure("FramePred/%d" % k, plot_midi(trans["frame"].detach().cpu()[0]), step)
                            sw.add_figure("FrameGT/%d" % k, plot_midi(frame.cpu()[0]), step)
                            sw.add_figure("FrameLabel/%d" % k, plot_midi(b["frame"].detach().cpu()[0]), step)
                            sw.add_figure("OnsetPred/%d" % k, plot_midi(trans["onset"].detach().cpu()[0]), step)
                            sw.add_figure("OnsetGT/%d" % k, plot_midi(onset.cpu()[0]), step)
                            # sw.add_figure("PredOffset/%d" % k, plot_midi(trans["offset"].detach().cpu()[0]), step)
                            # sw.add_figure("GTOffset/%d" % k, plot_midi(offset.cpu()[0]), step)


                    total_prec /= 100
                    total_recall /= 100
                    total_p_prec /= 100
                    total_p_recall /= 100

                    sw.add_scalar("Eval/Onset Precision", total_prec, step)
                    sw.add_scalar("Eval/Onset Recall", total_recall, step)
                    sw.add_scalar("Eval/Pitch Precision", total_p_prec, step)
                    sw.add_scalar("Eval/Pitch Recall", total_p_recall, step)

                transcriber.train()

            step += 1
                

        save_condition = epoch % checkpoint_interval == 1
        if save_condition:
            torch.save(transcriber, os.path.join(logdir, 'transcriber_{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
            torch.save({'instrument_mapping': dataset.instruments},
                       os.path.join(logdir, 'instrument_mapping.pt'.format(epoch)))


