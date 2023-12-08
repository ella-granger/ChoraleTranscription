from sacred import Experiment
import os
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from onsets_and_frames import *
from onsets_and_frames.dataset import EMDATASET
from onsets_and_frames.mel import melspectrogram
from torch.nn import DataParallel
from onsets_and_frames.transcriber import load_weights


ex = Experiment("test_transcription")

@ex.config
def cfg():
    # sequence_length = SEQ_LEN
    sequence_length = 3276800
    

@ex.automain
def my_main(logdir, iterations, checkpoint_interval, batch_size,
            train_data_path, labels_path, tsv_path, train_groups,
            sequence_length, device, learning_rate, learning_rate_decay_steps,
            clip_gradient_norm, epochs,  multi_ckpt, train_mode, transcriber_ckpt):

    sequence_length = 3276800
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
                        conversion_map=conversion_map)

    logdir = Path(logdir)
    ckpt = list(logdir.glob("transcriber_*.pt"))
    e = [int(x.stem.split("_")[1]) for x in ckpt]
    e.sort()
    e = e[-1]
    ckpt = logdir / ("transcriber_%d_%d000.pt" % (e, e))
    print(ckpt)
    _ = input()

    transcriber = torch.load(ckpt).to(device)

    loader = DataLoader(dataset, 1, shuffle=False, drop_last=False)

    POS = 1.1
    NEG = -0.1

    transcriber.eval()

    """
    dataset.update_pts(transcriber,
                       POS=POS,
                       NEG=NEG,
                       to_save="./test_align",
                       first=False,
                       update=True,
                       BEST_BON=False)
    """

    onset_p = np.zeros(10)
    onset_pp = np.zeros(10)
    onset_tp = np.zeros(10)
    frame_p = np.zeros(10)
    frame_pp = np.zeros(10)
    frame_tp = np.zeros(10)
    th = np.arange(10)
    for batch in loader:
        with torch.no_grad():
            transcription = transcriber.eval_on_batch(batch)
            # fig, axs = plt.subplots(5, 1, figsize=(10, 10))
            # i = 0
            # for k, v in transcription.items():
            #     print(k, v.shape, type(v))
            #     m = v.detach().cpu().numpy()[0].T
            #     m = m[:, :batch["real_length"][0]]
            #     axs[i].imshow(m, aspect="auto", origin="lower")
            #     axs[i].set_title(k)
            #     i += 1
            # gt = batch["real_label"].cpu().numpy()[0].T
            # axs[3].imshow(gt[:88, :batch["real_length"][0]], aspect="auto", origin="lower")
            # axs[3].set_title("gt")

            # audio_label = batch["audio"]
            # mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
            # mel = mel.detach().cpu()[0].T
            # axs[4].imshow(mel[:, :batch["real_length"][0]], origin="lower")
            # plt.tight_layout()
            # plt.savefig("sdtw_pred_out.png")
            # print(sequence_length)
            # _ = input()

            real_label = batch["real_label"]
            onset = (real_label == 3).float()
            offset = (real_label == 1).float()
            frame = (real_label > 1).float()
            shape = frame.shape
            keys = N_KEYS
            new_shape = shape[:-1] + (shape[-1] // keys, keys)
            frame, _ = frame.reshape(new_shape).max(axis=-2)

            transcription['onset'] = transcription['onset'][:, :batch["real_length"][0], :]
            transcription['offset'] = transcription['offset'][:, :batch["real_length"][0], :]
            transcription['frame'] = transcription['frame'][:, :batch["real_length"][0], :]

            onset = onset[:, :batch["real_length"][0], :]
            frame = frame[:, :batch["real_length"][0], :]

            for t in th:
                onset_pred = transcription['onset'].detach() > (t / 10)
                onset_tp_i = onset_pred * onset.detach() # batch['onset'].detach()
                onset_p = onset.detach() # batch['onset'].detach()
                frame_pred = transcription['frame'].detach() > (t / 10)
                frame_tp_i = frame_pred * frame.datach()
                frame_p = frame.detach()

                onset_p[int(t)] += onset_p.sum().item()
                onset_pp[int(t)] += onset_pred.sum().item()
                onset_tp[int(t)] += onset_tp_i.sum().item()

                frame_p[int(t)] += frame_p.sum().item()
                frame_pp[int(t)] += frame_pred.sum().item()
                frame_tp[int(t)] += frame_tp_i.sum().item()
                
                # onset_recall = (onset_tp.sum() / batch['onset'].detach().sum()).item()
                onset_recall = (onset_tp.sum() / onset.detach().sum()).item()
                onset_precision = (onset_tp.sum() / onset_pred.sum()).item()

                pitch_onset_recall = (onset_tp[..., -N_KEYS:].sum() / onset_p[..., -N_KEYS:].sum()).item()
                pitch_onset_precision = (onset_tp[..., -N_KEYS:].sum() / onset_pred[..., -N_KEYS:].sum()).item()

                total_p[int(t)] += onset_precision
                total_r[int(t)] += onset_recall

                total_pp[int(t)] = pitch_onset_precision
                total_pr[int(t)] = pitch_onset_recall

    print("th\tn_acc\tn_rec\tn_f1\tf_acc\tf_rec\tf_f1")
    for t in th:
        idx = int(t)
        onset_acc = onset_tp[idx] / onset_pp[idx]
        onset_rec = onset_tp[idx] / onset_p[idx]
        onset_f1 = 2 * onset_acc * onset_rec / (onset_acc + onset_rec)
        frame_acc = frame_tp[idx] / frame_pp[idx]
        frame_rec = frame_tp[idx] / frame_p[idx]
        frame_f1 = 2 * frame_acc * frame_rec / (frame_acc + frame_rec)
        print("%.1f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (t / 10, onset_acc, onset_rec, onset_f1, frame_acc, frame_rec, frame_f1))

