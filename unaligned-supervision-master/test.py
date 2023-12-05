from sacred import Experiment
import os
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from onsets_and_frames import *
from onsets_and_frames.dataset import EMDATASET
from torch.nn import DataParallel
from onsets_and_frames.transcriber import load_weights


ex = Experiment("test_transcription")

@ex.config
def cfg():
    sequence_length = SEQ_LEN
    

@ex.automain
def my_main(logdir, iterations, checkpoint_interval, batch_size,
            train_data_path, labels_path, tsv_path, train_groups,
            sequence_length, device, learning_rate, learning_rate_decay_steps,
            clip_gradient_norm, epochs,  multi_ckpt, train_mode, transcriber_ckpt):

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
    ckpt.sort()
    ckpt = ckpt[-2]
    print(ckpt)

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

    total_p = np.zeros(10)
    total_r = np.zeros(10)
    total_pp = np.zeros(10)
    total_pr = np.zeros(10)
    for batch in loader:
        with torch.no_grad():
            transcription = transcriber.eval_on_batch(batch)
            fig, axs = plt.subplots(3, 1, figsize=(10, 10))
            i = 0
            for k, v in transcription.items():
                print(k, v.shape, type(v))
                m = v.detach().cpu().numpy()[0].T
                m = m[:, :batch["real_length"][0]]
                axs[i].imshow(m, aspect="auto", origin="lower")
                axs[i].set_title(k)
                i += 1
            plt.tight_layout()
            plt.savefig("sdtw_pred_out.png")
            _ = input()

            th = np.arange(10)

            for t in th:
                onset_pred = transcription['onset'].detach() > (t / 10)
                onset_tp = onset_pred * batch['onset'].detach()
                onset_p = batch['onset'].detach()

                onset_recall = (onset_tp.sum() / batch['onset'].detach().sum()).item()
                onset_precision = (onset_tp.sum() / onset_pred.sum()).item()

                pitch_onset_recall = (onset_tp[..., -N_KEYS:].sum() / onset_p[..., -N_KEYS:].sum()).item()
                pitch_onset_precision = (onset_tp[..., -N_KEYS:].sum() / onset_pred[..., -N_KEYS:].sum()).item()

                total_p[int(t)] += onset_precision
                total_r[int(t)] += onset_recall

                total_pp[int(t)] = pitch_onset_precision
                total_pr[int(t)] = pitch_onset_recall
    print(total_p / len(dataset))
    print(total_r / len(dataset))
    print(total_pp / len(dataset))
    print(total_pr / len(dataset))

