from sacred import Experiment
import os
from pathlib import Path
import numpy as np

from onsets_and_frames import *
from onsets_and_frames.dataset import EMDATASET
from torch.nn import DataParallel
from onsets_and_frames.transcriber import load_weights


ex = Experiment("test_transcription")

@ex.config
def cfg():
    sequence_length = SEQ_LEN
    

@ex.automain
def my_main(logdir, train_data_path, labels_path, tsv_path, train_groups,
            sequence_length, device):

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
    ckpt = list(logdir.glob("transcriber_*.pt")).sort()[-1]
    print(ckpt)

    transcriber = torch.load(ckpt).to(device)

    loader = DataLoader(dataset, 1, shuffle=False, drop_last=False)

    POS = 1.1
    NEG = -0.1

    transcriber.eval()

    dataset.update_pts(transcriber,
                       POS=POS,
                       NEG=NEG,
                       to_save="./test_align",
                       first=False,
                       update=True,
                       BEST_BON=False)

    total_p = np.zeros(10)
    total_r = np.zeros(10)
    total_pp = np.zeros(10)
    total_pr = np.zeros(10)
    for batch in loader:
        with torch.no_grad():
            transcription = transcriber.eval_on_batch(batch)

            th = np.arange(10)

            for t in th:
                onset_pred = transcription['onset'].detach() > (t / 10)
                onset_total_pp += onset_pred
                onset_tp = onset_pred * batch['onset'].detach()
                onset_total_tp += onset_tp
                onset_total_p += batch['onset'].detach()

                onset_recall = (onset_total_tp.sum() / onset_total_p.sum()).item()
                onset_precision = (onset_total_tp.sum() / onset_total_pp.sum()).item()

                pitch_onset_recall = (onset_total_tp[..., -N_KEYS:].sum() / onset_total_p[..., -N_KEYS:].sum()).item()
                pitch_onset_precision = (onset_total_tp[..., -N_KEYS:].sum() / onset_total_pp[..., -N_KEYS:].sum()).item()

                total_p[int(t)] += onset_precision
                total_r[int(t)] += onset_recall

                total_pp[int(t)] = pitch_onset_precision
                total_pr[int(t)] = pitch_onset_recall
    print(total_p)
    print(total_r)
    print(total_pp)
    print(total_pr)

