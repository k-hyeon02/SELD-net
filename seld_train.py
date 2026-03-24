import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plot
plot.switch_backend('agg')

import parameter
import evaluation_metrics
import utils
from seld_dataset import SELDDataset
from seld_model import SELDNet


def collect_test_labels(loader):
    """
    테스트 데이터 전체의 정답 레이블을 미리 수집
    평가 지표 계산 시 전체 정답이 필요하기 때문
    """
    all_sed, all_doa = [], []
    for _, sed_label, doa_label in loader:
        all_sed.append(sed_label.numpy())   # (batch, seq_len, nb_classes)
        all_doa.append(doa_label.numpy())   # (batch, seq_len, nb_classes*2 or *3)
    # 배치 전체를 이어붙임 → (전체샘플, seq_len, ...)
    gt_sed = np.concatenate(all_sed, axis=0)
    gt_doa = np.concatenate(all_doa, axis=0)
    # 2D로 reshape: (전체샘플 * seq_len, ...)
    gt_sed = evaluation_metrics.reshape_3Dto2D(gt_sed).astype(int)
    gt_doa = evaluation_metrics.reshape_3Dto2D(gt_doa)
    return gt_sed, gt_doa


def plot_functions(fig_name, tr_loss, val_loss, sed_loss, doa_loss, epoch_metric_loss):
    plot.figure()
    nb_epoch = len(tr_loss)
    plot.subplot(311)
    plot.plot(range(nb_epoch), tr_loss, label='train loss')
    plot.plot(range(nb_epoch), val_loss, label='val loss')
    plot.legend()
    plot.grid(True)

    plot.subplot(312)
    plot.plot(range(nb_epoch), epoch_metric_loss, label='metric')
    plot.plot(range(nb_epoch), sed_loss[:, 0], label='er')
    plot.plot(range(nb_epoch), sed_loss[:, 1], label='f1')
    plot.legend()
    plot.grid(True)

    plot.subplot(313)
    plot.plot(range(nb_epoch), doa_loss[:, 1], label='gt_thres')
    plot.plot(range(nb_epoch), doa_loss[:, 2], label='pred_thres')
    plot.legend()
    plot.grid(True)

    plot.savefig(fig_name)
    plot.close()


def main(argv):
    if len(argv) != 3:
        print('Usage: python seld_train.py <job-id> <task-id>')
        print('Using default parameters.')

    task_id = '1' if len(argv) < 3 else argv[-1]
    params = parameter.get_params(task_id)
    job_id = 1 if len(argv) < 2 else argv[1]

    # 모델 저장 폴더 및 고유 이름 설정
    model_dir = 'models/'
    utils.create_folder(model_dir)
    unique_name = '{}_ov{}_split{}_{}{}_3d{}_{}'.format(
        params['dataset'], params['overlap'], params['split'],
        params['mode'], params['weakness'], int(params['cnn_3d']), job_id
    )
    unique_name = os.path.join(model_dir, unique_name)
    print("unique_name: {}\n".format(unique_name))

    # Device 설정 (GPU 있으면 GPU, 없으면 CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))

    # Dataset & DataLoader 생성
    train_dataset = SELDDataset(
        datagen_mode='train', dataset=params['dataset'], ov=params['overlap'],
        split=params['split'], db=params['db'], nfft=params['nfft'],
        seq_len=params['sequence_length'], classifier_mode=params['mode'],
        weakness=params['weakness'], xyz_def_zero=params['xyz_def_zero'],
        azi_only=params['azi_only']
    )
    test_dataset = SELDDataset(
        datagen_mode='test', dataset=params['dataset'], ov=params['overlap'],
        split=params['split'], db=params['db'], nfft=params['nfft'],
        seq_len=params['sequence_length'], classifier_mode=params['mode'],
        weakness=params['weakness'], xyz_def_zero=params['xyz_def_zero'],
        azi_only=params['azi_only']
    )

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=params['batch_size'], shuffle=False)

    # 입출력 shape 확인
    feat_shape, label_shape = train_dataset.get_data_sizes()
    data_in  = (params['batch_size'],) + feat_shape          # (batch, 2*nb_ch, seq_len, feat_len)
    data_out = [(params['batch_size'],) + s for s in label_shape]
    print('data_in: {}\ndata_out: {}\n'.format(data_in, data_out))

    # 테스트 정답 레이블 미리 수집
    gt_sed, gt_doa = collect_test_labels(test_loader)

    # 모델 생성
    model = SELDNet(
        data_in=data_in, data_out=data_out,
        dropout_rate=params['dropout_rate'],
        nb_cnn2d_filt=params['nb_cnn2d_filt'],
        pool_size=params['pool_size'],
        rnn_size=params['rnn_size'],
        fnn_size=params['fnn_size']
    ).to(device)

    # Loss & Optimizer
    sed_criterion = nn.BCELoss()   # SED: binary cross entropy
    doa_criterion = nn.MSELoss()   # DOA: mean squared error
    sed_w, doa_w = params['loss_weights']  # [1., 50.]
    optimizer = torch.optim.Adam(model.parameters())

    # 학습 기록용 배열
    nb_epoch = 2 if params['quick_test'] else params['nb_epochs']
    tr_loss          = np.zeros(nb_epoch)
    val_loss         = np.zeros(nb_epoch)
    sed_loss         = np.zeros((nb_epoch, 2))
    doa_loss         = np.zeros((nb_epoch, 6))
    epoch_metric_loss = np.zeros(nb_epoch)

    best_metric  = 99999
    best_epoch   = -1
    patience_cnt = 0

    for epoch_cnt in range(nb_epoch):
        start = time.time()

        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        nb_train_batches = 2 if params['quick_test'] else len(train_loader)
        for batch_cnt, (feat, sed_label, doa_label) in enumerate(train_loader):
            if batch_cnt == nb_train_batches:
                break
            feat      = feat.to(device)       # (batch, 2*nb_ch, seq_len, feat_len)
            sed_label = sed_label.to(device)  # (batch, seq_len, nb_classes)
            doa_label = doa_label.to(device)  # (batch, seq_len, nb_classes*2 or *3)

            optimizer.zero_grad()
            sed_pred, doa_pred = model(feat)

            loss = sed_w * sed_criterion(sed_pred, sed_label) + \
                   doa_w * doa_criterion(doa_pred, doa_label)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        tr_loss[epoch_cnt] = train_loss_sum / nb_train_batches

        # --- Validation loss ---
        model.eval()
        val_loss_sum = 0.0
        nb_test_batches = 2 if params['quick_test'] else len(test_loader)
        all_sed_pred, all_doa_pred = [], []
        with torch.no_grad():
            for batch_cnt, (feat, sed_label, doa_label) in enumerate(test_loader):
                if batch_cnt == nb_test_batches:
                    break
                feat      = feat.to(device)
                sed_label = sed_label.to(device)
                doa_label = doa_label.to(device)

                sed_pred, doa_pred = model(feat)

                loss = sed_w * sed_criterion(sed_pred, sed_label) + \
                       doa_w * doa_criterion(doa_pred, doa_label)
                val_loss_sum += loss.item()

                all_sed_pred.append(sed_pred.cpu().numpy())
                all_doa_pred.append(doa_pred.cpu().numpy())

        val_loss[epoch_cnt] = val_loss_sum / nb_test_batches

        # --- 평가 지표 계산 ---
        sed_pred_all = np.concatenate(all_sed_pred, axis=0)  # (전체샘플, seq_len, nb_classes)
        doa_pred_all = np.concatenate(all_doa_pred, axis=0)

        sed_pred_2d = evaluation_metrics.reshape_3Dto2D(sed_pred_all) > 0.5  # threshold 0.5
        doa_pred_2d = evaluation_metrics.reshape_3Dto2D(doa_pred_all)

        n_pred = sed_pred_2d.shape[0]
        gt_sed_eval = gt_sed[:n_pred]
        gt_doa_eval = gt_doa[:n_pred]

        sed_loss[epoch_cnt, :] = evaluation_metrics.compute_sed_scores(
            sed_pred_2d, gt_sed_eval, train_dataset.nb_frames_1s()
        )
        if params['azi_only']:
            doa_loss[epoch_cnt, :], conf_mat = evaluation_metrics.compute_doa_scores_regr_xy(
                doa_pred_2d, gt_doa_eval, sed_pred_2d, gt_sed_eval
            )
        else:
            doa_loss[epoch_cnt, :], conf_mat = evaluation_metrics.compute_doa_scores_regr_xyz(
                doa_pred_2d, gt_doa_eval, sed_pred_2d, gt_sed_eval
            )

        # SELD 통합 지표 계산 (낮을수록 좋음)
        epoch_metric_loss[epoch_cnt] = np.mean([
            sed_loss[epoch_cnt, 0],                                          # ER (낮을수록 좋음)
            1 - sed_loss[epoch_cnt, 1],                                      # 1 - F1 (낮을수록 좋음)
            2 * np.arcsin(doa_loss[epoch_cnt, 1] / 2.0) / np.pi,            # DOA gt 오차
            1 - (doa_loss[epoch_cnt, 5] / float(gt_sed.shape[0]))           # 1 - good_frame_ratio
        ])

        plot_functions(unique_name, tr_loss, val_loss, sed_loss, doa_loss, epoch_metric_loss)

        # --- Best model 저장 ---
        patience_cnt += 1
        if epoch_metric_loss[epoch_cnt] < best_metric:
            best_metric  = epoch_metric_loss[epoch_cnt]
            best_epoch   = epoch_cnt
            patience_cnt = 0
            torch.save(model.state_dict(), '{}_model.pt'.format(unique_name))

        print(
            'epoch: %d, time: %.2fs, tr_loss: %.4f, val_loss: %.4f, '
            'F1: %.4f, ER: %.4f, doa_gt: %.4f, doa_pred: %.4f, '
            'metric: %.4f, best_metric: %.4f, best_epoch: %d' % (
                epoch_cnt, time.time() - start,
                tr_loss[epoch_cnt], val_loss[epoch_cnt],
                sed_loss[epoch_cnt, 1], sed_loss[epoch_cnt, 0],
                doa_loss[epoch_cnt, 1], doa_loss[epoch_cnt, 2],
                epoch_metric_loss[epoch_cnt], best_metric, best_epoch
            )
        )

        if patience_cnt > params['patience']:
            print('Early stopping at epoch {}'.format(epoch_cnt))
            break

    print('\n--- Training done ---')
    print('best_epoch: {}, best_metric: {:.4f}'.format(best_epoch, best_metric))
    print('DOA: doa_gt: {:.4f}, doa_pred: {:.4f}'.format(
        doa_loss[best_epoch, 1], doa_loss[best_epoch, 2]))
    print('SED: F1: {:.4f}, ER: {:.4f}'.format(
        sed_loss[best_epoch, 1], sed_loss[best_epoch, 0]))


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)