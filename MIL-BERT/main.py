import os
import argparse
import time
import numpy as np
import torch
import multiprocessing
from torch.multiprocessing import freeze_support
from torch.utils.data import DataLoader
from learner import Learner, Learner_multicrop
from loss import *
from dataset import *
from sklearn import metrics
from tqdm import tqdm

def divide_feature(feature, length, L2Norm, modality):
    """
    Divide a sequence feature into `length` segments, averaging over each segment,
    and optionally L2-normalize each segment.
    """
    feat = torch.squeeze(feature, 0)  # remove batch dim (for batch=1 only)
    divided_features = torch.zeros((feat.shape[0], length, feat.shape[-1]), device=feat.device)
    r = np.linspace(0, feat.shape[1], length + 1, dtype=int)

    for ind in range(feat.shape[0]):
        f = feat[ind]
        new_f = torch.zeros((length, f.shape[-1]), device=feat.device)
        for i in range(length):
            if r[i] != r[i + 1]:
                new_f[i, :] = torch.mean(f[r[i]:r[i + 1], :], dim=0)
            else:
                new_f[i, :] = f[r[i], :]
        divided_features[ind] = new_f

    if L2Norm > 0:
        if modality == 'RGB+Flow':
            half = divided_features.shape[-1] // 2
            norm1 = divided_features[:, :, :half].norm(p=2, dim=-1, keepdim=True)
            divided_features[:, :, :half] /= norm1
            norm2 = divided_features[:, :, half:].norm(p=2, dim=-1, keepdim=True)
            divided_features[:, :, half:] /= norm2
        else:
            norm = divided_features.norm(p=2, dim=-1, keepdim=True)
            divided_features /= norm

    return torch.unsqueeze(divided_features, 0)


def train_by_step(max_step, normal_loader, anomaly_loader, model, criterion, optimizer, scheduler, device, args, ckpt_dir, key):
    best_AUC = -1
    loadern_iter = iter(normal_loader)
    loadera_iter = iter(anomaly_loader)

    for step in tqdm(range(1, max_step + 1), total=max_step, dynamic_ncols=True):
        if (step - 1) % len(normal_loader) == 0:
            loadern_iter = iter(normal_loader)
        if (step - 1) % len(anomaly_loader) == 0:
            loadera_iter = iter(anomaly_loader)

        normal_inputs = next(loadern_iter)
        anomaly_inputs = next(loadera_iter)

        min_batch = min(anomaly_inputs.shape[0], normal_inputs.shape[0])
        anomaly_inputs = anomaly_inputs[:min_batch]
        normal_inputs = normal_inputs[:min_batch]

        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=0).to(device)
        batch_size = inputs.shape[0] // 2

        outputs, _ = model(inputs)
        loss_mil = criterion(outputs, batch_size)

        optimizer.zero_grad()
        loss_mil.backward()
        optimizer.step()

        if (step - 1) % len(normal_loader) == 0:
            scheduler.step()

        if step % 5 == 0 and step > 20:
            if args.dataset == 'XD-Violence':
                from __main__ import test_abnormal_xdviolence
                auc, ap, _, _ = test_abnormal_xdviolence(0)  # dummy epoch
                metric = ap
            else:
                from __main__ import test_abnormal
                auc, ap, _, _ = test_abnormal(0)
                metric = auc

            if metric > best_AUC:
                best_AUC = metric
                torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{key}-step-{step}-best.pkl'))
    return


def train(epoch, normal_loader, anomaly_loader, model, criterion, optimizer, scheduler, device):
    print(f'\nEpoch: {epoch}')
    model.train()
    train_loss = 0

    loadera_iter = iter(anomaly_loader)
    for batch_idx, normal_inputs in enumerate(normal_loader):
        if batch_idx % len(anomaly_loader) == 0:
            loadera_iter = iter(anomaly_loader)
        anomaly_inputs = next(loadera_iter)

        min_batch = min(anomaly_inputs.shape[0], normal_inputs.shape[0])
        anomaly_inputs = anomaly_inputs[:min_batch]
        normal_inputs = normal_inputs[:min_batch]

        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=0).to(device)
        batch_size = inputs.shape[0] // 2

        outputs, _ = model(inputs)
        loss_mil = criterion(outputs, batch_size)

        optimizer.zero_grad()
        loss_mil.backward()
        optimizer.step()

        train_loss += loss_mil.item()

    print(f'loss = {train_loss / len(normal_loader):.4f}')
    scheduler.step()


def train_video_level(epoch, normal_loader, anomaly_loader, model, criterion_video, optimizer, scheduler, device):
    print(f'\nEpoch: {epoch}')
    model.train()
    train_loss = 0

    loadera_iter = iter(anomaly_loader)
    for batch_idx, normal_inputs in enumerate(normal_loader):
        if batch_idx % len(anomaly_loader) == 0:
            loadera_iter = iter(anomaly_loader)
        anomaly_inputs = next(loadera_iter)

        min_batch = min(anomaly_inputs.shape[0], normal_inputs.shape[0])
        anomaly_inputs = anomaly_inputs[:min_batch]
        normal_inputs = normal_inputs[:min_batch]

        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=0).to(device)
        labels = torch.tensor([1] * anomaly_inputs.shape[0] + [0] * normal_inputs.shape[0], device=device)

        _, outputs_video = model(inputs)
        outputs_video = outputs_video.squeeze(-1).float()

        loss_video = criterion_video(outputs_video, labels.float())

        optimizer.zero_grad()
        loss_video.backward()
        optimizer.step()

        train_loss += loss_video.item()

    print(f'loss = {train_loss / len(normal_loader):.4f}')
    scheduler.step()


def train_joint(epoch, normal_loader, anomaly_loader, model, criterion, criterion_video, optimizer, scheduler, device, args):
    print(f'\nEpoch: {epoch}')
    model.train()
    train_loss = 0

    loadera_iter = iter(anomaly_loader)
    for batch_idx, normal_inputs in enumerate(normal_loader):
        if batch_idx % len(anomaly_loader) == 0:
            loadera_iter = iter(anomaly_loader)
        anomaly_inputs = next(loadera_iter)

        min_batch = min(anomaly_inputs.shape[0], normal_inputs.shape[0])
        anomaly_inputs = anomaly_inputs[:min_batch]
        normal_inputs = normal_inputs[:min_batch]

        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=0).to(device)
        labels = torch.tensor([1] * anomaly_inputs.shape[0] + [0] * normal_inputs.shape[0], device=device)

        outputs, outputs_video = model(inputs)
        loss_mil = criterion(outputs, inputs.shape[0] // 2)
        outputs_video = outputs_video.squeeze(-1).float()
        loss_video = criterion_video(outputs_video, labels.float())

        loss = loss_mil + loss_video
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f'loss = {train_loss / len(normal_loader):.4f}')
    scheduler.step()


def test_abnormal(epoch, anomaly_loader, normal_loader, model, device):
    model.eval()
    score_list_all = []
    gt_list_all = []

    with torch.no_grad():
        for data in anomaly_loader:
            inputs, gts, frames, _ = data
            inputs = inputs.to(device)
            score, _ = model(inputs)
            score = score.squeeze(0).cpu().numpy()

            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0] // 16, 33)).astype(int)
            for j in range(32):
                start = step[j] * 16
                end = min(step[j + 1] * 16, frames[0])
                score_list[start:end] = score[j]

            gt_list = np.zeros(frames[0])
            for k in range(len(gts) // 2):
                s, e = gts[2 * k], min(gts[2 * k + 1], frames[0] - 1)
                if s < 0 or e < 0: continue
                gt_list[s:e + 1] = 1

            score_list_all.extend(score_list.tolist())
            gt_list_all.extend(gt_list.tolist())

        for data in normal_loader:
            inputs, _, frames, _ = data
            inputs = inputs.to(device)
            score, _ = model(inputs)
            score = score.squeeze(0).cpu().numpy()
            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0] // 16, 33)).astype(int)
            for j in range(32):
                start = step[j] * 16
                end = min(step[j + 1] * 16, frames[0])
                score_list[start:end] = score[j]

            score_list_all.extend(score_list.tolist())
            gt_list_all.extend(np.zeros(frames[0]).tolist())

    fpr, tpr, _ = metrics.roc_curve(gt_list_all, score_list_all, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    ap = metrics.average_precision_score(gt_list_all, score_list_all, pos_label=1)
    print(f'auc = {auc:.4f}, ap = {ap:.4f}')
    return auc, ap, None, None


def test_abnormal_xdviolence(epoch, normal_loader, model, device):
    model.eval()
    score_list_all = []

    with torch.no_grad():
        for data in normal_loader:
            inputs, _, frames, _ = data
            inputs = inputs.to(device)
            score, _ = model(inputs)
            score = score.squeeze(0).cpu().numpy()

            if args.divideTo32:
                step = np.round(np.linspace(0, frames[0] // 16, 33)).astype(int)
                score_list = np.zeros(frames[0])
                for j in range(32):
                    start = step[j] * 16
                    end = min(step[j + 1] * 16, frames[0])
                    score_list[start:end] = score[j]
                score_list_all.extend(score_list.tolist())
            else:
                score_list_all.extend(
                    np.zeros(frames[0])  # fallback
                )

    # load ground truth from file for XD-Violence
    gt_list_all = list(np.load('/media/ubuntu/MyHDataStor3/datasets/violence/XD-Violence/gt.npy'))
    fpr, tpr, _ = metrics.roc_curve(gt_list_all, score_list_all, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    ap = metrics.average_precision_score(gt_list_all, score_list_all, pos_label=1)
    print(f'auc = {auc:.4f}, ap = {ap:.4f}')
    return auc, ap, None, None


def main():
    parser = argparse.ArgumentParser(description='MIL-BERT')
    parser.add_argument('--modality', default='RGB+Flow', help='RGB, Flow, or RGB+Flow')
    parser.add_argument('--batch-size', type=int, default=30)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pretrained', default='model_best-joint-e2e-RGB+Flow.pkl')
    parser.add_argument('--lr', '--learning-rate', type=float, default=1e-4)
    parser.add_argument('--dataset', default='UCF-Crime')
    parser.add_argument('--epochs', type=int, default=75)
    parser.add_argument('--train_mode', type=int, default=2)
    parser.add_argument('-r', '--resume', action='store_true')
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('--divideTo32', action='store_true')
    parser.add_argument('--L2Norm', type=int, default=0)
    parser.add_argument('--multiCrop', action='store_true')
    parser.add_argument('--train_by_step', action='store_true')
    global args
    args = parser.parse_args()
    print('args =', args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # --- Dataset and model setup ---
    if args.dataset == 'XD-Violence':
        normal_train_dataset = Normal_Loader(1, args.dataset, args.modality, args.divideTo32, args.L2Norm)
        normal_test_dataset = Normal_Loader(0, args.dataset, args.modality, args.divideTo32, args.L2Norm)
        anomaly_train_dataset = Anomaly_Loader(1, args.dataset, args.modality, args.divideTo32, args.L2Norm)
        anomaly_test_dataset = Anomaly_Loader(0, args.dataset, args.modality, args.divideTo32, args.L2Norm)
        model_cls = Learner_multicrop if args.multiCrop or args.train_mode != 0 else Learner
        model = model_cls(feature_dim=1024 if 'UCF' not in args.dataset else 2048,
                          modality=args.modality,
                          BERT_Enable=(args.train_mode != 0))
    elif args.dataset == 'UCF-Crime-RTFM':
        args.modality = 'RGB'
        normal_train_dataset = Normal_Loader(1, args.dataset, args.modality, args.divideTo32, args.L2Norm, args.multiCrop)
        normal_test_dataset = Normal_Loader(0, args.dataset, args.modality, args.divideTo32, args.L2Norm, args.multiCrop)
        anomaly_train_dataset = Anomaly_Loader(1, args.dataset, args.modality, args.divideTo32, args.L2Norm, args.multiCrop)
        anomaly_test_dataset = Anomaly_Loader(0, args.dataset, args.modality, args.divideTo32, args.L2Norm, args.multiCrop)
        model_cls = Learner_multicrop if args.multiCrop or args.train_mode != 0 else Learner
        model = model_cls(feature_dim=2048, modality=args.modality, BERT_Enable=(args.train_mode != 0))
    else:
        args.divideTo32 = True
        args.L2Norm = 1
        args.multiCrop = False
        normal_train_dataset = Normal_Loader(1, args.dataset, args.modality)
        normal_test_dataset = Normal_Loader(0, args.dataset, args.modality)
        anomaly_train_dataset = Anomaly_Loader(1, args.dataset, args.modality)
        anomaly_test_dataset = Anomaly_Loader(0, args.dataset, args.modality)
        model = Learner(feature_dim=1024, modality=args.modality, BERT_Enable=(args.train_mode != 0))

    key = f'{args.dataset}-{args.modality}-trainmode-{args.train_mode}-divide32-{args.divideTo32}-L2Norm-{args.L2Norm}-multiCrop-{args.multiCrop}'
    print('key =', key)

    normal_train_loader = DataLoader(normal_train_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.workers, pin_memory=True)
    normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=False,
                                    num_workers=4, pin_memory=True)
    anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.workers, pin_memory=True)
    anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=False,
                                     num_workers=4, pin_memory=True)

    if torch.cuda.device_count() > 1:
        print("Using multiple GPUs")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 200])
    criterion = MIL
    criterion_video = torch.nn.BCELoss()

    best_AUC = -1
    ckpt_dir = f'./ckpt/{args.dataset}/'
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.test:
        print('Loading model =', args.pretrained)
        model.load_state_dict(torch.load(args.pretrained))
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("number of parameters =", pytorch_total_params)
        if args.dataset == 'XD-Violence':
            test_abnormal_xdviolence(0, normal_test_loader, model, device)
        else:
            test_abnormal(0, anomaly_test_loader, normal_test_loader, model, device)
        return

    if args.resume:
        print('Resuming from', args.pretrained)
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, args.pretrained)))

    train_mode = args.train_mode

    if train_mode == 0:
        if args.train_by_step:
            train_by_step(150_000, normal_train_loader, anomaly_train_loader, model,
                          criterion, optimizer, scheduler, device, args, ckpt_dir, key)
            return
        for epoch in range(args.epochs):
            train(epoch, normal_train_loader, anomaly_train_loader, model,
                  criterion, optimizer, scheduler, device)
            auc, ap, _, _ = test_abnormal(epoch, anomaly_test_loader, normal_test_loader, model, device)
            if auc > best_AUC:
                best_AUC = auc
                torch.save(model.state_dict(), os.path.join(ckpt_dir, f'epoch{epoch}-best.pkl'))

    elif train_mode == 1:
        for epoch in range(args.epochs // 5):
            train_video_level(epoch, normal_train_loader, anomaly_train_loader,
                              model, criterion_video, optimizer, scheduler, device)
            # implement test_abnormal_video_level if needed
        for epoch in range(args.epochs):
            train_joint(epoch, normal_train_loader, anomaly_train_loader,
                        model, criterion, criterion_video, optimizer, scheduler, device, args)
            # implement joint testing if needed

    elif train_mode == 2:
        if args.train_by_step:
            train_by_step(150_000, normal_train_loader, anomaly_train_loader, model,
                          criterion, optimizer, scheduler, device, args, ckpt_dir, key)
            return
        for epoch in range(args.epochs):
            train_joint(epoch, normal_train_loader, anomaly_train_loader,
                        model, criterion, criterion_video, optimizer, scheduler, device, args)
            auc, ap, _, _ = test_abnormal(epoch, anomaly_test_loader, normal_test_loader, model, device)
            if auc > best_AUC:
                best_AUC = auc
                torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{key}-epoch-{epoch}-best.pkl'))

    else:
        pass


if __name__ == '__main__':
    freeze_support()
    multiprocessing.set_start_method('spawn', force=True)
    main()
