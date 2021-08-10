import torch
import math
import sys
import time
import datetime
from .utils import AverageMeter, accuracy


def train_one_epoch_classifier(epoch, iterator, data, model, device, opt, criterion,
                               tensorboard, start_time, args):
    print('--------------------------Start training at epoch:{}--------------------------'.format(epoch + 1))
    model.to(device)

    dict_log = {'loss': AverageMeter(), 'acc': AverageMeter()}
    criterion = criterion.to(device)

    model.train()
    data, data_labels = data
    steps = data.shape[0] // args.batch_size + 1 if data.shape[0] % args.batch_size else data.shape[
                                                                                             0] // args.batch_size
    step = 0
    for features, labels in iterator.get_batches(data, data_labels, shuffle=True):
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)
        if not math.isfinite(loss.item()):
            print("Loss is {} at step{}/{}, stopping training.".format(loss.item(), step, epoch))
            print(loss.item())
            sys.exit(1)
        acc = accuracy(logits.detach(), labels.detach())[0]
        dict_log['loss'].update(loss.item(), len(features))
        dict_log['acc'].update(acc.item(), len(features))
        opt.zero_grad()
        loss.backward()
        opt.step()
        all_steps = epoch * steps + step + 1
        if 0 == (all_steps % args.print_freq):
            lr = list(opt.param_groups)[0]['lr']
            now_time = time.time() - start_time
            et = str(datetime.timedelta(seconds=now_time))[:-7]
            print_information = 'time consumption:{}    epoch:{}/{}   step:{}/{}  lr:{}    '.format(
                et, epoch + 1, args.epochs, all_steps, steps, lr)
            for key, value in dict_log.items():
                loss_info = "{}(val/avg):{:.3f}/{:.3f}  ".format(key, value.val, value.avg)
                print_information = print_information + loss_info
                tensorboard.add_scalar(key, value.val, all_steps)
            print(print_information)
        step = step + 1
    print('--------------------------End training at epoch:{}--------------------------'.format(epoch + 1))


def evaluate_one_epoch_classifier(epoch, iterator, data, model, device, criterion, tensorboard, args, start_time):
    print('--------------------------Start evaluating at epoch:{}--------------------------'.format(epoch + 1))
    model.to(device)
    dict_log = {'loss': AverageMeter(), 'acc': AverageMeter()}
    model.eval()
    data, data_labels = data
    step = 0
    for features, labels in iterator.get_batches(data, data_labels, shuffle=True):
        features = features.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits = model(features)
            loss = criterion(logits, labels)
        if not math.isfinite(loss.item()):
            print("Loss is {} at step{}/{}, stopping training.".format(loss.item(), step, epoch))
            print(loss.item())
            sys.exit(1)
        acc = accuracy(logits.detach(), labels.detach())[0]
        dict_log['acc'].update(acc.item(), len(features))
        dict_log['loss'].update(loss.item(), len(features))
    now_time = time.time() - start_time
    et = str(datetime.timedelta(seconds=now_time))[:-7]
    print_information = 'time consumption:{}    epoch:{}/{}   '.format(et, epoch + 1, args.epochs, len(data))
    for key, value in dict_log.items():
        loss_info = "{}(avg):{:.3f} ".format(key, value.avg)
        print_information = print_information + loss_info
        tensorboard.add_scalar(key, value.val, epoch)
    print(print_information)
    print('--------------------------Ending evaluating at epoch:{}--------------------------'.format(epoch + 1))
    return dict_log['acc'].avg, dict_log['loss'].avg
