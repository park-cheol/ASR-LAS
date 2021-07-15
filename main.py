import os
import json
import math
import random
import argparse
import numpy as np
import warnings
import os

from tqdm import tqdm


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import Levenshtein as Lev
# Levenshtein distance: 두 시퀀스 간의 차이를 측정하기 위한 문자열 메트릭

from Dataset import *
from model import EncoderRNN, DecoderRNN, Seq2Seq
from utils import *

parser = argparse.ArgumentParser
parser.add_argument('--model-name', type=str, default='LAS')
# Dataset
parser.add_argument('--train-file', type=str,
                    help='data list about train dataset', default='data/train_ClovaCall.json')
parser.add_argument('--test-file-list', nargs='*',
                    help='data list about test dataset', default='data/test_ClovCall.json')
parser.add_argument('--labels-path', default='data/kor_syllable.json', help='Contains large characters over korean')
parser.add_argument('--dataset-path', default='data', help='Target dataset path')
# Hyperparameters
parser.add_argument('--rnn-type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--encoder-layers', type=int, default=3, help='number of layers of model (default: 3)')
parser.add_argument('--encoder-size', type=int, default=512, help='hidden size of model (default: 512)')
parser.add_argument('--decoder-layers', type=int, default=2, help='number of pyramidal layers (default: 2)')
parser.add_argument('--decoder-size', type=int, default=512, help='hidden size of model (default: 512)')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate in training (default: 0.3)')
parser.add_argument('--bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size in training (default: 32)')
parser.add_argument('--num-workers', type=int, default=4, help='Number of workers in dataset loader (default: 4)')
parser.add_argument('--num-gpu', type=int, default=1, help='Number of gpus (default: 1)')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--epochs', type=int, default=100, help='Number of max epochs in training (default: 100)')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing learning rate every epoch')
parser.add_argument('--teacher-forcing', type=float, default=1.0,
                    help='Teacher forcing ratio in decoder (default: 1.0)')
parser.add_argument('--max-len', type=int, default=80, help='Maximum characters of sentence (default: 80)')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
# Audio Config
parser.add_argument('--sample-rate', default=16000, type=int, help='Sampling Rate')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram')
# System
parser.add_argument('--print-freq', default=1, type=int)
parser.add_argument('--resume', default=None, type=str, metavar='PATH' )
parser.add_argument('--save-folder', default='saved_models', help='Location to save epoch models')
parser.add_argument('--log-path', default='log/', help='path to predict log about valid and test dataset')
parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
parser.add_argument('--mode', type=str, default='train', help='Train or Test')
parser.add_argument('--finetune', dest='finetune', action='store_true', default=False,
                    help='Finetune the model after load model')


char2index = dict()
index2char = dict()
SOS_token = 0
EOS_token = 0
PAD_token = 0

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True  # TODO 원리
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count() # node: server(기계)라고 생각

    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global char2index
    global index2char
    global SOS_token
    global EOS_token
    global PAD_token

    args.gpu = gpu

    char2index, index2char = load_label_json(args.labels_path) # data/kor_syllable.json
    SOS_token = char2index['<s>'] # todo char2index index2char 찍어보기
    EOS_token = char2index['</s>']
    PAD_token = char2index['_']

    # audio 설정
    audio_conf = dict(sample_rate=args.sample_rate, # 16,000
                      window_size=args.window_size, # .02
                      window_stride=args.window_stride) # .01

    batch_size = args.batch_size * args.num_gpu # 32 * 1

    # Train dataset/ loader
    trainData_list = []
    with open(args.train_file, 'r', encoding='utf-8') as f:
        trainData_list = json.load(f) # data/train_ClovaCall.json
    # todo print

    if args.num_gpu != 1: # Multi GPU라면
        last_batch = len(trainData_list) % batch_size # 마지막 찌꺼기 batch(batch_size 보다 작음)
        if last_batch != 0 and last_batch < args.num_gpu: # 찌꺼기가 있고 사용할 gpu 수보다 작을시
            trainData_list = trainData_list[:-last_batch] # 삭제

    train_dataset_path = os.path.join(args.dataset_path, "wavs_train")
    train_dataset = SpectrogramDataset(audio_conf=audio_conf,
                                       dataset_path=train_dataset_path,
                                       data_list=trainData_list, # 파일명, text, speaker id
                                       char2index=char2index, sos_id=SOS_token, eos_id=EOS_token,
                                       normalize=True)

    train_sampler = BucketingSampler(train_dataset, batch_size=batch_size)
    train_loader = AudioDataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=train_sampler)

    # Test dataset/ loader
    testLoader_dict = {} # todo 왜 loader_dict이 있는지
    test_dataset_path = os.path.join(args.dataset_path, "wavs_test")
    for test_file in args.test_file_list:
        testData_list = []
        with open(test_file, 'r', encoding='utf-8') as f:
            testData_list = json.load(f)

        test_dataset = SpectrogramDataset(audio_conf=audio_conf,
                                          dataset_path=test_dataset_path,
                                          data_list=testData_list,
                                          char2index=char2index, sos_id=SOS_token, eos_id=EOS_token,
                                          normalize=True)
        testLoader_dict[test_file] = AudioDataLoader(test_dataset, batch_size=1, num_workers=args.num_workers)


    # Model
    input_size = int(math.floor((args.sample_rate * args.window_size) / 2) + 1)
    enc = EncoderRNN(input_size, args.encoder_size, n_layers=args.encoder_layers,
                     dropout_p=args.dropout, bidirectional=args.bidirectional,
                     rnn_cell=args.rnn_type, variable_lengths=False)

    dec = DecoderRNN(len(char2index), args.max_len, args.decoder_size, args.encoder_size,
                     SOS_token, EOS_token,
                     n_layers=args.decoder_layers, rnn_cell=args.rnn_type,
                     dropout_p=args.dropout, bidirectional_encoder=args.bidirectional)

    model = Seq2Seq(enc, dec)
    print("Number of parameters: %d" % Seq2Seq.get_param_size(model))

    if args.num_gpu != 1:
        model = nn.DataParallel(model).cuda(args.gpu)
    else:
        model = model.cuda(args.gpu)

    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))
    else:
        print("No saved model")

    # Optimizer / Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(reduction='mean').cuda(args.gpu)

    # Test
    if args.mode != "train":
        for test_file in args.test_file_list:
            test_loader = testLoader_dict[test_file]
            test_loss, test_cer, transcripts_list = evaluate(model, test_loader, criterion, args, save_output=True)

            for line in transcripts_list:
                print(line)

            print("Test {} CER : {}".format(test_file, test_cer))

    # Train
    else:
        best_cer = 1e10
        begin_epoch = 0

        for epoch in range(begin_epoch, args.epochs):
            train_loss, train_cer = train(model, train_loader, criterion, optimizer, args, epoch, train_sampler,
                                          args.max_norm, args.teacher_forcing)

            cer_list = []
            for test_file in args.test_file_list:
                test_loader = testLoader_dict[test_file]
                test_loss, test_cer, _ = evaluate(model, test_loader, criterion, args, save_output=False)
                test_log = 'Test({name}) Summary Epoch: [{0}]\tAverage Loss {loss:.3f}\tAverage CER {cer:.3f}\t'.format(
                    epoch + 1, name=test_file, loss=test_loss, cer=test_cer)
                print(test_log)

                cer_list.append(test_cer)

            if best_cer > cer_list[0]:
                print("Found better validated model)
                torch.save(model.state_dict(), "saved_mdoels/model_%d.pth" (epoch + 1))
                best_cer = cer_list[0]

            print("Shuffling batches...")
            train_sampler.shuffle(epoch)

            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / args.learning_anneal
            print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))


def train(model, data_loader, criterion, optimizer, args, epoch, train_sampler, max_norm=400,
              teacher_forcing_ratio=1):
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0

    model.train()
    for i, (data) in enumerate(data_loader):
        feats, scripts, feat_lengths, script_lengths = data

        optimizer.zero_grad()

        feats = feats.cuda(args.gpu)
        scripts = scripts.cuda(args.gpu)
        feat_lengths = feat_lengths.cuda(args.gpu)

        src_len = scripts.size(1)
        target = scripts[:, 1:]

        logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=teacher_forcing_ratio)

        logit = torch.stack(logit, dim=1).cuda(args.gpu)
        y_hat = logit.max(-1)[1]

        loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
        total_loss += loss.item()
        total_num += sum(feat_lengths).item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        dist, length, _ = get_distance(target, y_hat)
        total_dist += dist
        total_length += length
        cer = float(dist / length) * 100

        total_sent_num += target.size(0)

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}\t'
                  'Cer {cer:.4f}'.format(
                (epoch + 1), (i + 1), len(train_sampler), loss=loss, cer=cer))

    return total_loss / total_num, (total_dist / total_length) * 100


def evaluate(model, data_loader, criterion, args, save_output=False):
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0
    transcripts_list = []

    model.eval()
    with torch.no_grad():
        for i, (data) in tqdm(enumerate(data_loader), total=len(data_loader)):
            feats, scripts, feat_lengths, script_lengths = data

            feats = feats.cuda(args.gpu)
            scripts = scripts.cuda(args.gpu)
            feat_lengths = feat_lengths.cuda(args.gpu)

            src_len = scripts.size(1)
            target = scripts[:, 1:]

            logit = model(feats, feat_lengths, None, teacher_forcing_ratio=0.0)
            logit = torch.stack(logit, dim=1).cuda(args.gpu)
            y_hat = logit.max(-1)[1]

            logit = logit[:,:target.size(1),:] # cut over length to calculate loss
            loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
            total_loss += loss.item()
            total_num += sum(feat_lengths).item()

            dist, length, transcripts = get_distance(target, y_hat)
            cer = float(dist / length) * 100

            total_dist += dist
            total_length += length
            if save_output == True:
                transcripts_list += transcripts
            total_sent_num += target.size(0)


    aver_loss = total_loss / total_num
    aver_cer = float(total_dist / total_length) * 100
    return aver_loss, aver_cer, transcripts_list



def get_distance(ref_labels, hyp_labels):
    total_dist = 0
    total_length = 0
    transcripts = []
    for i in range(len(ref_labels)):
        ref = label_to_string(ref_labels[i])
        hyp = label_to_string(hyp_labels[i])

        transcripts.append('{hyp}\t{ref}'.format(hyp=hyp, ref=ref))

        dist, length = char_distance(ref, hyp)
        total_dist += dist
        total_length += length

    return total_dist, total_length, transcripts


def label_to_string(labels):
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            if i.item() == EOS_token:
                break
            sent += index2char[i.item()]
        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == EOS_token:
                    break
                sent += index2char[j.item()]
            sents.append(sent)

        return sents


def char_distance(ref, hyp):
    ref = ref.replace(' ', '')
    hyp = hyp.replace(' ', '')

    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', ''))

    return dist, length


if __name__ == "__main__":
    main()




















