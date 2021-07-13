"""Audio File Dataset / Dataloader"""
import os
import math
import numpy as np
import scipy.signal
import librosa

import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

def load_audio(path):
    sound = np.memmap(path, dtype='h', mode='r')
    # np.memmap: 디스크에 binary file로 저장된 array 에 대한 memory-map 생성 / 'h': short = int16 int 범위일치
    # 원래 하면 (220, ) 에서 'h' 추가 시 (110, )
    sound = sound.astype('float32') / 32767 # 내 생각) 32767: int 범위 (-32768, 32767)

    assert len(sound) # 있어야함

    sound = torch.from_numpy(sound).view(-1, 1).type(torch.FloatTensor)
    sound = sound.numpy() # todo 왜 Tensor로 가서 view하고 numpy로 다시 오는 지??

    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1) # 여러 채널 평균
            # numpy 에서 axis=1
            # [[3,2,1],
            #  [4,5,6]] ==> [2., 5.] 으로 나옴
    # todo 코드 상 보면 딱히 변한게 없어 보이는데 print 해볼 것
    return sound

class SpectrogramDataset(Dataset):

    def __init__(self, audio_conf, dataset_path, data_list, char2index, sos_id, eos_id, normalize=False):
        """
        Dataset 은 wav_name, transcripts, speaker_id 가 dictionary 로 담겨져있는 list으로부터 data 를 load
        :param audio_conf: Sample rate, window, window size나 length, stride 설정
        :param data_list: dictionary . key: 'wav', 'text', 'speaker_id'
        :param char2index: character 에서 index 로 mapping 된 Dictionary
        :param normalize: Normalized by instance-wise standardazation
        """
        super(SpectrogramDataset, self).__init__()
        self.audio_conf = audio_conf
        self.data_list = data_list
        self.size = len(self.data_list)
        self.char2index = char2index
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.PAD = 0
        self.normalize = normalize
        self.dataset_path = dataset_path

    def __getitem__(self, index):
        wav_name = self.data_list[index]['wav']
        audio_path = os.path.join(self.dataset_path, wav_name)

        transcript = self.data_list[index]['text']

        spect = self.parse_audio(audio_path)
        transcript = self.parse_transcript(transcript)

        return spect, transcript

    def parse_audio(self, audio_path):
        y = load_audio(audio_path)

        n_fft = int(self.audio_conf['sample_rate'] * self.audio_conf['window_size']) # deault:320
        window_size = n_fft
        stride_size = int(self.audio_conf['sample_rate'] * self.audio_conf['window_stride']) # default: 160

        # STFT (Short Time Fourier Transform)
        # 음원을 특정 시간 주기(window, 또는 fream)로 쪼갠 뒤, 해당 주기별로 FFT 진행하면 해당 주기만큼 주파수 분석 그래프 얻음
        # 이를 다시 시간 단위로 배열하면 3차원 칼라맵 나옴
        D = librosa.stft(y, n_fft=n_fft, hop_length=stride_size, win_length=window_size, window=scipy.signal.windows.hamming)
        # https://kaen2891.tistory.com/39
        """
        n_fft: 음성의 길이를 얼마 만큼으로 자를 것인가(window)
        e.g) 16kHz에 n_fft=512 이면 1개의 n_fft는 16000/512 = 약 32 총 음성 데이터 길이가 500이면 32씩 1칸으로자름
        보유한 음성 데이터의 sampling rate와 n_fft 사용시 아래 공식사용
        n_Fft = int(sampling rate * window_length(size))
        hop_length: 음성을 얼만큼 겹친 상태로 잘라서 칸으로 나타낼 것인지?
        즉, window_length - stride 라고 보면 됌
        why: 초기신호를 window_length 만큼 쪼개기 때문에 당연히 freq resolution 악화 반대로 window_ length를
        늘리면 time resolution 이 악화 되는 trage off 관계를 가짐
        이를 조금 개선 하기 위해 overlap 을 적용
        """
        spect, phase = librosa.magphase(D) # todo print

        # S = log(S+1) todo 내 생각으로는 log scale로 바꿔주고 또한 zero point를 0으로 옮겨주기 위해서
        spect = np.log1p(spect)
        if self.normalize:
            mean = np.mean(spect)
            std = np.std(spect)
            spect -= mean
            spect /= std
            # 이는 원래 normalize 식을 적용

        spect = torch.FloatTensor(spect)

        return spect

    def parse_transcript(self, transcript):
        transcript = list(filter(None, [self.char2index.get(x) for x in list(transcript)]))
        # filter(조건, 순횐 가능한 데이터): char2index 의 key 에 없는 것(None) 다 삭제 해버림
        transcript = [self.sos_id] + transcript + [self.eos_id]

        return transcript

    def __len__(self):
        return self.size



def _collate_fn(batch):
    # todo print(batch(
    def seq_length_(p):
        return p[0].size(1) # todo print
    def target_length_(p):
        return len(p[1]) # todo 왜 len

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    # todo print
    # e.g) Tensor([3,2,1]) 이여도 for s in Tensor ==> batch 수로 순환
    seq_lengths = [s[0].size(1) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_size = max(seq_lengths)
    max_target_size = max(target_lengths)

    feat_size = batch[0][0].size(0)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, 1, feat_size, max_seq_size)
    targets = torch.zeros(batch_size, max_target_size).to(torch.long)

    # todo shape를 알아야 확인 가능
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        seqs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    seq_lengths = torch.IntTensor(seq_lengths)

    return seqs, targets, seq_lengths, target_lengths

class AudioDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__()
        self.collate_fn = _collate_fn()

class BucketingSampler(Sampler):

    def __init__(self, data_source, batch_size=1):
        """
        비슷한 크기의 samples과 함께 순서대로 배치
        """
        super(BucketingSampler, self).__init__()
        self.data_source = data_source
        ids = list(range(0, len(data_source))) # idx 만듬
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
        # batch_size 만큼 쪼개짐
        # e.g) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] 와 batch_size=3
        # -> [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19]]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids
            # todo yield 다시 공부

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)








































