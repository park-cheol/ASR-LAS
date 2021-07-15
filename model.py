import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from utils import *
###################
# Encoder
###################
class EncoderRNN(nn.Module):

    def __int__(self, input_size, hidden_size, n_layers=1, dropout_p=0,
                bidirectional=False, rnn_cell='gru', variable_lengths=False):

        super(EncoderRNN, self).__int__()

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.variable_lengths = variable_lengths

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        outputs_channel = 32

        # todo kerenl size 등 인자들이 어떻게 정해지는 지
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, outputs_channel, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(outputs_channel),
            nn.Hardtanh(0, 20, inplace=True), # paper) min(max(x, 0), 20)
            nn.Conv2d(outputs_channel, outputs_channel, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(outputs_channel),
            nn.Hardtanh(0, 20, inplace=True)
        ))

        # size : (input_size + 2 * pad - filter) / stride + 1
        rnn_input_dims = int(math.floor(input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_dims = int(math.floor(rnn_input_dims + 2 * 10 - 21) / 2 + 1)
        rnn_input_dims *= outputs_channel

        self.rnn = self.rnn_cell(rnn_input_dims, self.hidden_size, self.n_layers, dropout=self.dropout_p,
                                 bidirectional=self.bidirectional)

    def forward(self, input, input_lengths=None):
        """
        :param input: Spectrogram shape(B, 1, D, T)
        :param input_lengths: zero-pad 적용 되지 않은 inputs sequence length
        """

        output_lengths = self.get_seq_lens(input_lengths)

        x = input # (B, 1, D, T)
        x, _ = self.conv(x, output_lengths) # (B, C, D, T) C: output_channel

        x_size = x.size()
        x = x.view(x_size[0], x_size[1] * x_size[2], x_size[3]) # (B, C * D, T)
        # rnn 에 넣기 위한 reshape
        x = x.permute(0, 2, 1).contiguous() # (B, T, D)

        total_length = x_size[3] # T

        # 패딩된 문장을 패킹(패딩은 연산 안들어가게)
        # packed: B * T, E
        x = nn.utils.rnn.pack_padded_sequence(x,
                                              output_lengths.cpu(), # 각각 batch 요소들의 sequence length 의 list
                                              batch_first=True,
                                              enforce_sorted=False) # True이면 감소하는 방향으로 정렬
        x, h_state = self.rnn(x)
        # 다시 패킹된 문장을 pad
        # unpacked: B, T, H
        x, _ = nn.utils.rnn.pad_packed_sequence(x,
                                                batch_first=True,
                                                total_length=total_length)

        return x, h_state

    def get_seq_lens(self, input_length):
        seq_len = input_length

        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)

        return seq_len.int()


#####################
# Attention
#####################
class Attention(nn.Module):
    """
    Location-based
    자세한것:
    https://arxiv.org/pdf/1506.07503.pdf
    https://arxiv.org/pdf/1508.01211.pdf
    """
    def __init__(self, dec_dim, enc_dim, conv_dim, attn_dim, smoothing=False):
        super(Attention, self).__init__()
        self.dec_dim = dec_dim
        self.enc_dim = enc_dim
        self.conv_dim = conv_dim
        self.attn_dim = attn_dim
        self.smoothing = smoothing
        self.conv = nn.Conv1d(in_channels=1, out_channels=self.attn_dim, kernel_size=3, padding=1)

        self.W = nn.Linear(self.dec_dim, self.attn_dim, bias=False)
        self.V = nn.Linear(self.enc_dim, self.attn_dim, bias=False)

        self.fc = nn.Linear(attn_dim, 1, bias=True)
        self.b = nn.Parameter(torch.randn(attn_dim))

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.mask = None

    def set_mask(self, mask):
        """
        mask 지정
        """
        self.mask = mask

    def forward(self, queries, values, last_attn):
        """
        :param queries: Decoder hidden state (s_i), shape=(B, 1, dec_D)
        :param values: Encoder output (h_i), shape=(B, enc_T, enc_D)
        :param last_attn: 이전 step 의 weight Attention, shape=(batch, enc_T)
        """
        # todo 인자들 확인
        batch_size = queries.size(0)
        dec_feat_dim = queries.size(2)
        enc_feat_dim = values.size(1)

        conv_attn = torch.transpose(self.conv(last_attn.unsqueeze(dim=1)), 1, 2) # [B, enc_T, conv_D]
        # todo print
        # paper 내용참고
        score = self.fc(self.tanh(
            self.W(queries) + self.V(values) + conv_attn + self.b
        )).squeeze(dim=-1) # [B, enc_T]
        # todo print

        if self.mask is not None:
            score.data.masked_fill_(self.mask, -float('inf'))

        # attn_weight : (B, enc_T)
        if self.smoothing:
            score = torch.sigmoid(score)
            attn_weight = torch.div(score, score.sum(dim=-1).unsqueeze(dim=-1))
            # div(x,y) = x/y
        else:
            attn_weight = self.softmax(score)

        # (B, 1, enc_T) * (B, enc_T, enc_D) -> (B, 1, enc_D)
        context = torch.bmm(attn_weight.unsqueeze(dim=-1), values)
        # C_i = context

        return context, attn_weight

#####################
# Decoder
#####################
# todo Decoder는 다 다시 검토
class DecoderRNN(nn.Module):

    def __init__(self, vocab_size, max_len, hidden_size, encoder_size,
                 sos_id, eos_id,
                 n_layers=1, rnn_cell='gru',
                 bidirectional_encoder=False, bidirectional_decoder=False,
                 dropout_p=0, use_attention=True):
        super(DecoderRNN, self).__init__()

        self.output_size = vocab_size
        self.vocab_size = vocab_size # todo print this len(char2index)
        self.hidden_size = hidden_size
        self.bidirectional_encoder = bidirectional_encoder
        self.bidirectional_decoder = bidirectional_decoder
        self.encoder_output_size = encoder_size * 2 if self.bidirectional_encoder else encoder_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.init_input = None
        self.rnn = self.rnn_cell(self.hidden_size + self.encoder_output_size, self.hidden_size,
                                 self.n_layers, batch_first=True, dropout=dropout_p,
                                 bidirectional=self.bidirectional_decoder)
        # S_i = RNN(s_i-1, y_i-1, c_i-1)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.input_dropout = nn.Dropout(self.dropout_p)

        self.attention = Attention(dec_dim=self.hidden_size, enc_dim=self.encoder_output_size,
                                   conv_dim=1, attn_dim=self.hidden_size)

        self.fc = nn.Linear(self.hidden_size + self.encoder_output_size, self.output_size)


    # 확인 절차
    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        # attention 사용 할꺼면 encoder output 있는지
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        batch_size = encoder_outputs.size(0)

        # input이 없다면 지정
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length


    # todo 어떻게 흘러가는지 제대로 파악
    def forward_step(self, input, hidden, encoder_outputs, context, attn_w, function):
        # todo print argument
        batch_size = input.size(0)
        dec_len = input.size(1)
        enc_len = encoder_outputs.size(1)
        enc_dim = encoder_outputs.size(2)
        embedded = self.embedding(input) # (B, dec_T, voc_D) -> (B, dec_T, dec_D)
        embedded = self.input_dropout(embedded)

        y_all = []
        attn_w_all = []

        for i in range(embedded.size(1)):
            embedded_inputs = embedded[:, i, :] # (B, dec_D)

            rnn_input = torch.cat([embedded_inputs, context], dim=1) # (B, dec_D + enc_D)
            rnn_input = rnn_input.unsqueeze(1)
            output, hidden = self.rnn(rnn_input, hidden) # (B, 1, dec_D)

            # queries, values, last_attn
            context, attn_w = self.attention(output, encoder_outputs, attn_w)
            # C_i=[B, 1, enc_D], a_i=[B, enc_T]
            attn_w_all.append(attn_w)

            context = context.squeeze(1)
            output = output.squeeze(1) # [B, 1, dec_D] -> [B, dec_D]
            context = self.input_dropout(context)
            output = self.input_dropout(output)
            output = torch.cat((output, context), dim=1) # [B, dec_D + enc_D]

            pred = function(self.fc(output), dim=-1)
            y_all.append(pred)

        if embedded.size(1) != 1:
            y_all = torch.stack(y_all, dim=1) # (B, dec_T, out_D)
            # torch.stack: cat가 다르게 차원을 확장하여 Tensor 쌓음
            # e.g) [M, N, K] satack [N, N, K] -> [M, 2, N, K]
            attn_w_all = torch.stack(attn_w_all, dim=1)  # (B, dec_T, enc_T)
        else:
            y_all = y_all[0].unsqueeze(1) # (B, 1, out_D)
            attn_w_all = attn_w_all[0] # [B, 1, enc_T]

        return y_all, hidden, context, attn_w_all

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                function=F.log_softmax, teacher_forcing_ratio=0):
        """
        :param inputs: [B, dec_T]
        :param encoder_hidden: Encoder last hidden states
        :param encoder_outputs: Encdoer output, [B, enc_T, enc_D]
        """
        # 지정한 확률로 teacher forcing 사용
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if teacher_forcing_ratio != 0:
            inputs, batch_size, max_len = _validate_args(inputs, encoder_hidden, encoder_outputs,
                                                         function, teacher_forcing_ratio)
        else:
            batch_size = encoder_outputs.size(0)
            inputs= torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            inputs = inputs.cuda()
            max_length = self.max_length

        decoder_hidden = None
        context = encoder_outputs.new_zeros(batch_size, encoder_outputs.size(2))  # (B, D)
        attn_w = encoder_outputs.new_zeros(batch_size, encoder_outputs.size(1))  # (B, T)

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output):
            decoder_outputs.append(step_output)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, context, attn_w = self.forward_step(decoder_input,
                                                                                decoder_hidden,
                                                                                encoder_outputs,
                                                                                context,
                                                                                attn_w,
                                                                                function=function)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                decode(di, step_output)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_hidden, context, attn_w = self.forward_step(decoder_input,
                                                                                    decoder_hidden,
                                                                                    encoder_outputs,
                                                                                    context,
                                                                                    attn_w,
                                                                                    function=function)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output)
                decoder_input = symbols

        return decoder_outputs




#################
# Seq2Seq
#################
# todo 이것도 다시 검토
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, decode_function=F.log_softmax):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function

    def flatten_parameters(self):
        pass

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0):

        self.encoder.rnn.flatten_parameters()
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)

        self.decoder.rnn.flatten_parameters()
        decoder_output = self.decoder(inputs=target_variable,
                                      encoder_hidden=None,
                                      encoder_outputs=encoder_outputs,
                                      function=self.decode_function,
                                      teacher_forcing_ratio=teacher_forcing_ratio)

        return decoder_output

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp




