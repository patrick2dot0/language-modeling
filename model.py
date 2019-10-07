import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils import weight_norm


class RNN(nn.Module):

    def __init__(self, vocab_size, embed_size, time_step, hidden_size,
                 dropout_rate, bos_idx, eos_idx, pad_idx):
        super(RNN, self).__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        # input
        self.embedding = nn.Embedding(vocab_size, embed_size,
                                      padding_idx=pad_idx)
        # RNN
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        # output
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq, length):
        # process input
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        input_seq = input_seq[sorted_idx]
        embedded_input = self.embedding(input_seq)
        drop_input = F.dropout(embedded_input, p=self.dropout_rate,
                               training=self.training)

        # RNN forward
        pack_input = pack_padded_sequence(drop_input, sorted_len + 1,
                                          batch_first=True)
        pack_output, _ = self.rnn(pack_input)
        output, _ = pad_packed_sequence(pack_output, batch_first=True)
        _, reversed_idx = torch.sort(sorted_idx)
        output = output[reversed_idx]

        # project output
        drop_output = F.dropout(output, p=self.dropout_rate,
                                training=self.training)
        batch_size, seq_len, hidden_size = drop_output.size()
        logit = self.output(drop_output.view(-1, hidden_size))
        logp = F.log_softmax(logit, dim=-1)
        logp = logp.view(batch_size, seq_len, -1)

        return logp


class LSTMEncoder(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, z_dim, pad_idx):
        super(LSTMEncoder, self).__init__()
        # input
        self.embedding = nn.Embedding(vocab_size, embed_size,
                                      padding_idx=pad_idx)
        # RNN
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        # output
        self.output = nn.Linear(hidden_size * 2, z_dim * 2)

    def forward(self, input_seq, length):
        # embed input
        embedded_input = self.embedding(input_seq)

        # RNN forward
        pack_input = pack_padded_sequence(embedded_input, length,
                                          batch_first=True)
        _, (h, c) = self.rnn(pack_input)

        # produce mu and logvar
        hidden = torch.cat([h, c], dim=-1).squeeze(0)
        mu, logvar = torch.chunk(self.output(hidden), 2, dim=-1)

        return mu, logvar


class RNNVAE(nn.Module):

    def __init__(self, vocab_size, embed_size, time_step, hidden_size, z_dim,
                 dropout_rate, bos_idx, eos_idx, pad_idx):
        super(RNNVAE, self).__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        # encoder
        self.encoder = LSTMEncoder(vocab_size, embed_size,
                                   hidden_size, z_dim, pad_idx)
        # decoder
        self.embedding = nn.Embedding(vocab_size, embed_size,
                                      padding_idx=pad_idx)
        self.init_h = nn.Linear(z_dim, hidden_size)
        self.init_c = nn.Linear(z_dim, hidden_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        # output
        self.output = nn.Linear(hidden_size, vocab_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + std * eps

    def forward(self, enc_input, dec_input, length):
        # process input
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        enc_input = enc_input[sorted_idx]
        dec_input = dec_input[sorted_idx]

        # encode
        mu, logvar = self.encoder(enc_input, sorted_len)
        z = self.reparameterize(mu, logvar)

        # decode
        embedded_input = self.embedding(dec_input)
        drop_input = F.dropout(embedded_input, p=self.dropout_rate,
                               training=self.training)
        pack_input = pack_padded_sequence(drop_input, sorted_len + 1,
                                          batch_first=True)
        h_0, c_0 = self.init_h(z), self.init_c(z)
        hidden = (h_0.unsqueeze(0), c_0.unsqueeze(0))
        pack_output, _ = self.rnn(pack_input, hidden)
        output, _ = pad_packed_sequence(pack_output, batch_first=True)
        _, reversed_idx = torch.sort(sorted_idx)
        output = output[reversed_idx]

        # project output
        batch_size, seq_len, hidden_size = output.size()
        logit = self.output(output.view(-1, hidden_size))
        logp = F.log_softmax(logit, dim=-1)
        logp = logp.view(batch_size, seq_len, -1)

        return logp, mu, logvar

    def inference(self, z):
        # set device
        tensor = torch.LongTensor
        if torch.cuda.is_available():
            tensor = torch.cuda.LongTensor

        # initialize hidden state
        batch_size = z.size(0)
        h_0, c_0 = self.init_h(z), self.init_c(z)
        hidden = (h_0.unsqueeze(0), c_0.unsqueeze(0))

        # RNN forward
        symbol = tensor(batch_size, self.time_step + 1).fill_(self.pad_idx)
        for t in range(self.time_step + 1):
            if t == 0:
                input_seq = tensor(batch_size, 1).fill_(self.bos_idx)
            embedded_input = self.embedding(input_seq)
            output, hidden = self.rnn(embedded_input, hidden)
            logit = self.output(output)
            _, sample = torch.topk(logit, 1, dim=-1)
            input_seq = sample.squeeze(-1)
            symbol[:, t] = input_seq.squeeze(-1)

        return symbol


class pBLSTMLayer(nn.Module):

    def __init__(self, input_dim, hidden_size, factor=2):
        super(pBLSTMLayer, self).__init__()
        self.factor = factor
        self.rnn = nn.LSTM(input_dim * factor, hidden_size,
                           batch_first=True, bidirectional=True)

    def forward(self, input_seq):
        batch_size, time_step, input_dim = input_seq.size()
        input_seq = input_seq.contiguous().view(batch_size,
                                                time_step // self.factor,
                                                input_dim * self.factor)
        output, _ = self.rnn(input_seq)

        return output


class pBLSTMEncoder(nn.Module):

    def __init__(self, vocab_size, embed_size, time_step,
                 hidden_size, z_dim, pad_idx):
        super(pBLSTMEncoder, self).__init__()
        # input
        self.embedding = nn.Embedding(vocab_size, embed_size,
                                      padding_idx=pad_idx)
        # pBLSTM
        self.pBLSTM = nn.Sequential(
            pBLSTMLayer(embed_size + z_dim, hidden_size),
            pBLSTMLayer(hidden_size * 2, hidden_size),
            pBLSTMLayer(hidden_size * 2, hidden_size)
        )
        # ouput
        self.output = nn.Sequential(
            nn.Linear(hidden_size * 2 * time_step // 8, hidden_size * 2),
            nn.Linear(hidden_size * 2, z_dim * 2)
        )

    def forward(self, input_seq, z):
        # process input
        embedded_input = self.embedding(input_seq)
        z_expand = z.unsqueeze(1).expand(embedded_input.size(0),
                                         embedded_input.size(1), z.size(-1))
        new_input = torch.cat([embedded_input, z_expand], dim=-1)

        # pBLSTM forward
        output = self.pBLSTM(new_input)

        # produce mu and logvar
        output = output.contiguous().view(output.size(0), -1)
        mu, logvar = torch.chunk(self.output(output), 2, dim=-1)

        return mu, logvar


class HRNNVAE(nn.Module):

    def __init__(self, vocab_size, embed_size, time_step, hidden_size, z_dim,
                 dropout_rate, bos_idx, eos_idx, pad_idx):
        super(HRNNVAE, self).__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        # encoder
        self.encoder_zg = LSTMEncoder(vocab_size, embed_size,
                                      hidden_size, z_dim, pad_idx)
        self.encoder_zl = pBLSTMEncoder(vocab_size, embed_size, time_step,
                                        hidden_size, z_dim, pad_idx)
        # decoder
        self.embedding = nn.Embedding(vocab_size, embed_size,
                                      padding_idx=pad_idx)
        self.rnn = nn.LSTM(embed_size + z_dim * 2,
                           hidden_size, batch_first=True)
        # output
        self.output = nn.Linear(hidden_size, vocab_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + std * eps

    def forward(self, enc_input, dec_input, length):
        # process input
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        enc_input = enc_input[sorted_idx]
        dec_input = dec_input[sorted_idx]

        # encoder 1 phrase
        mu_zg, logvar_zg = self.encoder_zg(enc_input, sorted_len)
        zg = self.reparameterize(mu_zg, logvar_zg)

        # encoder 2 phrase
        mu_zl, logvar_zl = self.encoder_zl(enc_input, zg)
        zl = self.reparameterize(mu_zl, logvar_zl)

        # decoder
        embedded_input = self.embedding(dec_input)
        drop_input = F.dropout(embedded_input, p=self.dropout_rate,
                               training=self.training)
        z = torch.cat([zg, zl], dim=-1)
        z_expand = z.unsqueeze(1).expand(embedded_input.size(0),
                                         embedded_input.size(1), z.size(-1))
        new_input = torch.cat([drop_input, z_expand], dim=-1)
        pack_input = pack_padded_sequence(new_input, sorted_len + 1,
                                          batch_first=True)
        packed_output, _ = self.rnn(pack_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        _, reversed_idx = torch.sort(sorted_idx)
        output = output[reversed_idx]

        # output
        batch_size, seq_len, hidden_size = output.size()
        logit = self.output(output.view(-1, hidden_size))
        logp = F.log_softmax(logit, dim=-1)
        logp = logp.view(batch_size, seq_len, -1)

        return logp, mu_zg, logvar_zg, mu_zl, logvar_zl

    def inference(self, z):
        # set device
        tensor = torch.LongTensor
        if torch.cuda.is_available():
            tensor = torch.cuda.LongTensor

        # concatenate latent variable and initialize hidden state
        batch_size = z.size(0)
        hx = torch.zeros(batch_size, self.hidden_size, device=z.device)
        hidden = (hx.unsqueeze(0), hx.unsqueeze(0))

        # RNN forward
        symbol = tensor(batch_size, self.time_step + 1).fill_(self.pad_idx)
        for t in range(self.time_step + 1):
            if t == 0:
                input_seq = tensor(batch_size, 1).fill_(self.bos_idx)
            embedded_input = self.embedding(input_seq)
            new_input = torch.cat([embedded_input, z.unsqueeze(1)], dim=-1)
            output, hidden = self.rnn(new_input, hidden)
            logit = self.output(output)
            _, sample = torch.topk(logit, 1, dim=-1)
            input_seq = sample.squeeze(-1)
            symbol[:, t] = input_seq.squeeze(-1)

        return symbol


class SelfVAE(nn.Module):

    def __init__(self, vocab_size, embed_size, time_step, hidden_size, z_dim,
                 dropout_rate, bos_idx, eos_idx, pad_idx):
        super(SelfVAE, self).__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        # encoder
        self.enc_embedding = nn.Embedding(vocab_size, embed_size,
                                          padding_idx=self.pad_idx)
        self.enc_rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.latent = nn.Linear(hidden_size * 2, z_dim * 2)

        # decoder
        self.dec_embedding = nn.Embedding(vocab_size, embed_size,
                                          padding_idx=self.pad_idx)
        self.attn = nn.Linear(hidden_size + embed_size, self.time_step)
        self.combine = nn.Linear(hidden_size + embed_size + z_dim * 2,
                                 hidden_size)
        self.dec_rnn = nn.LSTMCell(hidden_size, hidden_size)

        # variational inference
        self.pri = nn.Linear(hidden_size, z_dim * 2)
        self.inf = nn.Linear(hidden_size * 2, z_dim * 2)
        self.aux = nn.Linear(z_dim, hidden_size)

        # output
        self.output = nn.Linear(hidden_size, vocab_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + std * eps

    def gaussian_kld(self, mu_left, logvar_left, mu_right, logvar_right):
        """
        compute KL(N(mu_left, logvar_left) || N(mu_right, logvar_right))
        """
        gauss_klds = 0.5 * (logvar_right - logvar_left +
                            logvar_left.exp() / logvar_right.exp() +
                            (mu_left - mu_right).pow(2) / logvar_right.exp() -
                            1.)
        return torch.sum(gauss_klds, 1)

    def forward(self, enc_input, dec_input, length):
        # process input
        batch_size = enc_input.size(0)
        max_len = torch.max(length)
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        enc_input = enc_input[sorted_idx]
        dec_input = dec_input[sorted_idx]
        att_mask = enc_input==self.pad_idx

        # encode
        enc_embedded = self.enc_embedding(enc_input)
        enc_input = pack_padded_sequence(enc_embedded, sorted_len,
                                         batch_first=True)
        pack_output, (h, c) = self.enc_rnn(enc_input)
        enc_output, _ = pad_packed_sequence(pack_output, batch_first=True)
        if max_len != self.time_step:
            padding = torch.zeros([batch_size,
                                   self.time_step - max_len,
                                   self.hidden_size], device=enc_output.device)
            enc_output = torch.cat([enc_output, padding], dim=1)
        hidden = torch.cat([h, c], dim=-1).squeeze(0)
        mu, logvar = torch.chunk(self.latent(hidden), 2, dim=-1)
        z = self.reparameterize(mu, logvar)

        # decode
        dec_embedded = self.dec_embedding(dec_input)
        dec_input = F.dropout(dec_embedded, p=self.dropout_rate,
                              training=self.training)
        hx = torch.zeros(batch_size, self.hidden_size, device=z.device)
        hidden = (hx, hx)

        outputs, klds, aux_cs = [], [], []
        for t in range(max_len + 1):
            x_step = dec_input[:, t]
            mask = (length + 1 > t).float()

            # prior
            pri_mu, pri_logvar = torch.chunk(self.pri(hidden[0]), 2, dim=-1)

            # attention mechanism
            scale = self.attn(torch.cat([x_step, hidden[0]], dim=-1))
            scale = scale.masked_fill(att_mask, -math.inf)
            attn_weight = F.softmax(scale, dim=-1)
            context = torch.bmm(attn_weight.unsqueeze(1),
                                enc_output).squeeze(1)

            # inference
            inf_mu, inf_logvar = torch.chunk(
                self.inf(torch.cat([hidden[0], context], dim=-1)), 2, dim=-1)
            klds.append(self.gaussian_kld(inf_mu, inf_logvar,
                                          pri_mu, pri_logvar) * mask)
            if self.training:
                z_step = self.reparameterize(inf_mu, inf_logvar)
            else:
                z_step = pri_mu

            # auxiliary
            aux_mu = self.aux(z_step)
            aux_cs.append(
                torch.sum((context.detach() - aux_mu).pow(2), 1))

            # RNN forward
            input = self.combine(
                torch.cat([x_step, aux_mu, z_step, z], dim=-1))
            hidden = self.dec_rnn(input, hidden)
            outputs.append(hidden[0])
        output = torch.stack(outputs, dim=1)
        kld = torch.stack(klds, dim=1).sum()
        aux_loss = torch.stack(aux_cs, dim=1).sum()
        _, reversed_idx = torch.sort(sorted_idx)
        output = output[reversed_idx, :max_len + 1]

        # project output
        batch_size, seq_len, hidden_size = output.size()
        logit = self.output(output.view(-1, hidden_size))
        logp = F.log_softmax(logit, dim=-1)
        logp = logp.view(batch_size, seq_len, -1)

        return logp, mu, logvar, kld, aux_loss

    def inference(self, z):
        # set device
        tensor = torch.LongTensor
        if torch.cuda.is_available():
            tensor = torch.cuda.LongTensor

        # initialize hidden state
        batch_size = z.size(0)
        hx = torch.zeros(batch_size, self.hidden_size, device=z.device)
        hidden = (hx, hx)

        # RNN forward
        symbol = tensor(batch_size, self.time_step + 1).fill_(self.pad_idx)
        for t in range(self.time_step + 1):
            if t == 0:
                input_seq = tensor(batch_size).fill_(self.bos_idx)
            x_step = self.dec_embedding(input_seq)
            z_step, _ = torch.chunk(self.pri(hidden[0]), 2, dim=-1)
            aux_mu = self.aux(z_step)
            input = self.combine(
                torch.cat([x_step, aux_mu, z_step, z], dim=-1))
            hidden = self.dec_rnn(input, hidden)
            logit = self.output(hidden[0])
            _, sample = torch.topk(logit, 1, dim=-1)
            input_seq = sample.squeeze()
            symbol[:, t] = input_seq

        return symbol

    def get_att_weight(self, enc_input, dec_input, length):
        # process input
        batch_size = enc_input.size(0)
        max_len = torch.max(length)
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        enc_input = enc_input[sorted_idx]
        dec_input = dec_input[sorted_idx]
        att_mask = enc_input==self.pad_idx

        # encode
        enc_embedded = self.enc_embedding(enc_input)
        enc_input = pack_padded_sequence(enc_embedded, sorted_len,
                                         batch_first=True)
        pack_output, (h, c) = self.enc_rnn(enc_input)
        enc_output, _ = pad_packed_sequence(pack_output, batch_first=True)
        if max_len != self.time_step:
            padding = torch.zeros([batch_size,
                                   self.time_step - max_len,
                                   self.hidden_size], device=enc_output.device)
            enc_output = torch.cat([enc_output, padding], dim=1)
        hidden = torch.cat([h, c], dim=-1).squeeze(0)
        mu, logvar = torch.chunk(self.latent(hidden), 2, dim=-1)

        # repareameterize
        z = self.reparameterize(mu, logvar)

        # decode
        dec_embedded = self.dec_embedding(dec_input)
        hx = torch.zeros(batch_size, self.hidden_size, device=z.device)
        hidden = (hx, hx)

        att_weights = []
        for t in range(max_len + 1):
            x_step = dec_embedded[:, t]

            # attention mechanism
            score = self.attn(torch.cat([x_step, hidden[0]], dim=-1))
            score = score.masked_fill(att_mask, -math.inf)
            attn_weight = F.softmax(score, dim=-1)
            context = torch.bmm(attn_weight.unsqueeze(1),
                                enc_output).squeeze(1)
            att_weights.append(attn_weight[:, :max_len])

            # inference
            inf_mu, inf_logvar = torch.chunk(
                self.inf(torch.cat([hidden[0], context], dim=-1)), 2, dim=-1)
            z_step = self.reparameterize(inf_mu, inf_logvar)

            # auxiliary
            aux_mu = self.aux(z_step)

            # RNN forward
            input = self.combine(
                torch.cat([x_step, aux_mu, z_step, z], dim=-1))
            hidden = self.dec_rnn(input, hidden)
        att_weight = torch.stack(att_weights, dim=1)

        return att_weight


criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=0)
# negative log likelihood
def NLL(logp, target, length):
    target = target[:, :torch.max(length).item()].contiguous().view(-1)
    logp = logp[:, :torch.max(length).item(),:].contiguous().view(-1, logp.size(-1)) # logp = logp.view(-1, logp.size(-1))
    return criterion(logp, target)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.num_levels = len(num_channels)
        for i in range(self.num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        count = 0
        final_d = x
        for layer in self.network:
            final_d = layer(final_d)
            if count is 0:
                d = torch.unsqueeze(final_d,0) #size = (1,batch_size,hidden_states,length)
            elif count is (self.num_levels-1):
                break
                #last_d = torch.unsqueeze(temp_d,0) #size = (1,batch_size,embedding_size,length)
            else:
                d = torch.cat((d,torch.unsqueeze(final_d,0)), 0) #size = (count+1,batch_size,hidden_states,length)
            count = count+1
        return final_d, d
        #for layer in self.network:
        #    x=layer(x)
        #return x
        #return self.network(x)
    
    

class TCN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_channels, bos_idx, eos_idx, pad_idx, kernel_size=2, dropout=0.2, emb_dropout=0.2):
        super(TCN, self).__init__()
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.decoder = nn.Linear(embed_size, vocab_size)
        self.decoder.weight = self.encoder.weight
        self.drop = nn.Dropout(emb_dropout)
        self.init_weights()
        

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, length, target):
        # input has dimension (N, L_in), and emb has dimension (N, L_in, C_in)
        emb = self.drop(self.encoder(x))
        y, __ = self.tcn(emb.transpose(1, 2))
        o = self.decoder(y.transpose(1, 2))
        o = self.drop(o)
        
        logp = o.contiguous()
        NLL_loss = NLL(logp, target, length + 1)
        return logp, NLL_loss


from torch.autograd import Variable
    
def _reparameterized_sample(mean, std):
    """using std to sample"""
    eps = torch.cuda.FloatTensor(std.size()).normal_()
    eps = Variable(eps)
    return eps.mul(std).add_(mean)

    
def _kld_gauss(mean_1, std_1, mean_2, std_2, lengths):
    # size = (layers, batch size, time length, channels)
    # 1 represent the prior, 2 represent the variational distribution
    kld_element = 0
    """
    print('mean_1')
    print(mean_1.min(), mean_1.max())
    print('mean_2')
    print(mean_2.min(), mean_2.max())
    print('std_1')
    print(std_1.min(), std_1.max())
    print('std_2')
    print(std_2.min(), std_2.max())
    """
    for i, length in enumerate(lengths):
        length = length.long()
        if(length.item()==1):
            continue
        temp_mean_1 = mean_1[:,i, :(length-1),:]#.contiguous().view(-1)
        temp_mean_2 = mean_2[:,i, 1:length,:]#.contiguous().view(-1)
        temp_std_1 = std_1[:,i, :(length-1),:]#.contiguous().view(-1)
        temp_std_2 = std_2[:,i, 1:length,:]#.contiguous().view(-1)
        """Using std to compute KLD"""
        kld_element +=  (2 * torch.log(temp_std_2) - 2 * torch.log(temp_std_1) + (temp_std_1.pow(2) + (temp_mean_1 - temp_mean_2).pow(2)) / temp_std_2.pow(2) - 1).mean()
        #print(temp.item(), length,end='\t')
         
        
    return 0.5 * kld_element#torch.sum(kld_element)

    
class EncodeBlock(nn.Module):
    def __init__(self, n_inputs_d, n_inputs_z, n_outputs):
        super(EncodeBlock, self).__init__()
        #encoder
        self.n_inputs_z = n_inputs_z
        
        self.linear1 = nn.Linear(n_inputs_d, n_outputs)            
        self.linear3 = nn.Linear(n_outputs, n_outputs)
        
        self.enc = nn.Sequential(
            self.linear1,
            nn.ReLU())
        
        if n_inputs_z != 0:
            self.linear2 = nn.Linear(n_inputs_z, n_outputs)
            self.enc_z = nn.Sequential(
            self.linear2,
            nn.ReLU())
            
        self.enc_mean = nn.Linear(n_outputs, n_outputs)
        
        self.enc_std = nn.Sequential(
            self.linear3,
            nn.Softplus())

        self.init_weights()

    def init_weights(self):
        initrange = 0.01
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.fill_(0)
        if self.n_inputs_z != 0:
            self.linear2.weight.data.uniform_(-initrange, initrange)
            self.linear2.bias.data.fill_(0)
        self.linear3.weight.data.uniform_(-initrange, initrange)
        self.linear3.bias.data.fill_(0)
        self.enc_mean.weight.data.uniform_(-initrange, initrange)
        self.enc_mean.bias.data.fill_(0)
        #None

    def forward(self, d, z):
        e_d = self.enc(d)
        if self.n_inputs_z is 0:
            e = e_d
        else:
            e_z = self.enc_z(z)
            e = e_d + e_z
        mean = self.enc_mean(e)
        std = self.enc_std(e)
        std = torch.clamp(std, min=1e-3, max=5)
        return mean, std

    
class GenerativeNetwork(nn.Module):
    def __init__(self, num_inputs, num_channels):
        super(GenerativeNetwork, self).__init__()
        layers = []
        self.num_levels = len(num_channels)
        for i in range(self.num_levels):
            n_inputs_d = num_channels[self.num_levels - i - 1]
            n_inputs_z = 0 if i == 0 else num_channels[self.num_levels - i - 2]#num_inputs
            n_outputs = num_channels[self.num_levels - i - 1]#num_inputs
            layers += [EncodeBlock( n_inputs_d, n_inputs_z, n_outputs )]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, final_d, d):
        count = 0
        z = 0
        for layer in self.network:
            if count is 0:
                mean, std = layer(final_d.transpose(1,2), z) # size = [batch size, time length, channels]
                z = _reparameterized_sample(mean, std)
                all_mean = torch.unsqueeze(mean,0)
                all_std = torch.unsqueeze(std,0)
                all_z = torch.unsqueeze(z,0)
            else:
                mean, std = layer(d[count-1,:,:,:].transpose(1,2), z) # size = [batch size, time length, channels]
                z = _reparameterized_sample(mean, std)
                all_mean = torch.cat((all_mean,torch.unsqueeze(mean,0)), 0)
                all_std = torch.cat((all_std,torch.unsqueeze(std,0)), 0)
                all_z = torch.cat((all_z,torch.unsqueeze(z,0)), 0)
                
            count = count+1
            
        return all_z, all_mean, all_std
    
class InferenceNetwork(nn.Module):
    def __init__(self, num_inputs, num_channels):
        super(InferenceNetwork, self).__init__()
        layers = []
        self.num_levels = len(num_channels)
        for i in range(self.num_levels):
            n_inputs_d = num_channels[self.num_levels - i - 1]
            n_inputs_z = 0 if i == 0 else num_channels[self.num_levels - i - 2]#num_inputs
            n_outputs = num_channels[self.num_levels - i - 1]#num_inputs
            layers += [EncodeBlock( n_inputs_d, n_inputs_z, n_outputs )]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, final_d, d, prior_mean, prior_std):
        count = 0
        z = 0
        for layer in self.network:
            if count is 0:
                mean, std = layer(final_d.transpose(1,2), z) # size = [batch size, time length, channels]
                #std = 1/ ( std_.pow(-2) + prior_std[count,:,:,:].pow(-2) )
                #mean = torch.mul(std, torch.mul(mean_,std_.pow(-2)) + torch.mul(prior_mean[count,:,:,:], prior_std[count,:,:,:].pow(-2)) )
                z = _reparameterized_sample(mean, std)
                all_mean = torch.unsqueeze(mean,0)
                all_std = torch.unsqueeze(std,0)
                all_z = torch.unsqueeze(z,0)
            else:
                mean, std = layer(d[count-1,:,:,:].transpose(1,2), z) # size = [batch size, time length, channels]
                #std = 1/ ( std_.pow(-2) + prior_std[count,:,:,:].pow(-2) )
                #mean = torch.mul(std, torch.mul(mean_,std_.pow(-2)) + torch.mul(prior_mean[count,:,:,:], prior_std[count,:,:,:].pow(-2)) )
                z = _reparameterized_sample(mean, std)
                all_mean = torch.cat((all_mean,torch.unsqueeze(mean,0)), 0)
                all_std = torch.cat((all_std,torch.unsqueeze(std,0)), 0)
                all_z = torch.cat((all_z,torch.unsqueeze(z,0)), 0)
                
            count = count+1
        
        return all_mean, all_std
               
    
class STCN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_channels, bos_idx, eos_idx, pad_idx, kernel_size=2, dropout=0.2, emb_dropout=0.2):
        super(STCN, self).__init__()
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.num_levels = len(num_channels)
        self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.decoder = nn.Linear(sum(num_channels), vocab_size)
        self.generative= GenerativeNetwork(embed_size, num_channels)
        self.inference= InferenceNetwork(embed_size, num_channels)
        self.drop = nn.Dropout(emb_dropout)
        self.init_weights()
        

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        

    def forward(self, x, length, target):
        # input has dimension (N, L_in), and emb has dimension (N, L_in, C_in)
        emb = self.drop(self.encoder(x))
        final_d, d = self.tcn(emb.transpose(1, 2))
        all_z, prior_mean, prior_std = self.generative(final_d, d) # size = (layers, batch size, time length, channels)
        mean, std = self.inference(final_d, d, prior_mean, prior_std) # size = (layers, batch size, time length, channels)
        KL_loss = _kld_gauss(prior_mean, prior_std, mean, std, length)
        z = torch.cat((all_z[0,:,:,:],all_z[1,:,:,:]),2)
        for i in range(self.num_levels-2):
            z = torch.cat((z,all_z[i+2,:,:,:]),2) # size = (batch size, time length, channels * layers)
            
        o = self.decoder(z)
        o = self.drop(o)
        
        logp = o.contiguous()
        NLL_loss = NLL(logp, target, length + 1)
        return logp, NLL_loss, KL_loss