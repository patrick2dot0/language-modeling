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
    
class two_layer_RNN(nn.Module):

    def __init__(self, vocab_size, embed_size, time_step, hidden_size,
                 dropout_rate, bos_idx, eos_idx, pad_idx):
        super(two_layer_RNN, self).__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        # input
        self.embedding = nn.Embedding(vocab_size, embed_size,
                                      padding_idx=pad_idx)
        # RNN1
        self.rnn1 = nn.LSTM(embed_size, hidden_size, batch_first=True)
        # RNN2
        self.rnn2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
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
        pack_input1 = pack_padded_sequence(drop_input, sorted_len + 1,batch_first=True)
        pack_output1, _ = self.rnn1(pack_input1)
        pack_output2, _ = self.rnn2(pack_output1)
        output, _ = pad_packed_sequence(pack_output2, batch_first=True)
        _, reversed_idx = torch.sort(sorted_idx)
        output = output[reversed_idx]
        
        # project output
        drop_output = F.dropout(output, p=self.dropout_rate, training=self.training)
        batch_size, seq_len, hidden_size = drop_output.size()
        logit = self.output(drop_output.view(-1, hidden_size))
        logp = F.log_softmax(logit, dim=-1)
        logp = logp.view(batch_size, seq_len, -1)

        return logp





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
    
class CRN(nn.Module):

    def __init__(self, vocab_size, embed_size, num_channels , time_step, hidden_size,
                 dropout_rate, bos_idx, eos_idx, pad_idx, kernel_size=2, emb_dropout=0.2):
        super(CRN, self).__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.emb_dropout = emb_dropout
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        # input
        self.embedding = nn.Embedding(vocab_size, embed_size,padding_idx=pad_idx)
        
        # TCN
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size=kernel_size, dropout=dropout_rate)
        
        # RNN
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
        # output
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq, length, target):
        # process input
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        input_seq = input_seq[sorted_idx]
        embedded_input = self.embedding(input_seq)
        drop_input = F.dropout(embedded_input, p=self.emb_dropout, training=self.training)
        # TCN forward
        z,__ = self.tcn(drop_input.transpose(1, 2))
        z = F.dropout(z, p=self.emb_dropout, training=self.training)
        # RNN forward
        pack_input = pack_padded_sequence(z.transpose(1, 2), sorted_len + 1,batch_first=True)
        pack_output, _ = self.rnn(pack_input)
        output, _ = pad_packed_sequence(pack_output, batch_first=True)
        _, reversed_idx = torch.sort(sorted_idx)
        output = output[reversed_idx]

        # project output
        drop_output = F.dropout(output, p=self.emb_dropout,
                                training=self.training)
        batch_size, seq_len, hidden_size = drop_output.size()
        logit = self.output(drop_output.view(-1, hidden_size))
        logp = F.log_softmax(logit, dim=-1)
        logp = logp.view(batch_size, seq_len, -1)
        
        NLL_loss = NLL(logp, target, length + 1)

        return logp, NLL_loss
    
class RCN(nn.Module):

    def __init__(self, vocab_size, embed_size, num_channels , time_step, hidden_size,
                 dropout_rate, bos_idx, eos_idx, pad_idx, kernel_size=2, emb_dropout=0.2):
        super(RCN, self).__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.emb_dropout = emb_dropout
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        # input
        self.embedding = nn.Embedding(vocab_size, embed_size,padding_idx=pad_idx)
        
        # TCN
        self.tcn = TemporalConvNet(hidden_size, num_channels, kernel_size=kernel_size, dropout=dropout_rate)
        
        # RNN
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
        # output
        self.output = nn.Linear(embed_size, vocab_size)
        
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, input_seq, length, target):
        # process input
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        input_seq = input_seq[sorted_idx]
        embedded_input = self.embedding(input_seq)
        drop_input = F.dropout(embedded_input, p=self.emb_dropout,training=self.training)

        # RNN forward
        pack_input = pack_padded_sequence(drop_input, sorted_len + 1,
                                          batch_first=True)
        pack_output, _ = self.rnn(pack_input)
        z, _ = pad_packed_sequence(pack_output, batch_first=True)
        _, reversed_idx = torch.sort(sorted_idx)
        z = z[reversed_idx]
        
        # project output
        z = F.dropout(z, p=self.emb_dropout, training=self.training)
        
        
        # TCN forward
        y, __ = self.tcn(z.transpose(1, 2))
        o = self.output(y.transpose(1, 2))
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
    
class SCRN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_channels, hidden_size, bos_idx, eos_idx, pad_idx, kernel_size=2, dropout=0.2, emb_dropout=0.2):
        super(SCRN, self).__init__()
        self.emb_dropout = emb_dropout
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.num_levels = len(num_channels)
        # Embedding
        self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        # TCN
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        # RNN
        self.rnn = nn.LSTM(sum(num_channels), hidden_size, batch_first=True)
        # output
        self.output = nn.Linear(hidden_size, vocab_size)
        
        self.generative= GenerativeNetwork(embed_size, num_channels)
        self.inference= InferenceNetwork(embed_size, num_channels)
        self.drop = nn.Dropout(self.emb_dropout)
        self.init_weights()
        

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.output.bias.data.fill_(0)
        self.output.weight.data.uniform_(-initrange, initrange)
        

    def forward(self, x, length, target):
        # input has dimension (N, L_in), and emb has dimension (N, L_in, C_in)
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        x = x[sorted_idx]
        emb = self.drop(self.encoder(x))
        final_d, d = self.tcn(emb.transpose(1, 2))
        all_z, prior_mean, prior_std = self.generative(final_d, d) # size = (layers, batch size, time length, channels)
        mean, std = self.inference(final_d, d, prior_mean, prior_std) # size = (layers, batch size, time length, channels)
        KL_loss = _kld_gauss(prior_mean, prior_std, mean, std, length)
        z = torch.cat((all_z[0,:,:,:],all_z[1,:,:,:]),2)
        for i in range(self.num_levels-2):
            z = torch.cat((z,all_z[i+2,:,:,:]),2) # size = (laeyrs, batch size, time length, channels)

        # RNN forward
        pack_input = pack_padded_sequence(z, sorted_len + 1,batch_first=True)
        pack_output, _ = self.rnn(pack_input)
        output, _ = pad_packed_sequence(pack_output, batch_first=True)
        _, reversed_idx = torch.sort(sorted_idx)
        output = output[reversed_idx]

        # project output
        drop_output = F.dropout(output, p=self.emb_dropout, training=self.training)
        batch_size, seq_len, hidden_size = drop_output.size()
        logit = self.output(drop_output.view(-1, hidden_size))
        logp = F.log_softmax(logit, dim=-1)
        logp = logp.view(batch_size, seq_len, -1)
        
        NLL_loss = NLL(logp, target, length + 1)
        return logp, NLL_loss, KL_loss
    
class multi_resolution_CRN(nn.Module):

    def __init__(self, vocab_size, embed_size, num_channels , time_step, hidden_size,
                 dropout_rate, bos_idx, eos_idx, pad_idx, kernel_size=2, emb_dropout=0.2):
        super(multi_resolution_CRN, self).__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.emb_dropout = emb_dropout
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.num_levels = len(num_channels)

        # input
        self.embedding = nn.Embedding(vocab_size, embed_size,padding_idx=pad_idx)
        
        # TCN
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size=kernel_size, dropout=dropout_rate)
        
        # RNN
        RNNs = []
        for i in range(self.num_levels):
            RNNs += [nn.LSTM(num_channels[i], hidden_size, batch_first=True)]
            
        self.network = nn.Sequential(*RNNs)
        # output
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq, length, target):
        # process input
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        input_seq = input_seq[sorted_idx]
        embedded_input = self.embedding(input_seq)
        drop_input = F.dropout(embedded_input, p=self.emb_dropout, training=self.training)
        # TCN forward
        final_z,z = self.tcn(drop_input.transpose(1, 2))
        z = F.dropout(z, p=self.emb_dropout, training=self.training)
        final_z = F.dropout(final_z, p=self.emb_dropout, training=self.training)
        
        
        # RNN forward
        for i,layer in enumerate(self.network):
            if i==(self.num_levels-1):
                pack_input = pack_padded_sequence(final_z.transpose(1, 2), sorted_len + 1,batch_first=True)
                pack_output, _ = layer(pack_input)
                output, _ = pad_packed_sequence(pack_output, batch_first=True)
                _, reversed_idx = torch.sort(sorted_idx)
                output = output[reversed_idx]
                all_out=all_out.add(output)
            elif i==0:
                pack_input = pack_padded_sequence(z[i,:,:,:].transpose(1, 2), sorted_len + 1,batch_first=True)
                pack_output, _ = layer(pack_input)
                output, _ = pad_packed_sequence(pack_output, batch_first=True)
                _, reversed_idx = torch.sort(sorted_idx)
                output = output[reversed_idx]
                all_out=output
            else:
                pack_input = pack_padded_sequence(z[i,:,:,:].transpose(1, 2), sorted_len + 1,batch_first=True)
                pack_output, _ = layer(pack_input)
                output, _ = pad_packed_sequence(pack_output, batch_first=True)
                _, reversed_idx = torch.sort(sorted_idx)
                output = output[reversed_idx]
                all_out=all_out.add(output)
                

        # project output
        drop_output = F.dropout(all_out, p=self.emb_dropout,
                                training=self.training)
        batch_size, seq_len, hidden_size = drop_output.size()
        logit = self.output(drop_output.view(-1, hidden_size))
        logp = F.log_softmax(logit, dim=-1)
        logp = logp.view(batch_size, seq_len, -1)
        
        NLL_loss = NLL(logp, target, length + 1)

        return logp, NLL_loss
    
class multi_resolution_SCRN(nn.Module):

    def __init__(self, vocab_size, embed_size, num_channels , time_step, hidden_size,
                 dropout_rate, bos_idx, eos_idx, pad_idx, kernel_size=2, emb_dropout=0.2):
        super(multi_resolution_SCRN, self).__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.emb_dropout = emb_dropout
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.num_levels = len(num_channels)

        # input
        self.embedding = nn.Embedding(vocab_size, embed_size,padding_idx=pad_idx)
        
        # TCN
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size=kernel_size, dropout=dropout_rate)
        
        # RNN
        RNNs = []
        for i in range(self.num_levels):
            RNNs += [nn.LSTM(num_channels[i], hidden_size, batch_first=True)]
            
        self.network = nn.Sequential(*RNNs)
        
        self.generative= GenerativeNetwork(embed_size, num_channels)
        self.inference= InferenceNetwork(embed_size, num_channels)
        
        # output
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq, length, target):
        # process input
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        input_seq = input_seq[sorted_idx]
        embedded_input = self.embedding(input_seq)
        drop_input = F.dropout(embedded_input, p=self.emb_dropout, training=self.training)
        # TCN forward
        final_d,d = self.tcn(drop_input.transpose(1, 2))
        d = F.dropout(d, p=self.emb_dropout, training=self.training)
        final_d = F.dropout(final_d, p=self.emb_dropout, training=self.training)
        
        all_z, prior_mean, prior_std = self.generative(final_d, d) # size = (layers, batch size, time length, channels)
        mean, std = self.inference(final_d, d, prior_mean, prior_std) # size = (layers, batch size, time length, channels)
        KL_loss = _kld_gauss(prior_mean, prior_std, mean, std, length)
        
        # RNN forward
        for i,layer in enumerate(self.network):
            if i==0:
                pack_input = pack_padded_sequence(all_z[i,:,:,:], sorted_len + 1,batch_first=True)
                pack_output, _ = layer(pack_input)
                output, _ = pad_packed_sequence(pack_output, batch_first=True)
                _, reversed_idx = torch.sort(sorted_idx)
                output = output[reversed_idx]
                all_out=output
            else:
                pack_input = pack_padded_sequence(all_z[i,:,:,:], sorted_len + 1,batch_first=True)
                pack_output, _ = layer(pack_input)
                output, _ = pad_packed_sequence(pack_output, batch_first=True)
                _, reversed_idx = torch.sort(sorted_idx)
                output = output[reversed_idx]
                all_out=all_out.add(output)
                

        # project output
        drop_output = F.dropout(all_out, p=self.emb_dropout,
                                training=self.training)
        batch_size, seq_len, hidden_size = drop_output.size()
        logit = self.output(drop_output.view(-1, hidden_size))
        logp = F.log_softmax(logit, dim=-1)
        logp = logp.view(batch_size, seq_len, -1)
        
        NLL_loss = NLL(logp, target, length + 1)

        return logp, NLL_loss, KL_loss