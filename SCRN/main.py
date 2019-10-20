
import sys
sys.path.append('..')



import os
import time
import math
import scipy.io as sio
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from multiprocessing import cpu_count



from ptb import PTB
from model import SCRN



# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('GPU')
else:
    print('CPU')



# Penn TreeBank (PTB) dataset
data_path = '../data'
max_len = 96
splits = ['train', 'valid', 'test']
datasets = {split: PTB(root=data_path, split=split) for split in splits}



# data loader
batch_size = 20 #32
dataloaders = {split: DataLoader(datasets[split],
                                 batch_size=batch_size,
                                 shuffle=split=='train',
                                 num_workers=cpu_count(),
                                 pin_memory=torch.cuda.is_available())
                                 for split in splits}
symbols = datasets['train'].symbols


# SCRN model
embedding_size = 300 # dimension of character embeddings
dropout_rate = 0.1
emb_dropout_rate = 0.4
levels = 3    # # of levels
nhid = 450    # number of hidden units per layer
num_chans = [nhid] * (levels)#[nhid] * (levels - 1) + [embedding_size]
model = SCRN(vocab_size=datasets['train'].vocab_size,
            embed_size=embedding_size,
            num_channels=num_chans,
            time_step=max_len,
            hidden_size = nhid,
            bos_idx=symbols['<bos>'],
            eos_idx=symbols['<eos>'],
            pad_idx=symbols['<pad>'],
            dropout=dropout_rate,
            emb_dropout = emb_dropout_rate)
model = model.to(device)
print(model)


# folder to save model
save_path = 'model'
if not os.path.exists(save_path):
    os.makedirs(save_path)


# objective function
learning_rate = 10
criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=symbols['<pad>'])
optimizer = optim.SGD(model.parameters(), lr=learning_rate) #Adam
#optimizer = optim.Adam(model.parameters(), lr=learning_rate) #Adam


# negative log likelihood
def NLL(logp, target, length):
    target = target[:, :torch.max(length).item()].contiguous().view(-1)
    logp = logp[:, :torch.max(length).item(),:].contiguous().view(-1, logp.size(-1)) # logp = logp.view(-1, logp.size(-1))
    return criterion(logp, target)

# training setting
epoch = 20
print_every = 50



# training interface
step = 0
NLL_tracker = {'NLL': []}
KL_tracker = {'KL': []}
start_time = time.time()
beta = 1#1.0/epoch
beta_increase = beta
for ep in range(epoch):
    # learning rate decay
    if (ep % 5 == 0) and (learning_rate>0.1) and (ep>=10):
        learning_rate = learning_rate * 0.5#0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    for split in splits:
        dataloader = dataloaders[split]
        model.train() if split == 'train' else model.eval()
        totals = {'NLL': 0.,'KL': 0., 'words': 0}

        for itr, (_, dec_inputs, targets, lengths) in enumerate(dataloader):
            bsize = dec_inputs.size(0)
            dec_inputs = dec_inputs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            # forward
            logp, NLL_loss, KL_loss = model(dec_inputs, lengths, targets) #, lengths

            # calculate loss
            #NLL_loss = NLL(logp, targets, lengths + 1)
            KL_loss = KL_loss
            loss = NLL_loss + beta*KL_loss

            # cumulate
            totals['NLL'] += NLL_loss.item()
            totals['KL'] += KL_loss.item()
            totals['words'] += torch.sum(lengths).item()
            NLL_loss = NLL_loss / bsize

            # backward and optimize
            if split == 'train':
                step += 1

                # track
                NLL_tracker['NLL'].append(NLL_loss.item())
                KL_tracker['KL'].append(KL_loss.item())

                # print statistics
                if itr % print_every == 0 or itr + 1 == len(dataloader):
                    print("%s Batch %04d/%04d, NLL-Loss %.4f, KL-Loss %.4f, "
                          % (split.upper(), itr, len(dataloader),
                             NLL_tracker['NLL'][-1], KL_tracker['KL'][-1]))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25) #5
                optimizer.step()

        samples = len(datasets[split])
        print("%s Epoch %02d/%02d, NLL %.4f, PPL %.4f"
              % (split.upper(), ep, epoch, totals['NLL'] / samples,
                 math.exp(totals['NLL'] / totals['words'])))
        # KL annealing
        #if beta<=1:
        #    beta+=beta_increase

    # save checkpoint
    checkpoint_path = os.path.join(save_path, "E%02d.pkl" % ep)
    torch.save(model.state_dict(), checkpoint_path)
    print("Model saved at %s\n" % checkpoint_path)
end_time = time.time()
print('Total cost time',
      time.strftime("%H hr %M min %S sec", time.gmtime(end_time - start_time)))

print('# of parameters:', sum(param.numel() for param in model.parameters()))


# save learning results
sio.savemat("results.mat", NLL_tracker)

