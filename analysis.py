import os

import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence_

from bpe import BytePairEncoding
from model import BaseModel
import utils
from itertools import chain

import sentencepiece as spm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

processor = spm.SentencePieceProcessor()
processor.load(os.path.join('sentencepiece', 'imdb.model')) 

PAD = BytePairEncoding.PAD_token_idx
CLS = BytePairEncoding.CLS_token_idx
SEP = BytePairEncoding.SEP_token_idx

# Example stentences, choose your samples
sentences = ['i shut the book.', 'i close the book.', 'i read the book.', \
             'i am near to you.', 'i am close to you.', 'i am far to you.']
sentences = list(map(processor.EncodeAsPieces, sentences))
compare_list = [(0, 2, 1, 2), (1, 2, 2, 2), (3, 3, 4, 3), (4, 3, 5, 3), (0, 1, 3, 1), (1, 2, 4, 3)]

tokens = [[CLS] + [processor.piece_to_id(subword) for subword in sentence] + [SEP] for sentence in sentences]
sentences = [['<cls>'] + sentence + ['<sep>'] for sentence in sentences]
tokens = pad_sequence([torch.Tensor(sentence).to(torch.long) for sentence in tokens], padding_value=PAD)

pretrained_model_name = 'pretrained_final.pth'

# You can use a model which have been pretrained over 200 epochs by TA
# If you use this saved model, you should mention it in the report
#
# pretrained_model_name = 'pretrained_byTA.pth'

model = BaseModel(token_num = len(processor))
model.load_state_dict(torch.load(pretrained_model_name, map_location='cpu'), strict=False)

model.eval()

output = model(tokens)
output = pack_padded_sequence(output, (output[..., 0] != PAD).sum(0), enforce_sorted=False)
temp = (output.data - output.data.mean(dim=0)) # normalized output
covariance = 1.0 / len(output.data) * temp.T @ temp # covariance of 45 tokens
U, S, V = covariance.svd()
output = PackedSequence_(temp @ U[:, :7], output.batch_sizes, output.sorted_indices, output.unsorted_indices) # output summed into 7 dim
output, _ = pad_packed_sequence(output, batch_first=True, padding_value=PAD)

_, ax = plt.subplots(nrows=2, ncols=len(sentences) // 2)
ax = list(chain.from_iterable(ax))
for i, sentence, state in zip(range(len(sentences)), sentences, output):
    state = state[:len(sentence), :]

    _ = ax[i].pcolor(state.T.detach())

    ax[i].set_xticks([tick + .5 for tick in range(len(sentence))], minor=False)
    ax[i].xaxis.tick_top()
    ax[i].set_xticklabels(sentence, rotation=90, minor=False)

plt.savefig('analysis.png')

print("Cosine Similarities:")
for s_id1, w_id1, s_id2, w_id2 in compare_list:
    word1, out1 = sentences[s_id1][w_id1], output[s_id1][w_id1]
    word2, out2 = sentences[s_id2][w_id2], output[s_id2][w_id2]
    similarity = torch.cosine_similarity(out1, out2, dim=0).item()

    print('sentence{} {} & sentence{} {}: {:.4f}'.format(s_id1, word1, s_id2, word2, similarity))
