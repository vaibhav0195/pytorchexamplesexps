###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model.
#
###############################################################################
import argparse
import torch

import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')
# Model parameters.
parser.add_argument('--data', type=str, default='/home/yoda/data/textmasking/data',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='/home/yoda/data/textmasking/models/trainedpytorch/model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3.")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f, map_location=device)
model.eval()

corpus = data.Corpus(args.data,eval="/home/yoda/data/textmasking/models/trainedpytorch/idx2word.json")
ntokens = len(corpus.dictionary)
print(ntokens)
is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    hidden = model.init_hidden(1)

inpSentence = "tumor suppressor gene in various types of ca however its role in MASKTOKEN remains poorly"
input = [corpus.dictionary.word2idx[word] for word in inpSentence.split(" ")]
# input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
# the 2nd dimension is the batch. this is how this model was trained some design decision at pytorch to improve the speed.
input = torch.LongTensor(input).to(device).view(-1,1)
orignalsize = input.size(0)

with open(args.outf, 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(args.words):
            if is_transformer_model:
                output = model(input, False)
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                input = torch.cat([input, word_tensor], 0)
            else:
                output, hidden = model(input, hidden)
                # p = torch.nn.functional.softmax(output, dim=1).detach().numpy()
                # word_index = np.random.choice(len(last_word_logits), p=p)
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                # put the new word idx at the end of input
                word_idx = word_idx.to(device)
                input = torch.cat((input,word_idx.view(1,1)),dim=0)
                # slice the input from the first to the new word
                input = input[1:orignalsize,:]

            word = corpus.dictionary.idx2word[word_idx]

            # outf.write(word + ('\n' if i % 20 == 19 else ' '))
            outf.write(word + ('\n' if "eos" in word else ' '))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
