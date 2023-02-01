import argparse
import torch
import pickle
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
import dgl

from openks.models.pytorch.DisenCite.models import CitationGeneration
from openks.models.pytorch.DisenCite.data import S2orcDataset
from openks.models.pytorch.DisenCite.bleu import compute_bleu
from openks.models.pytorch.DisenCite.rouge import Rouge
from collections import defaultdict
from sklearn import metrics
from sklearn.metrics import classification_report
import time


def parse_args():
    parser = argparse.ArgumentParser(description='DisenPaper')
    parser.add_argument('--dataset', default='Computer Science8',
                        help='dataset name, medicine, computer science can choose')
    parser.add_argument('--mode', choices=['train', 'test'], default='test',
                        help='running mode')
    parser.add_argument('--pretrained', type=bool, default=True,
                        help='whether use pretrained embedding')
    parser.add_argument('--is_dec_attention', type=bool, default=True,
                        help='whether use decoder attention')
    parser.add_argument('--is_pointer_gen', type=bool, default=False,
                        help='whether use pointer generator')
    parser.add_argument('--is_coverage', type=bool, default=False,
                        help='whether converage for pointer generator')
    parser.add_argument('--cov_loss_wt', type=float, default=1.0,
                        help='coverage weight')
    parser.add_argument('--use_norm', type=bool, default=True,
                        help='whether use normalization')
    parser.add_argument('--n_hid', type=int, default=50,
                        help='dimension of embeddings')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='initial learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=5.0,
                        help='max grad norm for clip lr')
    parser.add_argument('--batch_size', type=int, default=48,
                        help='batch size')
    parser.add_argument('--n_textenc_layer', type=int, default=1,
                        help='number of layers for the text encoder')
    parser.add_argument('--n_neigh_layer', type=int, default=1,
                        help='number of neighs')
    parser.add_argument('--n_graphenc_layer', type=int, default=1,
                        help='number of layers for the graph encoder')
    parser.add_argument('--n_dec_layer', type=int, default=1,
                        help='number of layers for the decoder')
    parser.add_argument('--n_max_neigh', nargs='?', default=[20],
                        help='max number of neighs for each layer')
    parser.add_argument('--n_max_len_section', type=int, default=200,
                        help='max number of word for each section')
    parser.add_argument('--n_max_len_citation', type=int, default=50,
                        help='max number of word for each citation')
    parser.add_argument('--vocab_size', type=int, default=20000,
                        help='size of vocabs')
    # parser.add_argument('--bow_smoothing', type=float, default=1e-1,
    #                     help='bag of words smoothing')
    parser.add_argument('--use_mutual', type=bool, default=True,
                        help='use mutual information')
    parser.add_argument('--n_neg', type=int, default=1,
                        help='number of negative samples')
    parser.add_argument('--alpha1', type=float, default=0.1, #1e-1
                        help='mutual info weight alpha1')
    parser.add_argument('--alpha2', type=float, default=0.1,
                        help='mutual info weight alpha2')
    parser.add_argument('--alpha3', type=float, default=0.0,
                        help='mutual info weight alpha3')
    parser.add_argument('--dropout', type=float, default=0.35,
                        help='dropout rate (1-keep probability)')
    parser.add_argument('--is_drop_emb', type=bool, default=False,
                        help='whether drop out for embedding')
    parser.add_argument('--n_beams', type=int, default=5,
                        help='number of beams when doing beam search')
    parser.add_argument('--decay', type=float, default=0.98,
                        help='learning rate decay rate')
    parser.add_argument('--decay_step', type=int, default=1,
                        help='learning rate decay step')
    parser.add_argument('--log_step', type=int, default=1e2,
                        help='log print step')
    parser.add_argument('--epochs', type=int, default=30,
                        help='upper epoch limit')
    parser.add_argument('--patience', type=int, default=80,
                        help='extra iterations before early-stopping')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use GPU for training')
    parser.add_argument('--save', type=str, default='model/',
                        help='path to save the final model')
    parser.add_argument('--is_checkpoint', type=bool, default=False,
                        help='checkpoint for resuming training')

    args = parser.parse_args()
    args.save = args.save + args.dataset
    args.save = args.save + '_batch{}'.format(args.batch_size)
    args.save = args.save + '_hid{}'.format(args.n_hid)
    args.save = args.save + '_n_max_neigh{}'.format(str(args.n_max_neigh))
    args.save = args.save + '_n_max_len_section{}'.format(args.n_max_len_section)
    args.save = args.save + '_n_max_len_citation{}'.format(args.n_max_len_citation)
    args.save = args.save + '_n_textenc_layer{}'.format(args.n_textenc_layer)
    args.save = args.save + '_n_graphenc_layer{}'.format(args.n_graphenc_layer)
    args.save = args.save + '_n_dec_layer{}'.format(args.n_dec_layer)
    args.save = args.save + '_alpha1{}'.format(args.alpha1)
    args.save = args.save + '_alpha2{}'.format(args.alpha2)
    args.save = args.save + '_alpha3{}'.format(args.alpha3)
    args.save = args.save + '_lr{}'.format(args.lr)
    args.save = args.save + '_dropout{}'.format(args.dropout)
    args.save = args.save + '_decay{}'.format(args.decay)
    args.save = args.save + '_decaystep{}'.format(args.decay_step)
    args.save = args.save + '_patience{}_pretrained_newest8_4.pt'.format(args.patience)

    # args.save = 'model/Computer Science_batch48_hid50_n_max_neigh20_n_max_len_section200_n_max_len_citation50_n_textenc_layer1_n_graphenc_layer1_n_dec_layer1_alpha10.0_alpha20.1_alpha30.1_lr0.005_dropout0.35_decay0.98_decaystep1_patience80_pretrained_newest4.pt'
    # args.save = 'model/Computer Science3_batch48_hid50_n_max_neigh[20]_n_max_len_section200_n_max_len_citation50_n_textenc_layer1_n_graphenc_layer1_n_dec_layer1_alpha10.1_alpha20.1_alpha30.1_lr0.005_dropout0.35_decay0.98_decaystep1_patience80_pretrained_newest7.pt'
    # args.save = 'model/Computer Science_batch48_hid50_n_max_neigh[20]_n_max_len_section200_n_max_len_citation50_n_textenc_layer1_n_graphenc_layer1_n_dec_layer1_alpha10.0_alpha20.1_alpha30.1_lr0.005_dropout0.35_decay0.98_decaystep1_patience80_pretrained_newest6.pt'
    return args

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def graph_collate(batch):
    batch_sub_g = dgl.batch([item[0] for item in batch])
    batch_pair_og = default_collate([item[9] for item in batch])
    batch_pair = default_collate([item[1] for item in batch])
    batch_current_pos = default_collate([item[2] for item in batch])
    batch_citation_pos = default_collate([item[3] for item in batch])
    batch_shuf_paper = torch.cat([item[4] for item in batch], dim=0)
    batch_enc_extend_vocab = default_collate([item[5] for item in batch])
    # batch_citation_bow = default_collate([item[6] for item in batch])
    batch_citation_inp = [item[6] for item in batch]
    batch_citation_target = [item[7] for item in batch]
    sub_g_oovs = [item[8] for item in batch]
    return batch_sub_g, batch_pair, batch_current_pos, batch_citation_pos, batch_shuf_paper, batch_enc_extend_vocab, \
           batch_citation_inp, batch_citation_target, sub_g_oovs, batch_pair_og

def prepare_batch(batch, device, pad_token):
    batch_sub_g, batch_pair, batch_current_pos, batch_citation_pos, batch_shuf_paper, batch_enc_extend_vocab, \
        batch_citation_inp, batch_citation_target, sub_g_oovs, batch_pair_og = batch

    batch_size = len(sub_g_oovs)
    max_n_sub_g_oovs = max(len(sub_g_oov) for sub_g_oov in sub_g_oovs)
    batch_citation_len = [len(citation_inp) for citation_inp in batch_citation_inp]
    citation_len_max = max(batch_citation_len)
    batch_citation_len = torch.tensor(batch_citation_len, dtype=torch.float32).to(device)
    extra_zeros = None
    if max_n_sub_g_oovs > 0:
        extra_zeros = torch.zeros((batch_size, max_n_sub_g_oovs)).to(device)

    batch_paper_size_ = batch_sub_g.batch_num_nodes('paper')
    batch_paper_size_ = torch.cat([torch.zeros(1, dtype=torch.int64), batch_paper_size_[:-1]], dim=0)
    batch_paper_size_ = torch.cumsum(batch_paper_size_, dim=0)
    batch_paper_size = dgl.broadcast_nodes(batch_sub_g, batch_paper_size_, ntype='paper')

    batch_shuf_paper += batch_paper_size.unsqueeze(1)
    batch_shuf_paper = batch_shuf_paper.to(device)

    batch_pair += batch_paper_size_.unsqueeze(1)
    batch_pair = batch_pair.to(device)
    batch_enc_extend_vocab = batch_enc_extend_vocab.to(device)
    batch_current_pos = batch_current_pos.to(device)
    batch_citation_pos = batch_citation_pos.to(device)
    # batch_citation_bow = batch_citation_bow.to(device)
    batch_citation_inp = pad_sequence(batch_citation_inp, padding_value=pad_token, batch_first=True).to(device)
    batch_citation_target = pad_sequence(batch_citation_target, padding_value=pad_token, batch_first=True).to(device)
    enc_padding_mask = (batch_enc_extend_vocab > 0).type(torch.int).to(device)
    dec_padding_mask = (batch_citation_target > 0).type(torch.int).to(device)
    batch_sub_g = batch_sub_g.to(device)

    return (batch_sub_g, batch_pair, batch_current_pos, batch_citation_pos,
            batch_shuf_paper, batch_enc_extend_vocab,
            batch_citation_inp, batch_citation_target,
            enc_padding_mask, dec_padding_mask,
            batch_citation_len, citation_len_max, extra_zeros, sub_g_oovs, batch_pair_og)

def train_one_epoch(model, train_data_loader, optimizer, epoch, device, args):
    model.train()
    epoch_loss = []
    pad_token = model.vocab2index['PAD']
    for step, batch_data in enumerate(train_data_loader):
        batch_sub_g, batch_pair, batch_current_pos, batch_citation_pos, \
        batch_shuf_paper, batch_enc_extend_vocab, \
        batch_citation_inp, batch_citation_target, \
        enc_padding_mask, dec_padding_mask, \
        batch_citation_len, citation_len_max, extra_zeros, sub_g_oovs, batch_pair_og = prepare_batch(batch_data, device, pad_token)

        # if batch_sub_g.num_nodes('paper') < 96:
        #     print(batch_sub_g.batch_num_nodes('paper'))

        optimizer.zero_grad()
        loss = model(batch_sub_g, batch_pair, batch_current_pos, batch_citation_pos,
                     batch_shuf_paper, batch_enc_extend_vocab,
                     batch_citation_inp, batch_citation_target,
                     enc_padding_mask, dec_padding_mask, batch_citation_len, citation_len_max, extra_zeros)
        loss.backward()
        clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        epoch_loss.append(loss.item())
        if (step % args.log_step == 0) and step > 0:
            print('Train epoch: {}[{}/{} ({:.0f}%)]\tLr:{:.6f}, Loss: {:.6f}, AvgL: {:.6f}'
                  .format(epoch, step, len(train_data_loader),
                          100. * step / len(train_data_loader),
                          get_lr(optimizer), loss.item(), np.mean(epoch_loss)))
    mean_epoch_loss = np.mean(epoch_loss)
    return mean_epoch_loss


def train(model, train_data_loader, valid_data_loader, test_data_loader, index2vocab, device, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay)
    best_bleu = 0.0
    best_epoch = -1
    # valid(model, valid_data_loader, index2vocab, device)
    inference(model, test_data_loader, index2vocab, device)
    for epoch in range(args.epochs):
        print('Start epoch:', epoch)
        train_one_epoch(model, train_data_loader, optimizer, epoch, device, args)
        bleu_score, rouge_score = valid(model, valid_data_loader, index2vocab, device)
        scheduler.step()
        inference(model, test_data_loader, index2vocab, device)
        if bleu_score[3] > best_bleu:
            best_epoch = epoch
            best_bleu = bleu_score[3]
            model.save(args.save)
            print('Model save for higher bleu scores %f in %s' % (best_bleu, args.save))
            # test(model, test_data_loader, index2vocab, device)
        if epoch - best_epoch >= args.patience:
            print('Stop training after %i epochs without improvement on validation.' % args.patience)
            break
    model.load(args.save)
    inference(model, test_data_loader, index2vocab, device)

def decode_one_batch(model, batch, index2vocab, device):
    pad_token = model.vocab2index['PAD']
    with torch.no_grad():
        batch_sub_g, batch_pair, batch_current_pos, batch_citation_pos, \
        batch_shuf_paper, batch_enc_extend_vocab, \
        batch_citation_inp, batch_citation_target, \
        enc_padding_mask, dec_padding_mask, \
        batch_citation_len, citation_len_max, extra_zeros, sub_g_oovs, batch_pair_og = prepare_batch(batch, device, pad_token)
        pred, pos_score = model.beam_search(batch_sub_g, batch_pair, batch_current_pos, batch_citation_pos, batch_shuf_paper,
                                 batch_enc_extend_vocab, enc_padding_mask, extra_zeros)

        oovs_index2vocab = [{idx: w for w, idx in oov.items()} for oov in sub_g_oovs]
        eos_token = model.vocab2index['EOS']

        def ids2sent(indices, oov):
            def idx2word(idx):
                if idx in index2vocab:
                    return index2vocab[idx]
                # print(f'__{oov[idx]}__')
                return f'__{oov[idx]}__'

            words = []
            for idx in indices.tolist():
                words.append(idx2word(idx))
                if idx == eos_token:
                    break
            return words

        tgts = [ids2sent(tgt, oov) for tgt, oov in zip(batch_citation_target, oovs_index2vocab)]
        preds = [ids2sent(prd, oov) for prd, oov in zip(pred, oovs_index2vocab)]
        return tgts, preds, batch_citation_pos, batch_current_pos, pos_score, batch_pair_og

def valid(model, valid_data_loader, index2vocab, device):
    print('Start Valid')
    bleu_score, rouge_score, f1_micro, f1_macro, hl, report, diveristy = decode(model, valid_data_loader, index2vocab, device)
    print('Valid:')
    print('\tbleu-1:{0[0]}, bleu-2:{0[1]}, bleu-3:{0[2]}, bleu-4:{0[3]}'.format(bleu_score))
    print('\trouge-1-f:{0[0]}, rouge-2-f:{0[1]}, rouge-l-f:{0[2]}'.format(rouge_score))
    print('f1_micro:{}, f1_macro:{}, hl:{}'.format(f1_micro, f1_macro, hl))
    print(report)
    return bleu_score, rouge_score


def inference(model, test_data_loader, index2vocab, device):
    print('Start Test')
    bleu_score, rouge_score, f1_micro, f1_macro, hl, report, diversity, all_decodes = decode(model, test_data_loader, index2vocab, device)
    print('Test:')
    print('\tbleu-1:{0[0]}, bleu-2:{0[1]}, bleu-3:{0[2]}, bleu-4:{0[3]}'.format(bleu_score))
    print('\trouge-1-f:{0[0]}, rouge-2-f:{0[1]}, rouge-l-f:{0[2]}'.format(rouge_score))
    print('f1_micro:{}, f1_macro:{}, hl:{}'.format(f1_micro, f1_macro, hl))
    print(report)
    print('diversity:{}'.format(diversity))
    return bleu_score, rouge_score, all_decodes

def decode(model, data_loader, index2vocab, device):
    model.eval()
    metrics_score = [[] for _ in range(7)]
    rouge = Rouge()
    # n_decode = len(data_loader)
    # if max_batch is None:
    #     max_batch = n_decode
    n_decode = len(data_loader)
    all_preds = torch.zeros(len(data_loader.dataset), 4).cpu()
    all_targets = torch.zeros(len(data_loader.dataset), 4).cpu()
    start_idx = 0
    all_decodes = {}


    human_samples = []
    human_idx = 0
    for step, batch_data in tqdm(enumerate(data_loader), total=n_decode):
        # if step >= max_batch:
        #     break
        tgts, preds, tgt_pos, current_pos, pred_pos, batch_pair_og = decode_one_batch(model, batch_data, index2vocab, device)
        # print(' '.join(tgts[0]))
        # print(' '.join(preds[0]))
        # time.sleep(5)
        src_og = batch_pair_og[0].tolist()
        dst_og = batch_pair_og[1].tolist()
        # current_pos_list = torch.argmax(current_pos, dim=1).tolist()
        # print(tgt_pos)
        # print(current_pos)
        # print(current_pos_list)
        # time.sleep(10)


        for tgt, pred, src, dst, pos in zip(tgts, preds, src_og, dst_og, current_pos.tolist()):
            pair = str(src) + '_' + str(dst)
            # print(pair)
            # print(pred)
            # time.sleep(5)
            # if human_idx >= 100:
            #     break

            # if not pos == 3:
            #     continue
            #
            # human_sample = {'src': src, 'dst': dst, 'tgt': ' '.join(tgt), 'pred': ' '.join(pred), 'pos': pos}
            # human_samples.append(human_sample)
            # human_idx += 1


            if pair in all_decodes:
                all_decodes[pair].append(pred)
            else:
                all_decodes[pair] = [pred]
            # BLEU
            bleu_1, bleu_2, bleu_3, bleu_4 = compute_bleu([' '.join(tgt)], [' '.join(pred)])
            metrics_score[0].append(bleu_1)
            metrics_score[1].append(bleu_2)
            metrics_score[2].append(bleu_3)
            metrics_score[3].append(bleu_4)
            # ROUGE
            rouge_1 = rouge.get_scores(' '.join(pred), ' '.join(tgt))
            metrics_score[4].append(rouge_1[0]['rouge-1']['f'])
            metrics_score[5].append(rouge_1[0]['rouge-2']['f'])
            metrics_score[6].append(rouge_1[0]['rouge-l']['f'])

        end_idx = start_idx + tgt_pos.shape[0]
        all_targets[start_idx:end_idx] = tgt_pos.data.cpu()
        all_preds[start_idx:end_idx] = pred_pos.data.cpu()
        start_idx = end_idx

    metrics_score = [sum(i) / len(i) for i in metrics_score]
    bleu_score = metrics_score[:4]
    rouge_score = metrics_score[4:7]

    f1_micro = metrics.f1_score(all_targets, all_preds > 0.5, average='micro')
    f1_macro = metrics.f1_score(all_targets, all_preds > 0.5, average='macro')
    hl = metrics.hamming_loss(all_targets, all_preds > 0.5)
    label_names = ['intro', 'related work', 'experiment', 'result']
    report = classification_report(all_targets, all_preds > 0.5, target_names=label_names, digits=4)
    diversity = cal_diversity(all_decodes)

    # save_path = '/home/wyf/project/DisenPaper/human.json'
    # print(human_idx)
    # with open(save_path, 'w', encoding='utf-8') as fw:
    #     for sample in human_samples:
    #         meta = json.dumps(sample, ensure_ascii=False)
    #         fw.write(meta + '\n')

    return bleu_score, rouge_score, f1_micro, f1_macro, hl, report, diversity, all_decodes

def cal_diversity(all_decodes):
    def diversity(str1, str2):
        w2idx = {}
        str = str1 + str2
        for w in str:
            if w not in w2idx:
                w2idx[w] = len(w2idx)
        str1_v = np.zeros(len(w2idx))
        str2_v = np.zeros(len(w2idx))
        for w in str1:
            str1_v[w2idx[w]] = 1
        for w in str2:
            str2_v[w2idx[w]] = 1

        num = float(np.dot(str1_v, str2_v))
        denom = np.linalg.norm(str1_v) * np.linalg.norm(str2_v)

        return 1 - (num / denom)

    diversity_vals = []
    for key, vals in all_decodes.items():
        diversity_val = []
        len_vals = len(vals)
        if len_vals == 1:
            continue
        for val_idx1 in range(len_vals):
            for val_idx2 in range(len_vals):
                if val_idx1 == val_idx2:
                    continue
                diversity_val.append(diversity(vals[val_idx1], vals[val_idx2]))
        diversity_vals.append(np.mean(diversity_val))
    return np.mean(diversity_vals)



def main():
    args = parse_args()
    FOS = args.dataset
    root_path = '/home/wyf/project/DisenPaper/s2orc/data/filted/'
    # pretrained_path = '../seq2seq/model/' + args.dataset + '_pretrained.pt'
    pretrained_path = '../seq2seq/model/Computer Science_pretrained.pt'
    # pretrained_path = '../seq2seq/model/' + args.dataset + \
    #                   '_batch48_hid50_n_max_len_section200_n_max_len_citation50_n_textenc_layer1_n_dec_layer1_lr0.005_dropout0.35_decay0.98_decaystep1_patience20.pt'
    # root_path = '/home2/wyf/Projects/DisenPaper/s2orc/data/filted/small/'
    # root_path = '/home2/wyf/Projects/DisenPaper/s2orc/data/small_1/'
    train_data_path = root_path + FOS + '/train_data.json'
    valid_data_path = root_path + FOS + '/valid_data.json'
    test_data_path = root_path + FOS + '/test_data.json'
    paper_withtext_path = root_path + FOS + '/papers_withtext.json'
    paper2index_path = root_path + FOS + '/id2index.pickle'
    vocab2index_path = root_path + FOS + '/vocab2index.pickle'
    with open(paper2index_path, 'rb') as f:
        paper2index = pickle.load(f)
    print(f'paper2index loading complete with {len(paper2index)} papers')
    with open(vocab2index_path, 'rb') as f:
        vocab2index = pickle.load(f)
    vocab2index = {w: idx for w, idx in vocab2index.items() if idx < args.vocab_size}
    index2vocab = {idx: w for w, idx in vocab2index.items()}
    print(f'vocab2index loading complete with {len(vocab2index)} vocabs')
    print(f'UNK token: {vocab2index["UNK"]}')
    print(f'PAD token: {vocab2index["PAD"]}')
    print(f'SOS token: {vocab2index["SOS"]}')
    print(f'EOS token: {vocab2index["EOS"]}\n')

    citation_adj_row_1 = []
    citation_adj_column_1 = []
    citation_adj_row_2 = []
    citation_adj_column_2 = []
    citation_adj_row_3 = []
    citation_adj_column_3 = []
    citation_adj_row_4 = []
    citation_adj_column_4 = []

    section_withtexts = []
    citation_edges_train = []
    citation_edges_valid = []
    citation_edges_test = []

    citation_pos_train = defaultdict(set)
    citation_pos_valid = defaultdict(set)
    citation_pos_test = defaultdict(set)
    citation_pos_all = defaultdict(set)

    idx = 0
    with open(paper_withtext_path, 'r', encoding='utf-8') as f:
        for line in f:
            paper = json.loads(line)
            # if idx == 30:
            #     print(paper['title'])
            #     print(' '.join(paper['body_text']['1']))
            #     print(' '.join(paper['body_text']['2']))
            #     print(' '.join(paper['body_text']['3']))
            #     print(' '.join(paper['body_text']['4']))
            #     time.sleep(100)

            section_withtext = {}
            for section, section_text in paper['body_text'].items():
                if section == '1':
                    section_name = 'intro'
                elif section == '2':
                    continue
                elif section == '3':
                    section_name = 'method'
                elif section == '4':
                    section_name = 'experiment'
                section_withtext[section_name] = section_text
            section_withtexts.append(section_withtext)
            # if idx == 769:
            #     print(paper['title'])
            #     print(paper['id'])
            # if paper['id'] == 1230:
            #     print(paper['title'])

            # print(paper['id'])
            idx += 1
            # time.sleep(1)

    print('Paper loading complete!')

    citations_show = defaultdict(dict)
    citation_len = [0, 0, 0, 0]
    with open(train_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            citation = json.loads(line)
            if citation['cite_label'] == 1:
                citation_adj_row_1.append(citation['src'])
                citation_adj_column_1.append(citation['dst'])
            elif citation['cite_label'] == 2:
                citation_adj_row_2.append(citation['src'])
                citation_adj_column_2.append(citation['dst'])
            elif citation['cite_label'] == 3:
                citation_adj_row_3.append(citation['src'])
                citation_adj_column_3.append(citation['dst'])
            elif citation['cite_label'] == 4:
                citation_adj_row_4.append(citation['src'])
                citation_adj_column_4.append(citation['dst'])
            else:
                print('Error!')
            citation_edges_train.append(citation)
            citation_pos_train[f"{citation['src']}_{citation['dst']}"].add(citation['cite_label'])
            citation_pos_all[f"{citation['src']}_{citation['dst']}"].add(citation['cite_label'])

            citation_len[citation['cite_label']-1] += 1
            if citation['dst'] not in citations_show:
                citations_show[citation['dst']] = {citation['cite_label']: [str(citation['src'])+':'+' '.join(citation['cite_text'])]}
            else:
                if citation['cite_label'] not in citations_show[citation['dst']]:
                    citations_show[citation['dst']][citation['cite_label']] = [str(citation['src'])+':'+' '.join(citation['cite_text'])]
                else:
                    citations_show[citation['dst']][citation['cite_label']].append(str(citation['src'])+':'+' '.join(citation['cite_text']))

                # papers[citation['dst']].append(' '.join(citation['cite_text'][:30]))

            # if citation['src'] == citation['dst']:
            #     print(citation['cite_label'])
            #     print(citation['cite_text'])
            # print(citation['src'], citation['dst'])
            # time.sleep(1)
        #     if int(citation['src']) == 2640 and int(citation['dst']) == 4140:
        #         print(citation['cite_label'])
        #         print(' '.join(citation['cite_text']))
        # time.sleep(100)

    with open(valid_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            citation = json.loads(line)
            citation_edges_valid.append(citation)

            citation_pos_valid[f"{citation['src']}_{citation['dst']}"].add(citation['cite_label'])
            citation_pos_all[f"{citation['src']}_{citation['dst']}"].add(citation['cite_label'])

            citation_len[citation['cite_label']-1] += 1
            if citation['dst'] not in citations_show:
                citations_show[citation['dst']] = {
                    citation['cite_label']: [str(citation['src']) + ':' + ' '.join(citation['cite_text'])]}
            else:
                if citation['cite_label'] not in citations_show[citation['dst']]:
                    citations_show[citation['dst']][citation['cite_label']] = [
                        str(citation['src']) + ':' + ' '.join(citation['cite_text'])]
                else:
                    citations_show[citation['dst']][citation['cite_label']].append(
                        str(citation['src']) + ':' + ' '.join(citation['cite_text']))

    # case_pos = defaultdict(set)
    # case_src = defaultdict(set)

    # 4645 4480 1,2 4
    # idx = 0

    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            citation = json.loads(line)
            citation_edges_test.append(citation)

            citation_pos_test[f"{citation['src']}_{citation['dst']}"].add(citation['cite_label'])
            citation_pos_all[f"{citation['src']}_{citation['dst']}"].add(citation['cite_label'])

            citation_len[citation['cite_label']-1] += 1
            if citation['dst'] not in citations_show:
                citations_show[citation['dst']] = {
                    citation['cite_label']: [str(citation['src']) + ':' + ' '.join(citation['cite_text'])]}
            else:
                if citation['cite_label'] not in citations_show[citation['dst']]:
                    citations_show[citation['dst']][citation['cite_label']] = [
                        str(citation['src']) + ':' + ' '.join(citation['cite_text'])]
                else:
                    citations_show[citation['dst']][citation['cite_label']].append(
                        str(citation['src']) + ':' + ' '.join(citation['cite_text']))

    # for dst, val in citations_show.items():
    #     if len(val)<4:
    #         continue
    #     if dst != 769:
    #         continue
    #     print(dst)
    #     for k, v in val.items():
    #         for vv in v:
    #             if ('default setting' in vv) and k==4:
    #                 print(dst)
    #
    #         print(k)
    #         print(v)
            # time.sleep(5)
        # print('---------')
        # time.sleep(20)

            # if citation['src'] == 2419 and citation['dst'] == 887 and citation['cite_label'] == 2:
            #     print(idx)
            # idx += 1
            # case_pos[f"{citation['src']}_{citation['dst']}"].add(citation['cite_label'])
            # case_src[f"{citation['src']}"].add(f"{citation['dst']}_{citation['cite_label']}")

    # for src_dst, label in case_pos.items():
    #     src = src_dst.split('_')[0]
    #     if len(label) >= 2 and len(case_src[src]) > 2:
    #         print(src_dst)
    #         print(label)
    #         print(case_src[src])


    print('Citation loading complete!')
    print(f'Train set: {len(citation_edges_train)}')
    print(f'Valid set: {len(citation_edges_valid)}')
    print(f'Test set: {len(citation_edges_test)}')
    print(f'Edges: {citation_len}\n')
    print(len(citation_pos_all))
    print(len([k for k, v in citation_pos_all.items() if len(v)>1]))

    use_cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    heter_graph = dgl.heterograph({
        ('paper', 'citing-intro', 'paper'): (citation_adj_row_1, citation_adj_column_1),
        ('paper', 'cited-intro', 'paper'): (citation_adj_column_1, citation_adj_row_1),
        ('paper', 'citing-relate', 'paper'): (citation_adj_row_2, citation_adj_column_2),
        ('paper', 'cited-relate', 'paper'): (citation_adj_column_2, citation_adj_row_2),
        ('paper', 'citing-method', 'paper'): (citation_adj_row_3, citation_adj_column_3),
        ('paper', 'cited-method', 'paper'): (citation_adj_column_3, citation_adj_row_3),
        ('paper', 'citing-experiment', 'paper'): (citation_adj_row_4, citation_adj_column_4),
        ('paper', 'cited-experiment', 'paper'): (citation_adj_column_4, citation_adj_row_4),
    }, num_nodes_dict={'paper': len(paper2index)})
    print('Graph building complete!\n')

    section_type2index = {}
    section_type2index['general'] = 0
    section_type2index['intro'] = 1
    section_type2index['method'] = 2
    section_type2index['experiment'] = 3
    section_type2index['relate'] = 4

    for ntype in heter_graph.ntypes:
        heter_graph.nodes[ntype].data['node_id'] = torch.arange(0, heter_graph.number_of_nodes(ntype))

    for etype in heter_graph.etypes:
        section = etype.split('-')[1]
        heter_graph.edges[etype].data['stype_id'] = \
            torch.ones(heter_graph.number_of_edges(etype), dtype=torch.long) * section_type2index[section]

    train_data_loader = DataLoader(
        dataset=S2orcDataset(heter_graph, section_withtexts, citation_edges_train, citation_pos_train, vocab2index, args),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=5,
        collate_fn=graph_collate,
        pin_memory=True
    )
    valid_data_loader = DataLoader(
        dataset=S2orcDataset(heter_graph, section_withtexts, citation_edges_valid, citation_pos_valid, vocab2index, args),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=5,
        collate_fn=graph_collate,
        pin_memory=True
    )
    test_data_loader = DataLoader(
        dataset=S2orcDataset(heter_graph, section_withtexts, citation_edges_test, citation_pos_test, vocab2index, args),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=5,
        collate_fn=graph_collate,
        pin_memory=True
    )

    vec = None
    if not args.pretrained:
        vec = GloVe(name='6B', dim=50)
        intersect = set(vocab2index.keys()) & set(vec.stoi.keys())
        print(f'{len(intersect)}/{len(vocab2index)} words found in pretrained vectors!\n')
        vec = vec.get_vecs_by_tokens(sorted(vocab2index, key=vocab2index.get))[:, :args.n_hid]

    model = CitationGeneration(device, heter_graph, section_type2index, vocab2index, vec, args)
    model = model.to(device)
    if args.pretrained:
        pretrained_dict = torch.load(pretrained_path)['model_state']
        model_dict = model.state_dict()
        # print(model_dict.keys())
        # print(pretrained_dict.keys())
        pretrained_dict_filted = {}
        for k, v in pretrained_dict.items():
            if 'encoder' in k:
                k_filted = k.replace('encoder', 'encoder.gcs_intra.text_encoder')
                pretrained_dict_filted[k_filted] = v
            # elif 'x_context' in k or 'out' in k:
            #     continue
            else:
                pretrained_dict_filted[k] = v
        # print(model_dict.keys())
        # print(pretrained_dict.keys())
        pretrained_dict_filted = {k: v for k, v in pretrained_dict_filted.items() if k in model_dict}
        print('Load pretrained model from seq2seq:')
        print(pretrained_dict_filted.keys())
        model_dict.update(pretrained_dict_filted)
        model.load_state_dict(model_dict)

        # for name, param in model.named_parameters():
        #     if name in pretrained_dict_filted:
        #         param.requires_grad = False


    # model.load('model/Computer Science_batch48_hid50_n_max_neigh20_n_max_len_section200_n_max_len_citation50_n_textenc_layer1_n_graphenc_layer1_n_dec_layer1_alpha10.1_alpha20.1_alpha30.1_lr0.005_dropout0.35_decay0.98_decaystep1_patience50_nopretrained.pt')
    # model.to(device)

    if args.mode == 'train':
        if args.is_checkpoint:
            print(f'Loading checkpoint from {args.save}')
            model.load(args.save)
            model.to(device)
        print('Start training')
        train(model, train_data_loader, valid_data_loader, test_data_loader, index2vocab, device, args)
    else:
        print(f'Loading model from {args.save}')
        model.load(args.save)
        model.to(device)
        if args.mode == 'test':
            bleu_score, rouge_score, all_decodes = inference(model, test_data_loader, index2vocab, device)
            return all_decodes

if __name__ == '__main__':
    main()
