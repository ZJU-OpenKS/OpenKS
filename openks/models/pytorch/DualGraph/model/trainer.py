import math
import time
import os
import copy
from datetime import datetime
from shutil import copyfile
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.utils import dropout_adj
from utils import torch_utils, scorer
from utils.torch_utils import arg_max

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(1)
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def sharpen(p, T):
    sharp_p = p**(1./T)
    sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
    return sharp_p


def idx_to_onehot(target, opt, confidence=None):
    sample_size, class_size = target.size(0), opt['num_classes']
    if confidence is None:
        y = torch.zeros(sample_size, class_size).to(device)
        y = y.scatter_(1, torch.unsqueeze(target.data, dim=1), 1)
    else:
        y = torch.ones(sample_size, class_size)
        y = y * (1 - confidence.data).unsqueeze(1).expand(-1, class_size)
        y[torch.arange(sample_size).long(), target.data] = confidence.data
    y = Variable(y)
    return y


def evaluate(model, dataset, evaluate_type='prf'):
    loader = DataLoader(dataset, batch_size=model.opt['batch_size'])
    if evaluate_type == 'prf':
        all_preds = []
        all_golds = []
        all_loss = 0
        for data in loader:
            data.to(device)
            target = data.y.long() # in case data.y is float
            _, preds, loss = model.predict(data, target)
            all_preds += preds
            all_loss += loss
            all_golds += target.tolist()
        acc = scorer.score(all_golds, all_preds)[0]
        return acc, all_loss

    elif evaluate_type == 'auc':
        logits, labels = [], []
        for data in loader:
            data.to(device)

            target = data.y.long() # in case data.y is float
            logits += model.predict(data)[0]
            labels += target.tolist()
        
        label_tmp = torch.LongTensor(labels).to(device)
        label_tmp = idx_to_onehot(label_tmp, model.opt) 
        logits = torch.FloatTensor(logits).to(device)
        sl_confidence = torch.ones(logits.shape[0])
        confidence = sl_confidence.unsqueeze(1).expand(-1, logits.size(1))
        if model.opt['cuda']:
            confidence = confidence.to(device)
        loss = F.binary_cross_entropy_with_logits(logits, label_tmp, weight=confidence)
        loss *= model.opt['num_classes']

        p, q = 0, 0
        for rel in range(model.opt['num_classes']):
            logits_rel = [logit[rel] for logit in logits]
            labels_rel = [1 if label == rel else 0 for label in labels]
            ranking = list(zip(logits_rel, labels_rel))
            ranking = sorted(ranking, key=lambda x: x[0], reverse=True)
            logits_rel, labels_rel = zip(*ranking)
            p += scorer.AUC(logits_rel, labels_rel)
            q += 1

        val_auc = p / q * 100
        return val_auc, loss


def calc_confidence(probs, exp):
    '''Calculate confidence score from raw probabilities'''
    return max(probs)**exp


class Trainer(object):
    """ A wrapper class for the training and evaluation of models. """

    def __init__(self, opt, model, tau = 2, model_type='predictor'):
        self.opt = opt
        self.model_type = model_type
        self.model = model
        self.tau = tau
        if model_type == 'predictor':
            self.criterion = nn.CrossEntropyLoss()
        elif model_type == 'pointwise':
            self.criterion = nn.BCEWithLogitsLoss()
        elif model_type == 'pairwise':
            self.criterion = nn.BCEWithLogitsLoss()  # Only a placeholder, will NOT use this criterion
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]

        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()

        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'], opt['weight_decay'])

    def train(self, dataset_train, dataset_val, dataset_unlabel):
        opt = self.opt.copy()
        train_label_loader = DataLoader(dataset_train, batch_size=opt['batch_size'], shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size=opt['batch_size'])
        val_acc_history = []

        # current_lr = opt['lr']
        # global_step = 0
        # format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
        # max_steps = len(train_label_loader) * opt['num_epoch']

        # start training
        epoch = 0
        patience = 0

        iter_supervised = None

        while True:
            epoch = epoch + 1
            train_loss = 0
            train_correct = 0
            train_unlabel_loss = 0

            # unlabel training for model_p
            if len(dataset_unlabel) >0:
                if self.model_type == 'predictor':   
                    train_unlabel_loader = DataLoader(dataset_unlabel, batch_size=self.opt["batch_size"], shuffle=True)
                    for data in train_unlabel_loader:
                        try:
                            sdata = next(iter_supervised)
                        except Exception:
                            iter_supervised = iter(train_label_loader)
                            sdata = next(iter_supervised)
                        finally:
                            sdata, _ = sdata
                                                  
                            # labels = sdata.y.long()
                            labels = torch.zeros(sdata.y.shape[0], self.opt['num_classes']).scatter_(1, sdata.y.unsqueeze(1), 1).to(device)
                            
                        loss = self.update_unlabel(data, sdata, labels, me_max=True,aug=self.opt['aug'])
                        train_unlabel_loss += loss

                # unlabel training for model_r
                else:
                    train_unlabel_loader = DataLoader(dataset_unlabel, batch_size=self.opt["batch_size"], shuffle=True)
                    for data in train_unlabel_loader:
                        loss = self.update_unlabel_r(data, aug = self.opt['aug'], temperature=0.5)
                        train_unlabel_loss += loss

            for data in train_label_loader:          
                loss, correct = self.update(data)
                train_loss += loss
                train_correct += correct
                # start_time = time.time()
                # global_step += 1  
                # if global_step % opt['log_step'] == 0:
                #     duration = time.time() - start_time
                #     print(format_str.format(datetime.now(), global_step, max_steps, epoch,
                #                           opt['num_epoch'], loss, duration, current_lr))
            
            
            # eval on val
            # print("Evaluating on val set...")
            if self.model_type == 'predictor':
                val_acc, val_loss = evaluate(self, dataset_val, evaluate_type="prf")
            else:
                val_acc, val_loss = evaluate(self, dataset_val, evaluate_type='auc')


            train_acc = train_correct / len(dataset_train)
            # train_loss = train_loss / len(train_label_loader) * opt['batch_size']  # avg loss per batch
            # val_loss = val_loss / len(val_loader) * opt['batch_size']
            train_loss = train_loss / len(dataset_train)
            val_loss = val_loss / len(dataset_val)
            if self.model_type == 'predictor':
                print("epoch {}: train_loss = {:.6f}, val_loss = {:.6f}, train_acc = {:.4f}, val_acc = {:.4f}".format(epoch, \
                    train_loss, val_loss, train_acc, val_acc))
            else:
                print("epoch {}: train_loss = {:.6f}, val_loss = {:.6f}, train_acc = {:.4f}, val_auc = {:.6f}".format(epoch, \
                    train_loss, val_loss, train_acc, val_acc))

            # save the current model
            model_file = opt['model_save_dir'] + '/checkpoint_epoch_{}.pt'.format(epoch)
            self.save(model_file, epoch)
            if epoch == 1 or val_acc > max(val_acc_history):  # new best
                path = opt["model_save_dir"] + "/best_model.pt"
                if os.path.exists(path):
                    os.remove(path)
                copyfile(model_file, path)
                patience = 0
            else:
                patience = patience + 1
            if epoch % opt['save_epoch'] != 0:
                os.remove(model_file)

            # # change learning rate
            # if len(val_score_history) > 10 and val_score <= val_score_history[-1] and \
            #         opt['optim'] in ['sgd', 'adam']:
            #     current_lr *= opt['weight_decay']
            #     self.update_lr(current_lr)

            val_acc_history += [val_acc]

            if opt['patience'] != 0:
                if patience == opt['patience'] and epoch > opt['num_epoch']:
                    break
            else:
                if epoch == opt['num_epoch']:
                    break
        print("Training ended with {} epochs.".format(epoch))

    def snn(self, query, supports, labels):
        """ Soft Nearest Neighbours similarity classifier """
        # Step 1: normalize embeddings
        query = torch.nn.functional.normalize(query)
        supports = torch.nn.functional.normalize(supports)

        # Step 3: compute similarlity between local embeddings
        out = F.softmax(query @ supports.T / self.tau) @ labels
        return out

    def update_unlabel_r(self, data, aug, temperature=0.5):
        self.model.train()
        self.optimizer.zero_grad()
        data, data_aug = data 
        node_num, _ = data.x.size()
        data = data.to(device)
        if aug == 'dnodes' or aug == 'subgraph' or aug == 'random2' or aug == 'random3' or aug == 'random4':
            # node_num_aug, _ = data_aug.x.size()
            edge_idx = data_aug.edge_index.numpy()
            _, edge_num = edge_idx.shape
            idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

            node_num_aug = len(idx_not_missing)
            data_aug.x = data_aug.x[idx_not_missing]

            data_aug.batch = data.batch[idx_not_missing]
            idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
            edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
            data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)
        data_aug = data_aug.to(device)

        _ , v = self.model(data)
        _ , v_aug= self.model(data_aug)

        batch_size = v.shape[0]
    
        out = torch.cat([F.normalize(v), F.normalize(v_aug)], dim=0)
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        pos_sim = torch.exp(torch.sum(F.normalize(v) * F.normalize(v_aug), dim=-1) / temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        # labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0) # n_views=2
        # labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        # labels = labels.to(device)

        # features = F.normalize(torch.cat([v, v_aug], dim=0), dim=1)
        # similarity_matrix = torch.matmul(features, features.T)

        # # discard the main diagonal from both: labels and similarities matrix
        # mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        # labels = labels[~mask].view(labels.shape[0], -1)
        # similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # # select and combine multiple positives
        # positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        # # select only the negatives the negatives
        # negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # logits = torch.cat([positives, negatives], dim=1)
        # labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
        # logits = logits / temperature
        # criterion = torch.nn.CrossEntropyLoss().to(device)
        # loss = criterion(logits, labels)
        # # loss = self.criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return loss.item()


    def update_unlabel(self, data, sdata, labels, me_max=True, aug='random2'):
        self.model.train()
        self.optimizer.zero_grad()
        data, data_aug = data 
        node_num, _ = data.x.size()
        data = data.to(device)
        # Remove the nodes that don't have edges 
        if aug == 'dnodes' or aug == 'subgraph' or aug == 'random2' or aug == 'random3' or aug == 'random4':
            # node_num_aug, _ = data_aug.x.size()
            edge_idx = data_aug.edge_index.numpy()
            _, edge_num = edge_idx.shape
            idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

            node_num_aug = len(idx_not_missing)
            data_aug.x = data_aug.x[idx_not_missing]

            data_aug.batch = data.batch[idx_not_missing]
            idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
            edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
            data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)
        data_aug = data_aug.to(device)
        sdata.to(device)

        '''
        anchor_views, 第一个view的rep
        anchor_supports, support的rep
        anchor_support_labels, support的label
        target_views, 第二个view的rep
        target_supports, support的rep
        target_support_labels, support的label
        '''
        anchor_views, _ = self.model(data)
        target_views, _ = self.model(data_aug)
        anchor_supports , _ = self.model(sdata)
        target_supports = anchor_supports

        
        # Step 1: compute anchor predictions
        probs = self.snn(anchor_views, anchor_supports, labels)

        # Step 2: compute targets for anchor predictions
        with torch.no_grad():
            targets = self.snn(target_views, target_supports, labels)
            targets = sharpen(targets, T=0.25)
          
            targets[targets < 1e-4] *= 0  # numerical stability

        # Step 3: compute cross-entropy loss H(targets, queries)
        closs = torch.mean(torch.sum(torch.log(probs**(-targets)), dim=1))

        # Step 4: compute me-max regularizer
        rloss = 0.
        if me_max:
            avg_probs = torch.mean(sharpen(probs, T=0.25), dim=0)
            rloss -= torch.sum(torch.log(avg_probs**(-avg_probs)))

        loss = closs + rloss
      
        # f = lambda x: torch.exp(x / tau)
        # refl_sim = f(self.sim(logits, logits))
        # between_sim = f(self.sim(logits, logits_aug))
        # loss = (-torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return loss.item()


    # train the model with a batch
    def update(self, data):
        """ Run a step of forward and backward model update. """
        self.model.train()
        self.optimizer.zero_grad()

        data, _ = data
        data.to(device)
        
        target = data.y.long() # in case data.y is float

        if self.model_type == 'pointwise':
            target = idx_to_onehot(target, self.opt) 

        encoding, logits = self.model(data)

        if self.model_type == 'pointwise':
            sl_confidence = torch.ones(logits.shape[0])
            
            confidence = sl_confidence.unsqueeze(1).expand(-1, logits.size(1))
            if self.opt['cuda']:
                confidence = confidence.cuda()
            loss = F.binary_cross_entropy_with_logits(logits, target, weight=confidence)
            loss *= self.opt['num_classes']
        elif self.model_type == 'pairwise':
            # Form a matrix with row_i indicate which samples are its negative samples (0, 1)
            matrix = torch.stack(
                [target.ne(rid) for rid in range(self.opt['num_classes'])])  # R * B matrix, ne:"!="
            matrix = matrix.index_select(0, target)  # B * B matrix
            sl_confidence = torch.ones(logits.shape[0])
            confidence = sl_confidence.unsqueeze(1).expand_as(matrix)
            if self.opt['cuda']:
                confidence = confidence.cuda()
            pos_logits = logits.gather(1, target.view(-1, 1))  # B * 1 logits
            # B * B logits out[i][j] = j-th sample's score on class y[i]
            neg_logits = logits.t().index_select(0, target)
            # calculate pairwise loss
            loss = F.binary_cross_entropy_with_logits(
                pos_logits - neg_logits, (matrix.float() * 1 / 2 + 1 / 2) * confidence
            )
            loss *= self.opt['num_classes']
        else:
            loss = self.criterion(logits, target)
            # loss = torch.mean(loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()

        if self.model_type == 'predictor':
            probs = F.softmax(logits, dim=-1)
            _, preds = probs.max(dim=1)
            correct = preds.eq(target).sum().item()
        else:
            correct = 0

        loss_train = loss.item()
        return loss_train, correct


    def predict(self, data, target=None):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        self.model.eval()
        _, logits = self.model(data)
        loss = None if target is None else self.criterion(logits, target).item()

        if self.model_type == 'predictor':
            probs = F.softmax(logits, dim=-1).data.cpu().numpy().tolist()
            preds = np.argmax(probs, axis=1).tolist()
        elif self.model_type == 'pointwise':
            probs = torch.sigmoid(logits).data.cpu().numpy().tolist()
            preds = logits.data.cpu().numpy().tolist()
        elif self.model_type == 'pairwise':
            probs = torch.sigmoid(logits).data.cpu().numpy().tolist()
            preds = logits.data.cpu().numpy().tolist()
        return probs, preds, loss


    def retrieve(self, dataset, k_samples, label_distribution=None):
        if self.model_type != 'predictor' and label_distribution is None:
            raise ValueError('Retrival from selector cannot be done without label_distribution')
        
        train_unlabel_loader = DataLoader(dataset, batch_size=self.opt["batch_size"], shuffle=False)

        probs = []
        target = []
        for data in train_unlabel_loader:
            data, _ = data
            data.to(device)
            probs += self.predict(data)[0]
            target += data.y.tolist()

        meta_idxs = []
        confidence_idxs = []
        num_instance = len(dataset)

        if label_distribution:
            label_distribution = {k: math.ceil(v * k_samples) for k, v in label_distribution.items()}

        if self.model_type == 'predictor':
            # ranking
            ranking = list(zip(range(num_instance), probs))
            ranking = sorted(ranking, key=lambda x: calc_confidence(x[1], self.opt['alpha']), reverse=True) #用第一个元素排序
           
            # selection
            for eid, prod in ranking:
                if len(meta_idxs) == k_samples:
                    break
                class_id, _ = arg_max(prod) #对其进行分类，class_id是最大类
                val = calc_confidence(prod, self.opt['alpha'])
                if label_distribution:
                    if not label_distribution[class_id]:
                        continue
                    label_distribution[class_id] -= 1
                meta_idxs.append((eid, class_id, target[eid]))
                confidence_idxs.append((eid, val))
            return meta_idxs
        else:
            for class_id in range(self.opt['num_classes']): # given a label
                # 根据label进行ranking
                # ranking
                ranking = list(zip(range(num_instance), [probs[k][class_id] for k in range(num_instance)]))
                ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
                # selection
            
                cnt = min(len(ranking), label_distribution.get(class_id, 0))

                for k in range(cnt):
                    eid, val = ranking[k]
                    meta_idxs.append((eid, class_id, target[eid]))
                    confidence_idxs.append((eid, val**self.opt['beta']))
               
                # meta_idxs (79, 1, 1)的形式
                meta_idxs.sort(key=lambda t: probs[t[0]][t[1]], reverse=True)
            return meta_idxs


    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    # save the model
    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),  # model parameters
            'encoder': self.model.encoder.state_dict(),
            'classifier': self.model.classifier.state_dict(),
            'config': self.opt,  # options
            'epoch': epoch,  # current epoch
            'model_type': self.model_type  # current epoch
        }
        try:
            torch.save(params, filename)
            # print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    # load the model
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.encoder.load_state_dict(checkpoint['encoder'])
        self.model.classifier.load_state_dict(checkpoint['classifier'])
        self.opt = checkpoint['config']
        self.model_type = checkpoint['model_type']
        if self.model_type == 'predictor':
            self.criterion = nn.CrossEntropyLoss()
        elif self.model_type == 'pointwise':
            self.criterion = nn.BCEWithLogitsLoss()



