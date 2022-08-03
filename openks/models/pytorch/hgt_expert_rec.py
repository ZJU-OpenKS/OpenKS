import dgl
import torch.nn as nn
import pickle
import numpy as np
from .kg_modules.hgt import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from .kg_modules.metrics import *
from ..model import ExpertRecModel
from collections import defaultdict
from typing import Dict
import pickle
import json
import os
import logging
logger = logging.getLogger(__name__)


def graph_collate(batch):
    project_id = default_collate([item[0] for item in batch])
    batch_sub_g = dgl.batch([item[1] for item in batch])
    similar_id = default_collate([item[2] for item in batch])
    pos_person = default_collate([item[3] for item in batch])
    neg_person_list = default_collate([item[4] for item in batch])
    return project_id, batch_sub_g, similar_id, pos_person, neg_person_list


class NSFDataset(Dataset):
    def __init__(self, G, data, projects_text_emb, args):
        super(NSFDataset, self).__init__()
        self.data = data
        self.args = args
        self.G = G
        self.proj_text_emb = np.array(list(projects_text_emb.values()))
        self.proj_text_id = np.array(list(projects_text_emb.keys()))

    def Calculate_Similarity(self, project_text_emb):
        score = np.sum(project_text_emb * self.proj_text_emb, axis=1)
        indices = np.argpartition(score, -self.args.max_project)
        similar_id = self.proj_text_id[indices[-self.args.max_project:]]
        return similar_id

    def get_subgraph_from_heterograph(self, similar_id, person_list):

        subgraph_in = dgl.sampling.sample_neighbors(self.G, nodes={'project':similar_id, 'person':person_list}, fanout=self.args.n_max_neigh[0],
                                                        edge_dir='in')
        nodes_subgraph = defaultdict(set)
        nodes_subgraph['project'].update(similar_id)
        nodes_subgraph['person'].update(person_list)

        for layer in range(1, self.args.n_neigh_layer):
            subgraph = subgraph_in
            new_adj_nodes = defaultdict(set)
            for node_type_1, edge_type, node_type_2 in subgraph.canonical_etypes:
                nodes_id_1, nodes_id_2 = subgraph.all_edges(etype=edge_type)
                new_adj_nodes[node_type_1].update(set(nodes_id_1.numpy()).difference(nodes_subgraph[node_type_1]))
                new_adj_nodes[node_type_2].update(set(nodes_id_2.numpy()).difference(nodes_subgraph[node_type_2]))
                nodes_subgraph[node_type_1].update(new_adj_nodes[node_type_1])
                nodes_subgraph[node_type_2].update(new_adj_nodes[node_type_2])

            new_adj_nodes = {key: list(value) for key, value in new_adj_nodes.items()}

            subgraph_in = dgl.sampling.sample_neighbors(self.G, nodes=dict(new_adj_nodes),
                                                        fanout=self.args.n_max_neigh[layer],
                                                        edge_dir='in')
        nodes_sampled = {}
        for node_type in nodes_subgraph.keys():
            nodes_sampled[node_type] = torch.LongTensor(list(nodes_subgraph[node_type]))
        sub_g = self.G.subgraph(nodes_sampled)

        return sub_g

    def __getitem__(self, index):
        project_id = self.data[index][0]
        project_text_emb = self.data[index][1]
        pos_person = self.data[index][2]
        neg_person_list = self.data[index][3]

        similar_id = self.Calculate_Similarity(project_text_emb)

        person_list = torch.LongTensor([pos_person] + neg_person_list)
        similar_id = torch.LongTensor(similar_id)
        sub_g = self.get_subgraph_from_heterograph(similar_id, person_list)

        return project_id, sub_g, similar_id, pos_person, neg_person_list

    def __len__(self):
        return len(self.data)


@ExpertRecModel.register("HGTExpertRec", "PyTorch")
class HGTExpertRec(ExpertRecModel):
    def __init__(self, data_path, args: Dict):
        super().__init__(data_path)
        self.args = args
        use_cuda = torch.cuda.is_available() and args.cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        logger.info(self.device)

    def preprocess_data(self):
        root_path = self.data_path
        if os.path.exists(root_path + '/index.pkl'):
            logger.info("Preprocess have been finished. Skip.")
            return
        with open(root_path + 'entities_paper.pkl', 'rb') as f:
            papers = pickle.load(f)
        with open(root_path + 'entities_person.pkl', 'rb') as f:
            persons = pickle.load(f)
        with open(root_path + 'entities_project.pkl', 'rb') as f:
            projects = pickle.load(f)
        with open(root_path + 'train_rel_is_principal_investigator_of.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open(root_path + 'val_rel_is_principal_investigator_of.pkl', 'rb') as f:
            valid_data = pickle.load(f)
        with open(root_path + 'test_rel_is_principal_investigator_of.pkl', 'rb') as f:
            test_data = pickle.load(f)
        with open(root_path + 'project_emb_all_mpnet_base_v2.pkl', 'rb') as f:
            emb_data = pickle.load(f)

        train_project = []
        for train_one in train_data:
            train_project.append(train_one[2])

        project2index = {}
        index2project = {}
        for index in range(len(projects)):
            projects[index] = json.loads(projects[index])
            project2index[projects[index]['AwardID']] = index
            index2project[index] = projects[index]['AwardID']

        paper2index = {}
        index2paper = {}
        for index in range(len(papers)):
            paper2index[papers[index]['_id']] = index
            index2paper[index] = papers[index]['_id']

        person2index = {}
        index2person = {}
        index = 0
        for id in persons:
            person2index[id] = index
            index2person[index] = id
            index += 1

        project_main_row = []
        person_main_col = []
        project_co_row = []
        person_co_col = []
        for project in projects:
            if project['AwardID'] in train_project:
                for role in project['Investigator']:
                    if role['RoleCode'] == 'Principal Investigator':
                        project_main_row.append(project2index[project['AwardID']])
                        person_main_col.append(person2index[role['uid']])
                    elif role['RoleCode'] == 'Co-Principal Investigator':
                        project_co_row.append(project2index[project['AwardID']])
                        person_co_col.append(person2index[role['uid']])

        paper_auther_row = []
        author_col = []
        paper_ref_row = []
        paper_ref_col = []
        for paper in papers:
            for author in paper['authors']:
                paper_auther_row.append(paper2index[paper['_id']])
                author_col.append(person2index[author['_id']])
            try:
                #sometimes no references or no information about it
                temp = paper['references']
            except:
                continue
            for ref in temp:
                try:
                    paper_ref_col.append(paper2index[ref])
                except:
                    continue
                paper_ref_row.append(paper2index[paper['_id']])

        projects_text_emb = {}
        train_projects_text_emb ={}
        for i in range(len(emb_data)):
            if emb_data[i][2] is not None:
                projects_text_emb[project2index[emb_data[i][0]]] = emb_data[i][2]
                if emb_data[i][0] in train_project:
                    train_projects_text_emb[project2index[emb_data[i][0]]] = emb_data[i][2]

        train_dataset = []
        for index in range(len(train_data)):
            project_id = project2index[train_data[index][2]]
            pos_person = person2index[train_data[index][1]]
            project_text_emb = projects_text_emb[project_id]
            neg_person = []
            for i in range(len(train_data[index][4])):
                neg_person.append(person2index[train_data[index][4][i]])
            train_dataset.append((project_id, project_text_emb, pos_person, neg_person))

        valid_dataset = []
        for index in range(len(valid_data)):
            project_id = project2index[valid_data[index][2]]
            pos_person = person2index[valid_data[index][1]]
            project_text_emb = projects_text_emb[project_id]
            neg_person = []
            for i in range(len(valid_data[index][4])):
                neg_person.append(person2index[valid_data[index][4][i]])
            valid_dataset.append((project_id, project_text_emb, pos_person, neg_person))

        test_dataset = []
        for index in range(len(test_data)):
            project_id = project2index[test_data[index][2]]
            pos_person = person2index[test_data[index][1]]
            project_text_emb = projects_text_emb[project_id]
            neg_person = []
            for i in range(len(test_data[index][4])):
                neg_person.append(person2index[test_data[index][4][i]])
            test_dataset.append((project_id, project_text_emb, pos_person, neg_person))

        with open(root_path + '/index.pkl', 'wb') as f:
            pickle.dump(project2index, f)
            pickle.dump(index2project, f)
            pickle.dump(paper2index, f)
            pickle.dump(index2paper, f)
            pickle.dump(person2index, f)
            pickle.dump(index2person, f)

        with open(root_path + '/dgl_data.pkl', 'wb') as f:
            pickle.dump(project_main_row, f)
            pickle.dump(person_main_col, f)
            pickle.dump(project_co_row, f)
            pickle.dump(person_co_col, f)
            pickle.dump(paper_ref_row, f)
            pickle.dump(paper_ref_col, f)
            pickle.dump(paper_auther_row, f)
            pickle.dump(author_col, f)

        with open(root_path + '/train_dataset.pkl', 'wb') as f:
            pickle.dump(train_dataset, f)

        with open(root_path + '/valid_dataset.pkl', 'wb') as f:
            pickle.dump(valid_dataset, f)

        with open(root_path + '/test_dataset.pkl', 'wb') as f:
            pickle.dump(test_dataset, f)

        with open(root_path + '/projects_text_emb.pkl', 'wb') as f:
            pickle.dump(train_projects_text_emb, f)

    def load_data_and_model(self):
        index_path = self.data_path + '/index.pkl'
        dgl_data_path = self.data_path + '/dgl_data.pkl'
        emb_data_path = self.data_path + '/projects_text_emb.pkl'
        train_data_path = self.data_path + '/train_dataset.pkl'
        valid_data_path = self.data_path + '/train_dataset.pkl'
        test_data_path = self.data_path + '/train_dataset.pkl'
        with open(index_path, 'rb') as f:
            project2index = pickle.load(f)
            index2project = pickle.load(f)
            paper2index = pickle.load(f)
            index2paper = pickle.load(f)
            person2index = pickle.load(f)
            index2person = pickle.load(f)
        with open(dgl_data_path, 'rb') as f:
            project_main_row = pickle.load(f)
            person_main_col = pickle.load(f)
            project_co_row = pickle.load(f)
            person_co_col = pickle.load(f)
            paper_ref_row = pickle.load(f)
            paper_ref_col = pickle.load(f)
            paper_auther_row = pickle.load(f)
            author_col = pickle.load(f)
        with open(emb_data_path, "rb") as f:
            train_projects_text_emb = pickle.load(f)
        with open(train_data_path, "rb") as f:
            train_data = pickle.load(f)
        with open(valid_data_path, "rb") as f:
            valid_data = pickle.load(f)
        with open(test_data_path, "rb") as f:
            test_data = pickle.load(f)
        self.G = dgl.heterograph({
            ('project', 'investigated-by', 'person'): (project_main_row, person_main_col),
            ('person', 'investigate', 'project'): (person_main_col, project_main_row),
            ('project', 'co-investigated-by', 'person'): (project_co_row, person_co_col),
            ('person', 'co-investigate', 'project'): (person_co_col, project_co_row),
            ('paper', 'cite', 'paper'): (paper_ref_row, paper_ref_col),
            ('paper', 'cited-by', 'paper'): (paper_ref_col, paper_ref_row),
            ('paper', 'writed-by', 'person'): (paper_auther_row, author_col),
            ('person', 'write', 'paper'): (author_col, paper_auther_row),
        })
        logger.info(self.G)
        logger.info(self.G.ntypes)
        logger.info(self.G.etypes)
        self.node_dict = {}
        self.edge_dict = {}
        for ntype in self.G.ntypes:
            self.node_dict[ntype] = len(self.node_dict)
        for etype in self.G.etypes:
            self.edge_dict[etype] = len(self.edge_dict)
        self.node_emb = {}
        for ntype in self.G.ntypes:
            self.G.nodes[ntype].data['id'] = torch.arange(0, self.G.number_of_nodes(ntype))
            emb = nn.Embedding(self.G.number_of_nodes(ntype), self.args.n_dim) #, requires_grad=False
            nn.init.xavier_uniform_(emb.weight)
            emb.weight.data[0] = 0
            self.node_emb[ntype] = emb.to(self.device)
        self.train_data_loader = DataLoader(
            dataset=NSFDataset(self.G, train_data, train_projects_text_emb, self.args),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=graph_collate,
            pin_memory=True
        )
        self.valid_data_loader = DataLoader(
            dataset=NSFDataset(self.G, valid_data, train_projects_text_emb, self.args),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=graph_collate,
            pin_memory=True
        )
        self.test_data_loader = DataLoader(
            dataset=NSFDataset(self.G, test_data, train_projects_text_emb, self.args),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=graph_collate,
            pin_memory=True
        )
        self.model = HGT(self.node_emb,
            self.node_dict, self.edge_dict,
            n_inp=self.args.n_dim,
            n_hid=self.args.n_hid,
            n_out=self.args.n_dim,
            n_layers=self.args.n_neigh_layer,
            n_heads=self.args.n_head,
            use_norm=True).to(self.device)
	
    def train_expert(self):
        best_ndcg = 0.0
        best_epoch = -1
        n = len(self.train_data_loader)
        loss_fn = nn.BCELoss().to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.decay_step, gamma=self.args.decay)

        # eval(model, args, valid_data_loader)
        for epoch in np.arange(self.args.n_epoch) + 1:
            logger.info('Start epoch: %d' % epoch)
            self.model.train()
            for step, batch_data in tqdm(enumerate(self.train_data_loader), total = n):
                project_id, sub_g, similar_id, pos_person, neg_person_list = batch_data
                batch_size = project_id.shape[0]
                pos_label = torch.ones(batch_size).to(self.device)
                neg_label = torch.zeros(batch_size).to(self.device)
                sub_g = sub_g.to(self.device)
                project_emb, person_emb = self.model(sub_g, 'project', 'person')
                cur_emb = torch.zeros(batch_size, self.args.n_dim).to(self.device)
                for i in range(batch_size):
                    for j in range(self.args.max_project):
                        cur_emb[i] += project_emb[similar_id[i][j].item()]
                cur_emb /= self.args.max_project
                pos_person_emb = []
                for i in range(batch_size):
                    pos_person_emb.append(person_emb[pos_person[i].item()])
                pos_person_emb = torch.stack(pos_person_emb)
                neg_person_emb = []
                for i in range(batch_size):
                    neg_person_emb.append(person_emb[neg_person_list[i][0].item()])
                neg_person_emb = torch.stack(neg_person_emb)

                pos_score = torch.sigmoid(torch.sum(cur_emb * pos_person_emb, -1))
                neg_score = torch.sigmoid(torch.sum(cur_emb * neg_person_emb, -1))
                pos_loss = loss_fn(pos_score, pos_label)
                neg_loss = loss_fn(neg_score, neg_label)
                loss = pos_loss + neg_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                optimizer.step()
            scheduler.step()
            mean_p, mean_r, mean_h, mean_ndcg = eval(self.model, self.args, self.valid_data_loader)
            logger.info(f'Valid:\tprecision@{self.args.topk}:{mean_p:.6f}, recall@{self.args.topk}:{mean_r:.6f}, '
                  f'hr@{self.args.topk}:{mean_h:.6f}, ndcg@{self.args.topk}:{mean_ndcg:.6f}')
            if mean_ndcg > best_ndcg:
                best_epoch = epoch
                best_ndcg = mean_ndcg
                self.model.save(self.args.save)
                logger.info('Model save for higher ndcg %f in %s' % (best_ndcg, self.args.save))
            if epoch - best_epoch >= self.args.patience:
                logger.info('Stop training after %i epochs without improvement on validation.' % self.args.patience)
                break
        self.model.load(self.args.save)
        mean_p, mean_r, mean_h, mean_ndcg = eval(self.model, self.args, self.test_data_loader)
        logger.info(f'Test:\tprecision@{self.args.topk}:{mean_p:.6f}, recall@{self.args.topk}:{mean_r:.6f}, '
              f'hr@{self.args.topk}:{mean_h:.6f}, ndcg@{self.args.topk}:{mean_ndcg:.6f}')
	
    def train_team(self):
        return NotImplemented
	
    def inference_expert(self):
        return NotImplemented
	
    def inference_team(self):
        return NotImplemented
	
    def evaluate(self):
        self.model.eval()
        eval_p = []
        eval_r = []
        eval_h = []
        eval_ndcg = []
        eval_len = []
        n = len(self.eval_data_loader)
        with torch.no_grad():
            for step, batch_data in tqdm(enumerate(self.eval_data_loader), total = n):
                project_id, sub_g, similar_id, pos_person, neg_person_list = batch_data
                sub_g = sub_g.to(self.device)
                batch_size = project_id.shape[0]
                project_emb, person_emb = self.model(sub_g, 'project', 'person')

                cur_emb = torch.zeros(batch_size, self.args.n_dim).to(self.device)
                for i in range(batch_size):
                    for j in range(self.args.max_project):
                        cur_emb[i] += project_emb[similar_id[i][j].item()]
                cur_emb /= self.args.max_project

                neg_person_list = torch.transpose(torch.stack(neg_person_list), 0, 1)
                pos_person = pos_person.unsqueeze(1)
                person_list = torch.cat((pos_person, neg_person_list), dim=1)
                cur_emb = cur_emb.unsqueeze(1)

                eval_person_emb = []
                for i in range(batch_size):
                    temp =[]
                    for j in range(person_list.size(1)):
                        temp.append(person_emb[person_list[i][j].item()])
                    temp = torch.stack(temp)
                    eval_person_emb.append(temp)
                eval_person_emb = torch.stack(eval_person_emb)

                score = torch.sigmoid(torch.sum(cur_emb * eval_person_emb, -1))
                pred_person_index = torch.topk(score, self.args.topk)[1].tolist()
                for i in range(batch_size):
                    p_at_k = getP(pred_person_index[i], [0])
                    r_at_k = getR(pred_person_index[i], [0])
                    h_at_k = getHitRatio(pred_person_index[i], [0])
                    ndcg_at_k = getNDCG(pred_person_index[i], [0])
                    eval_p.append(p_at_k)
                    eval_r.append(r_at_k)
                    eval_h.append(h_at_k)
                    eval_ndcg.append(ndcg_at_k)
                    eval_len.append(1)
                if (step % self.args.log_step == 0) and step > 0:
                    logger.info('Valid epoch:[{}/{} ({:.0f}%)]\t Recall: {:.6f}, AvgRecall: {:.6f}'.format(
                        step, len(self.eval_data_loader),
                        100. * step / len(self.eval_data_loader),
                        r_at_k, np.mean(eval_r)
                    ))
            mean_p = np.mean(eval_p)
            mean_r = np.mean(eval_r)
            mean_h = np.sum(eval_h) / np.sum(eval_len)
            mean_ndcg = np.mean(eval_ndcg)
            return mean_p, mean_r, mean_h, mean_ndcg
	
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
