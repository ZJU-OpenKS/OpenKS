import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import multi_HAN
from torch.optim import lr_scheduler
from utils import YelpDataset, AmazonDataset, MovielensDataset
from metrics import *
import time

def parse_args():
    parser = argparse.ArgumentParser(description='DisenHAN')
    parser.add_argument('--cur_dim', type=int, default=100,
                        help='dimension of embeddings')
    parser.add_argument('--n_facet', type=int, default=[5,5],
                        help='number of facet for each embedding for each layer')
    parser.add_argument('--n_layer', type=int, default=2,
                        help='number of layers for the model')
    # yelp
    # parser.add_argument('--n_neigh', nargs='?', default=[[[5,5],[5,5,5],[5],[5]], [[5,5],[5,5,5],[5],[5]]],
    #                     help='number of neighbor to sample')

    # amazon
    # parser.add_argument('--n_neigh', nargs='?', default=[[[20],[20,20,1,3], [20], [20]], [[10],[10,10,1,1], [10], [10]]],
    #                     help='number of neighbor to sample')

    # movielens
    # parser.add_argument('--n_neigh', nargs='?', default=[[[20], [20, 10, 1, 1, 2], [10], [10], [20], [20]], [[10], [10, 5, 1, 1, 1], [5], [5], [10], [10]]],
    #                     help='number of neighbor to sample')
    # parser.add_argument('--n_neigh', nargs='?', default=[[[20], [20, 10, 10, 10, 10], [20], [20], [20], [20]],
    #                                                      [[10], [10, 10, 10, 10, 10], [10], [10], [10], [10]]],
    #                     help='number of neighbor to sample')

    parser.add_argument('--n_neigh', nargs='?', default=[20,10],
                        help='number of neighbor to sample')

    parser.add_argument('--n_iter', type=int, default=5,
                        help='number of iterations when routing')
    parser.add_argument('--n_neg', type=int, default=5,
                        help='number of negative instances to pair with a positive instance')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='initial learning rate')
    # parser.add_argument('--reg', type=float, default=1e-5,
    #                     help='reg rate')
    parser.add_argument('--train_prop', type=float, default=1.0,
                        help='train data use proportion')
    parser.add_argument('--dropout', type=float, default=0.35,
                        help='dropout rate (1 - keep probability)')
    parser.add_argument('--decay', type=float, default=0.98,
                        help='learning rate decay rate')
    parser.add_argument('--decay_step', type=int, default=1,
                        help='learning rate decay step')
    parser.add_argument('--log_step', type=int, default=1e2,
                        help='log print step')
    parser.add_argument('--epochs', type=int, default=200,
                        help='upper epoch limit')
    parser.add_argument('--patience', type=int, default=20,
                        help='extra iterations before early-stopping')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use GPU for training')
    parser.add_argument('--save', type=str, default='model/',
                        help='path to save the final model')
    parser.add_argument('--topk', type=int,default=10,
	                    help="compute metrics@top_k")
    parser.add_argument('--dataset', default='movielens',
                        help='dataset name, yelp, amazon, movielens can choose')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', type=str,
                        help='passing "test" will only run a single evaluation, otherwise full training will be performed')
    args = parser.parse_args()
    args.save = args.save + args.dataset
    args.save = args.save + '_batch{}'.format(args.batch_size)
    args.save = args.save + '_lr{}'.format(args.lr)
    args.save = args.save + '_emb{}'.format(args.cur_dim)
    args.save = args.save + '_iter{}'.format(args.n_iter)
    args.save = args.save + '_layer{}'.format(args.n_layer)
    args.save = args.save + '_neigh{}'.format(sum(args.n_neigh))
    args.save = args.save + '_facet{}'.format(sum(args.n_facet))
    args.save = args.save + '_negsize{}'.format(args.n_neg)
    args.save = args.save + '_trainsize{}'.format(args.train_prop)
    # args.save = args.save + '_reg{}'.format(args.reg)
    args.save = args.save + '_decay{}'.format(args.decay)
    args.save = args.save + '_decaystep{}'.format(args.decay_step)
    args.save = args.save + '_patience{}_test3.pt'.format(args.patience)
    return args

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def to_device(neighs_layers, device):
    if isinstance(neighs_layers, list):
        neighs_layers = [to_device(neighs_layer, device) for neighs_layer in neighs_layers]
    else:
        neighs_layers= neighs_layers.to(device)
    return neighs_layers

def train_one_epoch(model, train_data_loader, optimizer, epoch, device, args):
    model.train()
    epoch_loss = []
    for step, batch_data in enumerate(train_data_loader):
        label, user_neighs_layers, business_neighs_layers = batch_data
        label = label.to(device)
        user_neighs_layers = to_device(user_neighs_layers, device)
        business_neighs_layers = to_device(business_neighs_layers, device)
        optimizer.zero_grad()
        output = model(user_neighs_layers, business_neighs_layers)
        loss_fn = nn.BCELoss(reduction='none').to(device)
        loss = loss_fn(output, label)
        # try:
        loss = torch.mean(torch.sum(loss, 1))
        loss.backward()
        # print(loss)
        # except:
        #     print('error')
        #     print(loss)
        #     print(loss.shape)
        #     time.sleep(100)
        optimizer.step()
        epoch_loss.append(loss.item())
        if (step % args.log_step == 0) and step > 0:
            print('Train epoch: {}[{}/{} ({:.0f}%)]\tLr:{:.6f}, Loss: {:.6f}, AvgL: {:.6f}'.format(epoch, step, len(train_data_loader),
                                                    100. * step / len(train_data_loader), get_lr(optimizer), loss.item(), np.mean(epoch_loss)))

    mean_epoch_loss = np.mean(epoch_loss)
    return mean_epoch_loss

def eval(model, eval_data_loader, K, device):
    model.eval()
    eval_p = []
    eval_r = []
    eval_h = []
    eval_ndcg = []
    eval_len = []
    with torch.no_grad():
        for step, batch_data in enumerate(eval_data_loader):
            label, user_neighs_layers, business_neighs_layers = batch_data
            label = label.to(device)
            user_neighs_layers = to_device(user_neighs_layers, device)
            business_neighs_layers = to_device(business_neighs_layers, device)
            logit = model(user_neighs_layers, business_neighs_layers)
            pred_items = torch.topk(logit, K)[1]
            pred_items = pred_items.tolist()
            gt_items = torch.nonzero(label)[:, 1].tolist()
            p_at_k = getP(pred_items, gt_items)
            r_at_k = getR(pred_items, gt_items)
            h_at_k = getHitRatio(pred_items, gt_items)
            ndcg_at_k = getNDCG(pred_items, gt_items)
            eval_p.append(p_at_k)
            eval_r.append(r_at_k)
            eval_h.append(h_at_k)
            eval_ndcg.append(ndcg_at_k)
            eval_len.append(len(gt_items))
    mean_p = np.mean(eval_p)
    mean_r = np.mean(eval_r)
    mean_h = np.sum(eval_h)/np.sum(eval_len)
    mean_ndcg = np.mean(eval_ndcg)
    return mean_p, mean_r, mean_h, mean_ndcg

def valid(model, valid_data_loader, topk, device):
    print('Start Valid')
    mean_p, mean_r, mean_h, mean_ndcg = eval(model, valid_data_loader, topk, device)
    print('Valid:\tprecision@%d:%f, recall@%d:%f, hr@%d:%f, ndcg@%d:%f' % (topk, mean_p, topk, mean_r, topk, mean_h, topk, mean_ndcg))
    return mean_p, mean_r, mean_h, mean_ndcg

def inference(model, test_data_loader, topk, device):
    print('Start Test')
    mean_p, mean_r, mean_h, mean_ndcg = eval(model, test_data_loader, topk, device)
    print('Test:\tprecision@%d:%f, recall@%d:%f, hr@%d:%f, ndcg@%d:%f' % (topk, mean_p, topk, mean_r, topk, mean_h, topk, mean_ndcg))

if __name__ == '__main__':
    args = parse_args()
    if args.dataset == 'yelp':
        train_data_path = '../yelp_dataset/rates_1/rate_train.pickle'
        valid_data_path = '../yelp_dataset/rates_1/valid_with_neg.pickle'
        test_data_path = '../yelp_dataset/rates_1/test_with_neg.pickle'
        adj_paths = []
        adj_names = ['adj_UU.pickle', 'adj_UB.pickle', 'adj_BCi.pickle', 'adj_BCa.pickle']
        for name in adj_names:
            adj_paths.append('../yelp_dataset/adjs_1/' + name)
        index_to_id_paths = []
        index_to_ids = []
        index_to_id_names = ['index_to_userid.pickle', 'index_to_businessid.pickle', 'index_to_cityid.pickle', 'index_to_categoryid.pickle']
        for name in index_to_id_names:
            index_to_id_paths.append('../yelp_dataset/adjs_1/' + name)
        for path in index_to_id_paths:
            with open(path, 'rb') as f:
                index_to_ids.append(pickle.load(f))
        n_users, n_businesses, n_cities, n_categories = [len(index_to_id) for index_to_id in index_to_ids]
        n_nodes_list = [n_users, n_businesses, n_cities, n_categories]
        print(n_nodes_list)
        user_neighs_type = [0, 1]
        business_neighs_type = [0, 2, 3]
        city_neighs_type = [1]
        category_neighs_type = [1]
        neighs_type = [user_neighs_type, business_neighs_type, city_neighs_type, category_neighs_type]
        Dataset = YelpDataset
    elif args.dataset == 'amazon':
        train_data_path = '../amazon_dataset/Electronics/rates_1/train_data.pickle'
        valid_data_path = '../amazon_dataset/Electronics/rates_1/valid_with_neg_sample.pickle'
        test_data_path = '../amazon_dataset/Electronics/rates_1/test_with_neg_sample.pickle'
        adj_paths = []
        adj_names = ['adj_UI.pickle', 'adj_II.pickle', 'adj_IBr.pickle', 'adj_ICa.pickle']
        for name in adj_names:
            adj_paths.append('../amazon_dataset/Electronics/adjs_1/' + name)
        index_to_id_paths = []
        index_to_ids = []
        index_to_id_names = ['index2user_id.pickle', 'index2item_id.pickle', 'index2brand_id.pickle', 'index2category_id.pickle']
        for name in index_to_id_names:
            index_to_id_paths.append('../amazon_dataset/Electronics/adjs_1/' + name)
        for path in index_to_id_paths:
            with open(path, 'rb') as f:
                index_to_ids.append(pickle.load(f))
        n_users, n_items, n_brands, n_categories = [len(index_to_id) for index_to_id in index_to_ids]
        n_nodes_list = [n_users, n_items, n_brands, n_categories]
        print(n_nodes_list)
        user_neighs_type = [1]
        item_neighs_type = [0, 1, 2, 3]
        brand_neighs_type = [1]
        category_neighs_type = [1]
        neighs_type = [user_neighs_type, item_neighs_type, brand_neighs_type, category_neighs_type]
        Dataset = AmazonDataset
    elif args.dataset == 'movielens':
        train_data_path = './data/movielens_dataset/rates/train_data.pickle'
        valid_data_path = './data/movielens_dataset/rates/valid_with_neg_sample.pickle'
        test_data_path = './data/movielens_dataset/rates/test_with_neg_sample.pickle'
        adj_paths = []
        adj_names = ['adj_UI.pickle', 'adj_IA.pickle', 'adj_ID.pickle', 'adj_IC.pickle', 'adj_IG.pickle']
        for name in adj_names:
            adj_paths.append('./data/movielens_dataset/adjs/' + name)
        index_to_id_paths = []
        index_to_ids = []
        index_to_id_names = ['index2user_id.pickle', 'index2movie_id.pickle', 'index2actor_id.pickle', 'index2director_id.pickle', 'index2country_id.pickle', 'index2genre_id.pickle']
        for name in index_to_id_names:
            index_to_id_paths.append('./data/movielens_dataset/adjs/' + name)
        for path in index_to_id_paths:
            with open(path, 'rb') as f:
                index_to_ids.append(pickle.load(f))
        n_users, n_movies, n_actors, n_directors, n_countries, n_genres = [len(index_to_id) for index_to_id in index_to_ids]
        n_nodes_list = [n_users, n_movies, n_actors, n_directors, n_countries, n_genres]
        user_neighs_type = [1]
        movie_neighs_type = [0, 2, 3, 4, 5]
        actor_neighs_type = [1]
        director_neighs_type = [1]
        country_neighs_type = [1]
        genre_neighs_type = [1]
        neighs_type = [user_neighs_type, movie_neighs_type, actor_neighs_type, director_neighs_type, country_neighs_type, genre_neighs_type]
        Dataset = MovielensDataset
    use_cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = multi_HAN(n_nodes_list, neighs_type, args)

    # for name, param in model.named_parameters():
    #     print('name: ', name)
    #     print(type(param))
    #     print('param.shape: ', param.shape)
    #     print('param.requires_grad: ', param.requires_grad)
    #     print('=====')

    if args.mode == 'train':
        model = model.to(device)
        train_data_loader = DataLoader(dataset=Dataset(n_nodes_list, train_data_path, adj_paths, args.n_layer, args.n_neigh, args.n_neg, args.train_prop, 'train'),
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=20,
                                       pin_memory=True)

        valid_data_loader = DataLoader(dataset=Dataset(n_nodes_list, valid_data_path, adj_paths, args.n_layer, args.n_neigh, args.n_neg, args.train_prop, 'valid'),
                                       batch_size=1,
                                       shuffle=False,
                                       num_workers=20,
                                       pin_memory=True)

        test_data_loader = DataLoader(dataset=Dataset(n_nodes_list, test_data_path, adj_paths, args.n_layer, args.n_neigh, args.n_neg, args.train_prop, 'test'),
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=20,
                                      pin_memory=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay)
        best_ndcg = 0.0
        best_epoch = -1
        for epoch in range(args.epochs):
            print('Start epoch: ', epoch)
            mean_loss = train_one_epoch(model, train_data_loader, optimizer, epoch, device, args)
            valid_precision, valid_recall, valid_hr, valid_ndcg = valid(model, valid_data_loader, args.topk, device)
            inference(model, test_data_loader, args.topk, device)
            scheduler.step()
            if valid_ndcg > best_ndcg:
                best_epoch = epoch
                best_ndcg = valid_ndcg
                with open(args.save, 'wb') as f:
                    torch.save(model.state_dict(), f)
                print('Model save for higher ndcg %f in %s' % (best_ndcg, args.save))
            if epoch - best_epoch >= args.patience:
                print('Stop training after %i epochs without improvement on validation.' % args.patience)
                break
        model.load_state_dict(torch.load(args.save))
        model.to(device)
        inference(model, test_data_loader, args.topk, device)
    else:
        # test_data_path = 'yelp_dataset/rates/test_with_neg'
        # with open(test_data_path, 'rb') as f:
        #     test_data = pickle.load(f)

        test_data_loader = DataLoader(
            dataset=Dataset(n_nodes_list, test_data_path, adj_paths, args.n_layer, args.n_neigh, args.n_neg, args.train_prop, 'test'),
            batch_size=1,
            shuffle=True,
            num_workers=20,
            pin_memory=True)
        model.load_state_dict(torch.load(args.save))
        model.to(device)
        # user_emb = model.user_emb.squeeze(0).cpu().detach().numpy()
        # item_emb = model.item_emb.squeeze(0).cpu().detach().numpy()
        # np.save('user_embedding_1.npy', user_emb)
        # np.save('item_embedding_1.npy', item_emb)
        # print('embedding saved')
        # time.sleep(1000)




        user_embs = []
        item_embs = []
        selected_users = []
        selected_items = {}
        weight_1 = []
        weight_2 = []
        weight_3 = []
        weight_4 = []
        weight_5 = []
        weight_6 = []
        weight_7 = []
        # weight_8 = []



        for step, batch_data in enumerate(test_data_loader):
            label, user_neighs_layers, business_neighs_layers = batch_data

            # if len(business_neighs_layers[0][0][:-100].numpy())<10 or len(business_neighs_layers[0][0][:-100].numpy())>15:
            #     continue

            # if user_neighs_layers[0][0].numpy()[0] not in [9714,5115,10140,9145,4225,3496]:
            #     continue
            current_user = user_neighs_layers[0][0].numpy()[0]

            selected_users.extend(user_neighs_layers[0][0].numpy())
            selected_items[user_neighs_layers[0][0].numpy()[0]] = business_neighs_layers[0][0][:-100].numpy()
        #     # print(selected_users)
        #     # print(selected_items)
        #     # time.sleep(5)
        #
        #     label = label.to(device)
            user_neighs_layers = to_device(user_neighs_layers, device)
            business_neighs_layers = to_device(business_neighs_layers, device)
            logit = model(user_neighs_layers, business_neighs_layers)
            user_emb = model.user_emb.squeeze(0).cpu().detach().numpy()
            item_emb = model.item_emb.squeeze(0).cpu().detach().numpy()
            user_embs.append(user_emb)
            item_embs.append(item_emb[:-100])

        #     # if current_user not in [9145]:
        #     #     continue
        #     # idx = 0
        #     # print(len(selected_items[current_user]))

            # print(current_user)
            # print(selected_items[current_user][idx])
            # print('**********************************')
            # print(model.user_weight_p[1].shape)
            # print(model.user_weight_p[0][0].shape)
            # print('u1--------------')
            # print(torch.max(model.user_weight_p[1].squeeze(), dim=-1))
            weight_1.append(model.user_weight_p[-1].squeeze().cpu().detach().numpy())
            # print('u2--------------')
            # print(torch.max(model.user_weight_p[0][0].squeeze().mean(dim=0), dim=-1))
            weight_2.append(model.user_weight_p[0][0].squeeze().mean(dim=0).cpu().detach().numpy())
            # weight_8.append(model.user_weight_p[0][1].squeeze().mean(dim=0).cpu().detach().numpy())
            # print('**********************************')
            for idx in range(len(selected_items[current_user])):
                # print(model.item_weight_p[1].shape)
                # print(model.item_weight_p[0][0].shape)
                # print(model.item_weight_p[0][1].shape)
                # print(model.item_weight_p[0][2].shape)
                # print(model.item_weight_p[0][3].shape)
                # print('i1--------------')
                # print(torch.max(model.item_weight_p[1].squeeze()[:-100][idx], dim=-1))
                weight_3.append(model.item_weight_p[-1].squeeze()[:-100][idx].cpu().detach().numpy())
                # print('i2--------------')
                # print(torch.max(model.item_weight_p[0][0].squeeze()[:-100][idx].mean(dim=0), dim=-1))
                weight_4.append(model.item_weight_p[0][0].squeeze()[:-100][idx].mean(dim=0).cpu().detach().numpy())
                # print('i3--------------')
                # print(torch.max(model.item_weight_p[0][1].squeeze()[:-100][idx].mean(dim=0), dim=-1))
                weight_5.append(model.item_weight_p[0][1].squeeze()[:-100][idx].mean(dim=0).cpu().detach().numpy())
                # print('i4--------------')
                # print(torch.max(model.item_weight_p[0][2].squeeze()[:-100][idx].mean(dim=0), dim=-1))
                weight_6.append(model.item_weight_p[0][2].squeeze()[:-100][idx].mean(dim=0).cpu().detach().numpy())
                # print('i5--------------')
                # print(torch.max(model.item_weight_p[0][3].squeeze()[:-100][idx].mean(dim=0), dim=-1))
                weight_7.append(model.item_weight_p[0][3].squeeze()[:-100][idx].mean(dim=0).cpu().detach().numpy())
                # time.sleep(10)
            # if step == 10:
            #     break
        weight_1 = np.stack(weight_1, axis=0)
        weight_2 = np.stack(weight_2, axis=0)
        weight_3 = np.stack(weight_3, axis=0)
        weight_4 = np.stack(weight_4, axis=0)
        weight_5 = np.stack(weight_5, axis=0)
        weight_6 = np.stack(weight_6, axis=0)
        weight_7 = np.stack(weight_7, axis=0)
        # weight_8 = np.stack(weight_8, axis=0)
        weight_1 = weight_1.mean(axis=0)
        weight_2 = weight_2.mean(axis=0)
        weight_3 = weight_3.mean(axis=0)
        weight_4 = weight_4.mean(axis=0)
        weight_5 = weight_5.mean(axis=0)
        weight_6 = weight_6.mean(axis=0)
        weight_7 = weight_7.mean(axis=0)
        # weight_8 = weight_8.mean(axis=0)

        # print(weight_1)
        print(np.max(weight_1, axis=-1))
        print(np.argmax(weight_1, axis=-1))

        print(np.max(weight_2, axis=-1))
        print(np.argmax(weight_2, axis=-1))

        # print(np.max(weight_8, axis=-1))
        # print(np.argmax(weight_8, axis=-1))

        # print(weight_3)
        print(np.max(weight_3, axis=-1))
        print(np.argmax(weight_3, axis=-1))

        print(np.max(weight_4, axis=-1))
        print(np.argmax(weight_4, axis=-1))
        #
        print(np.max(weight_5, axis=-1))
        print(np.argmax(weight_5, axis=-1))
        #
        print(np.max(weight_6, axis=-1))
        print(np.argmax(weight_6, axis=-1))
        #
        print(np.max(weight_7, axis=-1))
        print(np.argmax(weight_7, axis=-1))

        time.sleep(100000)


            #
            # print(len(selected_users))
            # if len(selected_users) == 6:
            #     break
        user_embedding = np.concatenate(user_embs, axis=0)
        item_embedding = np.concatenate(item_embs, axis=0)
        selected_item_embs = []
        reselected_items = []
        selected_user_items_idx = {}
        selected_user_items = {}
        i = 0
        j = 0
        k = 0
        for user, items in selected_items.items():
            user_item_idx = []
            user_item = []
            for item in items:
                if item not in reselected_items:
                    reselected_items.append(item)
                    selected_item_embs.append(item_embedding[k])
                    user_item.append(item)
                    user_item_idx.append(j)
                    j += 1
                k += 1
            selected_user_items_idx[i] = user_item_idx
            selected_user_items[user] = user_item
            i += 1
        item_embedding = np.stack(selected_item_embs, axis=0)
        np.save('user_embedding.npy', user_embedding)
        np.save('item_embedding.npy', item_embedding)
        with open('user_item_idx.pickle', 'wb') as f:
            pickle.dump(selected_user_items_idx, f)
        with open('user_item.pickle', 'wb') as f:
            pickle.dump(selected_user_items, f)
        print(selected_user_items_idx)
        print(selected_user_items)
        # np.save('user.npy', np.array(selected_users))
        # np.save('item.npy', np.array(reselected_items))
        print('embedding saved')

        # user_embedding = model.emb_init[0].weight.cpu().detach().numpy()
        # item_embedding = model.emb_init[1].weight.cpu().detach().numpy()
        # np.save('user_embedding.npy', user_embedding)
        # np.save('item_embedding.npy', item_embedding)
        # print('embedding saved')



        # test_data_loader = DataLoader(
        #     dataset=Dataset(n_nodes_list, test_data_path, adj_paths, args.n_layer, args.n_neigh, args.n_neg, args.train_prop, 'test'),
        #     batch_size=1,
        #     shuffle=False,
        #     num_workers=20,
        #     pin_memory=True)
        # model.load_state_dict(torch.load(args.save))
        # model.to(device)
        # test(model, test_data_loader, args.topk, device)


        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # # Print model's state_dict
        # print("Model's state_dict:")
        # for param_tensor in model.state_dict():
        #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        #
        # # Print optimizer's state_dict
        # print("Optimizer's state_dict:")
        # for var_name in optimizer.state_dict():
        #     print(var_name, "\t", optimizer.state_dict()[var_name])
        #
        # for name, param in model.named_parameters():
        #     print('name: ', name)
        #     print(type(param))
        #     print('param.shape: ', param.shape)
        #     print('param.requires_grad: ', param.requires_grad)
        #     print('=====')

        # for name, child in model.named_children():
        #     print('name: ', name)
        #     print('isinstance({}, nn.Module): '.format(name), isinstance(child, nn.Module))
        #     print('=====')



