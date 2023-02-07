# import json
#
# if __name__ == '__main__':
#     paper_withtext_path = '/home/wyf/project/DisenPaper/s2orc/data/filted/Computer Science3/papers_withtext.json'
#     sample_path = '/home/wyf/project/DisenPaper/human.json'
#     save_path = '/home/wyf/project/DisenPaper/human_disencite.json'
#
#     human_samples = []
#     papers = {}
#
#     with open(paper_withtext_path, 'r', encoding='utf-8') as fr:
#         for line in fr:
#             paper = json.loads(line)
#             papers[paper['id']] = paper
#
#     with open(sample_path, 'r', encoding='utf-8') as fr:
#         for line in fr:
#             sample = json.loads(line)
#             sample['src_title'] = papers[sample['src']]['title']
#             sample['dst_title'] = papers[sample['dst']]['title']
#             sample['src_abstract'] = ' '.join(papers[sample['src']]['abstract'])
#             sample['dst_abstract'] = ' '.join(papers[sample['dst']]['abstract'])
#             human_samples.append(sample)
#
#     with open(save_path, 'w', encoding='utf-8') as fw:
#         for sample in human_samples:
#             meta = json.dumps(sample, ensure_ascii=False)
#             fw.write(meta + '\n')


import json
import xlwt
import numpy as np
import time



def fleiss_kappa(testData, N, k, n):  # testData表示要计算的数据，（N,k）表示矩阵的形状，说明数据是N行j列的，一共有n个标注人员
    dataMat = np.mat(testData, float)
    oneMat = np.ones((k, 1))
    sum = 0.0
    P0 = 0.0
    for i in range(N):
        temp = 0.0
        for j in range(k):
            sum += dataMat[i, j]
            temp += 1.0 * dataMat[i, j] ** 2
        temp -= n
        temp /= (n - 1) * n
        P0 += temp
    P0 = 1.0 * P0 / N
    ysum = np.sum(dataMat, axis=0)
    for i in range(k):
        ysum[0, i] = (ysum[0, i] / sum) ** 2
    Pe = ysum * oneMat * 1.0
    ans = (P0 - Pe) / (1 - Pe)
    return ans[0, 0]

def cal_std():
    t0 = [1.35, 0.8, 1.3]
    t1 = [1.4, 0.95, 0.85]
    t2 = [1.25, 0.55, 0.7]
    t3 = [1.55, 0.95, 0.9]
    t4 = [1.85, 1.3, 1.0]
    t5 = [0.9, 0.5, 1.0]
    t6 = [0.75, 0.5, 1.05]
    t7 = [1.8, 1.15, 1.7]
    t8 = [1.8, 1.15, 1.7]
    t9 = [1.95, 1.15, 1.7]
    t = [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9]
    print(np.std(t, axis=1))
    print(np.std(t, axis=1).mean())

def cal_kappa():

    dataArr_ = [[2,	2],
               [1,	2],
               [2,	2],
               [2,	2],
               [2,	0],
               [2,	1],
               [1,	1],
               [2,	1],
               [0,	1],
               [1,	2],
               [1,	1],
               [2,	1],
               [0,	1],
               [1,	0],
               [1,	1],
               [0,	1],
               [0,	1],
               [2,	2],
               [1,	2],
               [1,	2]]

    dataArr = []
    for data in dataArr_:
        d_coun = [data.count(0), data.count(1), data.count(2)]
        dataArr.append(d_coun)

    dataArr2 = [[0, 0, 0, 0, 14],
                [0, 2, 6, 4, 2],
                [0, 0, 3, 5, 6],
                [0, 3, 9, 2, 0],
                [2, 2, 8, 1, 1],
                [7, 7, 0, 0, 0],
                [3, 2, 6, 3, 0],
                [2, 5, 3, 2, 2],
                [6, 5, 2, 1, 0],
                [0, 2, 2, 3, 7]]

    print(dataArr)


    print(fleiss_kappa(dataArr, 20, 3, 2))
    print(fleiss_kappa(dataArr2, 10, 5, 14))

    # dataArr = [[0, 2, 1],
    #            [1, 0, 2],
    #            [0, 0, 3],
    #            [0, 1, 2],
    #            [0, 0, 3],
    #            [0, 2, 1],
    #            [0, 1, 2],
    #            [0, 0, 3],
    #            [0, 1, 2],
    #            [0, 0, 3],
    #            [0, 1, 2],
    #            [0, 1, 2],
    #            [0, 2, 1],
    #            [],
    #            [],
    #            [],
    #            [],
    #            [],
    #            [],
    #            []]



if __name__ == '__main__':
    # cal_std()
    cal_kappa()
    time.sleep(10000)


    paper_withtext_path = '/home/wyf/project/DisenPaper/s2orc/data/filted/Computer Science3/papers_withtext.json'
    disen_path = '/home/wyf/project/DisenPaper/human.json'
    seq2seq_path = '/home/wyf/project/DisenPaper/human_seq2seq_.json'
    ptgen_path = '/home/wyf/project/DisenPaper/human_ptgen_.json'
    autocite_path = '/home/wyf/project/DisenPaper/human_autocite_.json'
    scigen_path = '/home/wyf/project/DisenPaper/human_scigen_.json'
    gat_path = '/home/wyf/project/DisenPaper/human_gat_.json'
    hgt_path = '/home/wyf/project/DisenPaper/human_hgt_.json'
    extracite_random_path = '/home/wyf/project/DisenPaper/human_extracite_random_.json'
    extradst_random_path = '/home/wyf/project/DisenPaper/human_extradst_random_.json'
    extradst_first_path = '/home/wyf/project/DisenPaper/human_extradst_first_.json'

    save_path = '/home/wyf/project/DisenPaper/human_eval.xls'

    papers = {}
    with open(paper_withtext_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            paper = json.loads(line)
            papers[paper['id']] = paper

    workbook = xlwt.Workbook()
    model_idx = 1
    column_dict = {'src': 3, 'dst': 4, 'src_title': 5,  'dst_title': 6, 'src_abstract': 7,
                   'dst_abstract': 8, 'pred': 9, 'tgt': 10, 'pos': 11}
    pos_dict = {0: 'intro', 1: 'relate', 2: 'method', 3: 'experiment'}

    samples = []
    idx = 0
    with open(disen_path, 'r') as fr:
        for line in fr:
            sample = json.loads(line)
            sample['src_title'] = papers[sample['src']]['title']
            sample['dst_title'] = papers[sample['dst']]['title']
            sample['src_abstract'] = ' '.join(papers[sample['src']]['abstract'])
            sample['dst_abstract'] = ' '.join(papers[sample['dst']]['abstract'])
            sample['pos'] = pos_dict[sample['pos']]
            eval = str(sample['src']) + '_' + str(sample['dst']) + '_' + str(sample['pos'])
            samples.append(eval)
            idx += 1
            if idx == 30:
                break

    for path in [disen_path, seq2seq_path, ptgen_path, autocite_path, scigen_path, gat_path, hgt_path,
                 extracite_random_path, extradst_random_path, extradst_first_path]:
        human_samples = []
        with open(path, 'r') as fr:
            for line in fr:
                sample = json.loads(line)
                if path == ptgen_path:
                    sample['src'] = int(sample['src'].strip())
                    sample['dst'] = int(sample['dst'].strip())
                    sample['pos'] = int(sample['pos'].strip())

                sample['src_title'] = papers[sample['src']]['title']
                sample['dst_title'] = papers[sample['dst']]['title']
                sample['src_abstract'] = ' '.join(papers[sample['src']]['abstract'])
                sample['dst_abstract'] = ' '.join(papers[sample['dst']]['abstract'])
                sample['pos'] = pos_dict[sample['pos']]
                human_samples.append(sample)


        sheet = workbook.add_sheet('model'+str(model_idx))

        sheet.write(0, 0, label='quality')
        sheet.write(0, 1, label='related')
        sheet.write(0, 2, label='section-f1')

        sheet.write(0, 3, label='src')
        sheet.write(0, 4, label='dst')
        sheet.write(0, 5, label='src_title')
        sheet.write(0, 6, label='dst_title')
        sheet.write(0, 7, label='src_abstract')
        sheet.write(0, 8, label='dst_abstract')

        sheet.write(0, 9, label='pred')
        sheet.write(0, 10, label='tgt')
        sheet.write(0, 11, label='pos')

        for eval_idx in range(len(samples)):
            for idx in range(len(human_samples)):
                sample = human_samples[idx]
                sample_eval = str(sample['src']) + '_' + str(sample['dst']) + '_' + str(sample['pos'])
                if sample_eval == samples[eval_idx]:
                    for k, v in sample.items():
                        sheet.write(eval_idx+1, column_dict[k], str(v))
                    break

        model_idx += 1

    workbook.save(save_path)




