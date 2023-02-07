from glob import glob
from rouge import Rouge
from bleu import compute_bleu

root_path = '/home2/kyfd/PaperPretrain/ptgen/output'
preds, tgts = [], []
for file_name in sorted(glob(root_path + '/decoded/*.txt')):
    with open(file_name, 'r') as f:
        preds.append(f.readline().strip())

for file_name in sorted(glob(root_path + '/reference/*.txt')):
    with open(file_name, 'r') as f:
        tgts.append(f.readline().strip())

metrics_score = [[] for _ in range(7)]
rouge = Rouge()
for tgt, pred in zip(tgts, preds):
    # BLEU
    bleu_1, bleu_2, bleu_3, bleu_4 = compute_bleu([tgt], [pred])
    metrics_score[0].append(bleu_1)
    metrics_score[1].append(bleu_2)
    metrics_score[2].append(bleu_3)
    metrics_score[3].append(bleu_4)

    # ROUGE
    rouge_score = rouge.get_scores(pred, tgt)
    metrics_score[4].append(rouge_score[0]['rouge-1']['f'])
    metrics_score[5].append(rouge_score[0]['rouge-2']['f'])
    metrics_score[6].append(rouge_score[0]['rouge-l']['f'])

metrics_score = [sum(i) / len(i) for i in metrics_score]
print('\tbleu-1:{0[0]}, bleu-2:{0[1]}, bleu-3:{0[2]}, bleu-4:{0[3]}'.format(metrics_score))
print('\trouge-1-f:{0[4]}, rouge-2-f:{0[5]}, rouge-l-f:{0[6]}'.format(metrics_score))

