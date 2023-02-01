from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def compute_bleu(references, candidates):
    references = [[item] for item in references]
    smooth_func = SmoothingFunction().method0  # method0-method7
    score1 = corpus_bleu(references, candidates, smoothing_function=smooth_func, weights=(1.0,))
    score2 = corpus_bleu(references, candidates, smoothing_function=smooth_func, weights=(1.0/2,)*2)
    score3 = corpus_bleu(references, candidates, smoothing_function=smooth_func, weights=(1.0/3,)*3)
    score4 = corpus_bleu(references, candidates, smoothing_function=smooth_func, weights=(1.0/4,)*4)
    return score1, score2, score3, score4