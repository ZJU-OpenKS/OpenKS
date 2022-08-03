import numpy as np
import json


# with open("pattern.json", "r") as fh:
#     patterns = json.load(fh)


class Pat_Match(object):
    def __init__(self, config, label_to_id, filt=None):
        self.config = config
        self.label_to_id = label_to_id
        self.patterns = config.patterns
        if filt is not None:
            patterns = [pattern for pattern in self.patterns if label_to_id[pattern[0]] not in filt]
            self.patterns = patterns

    def match(self, tokens):
        config = self.config
        num_pat = len(self.patterns)
        num_text = len(tokens)
        res = np.zeros([num_text, num_pat])
        pred = np.zeros([num_text, config.num_class])

        for i, pattern in enumerate(self.patterns):
            rel, pat = pattern
            rel = self.label_to_id[rel]

            for j, token in enumerate(tokens):
                text = " ".join(token)
                if pat in text:
                    # print(text)
                    # print(pat)
                    res[j, i] += 1
                    pred[j, rel] += 1
        
        none_zero = (np.amax(pred, axis=1) > 0).astype(np.int32)
        pred = np.argmax(pred, axis=1)
        pred = pred * none_zero
        return res, pred
