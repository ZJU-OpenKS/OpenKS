# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import logging
import argparse
import json

from ..model import KELearnModel

@KELearnModel.register("KELearn", "MLLib")
class KELearnMLLib(KELearnModel):
    def __init__(self, name='mllib-default', dataset=None, model=None, args=None):
        self.name = name
        self.dataset = dataset
        self.args = args
        self.model = model

    def data_reader(self):
        dataset = [item for item in self.dataset.bodies]
        train = dataset[0]
        valid = dataset[1]
        return train, valid

    def candidate_length_summarize(self, input_path):
        minimum_length = 100
        maximum_length = 0
        total_length = 0
        total_count = 0
        with open(input_path, "r") as f:
            for line in f:
                tempList = json.loads(line)
                for word in tempList:
                    minimum_length = min(minimum_length, len(word[0]))
                    maximum_length = max(maximum_length, len(word[0]))
                    total_length = total_length + len(word[0])
                    total_count = total_count + 1
        print(minimum_length, maximum_length, total_length / total_count)
        return [minimum_length, maximum_length, total_length / total_count]

    def evaluate(self, result_path, standard_path):
        f = open(result_path, 'r')
        p = open(standard_path, 'r')
        patents = [line for line in p]
        line_num = 0
        precision_num = 0
        total_standard = 0
        total = 0
        for line in f:
            patent = json.loads(patents[line_num])
            patent = list(set(patent))
            line_num = line_num + 1
            tempList = json.loads(line)
            
            total_standard = total_standard + len(patent)
            for word in tempList:
                total = total + 1
                if word[0].lower() in patent:
                    precision_num = precision_num + 1
        print("Precision: " + str(precision_num / total))
        print("Recall: " + str(precision_num / total_standard))


    def run(self):
        model = self.model(self.args)
        train, valid = self.data_reader()
        res = model.process(train)

        with open(self.args['result_dir'] + '/' + self.args['extractor'], "w") as out:
            for res_item in res:
                out.write(json.dumps(res_item, ensure_ascii=False) + '\n')

        result_path = self.args['result_dir'] + '/' + self.args['extractor']
        standard_path = self.args['result_dir'] + '/' + 'standard'
        self.evaluate(result_path, standard_path)
