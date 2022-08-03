#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
sys.path.append("./")
import json
import faiss
from tornado import web
from tornado import ioloop
from openks.apps.irqa import irqa


def load_model():
    de_model = 'zh_dureader_de'
    ce_model = 'zh_dureader_ce'

    de_conf = {
        "model": de_model,
        "use_cuda": True,
        "device_id": 0,
        "batch_size": 32
    }
    ce_conf = {
        "model": ce_model,
        "use_cuda": True,
        "device_id": 0,
        "batch_size": 32
    }
    dual_encoder = irqa.load_model(**de_conf)
    cross_encoder = irqa.load_model(**ce_conf)
    return dual_encoder, cross_encoder

def build_index(dual_encoder, title_list, para_list):

    para_embs = dual_encoder.encode_para(para=para_list, title=title_list)

    indexer = faiss.IndexFlatIP(768)
    indexer.add(para_embs.astype('float32'))
    faiss.write_index(indexer, ".irqa_faiss_index")


class FaissTool():
    """
    Faiss index tools
    """
    def __init__(self, text_filename):
        if not os.path.exists('.irqa_faiss_index'):
            print ("Index file not found, please run `build_index` function first !")
            return
        self.engine = faiss.read_index('.irqa_faiss_index')
        self.id2text = []
        for line in open(text_filename):
            self.id2text.append(line.strip())


    def search(self, q_embs, topk=5):
        res_dist, res_pid = self.engine.search(q_embs, topk)
        result_list = []
        for i in range(topk):
            result_list.append(self.id2text[res_pid[0][i]])
        return result_list


class IRQAServer(web.RequestHandler):

    def __init__(self, application, request, **kwargs):
        web.RequestHandler.__init__(self, application, request)
        self._faiss_tool = kwargs["faiss_tool"]
        self._dual_encoder = kwargs["dual_encoder"]
        self._cross_encoder = kwargs["cross_encoder"]

    def get(self):
        """
        Get request
        """

    def post(self):
        input_request = self.request.body
        output = {}
        output['error_code'] = 0
        output['error_message'] = ''
        output['answer'] = []
        if input_request is None:
            output['error_code'] = 1
            output['error_message'] = "Input is empty"
            self.write(json.dumps(output))
            return

        try:
            input_data = json.loads(input_request)
        except:
            output['error_code'] = 2
            output['error_message'] = "Load input request error"
            self.write(json.dumps(output))
            return

        if "query" not in input_data:
            output['error_code'] = 3
            output['error_message'] = "[Query] is missing"
            self.write(json.dumps(output))
            return

        query = input_data['query']
        topk = 5
        if "topk" in input_data:
            topk = input_data['topk']

        # encode query
        q_embs = self._dual_encoder.encode_query(query=[query])

        # search with faiss
        search_result = self._faiss_tool.search(q_embs, topk)

        titles = []
        paras = []
        queries = []
        for t_p in search_result:
            queries.append(query)
            t, p = t_p.split('\t')
            titles.append(t)
            paras.append(p)
        ranking_score = self._cross_encoder.matching(query=queries, para=paras, title=titles)

        final_result = {}
        for i in range(len(paras)):
            final_result[query + '\t' + titles[i] + '\t' + paras[i]] = ranking_score[i]
        sort_res = sorted(final_result.items(), key=lambda a:a[1], reverse=True)

        for qtp, score in sort_res:
            one_answer = {}
            one_answer['probability'] = score
            q, t, p = qtp.split('\t')
            one_answer['title'] = t
            one_answer['para'] = p
            output['answer'].append(one_answer)

        result_str = json.dumps(output, ensure_ascii=False)
        self.write(result_str)


def create_irqa_app(dual_encoder, cross_encoder, sub_address, irqa_server, data_file):
    """
    Create IRQA server application
    """
    faiss_tool = FaissTool(data_file)
    print ('Load index done')

    return web.Application([(sub_address, irqa_server, \
                        dict(faiss_tool=faiss_tool, \
                              dual_encoder=dual_encoder, \
                              cross_encoder=cross_encoder))])



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print ("USAGE: ")
        print ("      python3 irqa_service.py ${data_file}")
        exit()

    data_file = sys.argv[1]
    para_list = []
    title_list = []
    for line in open(data_file):
        t, p = line.strip().split('\t')
        para_list.append(p)
        title_list.append(t)

    dual_encoder, cross_encoder = load_model()
    build_index(dual_encoder, title_list, para_list)

    sub_address = r'/irqa'
    port = '8888'
    app = create_irqa_app(dual_encoder, cross_encoder, sub_address, IRQAServer, data_file)
    app.listen(port)
    ioloop.IOLoop.current().start()

