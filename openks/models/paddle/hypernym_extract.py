from ..model import HypernymExtractModel
from bs4 import BeautifulSoup
import requests


@HypernymExtractModel.register("HypernymExtract", "Paddle")
class HypernymExtractPaddle(HypernymExtractModel):
    def __init__(self, name: str = 'HypernymExtractModel', ):
        super().__init__()
        self.name = name

    def entity2hyper_lst(self, entity: str):
        url = f'http://101.200.120.155/getEnAPI?q={entity}'
        res = requests.get(url)
        if res.ok:
            bs = BeautifulSoup(res.text, 'html.parser')
            hyper_set = set()
            hyper_path_lst = filter(None, bs.find_all('table')[1].text.replace(' ', '').split('\n'))  # 获取所有上位词
            for hyper_path in hyper_path_lst:
                hypers = hyper_path.split('->')[1:]
                hyper_set = hyper_set.union(hypers)
            return [(entity, hyper) for hyper in hyper_set]
        print(f'大词林接口请求失败：{entity}；{res.status_code}')
        return []
