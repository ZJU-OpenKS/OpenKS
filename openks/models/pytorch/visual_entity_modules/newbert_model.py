import torch
import clip
from torch import nn
from nn import MLP, Bilinear, BaseModule
from transformers import AutoModel
import numpy as np
from torch.nn import functional as F
from transformers import AutoTokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"

def dep_loss(muls, label, mask):
    #muls , batch * 20 * 2
    head_loss = nn.CrossEntropyLoss() #交叉熵函数
    mseloss = nn.MSELoss()

    real_muls = muls[mask] # x * 2
    real_labels = label[mask] #x
    #loss = head_loss(real_muls, real_labels)
    """
    print("muls")
    print(muls)
    print(muls.shape)
    print("mask")
    print(mask)
    print(mask.shape)
    print("label")
    print(label)
    print(label.shape)
    print("real_muls")
    print(real_muls)
    print(real_muls.shape)
    print("real_labels")
    print(real_labels)
    print(real_labels.shape)
    """
    mse_labels = torch.ones((real_muls.shape[0],real_muls.shape[1])).to(device)

    for i in range(0,mse_labels.shape[0]):
        if real_labels[i] == 1:
            mse_labels[i][0] = 0
            mse_labels[i][1] = 1
        else:
            mse_labels[i][0] = 1
            mse_labels[i][1] = 0

    #real_muls = torch.sigmoid(real_muls)
   #用这个的话test就得用argmax来选
    real_muls = torch.nn.functional.softmax(real_muls,dim = -1) ##############
    loss = mseloss(real_muls, mse_labels)

    return loss

class TransformerBiaffine(BaseModule):
    def __init__(self, hparams,loss_func=dep_loss):
        super().__init__()
        self.save_hyperparameters(hparams)
        #self.graphmodel,self.preprocess = clip.load("ViT-B/32", device=device)#AutoModel.from_pretrained("clip-vit-base-patch32 ")
        #self.graphmodel.train()  ######################
        self.transformer = AutoModel.from_pretrained("bert-base-uncased")
        self.loss_func = loss_func
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.dropout = nn.Dropout(self.hparams.dropout) #0.1
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.fc1r = nn.Linear(768, 512, bias=False)
        self.arc_atten = Bilinear(512, 512, 2, bias_x=True, bias_y=False, expand=True)
    def forward(
            self,
            shiti, #[batch个]  实体的文本表示
            data_shiti,
            data_tupian, #batch  *   (20) * 512   m+n个正例和负例的图片表示  )
            label, #batch * 20 匹配or不匹配，1or0的label
            mask #batch * 20 匹配or不匹配，1or0的label
    ) :
        #image_features = self.graphmodel.encode_image(image)
        #text_features = self.graphmodel.encode_text(text)
        #image_features = torch.zeros((data.shape[0],data[0][1].shape[0],512),device=device) #b * (m+n) * 512
        #text_features = torch.zeros((data.shape[0], 1, 512), device=device) #b * 1 * 512
        encoding = self.tokenizer(shiti, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        hidden_states = self.transformer(
            input_ids,
            attention_mask=attention_mask,
        )   # B * 1 * 768
        hidden_states = self.fc1r(hidden_states[1]) # B * 512

        sequence = hidden_states.unsqueeze(1) # B * 1 * 512
        # data_tupian       B * (m+n) * 512

        #B * (m+n) * 1
        s_arc = self.arc_atten(sequence, data_tupian).squeeze(2).permute(0, 2, 1) # batch * 2* 1* 20 -> batch * 20 * 2
        #new_s_arc = torch.nn.functional.softmax(s_arc,dim = -1) ##############
        loss = None
        if label is not None:
            # 下面要取得count对应的行
            loss = self.loss_func(s_arc, label, mask)
        return loss,s_arc