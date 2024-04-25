# -*- coding: utf-8 -*-
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, SpectralClustering, SpectralBiclustering, MiniBatchKMeans, SpectralCoclustering, BisectingKMeans
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from node2vec import Node2Vec
from gensim.models import KeyedVectors, Word2Vec
import networkx as nx
import pandas as pd
import numpy as np
import networkx as nx
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from node2vec import Node2Vec
from scipy.io import savemat
import itertools


###获取API目录的字典
def getdictofcategory():
    '''
    得到API和目录一级类别的字典
    '''
    API = pd.read_csv("./datasets/APIs.csv")
    # 对数据进行清洗 除去了在Categories中为nan值的一行
    API = API.dropna(subset=["Categories"])
    API = API.reset_index(drop=True)
    # 对sumbit_date行进行切割
    API['newCategories'] = API['Categories'].str.split("###")
    newCategories = API['newCategories']
    # 将newCategories中全部缩减为只取一级标签
    for i in range(len(newCategories)):
        newCategories[i] = newCategories[i][0]
    # 将一级标签的series赋值于API数据中
    API['newCategories'] = newCategories
    keys = list(API['APIName'])
    values = list(API['newCategories'])
    dictofAPICategories = dict(zip(keys, values))
    return dictofAPICategories


category_dict = getdictofcategory()



#####数据预处理######
API = pd.read_csv("./datasets/APIs.csv")
# 对数据进行清洗 除去了在Categories中为nan值的一行
API = API.dropna(subset=["Categories"])
API = API.reset_index(drop=True)
# 对数据进行清洗 除去了在Description中为nan值的一行
API = API.dropna(subset=["Description"])
API = API.reset_index(drop=True)
# 对sumbit_date行进行切割
API['newCategories'] = API['Categories'].str.split("###")
newCategories = API['newCategories']
# 将newCategories中全部缩减为只取一级标签
for i in range(len(newCategories)):
    newCategories[i] = newCategories[i][0]
# 将一级标签的series赋值于API数据中
API['newCategories'] = newCategories

# 根据保存的图结构 构建图
txt = []
with open('PSO_edges_noweights.txt', 'r', encoding='utf-8') as f:
    for line in f:
        txt.append(line)

edges = []
for item in txt:
    a = item.strip().split('\t')
    # a[2] = int(a[2])
    edges.append(tuple(a))

G = nx.Graph()
G.add_edges_from(edges)

nodes = list(G.nodes)
# 预处理 将网络中的节点不在API的datafram中 那么就将该节点删掉
full_API = list(API["APIName"])


#在这里把图中节点不存在API里面的节点去除掉
for none_node in nodes:
    if none_node not in full_API:
        G.remove_node(none_node)


# 将节点的描述信息放入node_description
node_description = []
node_label = []
#new_nodes为处理之后的图, 需要用这个图, 因为这个图他的每一个节点都是有描述和category
new_nodes = list(G.nodes)


for no in new_nodes:
    aa = API[API.APIName == no].index.tolist()[0]
    node_label.append(API['newCategories'][aa])
    node_description.append(API['Description'][aa])
####预处理结束####



####目标: 利用bert模型得到每一个API的文本特征向量
#####对预处理数据进行深度学习
max_length = 32
embedding_size = 768
# 加载BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('E:/IntelligentServiceLab/huggingface/bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', mirror= 'https://mirros.tuna.tsinghua.edu.cn/hugging-face-models')
model = BertModel.from_pretrained('E:/IntelligentServiceLab/huggingface/bert-base-uncased')

# 将需求的话语嵌入为向量
query = "recent jobs vacancies close to location and message"

inputs_query = tokenizer(query, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
outputs_query = model(**inputs_query)

# Get the last hidden state of the first token
embedding_query = outputs_query.last_hidden_state[:, 0, :]
# Resize the embedding to the desired size
embedding_query = torch.nn.functional.pad(embedding_query, (0, embedding_size - embedding_query.size(1)), 'constant', 0)
# Convert to a numpy array
embedding_query = embedding_query.detach().numpy()
embedding_query = torch.Tensor(embedding_query[0])




# 构建节点特征矩阵
node_feature = []

for sentence in node_description:
    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    outputs = model(**inputs)

    # Get the last hidden state of the first token
    embedding = outputs.last_hidden_state[:, 0, :]
    # Resize the embedding to the desired size
    embedding = torch.nn.functional.pad(embedding, (0, embedding_size - embedding.size(1)), 'constant', 0)
    # Convert to a numpy array
    embedding = embedding.detach().numpy()
    node_feature.append(embedding[0])

#####这里的文本特征为对每一个API的文本进行嵌入
text_features = torch.Tensor(np.array(node_feature))
#######API的文本特征向量的构造结束
# features = dict(zip(new_nodes, node_feature))
# nx.set_node_attributes(G, features, "features")
#
# G1 = nx.convert_node_labels_to_integers(G)



######目标: 利用Node2vec模型得到每一个API的机构特征向量
model_X2 = Word2Vec.load('Node2vec_PSO16_noweights.model')

dict_vec = model_X2.wv.key_to_index
vec = model_X2.wv.vectors

node2vec_features = []

###处理所得到的Node2vecAPI嵌入向量的表征和之前的networkx的图嵌入的API表征的对齐 因为这里的dict_vec第一个为Twitter但是new_node第一个API为Google 所以要对齐
for i in new_nodes:
    node2vec_features.append(vec[dict_vec[i]])

node2vec_features = torch.Tensor(node2vec_features)
########利用Node2vec模型得到每一个API的机构特征向量结束


test_X = torch.cat([text_features, node2vec_features], dim=1)



# ######目标: 利用所得到的文本特征向量和结构特征向量进行多模态特征融合
# n = node2vec_features.shape[0]
# # 用 1 扩充维度
# A = torch.cat([text_features, torch.ones(n, 1)], dim=1)
# B = torch.cat([node2vec_features, torch.ones(n, 1)], dim=1)
#
# # 计算笛卡尔积
# A = A.unsqueeze(2)  # [n, A, 1]
# B = B.unsqueeze(1)  # [n, 1, B]
#
# fusion_AB = torch.einsum('nxt, nty->nxy', A, B)  # [n, A, B]
# fusion_AB = fusion_AB.flatten(start_dim=1).unsqueeze(1) # [n, AxB, 1]
#
# n_query = embedding_query.shape[0]
# oo = torch.zeros(12305)
# embedding_query = torch.cat((embedding_query,oo), dim=0)
# #######利用所得到的文本特征向量和结构特征向量进行多模态特征融合结束


X2_truelabel = []


for i in dict_vec.keys():
    if i in category_dict.keys():
        X2_truelabel.append(category_dict[i])


all_label_category = list(set(X2_truelabel))

label = []

for i in X2_truelabel:
    label.append(all_label_category.index(i))


cluster2 = KMeans(n_clusters=329, random_state=9).fit(node2vec_features)
pre_label = cluster2.labels_

pre_label_spect = SpectralClustering(n_clusters=329, gamma=0.1).fit_predict(node2vec_features)
spectbi = SpectralBiclustering(n_clusters=329, random_state=0).fit(test_X)
pre_label_spectbi = spectbi.row_labels_


minibatchkmeans = MiniBatchKMeans(n_clusters=329, random_state=0, batch_size=6).fit(node2vec_features)
pre_label_minibatch = minibatchkmeans.labels_


# spectco = SpectralCoclustering(n_clusters=329, random_state=0).fit(test_X)
# pre_label_spectco = spectco.row_labels_

Bisect = BisectingKMeans(n_clusters=329, random_state=0).fit(test_X)
pre_label_Bisect = Bisect.labels_



ACC_kmeans = metrics.accuracy_score(label, pre_label)
NMI_kmeans = metrics.normalized_mutual_info_score(label, pre_label)
ARI_kmeans = metrics.adjusted_rand_score(label, pre_label)
AMI_kmeans = metrics.adjusted_mutual_info_score(label, pre_label)
MI_kmeans = metrics.mutual_info_score(label, pre_label)
R_kmeans = metrics.rand_score(label, pre_label)
CS_kmeans = metrics.completeness_score(label, pre_label)



spect_ACC = metrics.accuracy_score(label, pre_label_spect)
spect_NMI = metrics.normalized_mutual_info_score(label, pre_label_spect)
spect_ARI = metrics.adjusted_rand_score(label, pre_label_spect)
spect_AMI = metrics.adjusted_mutual_info_score(label, pre_label_spect)
spect_MI = metrics.mutual_info_score(label, pre_label_spect)
spect_R = metrics.rand_score(label, pre_label_spect)
spect_CS = metrics.completeness_score(label, pre_label_spect)


spectbi_ACC = metrics.accuracy_score(label, pre_label_spectbi)
spectbi_NMI = metrics.normalized_mutual_info_score(label, pre_label_spectbi)
spectbi_ARI = metrics.adjusted_rand_score(label, pre_label_spectbi)
spectbi_AMI = metrics.adjusted_mutual_info_score(label, pre_label_spectbi)
spectbi_MI = metrics.mutual_info_score(label, pre_label_spectbi)
spectbi_R = metrics.rand_score(label, pre_label_spectbi)
spectbi_CS = metrics.completeness_score(label, pre_label_spectbi)


minibatch_ACC = metrics.accuracy_score(label, pre_label_minibatch)
minibatch_NMI = metrics.normalized_mutual_info_score(label, pre_label_minibatch)
minibatch_ARI = metrics.adjusted_rand_score(label, pre_label_minibatch)
minibatch_AMI = metrics.adjusted_mutual_info_score(label, pre_label_minibatch)
minibatch_MI = metrics.mutual_info_score(label, pre_label_minibatch)
minibatch_R = metrics.rand_score(label, pre_label_minibatch)
minibatch_CS = metrics.completeness_score(label, pre_label_minibatch)


Bisect_ACC = metrics.accuracy_score(label, pre_label_Bisect)
Bisect_NMI = metrics.normalized_mutual_info_score(label, pre_label_Bisect)
Bisect_ARI = metrics.adjusted_rand_score(label, pre_label_Bisect)
Bisect_AMI = metrics.adjusted_mutual_info_score(label, pre_label_Bisect)
Bisect_MI = metrics.mutual_info_score(label, pre_label_Bisect)
Bisect_R = metrics.rand_score(label, pre_label_Bisect)
Bisect_CS = metrics.completeness_score(label, pre_label_Bisect)



# spectco_ACC = metrics.accuracy_score(label, pre_label_spectco)
# spectco_NMI = metrics.normalized_mutual_info_score(label, pre_label_spectco)
# spectco_ARI = metrics.adjusted_rand_score(label, pre_label_spectco)
# spectco_AMI = metrics.adjusted_mutual_info_score(label, pre_label_spectco)
# spectco_MI = metrics.mutual_info_score(label, pre_label_spectco)
# spectco_R = metrics.rand_score(label, pre_label_spectco)
# spectco_CS = metrics.completeness_score(label, pre_label_spectco)





print("aaaa")






