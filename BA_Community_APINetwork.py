# -*- coding: utf-8 -*-
import json

import numpy as np
import networkx as nx
import pandas as pd
import itertools
import random
import community as community_louvain
import pickle




from gensim.models import Word2Vec
from node2vec import Node2Vec
from scipy.io import savemat
from sklearn.metrics.pairwise import cosine_similarity




def create_coperation_networkx():
    '''
    从csv文件中将数据装换为协作网络图
    :param filepath:文件路径
    :return: 所得的图
    '''
    #读取mashup文件
    data = pd.read_csv("./datasets/Mashups.csv")
    #将mashup中的相关API部分进行切割爆炸然后获取不相同的API列表，那么这个列表就是我们的节点
    # 对相关的api进行切割
    data['newcol'] = data['related_apis'].str.split("###")
    #newcol进行保存也就是节点之间存在的边
    coperation = data['newcol']
    #对coperation进行数据处理将空值删除掉
    coperation = coperation.fillna('None')
    # 对切割的newcol进行爆炸处理
    data = data.explode('newcol')
    #制作节点,将newcol单独提取出来进行数据清洗删去所有重复的行
    newcol = data['newcol']
    newcol.duplicated()
    nodes = newcol.drop_duplicates()
    #把nodes的空值去掉
    nodes = nodes.fillna('None')
    #转为list
    nodes = list(nodes)
    #建立没有边的图
    G = nx.Graph()
    G.add_nodes_from(nodes)
    #对于还没有爆炸处理的newcol,那么比如['A','B','C']就称他们之间有连接的边,对图添加边，其中判断如果有边则权值加1，无边就之间加
    #其中难点就是对于['A','B','C']中要判断三组边 该如何去表示
    for i in range(coperation.size):
        #对coperation每个元素制作边,其中这里面coperation数据存在NAN，此时要进行处理
        Totaledge = itertools.combinations(coperation[i], r=2)          #这里用组合函数
        Totaledge = list(Totaledge)
        for j in range(len(Totaledge)):
            #如果图中存在该边,该边其权值
            edge = Totaledge[j]
            if G.has_edge(edge[0], edge[1]):
                edgedata = G.get_edge_data(edge[0],edge[1])
                G.add_edge(edge[0], edge[1], weight = (edgedata["weight"]+1))
            #如果图中不存在该边,添加边然后设置权值为1
            else:
                G.add_edge(edge[0], edge[1], weight = 1)
    return G




def getdictoftime():
    '''
    得到API和对应提交时间的字典
    '''
    API = pd.read_csv("./datasets/APIs.csv")
    #对数据进行清洗 除去了在Categories中为nan值的一行
    API = API.dropna(subset=["SubmittedDate"])
    API = API.reset_index(drop=True)
    #对sumbit_date行进行切割
    API['newdate'] = API['SubmittedDate'].str.split("###")
    newdate = API['newdate']
    #将newCategories中全部缩减为只取一级标签
    for i in range(len(newdate)):
        newdate[i] = newdate[i][0]
    # 将series转换为时间格式,其中的format一定得是和文本格式一样
    newdate = pd.to_datetime(newdate, format="%m.%d.%Y")
    #将一级标签的series赋值于API数据中
    API['newdate'] = newdate
    keys = list(API['APIName'])
    values = list(API['newdate'])
    dictofAPIDate = dict(zip(keys, values))
    return dictofAPIDate



def getnode_by_probability(G):
    '''
    这个函数为计算出图中每个节点的概率 最后返回合适的节点用于与原节点形成边
    :return: node
    '''
    #计算度值得到字典
    degree = dict(nx.degree(G))
    #对字典的每一个值除以其总和 得到其概率
    total = sum(degree.values(), 0.0)
    degree_probability = {k: v / total for k, v in degree.items()}
    #对字典进行降序排序
    degree_probability_sort = sorted(degree_probability.items(), key=lambda x:x[1], reverse=True)
    #随机生成0到3的一个整数
    random_value = random.randint(0, 3)
    node = degree_probability_sort[random_value][0]
    return node


def create_Gba(G, m0):
    cooperationNetwork = create_coperation_networkx()
    # 获取API对应的时间字典用于后面的查询
    menu = getdictoftime()
    nodeDegree = dict(cooperationNetwork.degree)
    L1_list = sorted(nodeDegree.items(), key=lambda x: x[1], reverse=True)
    # L1为按节点的度值降序进行排序
    L1 = []
    for item in L1_list:
        L1.append(item[0])
    menu_by_time = sorted(menu.items(), key=lambda x: x[1])
    # 创建L2 L2为孤立的节点按时间升序来进行排序
    L2 = []
    for i in menu_by_time:
        if i[0] not in L1:
            L2.append(i[0])
    L1.extend(L2)
    # 初始化G0 G0为L1中前m0个节点的完全图
    init_list = L1[0:m0]
    init_edge = itertools.combinations(init_list, r=2)
    init_edge = list(init_edge)
    # 形成初始化的m0个节点的完全图
    G = nx.from_edgelist(init_edge)
    # 遍历L1中剩下的部分 将剩下的部分一个一个根据节点的popularity加入到网络G中
    for t in L1[m0:]:
        # 计算G中各个节点的度的权重占比 根据受欢迎性选择一个t要去连接的节点Y 这样就可以形成边(t,Y) 然后将其边添加到G中 形成最后的BA网络
        node_by_probability = getnode_by_probability(G)
        G.add_edge(node_by_probability, t)
    return G


def jaccard_similarity(g, h):
    i = set(g).intersection(h)
    return round(len(i) / (len(g) + len(h) - len(i)),3)


def create_tag_network():
    # 读取mashup文件
    data = pd.read_csv("./datasets/APIs.csv")
    data = data.dropna(axis=0, how='any', subset=['Categories'])
    data['newcol'] = data['Categories'].str.split("###")

    net_dict = data[["APIName", "newcol"]].set_index("APIName").to_dict(orient='dict')["newcol"]

    edge = []

    coperation_network = create_coperation_networkx()

    API_list = list(coperation_network.nodes)

    # 利用暴力法解决边的情况,在这里我们用edge去装我们的边
    for i in net_dict.keys():
        if i in API_list:
            for j in net_dict.keys():
                if (i != j) and (j in API_list):
                    # 判断其相应的值是否有交集,如果有交集那么就是会产生边,如果没有交集就说明没有边
                    i_value = net_dict[i]
                    j_value = net_dict[j]
                    i_value = [str(x) for x in i_value]
                    j_value = [str(y) for y in j_value]
                    and_tag = set(i_value).intersection(j_value)
                    if len(and_tag) != 0:
                        edge.append((i, j))
    G = nx.Graph()
    G.add_edges_from(edge)
    return G


def updata_community_probability(G):
    '''
    检测网络中社区的节点,然后更新网络中各个节点的社区概率
    :return: 网络中所有接
    '''
    # 处理社区这一个方面
    partition = community_louvain.best_partition(G)



    # 对于根据社区的所计算的概率是不断变动的 我们需要及时的更新其每个节点的总概率 这里同样定义一个有关社区的一个概率
    community_and_number = list(partition.values())
    # 对社区进行统计 不同的值就代表不同的社区 不同值的个数就代表着该社区的节点数为多少
    temp_community = pd.value_counts(community_and_number)

    community_probability = []

    for cn in range(len(community_and_number)):
        a = temp_community[community_and_number[cn]]
        community_probability.append(temp_community[community_and_number[cn]] / len(community_and_number))

    return community_probability



def updata_degree_probability(G):
    '''
    网络中添加节点后 更新网络中各节点度的概率
    :param G: 传入的网络
    :return: 返回新的有关节点的度的概率
    '''
    G_degree = dict(G.degree)  # 保存新构建的图的节点各度值
    degree_all = sum(G_degree.values())
    G_probability = {k: v / degree_all for k, v in G_degree.items()}  # 保存各节点的优先概率
    return list(G_probability.values())





if __name__ == "__main__":
    G = create_coperation_networkx()

    nodeDegree = dict(G.degree)

    # tag_cooperationNetwork = create_tag_network()
    model_Doc2vec = Word2Vec.load('Doc2vec_X1.model')
    API_dict = model_Doc2vec.docvecs.key_to_index
    API_vectors = model_Doc2vec.docvecs.vectors

    m = 2            #设定m为添加的边数

    # 获取API对应的时间字典用于后面的查询
    menu = getdictoftime()
    nodeDegree = dict(G.degree)


    L1 = list(nodeDegree.keys())
    menu_by_time = sorted(menu.items(), key=lambda x:x[1])



    L_cooperation = L1.copy()

    #创建L2 L2为孤立的节点按时间升序来进行排序
    L2 = []
    for u in menu_by_time:
        if u[0] not in L1:
            L2.append(u[0])

    L1.extend(L2)
    #处理L1中的元素 将L1中没有文本描述信息的项去掉 得到的L3为我们所需要的list 里面的每一项都是有文本描述的
    L3 = L1.copy()


    m0 = 1724


    #处理度这个方面
    G_degree = dict(G.degree)                                                         #保存新构建的图的节点各度值
    degree_all = sum(G_degree.values())
    G_probability = {k: v / degree_all for k, v in G_degree.items()}                  #保存各节点的优先概率


    #处理社区这一个方面
    partition = community_louvain.best_partition(G, resolution=1.4)

    p_list = list(partition.keys())



    # G_index = dict(zip(range(len(L3), str(L3))))
    G_degree_list = list(G_degree.values())
    #度的概率 G_probability_list
    G_probability_list = list(G_probability.values())

    # 对于根据社区的所计算的概率是不断变动的 我们需要及时的更新其每个节点的总概率 这里同样定义一个有关社区的一个概率
    community_and_number = list(partition.values())
    #对社区进行统计 不同的值就代表不同的社区 不同值的个数就代表着该社区的节点数为多少
    temp_community = pd.value_counts(community_and_number)
    #社区的概率

    community_probability = []

    for cn in range(len(community_and_number)):
        a = temp_community[community_and_number[cn]]
        community_probability.append(temp_community[community_and_number[cn]]/len(G_probability_list))

    all_add_edges = 5000

    similarity_dict = {}
    #遍历L1中剩下的部分 将剩下的部分一个一个根据节点的popularity加入到网络G中
    for i in range(m0, all_add_edges):

        # similarity_dict.clear()
        #计算G中各个节点的度的权重占比 根据受欢迎性选择一个t要去连接的节点Y 这样就可以形成边(t,Y) 然后将其边添加到G中 形成最后的BA网络
        # node_by_probability = getnode_by_probability(G)
        #在这里根据余弦相似度来计算出和t最相似的节点(这里最相似的应该从G中的节点进行寻找 而非L3中),将其与t形成边


        #计算t和com对应向量的余弦相似度
        a_list = list(range(len(community_probability)))
        probability = np.multiply(G_probability_list, community_probability)

        probability = np.array(probability)

        probability /= probability.sum()   #normalize   归一化

        index = np.random.choice(a_list, size=2, replace=False, p=probability)

        for temp_index in index:
            com_probability = G_probability_list[temp_index]
            # w = cooperationNetwork.get_edge_data(L2[temp_index], L2[i])             #在这里w可能不存在 也就是为None 这里就要注意一下 判断一下
            # if w:
            #     w_value = w["weight"]
            # else:
            #     w_value = 1
            new_node_list = list(G.nodes)
            x = new_node_list[temp_index]
            y = L3[i]

            if G.has_edge(x, y):
                edgedata = G.get_edge_data(x, y)
                G.add_edge(x, y, weight=(edgedata["weight"] + 1))
                # 如果图中不存在该边,添加边然后设置权值为1
            else:
                G.add_edge(x, y, weight=1)

            #添加一个节点之后 就更新节点度的概率和社区的概率
            updata_temp_probability = updata_community_probability(G)
            community_probability = updata_temp_probability.copy()

            degree_probability = updata_degree_probability(G)
            G_probability_list = degree_probability.copy()

        # cosineSimilarity = cosine_similarity(vec_t.reshape(1, -1), vec_com.reshape(1, -1))
        # similarity_dict[com] = cosineSimilarity[0][0]

    # 保存图为GraphML文件
    # nx.write_graphml(G, './new_Network_different_metric/new_BACommunity_edges2000_resolution_0.6.graphml', encoding='utf-8')
    # 将边的节点和权重信息写入本地文件中
    with open('./new_Network_different_metric/new_BACommunity_edges5000_resolution_1.4.txt', 'w', encoding='utf-8') as f:
        for edge in G.edges(data=True):
            f.write(f'{edge[0]}\t{edge[1]}\t{edge[2]["weight"]}\n')

