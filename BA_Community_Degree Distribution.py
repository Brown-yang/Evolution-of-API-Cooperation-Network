import networkx as nx
import matplotlib.pyplot as plt

# 根据保存的图结构 构建图
txt = []
with open('edges.txt', 'r', encoding='utf-8') as f:
    for line in f:
        txt.append(line)

edges = []
for item in txt:
    a = item.strip().split('\t')
    a[2] = int(a[2])
    edges.append(tuple(a))

G = nx.Graph()
G.add_weighted_edges_from(edges)

# 计算度分布
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
degree_count = {}
for degree in degree_sequence:
    if degree in degree_count:
        degree_count[degree] += 1
    else:
        degree_count[degree] = 1

# 绘制度分布图
fig, ax = plt.subplots()
ax.scatter( degree_count.values(),degree_count.keys())
plt.title("Degree distribution")
plt.xlabel("Degree")
plt.ylabel("Count")
plt.show()