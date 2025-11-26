import numpy as np
from similarity_search import search_cosine_similarity, add_vectors, search_euclidean_distance, kmeans_clustering

# 生成随机向量
np.random.seed(0)
vectors = np.random.rand(100, 128).astype('float32')

# 添加向量到索引中
index = add_vectors(vectors.tolist())

# 查询向量
query_vector = np.random.rand(128).astype('float32')

# 余弦相似度搜索
indices, similarities = search_cosine_similarity(index, query_vector, k=10)
print("余弦相似度搜索结果：", indices)
print("相似度：", similarities)

# 欧几里得距离搜索
indices, distances = search_euclidean_distance(index, query_vector, k=10)
print("欧几里得距离搜索结果：", indices)
print("距离：", distances)

# KMeans 聚类
labels = kmeans_clustering(vectors, 10)
print("KMeans 聚类结果：", labels)
