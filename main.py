import numpy as np
from similarity_search import searchCosineSimilarity, addVectors, searchEuclideanDistance, kmeansClustering
# 生成多组随机向量
np.random.seed(0)
numVectors = 1000
vectorDimension = 128
vectorsList = []
for _ in range(10):
    # 随机生成一组向量
    vectors = np.random.rand(numVectors, vectorDimension).astype('float32')
    vectorsList.append(vectors)



# 对每组向量进行查询和聚类
for i, vectors in enumerate(vectorsList):
    print(f"Group {i+1} Vectors:")
    # 添加向量到索引中
    index = addVectors(vectors.tolist())
    # 随机生成一个查询向量
    queryVector = np.random.rand(vectorDimension).astype('float32')
    # 余弦相似度搜索
    indices, similarities = searchCosineSimilarity(index, queryVector, k=10)
    print("Cosine Similarity Search Results:", indices)
    # 打印相似度
    print("Similarities:", similarities)
    # 欧几里得距离搜索
    indices, distances = searchEuclideanDistance(index, queryVector, k=10)
    print("Euclidean Distance Search Results:", indices)
    # 打印距离
    print("Distances:", distances)
    # KMeans 聚类
    labels = kmeansClustering(vectors, 10)
    # 打印聚类结果
    print("KMeans Clustering Results:", labels)
    print()
