import numpy as np

def searchCosineSimilarity(index, queryVector, k=10):
    """
    计算查询向量与索引中向量的余弦相似度，并返回最相似的k个向量索引和相似度。
    参数：
    - index：向量索引列表
    - queryVector：查询向量
    - k：返回最相似的向量数量（默认10）
    返回：
    - indices：最相似的k个向量索引
    - similarities：最相似的k个向量与查询向量的相似度
    """
    similarities = []
    for i, vector in enumerate(index):
        # 计算查询向量与当前向量的点积
        dotProduct = np.dot(queryVector, vector)
        # 计算查询向量和当前向量的模
        magnitude1 = np.linalg.norm(queryVector)
        magnitude2 = np.linalg.norm(vector)
        # 计算余弦相似度
        similarity = dotProduct / (magnitude1 * magnitude2)
        similarities.append((i, similarity))
    # 按相似度降序排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    # 获取最相似的k个向量索引和相似度
    indices = [x[0] for x in similarities[:k]]
    similarities = [x[1] for x in similarities[:k]]
    return indices, similarities

def addVectors(vectors, index=None):
    """
    将新向量添加到索引中。

    参数：
    - vectors：新向量列表
    - index：现有索引列表（默认None）

    返回：
    - 更新后的索引列表
    """
    if index is None:
        index = []
    for vector in vectors:
        index.append(vector)
    return index

def searchEuclideanDistance(index, queryVector, k=10):
    """
    计算查询向量与索引中向量的欧几里得距离，并返回最近的k个向量索引和距离。

    参数：
    - index：向量索引列表
    - queryVector：查询向量
    - k：返回最近的向量数量（默认10）
    返回：
    - indices：最近的k个向量索引
    - distances：最近的k个向量与查询向量的距离
    """
    distances = []
    for i, vector in enumerate(index):
        # 计算查询向量与当前向量的欧几里得距离
        distance = np.linalg.norm(np.array(queryVector) - np.array(vector))
        distances.append((i, distance))
    # 按距离升序排序
    distances.sort(key=lambda x: x[1])
    # 获取最近的k个向量索引和距离
    indices = [x[0] for x in distances[:k]]
    distances = [x[1] for x in distances[:k]]
    return indices, distances

def kmeansClustering(vectors, k):
    """
    对向量进行k-means聚类。

    参数：
    - vectors：向量矩阵
    - k：聚类数量

    返回：
    - labels：每个向量的聚类标签
    """
    np.random.seed(0)
    # 随机初始化k个聚类中心
    centroids = vectors[np.random.choice(vectors.shape[0], k, replace=False)]
    # 初始化聚类标签
    labels = np.zeros(vectors.shape[0])
    for _ in range(100):
        # 计算每个向量到聚类中心的距离
        distances = np.linalg.norm(vectors[:, np.newaxis] - centroids, axis=2)
        # 更新聚类标签
        labels = np.argmin(distances, axis=1)
        # 更新聚类中心
        newCentroids = []
        for i in range(k):
            pointsInCluster = vectors[labels == i]
            if pointsInCluster.size:
                centroid = pointsInCluster.mean(axis=0)
            else:
                centroid = centroids[i]
            newCentroids.append(centroid)
        newCentroids = np.array(newCentroids)
        # 检查聚类中心是否收敛
        if np.all(centroids == newCentroids):
            break
        centroids = newCentroids
    return labels
