import numpy as np

def search_cosine_similarity(index, query_vector, k=10):
    """
    使用余弦相似度搜索相似向量

    参数：
        index (list)：索引列表
        query_vector (numpy.array)：查询向量
        k (int)：返回前 k 个相似向量，默认为 10

    返回值：
        indices (list)：前 k 个相似向量的索引列表
        similarities (list)：前 k 个相似向量的相似度列表
    """
    similarities = []
    for i, vector in enumerate(index):
        # 计算点积
        dot_product = np.dot(query_vector, vector)

        # 计算向量模
        magnitude1 = np.linalg.norm(query_vector)
        magnitude2 = np.linalg.norm(vector)

        # 计算余弦相似度
        similarity = dot_product / (magnitude1 * magnitude2)

        similarities.append((i, similarity))

    # 按相似度降序排列
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 取前 k 个相似向量
    indices = [x[0] for x in similarities[:k]]
    similarities = [x[1] for x in similarities[:k]]

    return indices, similarities

def add_vectors(vectors, index=None):
    """
    添加向量到索引中

    参数：
        vectors (list)：要添加的向量列表
        index (list)：可选，现有索引列表，默认为 None

    返回值：
        index (list)：更新后的索引列表
    """
    if index is None:
        index = []
    for vector in vectors:
        index.append(vector)
    return index

def search_euclidean_distance(index, query_vector, k=10):
    """
    使用欧几里得距离搜索相似向量

    参数：
        index (list)：索引列表
        query_vector (numpy.array)：查询向量
        k (int)：返回前 k 个相似向量，默认为 10

    返回值：
        indices (list)：前 k 个相似向量的索引列表
        distances (list)：前 k 个相似向量的距离列表
    """
    distances = []
    for i, vector in enumerate(index):
        distance = np.linalg.norm(np.array(query_vector) - np.array(vector))
        distances.append((i, distance))

    distances.sort(key=lambda x: x[1])

    indices = [x[0] for x in distances[:k]]
    distances = [x[1] for x in distances[:k]]

    return indices, distances

def kmeans_clustering(vectors, k):
    """
    进行 KMeans 聚类

    参数：
        vectors (numpy.array)：要聚类的向量矩阵
        k (int)：聚类数量

    返回值：
        labels (numpy.array)：聚类标签列表
    """
    np.random.seed(0)
    centroids = vectors[np.random.choice(vectors.shape[0], k, replace=False)]
    labels = np.zeros(vectors.shape[0])
    for _ in range(100):
        distances = np.linalg.norm(vectors[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = []
        for i in range(k):
            points_in_cluster = vectors[labels == i]
            if points_in_cluster.size:
                centroid = points_in_cluster.mean(axis=0)
            else:
                centroid = centroids[i]
            new_centroids.append(centroid)
        new_centroids = np.array(new_centroids)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels
