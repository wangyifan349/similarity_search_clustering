import numpy as np
# 添加向量到索引中
def add_vectors(vectors, index=None):
    if index is None:
        index = []
    for vector in vectors:
        index.append(vector)
    return index
# 使用余弦相似度搜索相似向量
def search_cosine_similarity(index, query_vector, k=10):
    similarities = []
    for i, vector in enumerate(index):
        dot_product = np.dot(query_vector, vector)
        magnitude1 = np.linalg.norm(query_vector)
        magnitude2 = np.linalg.norm(vector)
        similarity = dot_product / (magnitude1 * magnitude2)
        similarities.append((i, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    indices = [x[0] for x in similarities[:k]]
    return indices, [x[1] for x in similarities[:k]]
# 使用欧几里得距离搜索相似向量
def search_euclidean_distance(index, query_vector, k=10):
    distances = []
    for i, vector in enumerate(index):
        distance = np.linalg.norm(query_vector - vector)
        distances.append((i, distance))
    distances.sort(key=lambda x: x[1])
    indices = [x[0] for x in distances[:k]]
    return indices, [x[1] for x in distances[:k]]
# 进行KMeans聚类
def kmeans_clustering(vectors, k):
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
# 示例用法：
if __name__ == "__main__":
    vectors = np.random.rand(100, 128).astype('float32')
    index = add_vectors(vectors.tolist())
    query_vector = np.random.rand(128).astype('float32')
    indices, similarities = search_cosine_similarity(index, query_vector, k=10)
    print("余弦相似度搜索结果：", indices)
    indices, distances = search_euclidean_distance(index, query_vector, k=10)
    print("欧几里得距离搜索结果：", indices)
    labels = kmeans_clustering(vectors, 10)
    print("KMeans聚类结果：", labels)


