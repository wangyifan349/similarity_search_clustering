# 相似度搜索与聚类库 🤖

## 简介 👋

本库提供高效、易用的高维向量相似度搜索与聚类功能 🎉。包括余弦相似度搜索、欧几里得距离搜索和 KMeans 聚类等 🌟。

## 特色 🌈

*   **余弦相似度搜索** 🔍：使用余弦相似度搜索相似向量。
*   **欧几里得距离搜索** 📏：使用欧几里得距离搜索相似向量。
*   **KMeans 聚类** 📊：对向量集合进行 KMeans 聚类。

## 安装 🚀

克隆本仓库并安装依赖即可 🤔：

```bash
git clone https://github.com/wangyifan349/similarity_search_clustering.git
cd similarity_search_clustering
pip install -r requirements.txt
```

`requirements.txt` 文件内容如下 📝：

```
numpy
```

## 使用指南 📚

### 添加向量到索引 📈

使用 `add_vectors` 函数将向量添加到索引中 📊：

```python
import numpy as np
from similarity_search_clustering import add_vectors

vectors = np.random.rand(100, 128).astype('float32')
index = add_vectors(vectors.tolist())
```

### 余弦相似度搜索 🔍

使用 `search_cosine_similarity` 函数进行余弦相似度搜索 🔎：

```python
from similarity_search_clustering import search_cosine_similarity

query_vector = np.random.rand(128).astype('float32')
indices, similarities = search_cosine_similarity(index, query_vector, k=10)
print("余弦相似度搜索结果：", indices)
```

### 欧几里得距离搜索 📏

使用 `search_euclidean_distance` 函数进行欧几里得距离搜索 📐：

```python
from similarity_search_clustering import search_euclidean_distance

indices, distances = search_euclidean_distance(index, query_vector, k=10)
print("欧几里得距离搜索结果：", indices)
```

### KMeans 聚类 📊

使用 `kmeans_clustering` 函数进行 KMeans 聚类 📈：

```python
from similarity_search_clustering import kmeans_clustering

labels = kmeans_clustering(vectors, 10)
print("KMeans 聚类结果：", labels)
```

## 许可证 📜

本库使用 MIT 许可证 🎉。查看 [LICENSE](LICENSE) 获取更多信息 🤔。

## 贡献指南 🤝

欢迎贡献！请提交拉取请求到 [https://github.com/wangyifan349/similarity_search_clustering](https://github.com/wangyifan349/similarity_search_clustering) 🌟

## 作者 👨‍💻

*   [wangyifan349](https://github.com/wangyifan349)

如果您有任何问题，欢迎联系我

wangyifangwbk@163.com

## 致谢 🙏

*   本库使用 [NumPy](https://numpy.org/) 进行高效的数值计算 💻。

## 待办事项 📝

*   添加更多相似度搜索算法（例如 L2 距离、内积）。
*   提高 KMeans 聚类算法的效率 🚀。
*   支持超大内存数据集 🌐。

## 引用 📚

如果您在研究中使用本库，请按以下格式引用 📝：

```
@misc{wangyifan3492024similarity,
  title={相似度搜索与聚类库},
  author={王一帆},
  year={2025},
  eprint={},
  archivePrefix={GitHub},
  primaryClass={cs.CV}
}
```
