# Online Robust PCA
The online robust PCA is implemented to be applied to anomaly detection from fundus images.
## The Mathematical Model
The mathematical model proposed by the authors is demonstrated below:

![The overall mathematical model](https://github.com/yuchendu/OR-PCA/blob/main/fig/overall.png)

each column in X represents an image, thus X could be break down into the sum of columns as shown below:

![Deviding the matrix to single vectors](https://github.com/yuchendu/OR-PCA/blob/main/fig/single_vector.png)

Rearrange the formula, we obtain:

![Format 2](https://github.com/yuchendu/OR-PCA/blob/main/fig/sum.png)
