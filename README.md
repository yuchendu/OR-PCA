# Online Robust PCA
We implemented the online robust PCA method from paper "Online Robust PCA via Stochastic Optimization". The codes I provided is designed for anomaly detection in fundus images.
## The Mathematical Model
The authors propose the following mathematical model:

![The overall mathematical model](https://github.com/yuchendu/OR-PCA/blob/main/fig/overall.png)

Each column in 𝑋 represents an image, and 𝑋 can be decomposed into a sum of columns, as illustrated below:

![Deviding the matrix to single vectors](https://github.com/yuchendu/OR-PCA/blob/main/fig/single_vector.png)

Rearranging the formula, we obtain:

![Format 2](https://github.com/yuchendu/OR-PCA/blob/main/fig/sum.png)

Dividing the entire formula by 𝑛, we get the re-formulated expression:

![re-formulated form](https://github.com/yuchendu/OR-PCA/blob/main/fig/reformulated.png)

From this equation, it is evident that as 𝑛 increases, the left part of the formula, which represents the average value associated with the reconstruction coefficient 𝑣𝑖 and sparse error 𝑒𝑖, approaches 0. Therefore, 𝜆1 and 𝜆2 serve as weights that influence the balance between 𝑣𝑖 and 𝑒𝑖, representing the low-rank constraint power and sparsity constraint power, respectively.
