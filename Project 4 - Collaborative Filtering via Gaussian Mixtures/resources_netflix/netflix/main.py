import numpy as np
import kmeans
import common
import naive_em
import em

# X = np.loadtxt("toy_data.txt")

# TODO: Your code here

###  2. K-Means

# cost_array = np.zeros([5,4])
# cost_array.fill(np.nan) # allocate non array
#
#
# for i in range(5):
#
#     for K in range(1,5):
#
#         [mixture, post] = common.init(X,K,i)
#         [mixture, post, cost] = kmeans.run(X, mixture, post)
#
#         cost_array[i, K-1] = cost
#         cost_min = np.min(cost_array, axis=0) # min for each column
#
#         # common.plot(X, mixture, post, 'clustering')
#
# print(cost_array)
# print(cost_min)

### 3. Expectation-Maximization Algorithm

# L_ = np.zeros([5,4])
# L_.fill(np.nan)
#
# for seed in range(5):
#     for K in range(1,5):
#         mixture, p = common.init(X, K, seed)
#         mixture, p, L = naive_em.run(X, mixture, p)
#
#         L_[seed, K-1] = L
#
#         L_min = np.min(L_, axis = 0)
#
#         common.plot(X, mixture, p, 'Clustering of the data')
#
# print(L_)
# print(L_min)

### 4. Comparing K-means and EM

#K = 3
# seed = 0
# mixture, post = common.init(X, K, seed)
# mixture, post, cost = kmeans.run(X, mixture, post)
#
# common.plot(X, mixture, post, 'Clustering')
#
# print(mixture)

# K-means: determining the centroids by comparing the cost
# em_k_dict = dict()
# em_total_likelihood_dict = dict()
# for seed in range(5):
#     em_total_likelihood = 0
#     for k in range(1, 5):
#         mixture, post = common.init(X=X, K=k, seed=seed)
#         likelihood = naive_em.run(X, mixture, post)[2]
#         em_total_likelihood += likelihood
#         em_k_dict.update({(seed, k): likelihood})
#     em_total_likelihood_dict.update({seed: em_total_likelihood})
#
# ### get the best seed and the best k size that minimizes the cost
#
# ## Best seed
# # Get the lowest cost
# optimal_seed_cost = em_total_likelihood_dict[0]
# for k, v in em_total_likelihood_dict.items():
#     if v > optimal_seed_cost:
#         optimal_seed_cost = v
#     else:
#         optimal_seed_cost = optimal_seed_cost
# # Get the seed associated with the lowest cost
# for k, v in em_total_likelihood_dict.items():
#     if v == optimal_seed_cost:
#         optimal_seed = k
# print(em_k_dict)

### 5. Bayesian information Criterion
#### Finding the best K

# BIC_values = np.array([])
# seed = 0
#
# for K in range(1,5):
#     mixture, post = common.init(X, K, seed)
#     mixture, post, L = naive_em.run(X, mixture, post)
#
#     BIC_actual = common.bic(X, mixture, L)
#
#     BIC_values = np.append(BIC_values, BIC_actual)
#
# BIC_best = max(BIC_values)
# K_best = list(BIC_values).index(BIC_best) + 1
#
# print(BIC_best, K_best)

### 8. Using the mixture model for collaborative filtering

X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt('netflix_complete.txt')

mixture, post = common.init(X, K=12, seed = 1   )

mixture, post, loglike = em.run(X, mixture, post)

X_pred = em.fill_matrix(X, mixture)

print(common.rmse(X_gold, X_pred))

print(mixture)
print(loglike)

em_k_dict = dict()
em_total_likelihood_dict = dict()
for seed in range(5):
    em_total_likelihood = 0
    for k in [1, 12]:
        mixture, post = common.init(X=X, K=k, seed=seed)
        likelihood = em.run(X, mixture, post)[2]
        em_total_likelihood += likelihood
        em_k_dict.update({(seed, k): likelihood})
    em_total_likelihood_dict.update({seed: em_total_likelihood})
#
# ### get the best seed and the best k size that minimizes the cost
#
# ## Best seed
# # Get the lowest cost
optimal_seed_cost = em_total_likelihood_dict[0]
for k, v in em_total_likelihood_dict.items():
    if v > optimal_seed_cost:
        optimal_seed_cost = v
    else:
        optimal_seed_cost = optimal_seed_cost
# Get the seed associated with the lowest cost
for k, v in em_total_likelihood_dict.items():
    if v == optimal_seed_cost:
        optimal_seed = k
print(em_k_dict)
#
