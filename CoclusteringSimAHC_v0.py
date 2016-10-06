'''
1st version, problems to be solved, code to be refined
tested by xywang jul 2016
1. give a matrix A, construct A' = D1^(-1/2) A D2^(-1/2)
2. compute l = ceil(log_2 k) singular vectors of An and construct matrix Z
3. run hierarchical clsutering on l-dimensional Z to obtain k-way hierarchic multipoatitioning
'''


import numpy as np
from sklearn.datasets import make_biclusters
from sklearn.utils.extmath import make_nonnegative, randomized_svd
from sklearn.cluster.bicluster import SpectralCoclustering, KMeans

# generate a bicluster dataset
n_clusters = 3
data, rows, columns = make_biclusters(
    shape=(80,80), n_clusters=n_clusters, noise=10, shuffle=True, random_state=0)

# step 1.
X = make_nonnegative(data)
row_diag = np.asarray(1.0 / np.sqrt(X.sum(axis=1)))
col_diag = np.asarray(1.0 / np.sqrt(X.sum(axis=0)))
normalized_data = row_diag[:, np.newaxis] * X * col_diag

# step 2.
n_sv = 1 + int(np.ceil(np.log2(n_clusters))) # =2
u, _, vt  = randomized_svd(normalized_data, n_sv)

n_discard = 1
u = u[:, n_discard:]

vt = vt[n_discard:]
v = vt.T

z = np.vstack((row_diag[:, np.newaxis] * u,
               col_diag[:, np.newaxis] * v)) # z.shape = (13,1)
# step 3.
model = KMeans(n_clusters)
model.fit(z)
centroid = model.cluster_centers_
labels = model.labels_  # the 1st 8 entries are row_lables

n_rows = X.shape[0]
row_labels_ = labels[:n_rows]
column_labels_ = labels[n_rows:]

rows_ = np.vstack(row_labels_ == c for c in range(n_clusters))
columns_ = np.vstack(column_labels_ == c for c in range(n_clusters))

from matplotlib import pyplot as plt

fit_data = data[np.argsort(row_labels_)]
fit_data = fit_data[:, np.argsort(column_labels_)]

plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.title("After clustering; rearranged to show clusters")

plt.show()

print z.shape
z_n, _ = z.shape
from matplotlib import pyplot as plt

plt.scatter(z[:,0], z[:,1])
plt.show()

# scipy ahc

from scipy.cluster.hierarchy import linkage, cophenet, dendrogram, fcluster
from scipy.spatial.distance import pdist

x = np.asarray(z)
Y = pdist(x,'euclidean')
res_linkage = linkage(x,"complete")
c, coph_dists = cophenet(res_linkage, Y)
print "cophenetic coefficient of dendrogram and distance matrix: %.2f" % c

# plot dendorgram using scipy.cluster.hierarchy

plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    res_linkage,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=3,  # show only the last p merged clusters
    show_leaf_counts=True,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()

def fancy_dendrogram(*args, **kwargs):
    
    # mark each merging distance in the dendrogram
    
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

#  plot using fancy_denrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
fancy_dendrogram(
    res_linkage,
    show_leaf_counts=True,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
    show_contracted=True,  # to get a distribution impression in truncated branches
    max_d=0.002,  # plot a horizontal cut-off line
)
plt.show()

last = res_linkage[:, 2]
last_rev = last[::-1] # reverse the array
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.show()
k = acceleration_rev.argmax() + 2  
print "clusters:", k  # k is selected when green line is at peak

# form flat clusters by using distances or number of clusters
# max_d = 0.05
# clusters = fcluster(res_linkage, max_d, criterion='distance')

clusters = fcluster(res_linkage, 3, criterion='maxclust')
print clusters

plt.figure(figsize=(10, 8))
plt.scatter(z[:,0], z[:,1], c=clusters, cmap='prism')  # plot points with cluster dependent colors
plt.show()

# sim-AHC

import lw_dist_v2
clust, n = lw_dist_v2.con_inp_to_class(z)
res_dict, _ = lw_dist_v2.build_lookup_tb(clust)

method = "complete"
i = 0
K = 3
den_arr = []

while (i < n-1):
    minVal, tup_ind = lw_dist_v2.build_mbp_tr(res_dict)
    new_temp = lw_dist_v2.temp(tup_ind[0],tup_ind[1], minVal)
    res_dict.pop(tup_ind)
    den_arr.append([new_temp.i, new_temp.j, new_temp.dist])
    upd_dict, del_dict, rest_dict = lw_dist_v2.build_del_upd_tbs(new_temp, res_dict)
    upd_dict_new = lw_dist_v2.update_dist(clust, upd_dict, del_dict, new_temp, method)
    if len(clust) == K:  # flatten dendrogram to k clusters
        break
    res_dict = lw_dist_v2.combine_tbs(upd_dict_new, rest_dict)
    i+=1
    
cophe_arr = lw_dist_v2.cophe_array(n, np.asarray(den_arr))
# squeezed_cophe_arr = np.squeeze(cophe_arr) # convert a numpy matrix into a list

# [start === to compare results of scipy ============
# for ward, centroid and median, scipy can't return correct results

from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist

x = np.asarray(z)
Y = pdist(x,'euclidean')
res_linkage = linkage(x,"complete")

_, coph_mat = cophenet(res_linkage, Y)
coeff_ = np.corrcoef(np.asarray(cophe_arr), coph_mat)[1,0]

print coeff_

# record dendrogram in a dictionary 

vis_dict = {}
for ele in den_arr:
    if (ele[0] not in vis_dict.keys()) & (ele[1] not in vis_dict.keys()):
        vis_dict[ele[0]] = [ele[0], ele[1]]
    elif (ele[0] in vis_dict.keys()) & (ele[1] not in vis_dict.keys()):
        vis_dict[ele[0]] = [vis_dict[ele[0]],ele[1]]
    elif (ele[0] not in vis_dict.keys()) & (ele[1] in vis_dict.keys()):
        vis_dict[ele[0]] = [ele[0], vis_dict[ele[1]]]
    elif (ele[0] in vis_dict.keys()) & (ele[1] in vis_dict.keys()):
        vis_dict[ele[0]] = [vis_dict[ele[0]], vis_dict[ele[1]]]
        vis_dict.pop(ele[1])

print len(vis_dict.keys())

# flatten dendorgram dictionary 

flatten=lambda l: sum(map(flatten,l),[]) if isinstance(l,list) else [l]
flat_dict= {key: flatten(vis_dict[key]) for key in vis_dict.keys() }

# label each input with predicted cluster label

lb_arr = np.zeros(z_n)

for key in flat_dict.keys():
    val_l = flat_dict[key]
    for ele in val_l:
        lb_arr[ele] = int(key)
        
print lb_arr

# plut flattened clusters with predicted labels

plt.figure(figsize=(10, 8))
plt.scatter(z[:,0], z[:,1], c=lb_arr, cmap='prism')  # plot points with cluster dependent colors
plt.show()
