import scipy.spatial
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

class ADLAKNCN:
    """
    Parameters
    ----------
    k_max: int, optional (default = 5), the max number of nearest neighbors
    X_train: array, shape (n_queries, n_features)

    Y_train: array, shape (n_queries)

    """
    def __init__(self, k_max=20):
        self.k_max = k_max
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def distance(self, X1, X2):
      return scipy.spatial.distance.euclidean(X1, X2)

    def centroid(self, X_test, nonCentroidDataSet, centroidDataSet):
        CentroidDataSet = centroidDataSet.sum(axis=1)
        NewDataSet = (nonCentroidDataSet + np.repeat(np.expand_dims(CentroidDataSet,1),nonCentroidDataSet.shape[1],axis=1)) / (centroidDataSet.shape[1])

        # Calculate the distance.
        distance = np.linalg.norm(NewDataSet-np.repeat(np.expand_dims(X_test,1),NewDataSet.shape[1],1),axis=-1)
        sortedDistIndices = np.argsort(distance,axis=1)[:,0]
        sortedDistance = np.sort(distance,axis=1)[:,0]

        # Return to centroid,sortedDistance and sortedDistIndices.
        centroid = np.array([nonCentroidDataSet[i,sortedDistIndices[i],:] for i in range(len(X_test))])

        return centroid, sortedDistance, sortedDistIndices

    def predict(self, X_test):
        """
        :param X_test: array, shape (n_queries, n_features)
        :return: final_output: list,shape (n_queries)
        """
        #Search the k_max nearest centroid neighbors of X_test
        centroidDataSet = np.zeros([len(X_test),1,X_test.shape[1]])
        nonCentroidDataSet = np.repeat(np.expand_dims(self.X_train,0),len(X_test),0)
        labelTrain = np.repeat(np.expand_dims(self.Y_train,0),len(X_test),0)
        neigh_labels=[]
        for index in range(self.k_max):
            NewCentroid, dist, label = self.centroid(X_test, nonCentroidDataSet, centroidDataSet)
            centroidDataSet = np.concatenate([centroidDataSet,np.expand_dims(NewCentroid,1)],axis=1)
            nonCentroidDataSet=np.array([np.delete(nonCentroidDataSet[i],label[i],axis=0) for i in range(len(label))])
            neigh_labels.append(np.array([labelTrain[i,label[i]] for i in range(len(label))]))
            labelTrain=np.array([np.delete(labelTrain[i],label[i],axis=0) for i in range(len(label))])

        neigh_nodes=centroidDataSet[:,1:,:]
        neigh_labels=list(np.array(neigh_labels).T)


        final_output=[]
        for x_test_index in range(len(neigh_labels)):
            #Selection of discrimination class of x_test
            cent_number_class_list = []
            for k in range(self.k_max):
                k_neigh_nodes = neigh_nodes[x_test_index, :k + 1, :]
                k_neigh_labels = neigh_labels[x_test_index][:k + 1]
                k_neigh_labels_sort = Counter(k_neigh_labels)
                k_neigh_labels_sort = sorted(k_neigh_labels_sort.items(), key=lambda x: x[1], reverse=True)
                label_list=[]
                cent_list=[]
                cent_label_list=[]
                for one_class in k_neigh_labels_sort:
                    major_class=[i for i in k_neigh_labels_sort if i[1]==k_neigh_labels_sort[0][1]] if k_neigh_labels_sort else []
                    major_class_index=[np.where(np.array(k_neigh_labels)==i[0]) for i in major_class]
                    major_class_cent = [k_neigh_nodes[i] for i in major_class_index] if major_class_index else []
                    major_class_cent = [np.array(i).mean(axis=0) for i in major_class_cent] if major_class_cent else []
                    major_class_cent_label = [i[0] for i in major_class] if major_class_cent else []
                    label_list.append(major_class)
                    cent_list.append(major_class_cent)
                    cent_label_list.append(major_class_cent_label)
                    for c in major_class:
                        k_neigh_labels_sort.remove(c)
                label_num_list=[i[0][1] for i in label_list if i]
                cent_list=[i for i in cent_list if i]
                cent_label_list=[i for i in cent_label_list if i]
                score=[]
                for label_num,cent,cent_label in zip(label_num_list,cent_list,cent_label_list):
                    distance=[self.distance(X_test[x_test_index],i) for i in cent]
                    label=cent_label[np.argmin(np.array(distance))]
                    score.append([np.min(np.array(distance)),cent[np.argmin(np.array(distance))],label_num,label])
                score=sorted(score,key=lambda x:x[0])
                cent_number_class_list.append((score[0][1],score[0][2],score[0][3]))

            #Generate rank_ratio and rank_d.
            ratio_k = []
            disc_k = []
            for k in range(self.k_max):
                cent_k, num_k,_  = cent_number_class_list[k]
                ratio_k.append(num_k / (k + 1))
                disc_k.append(self.distance(X_test[x_test_index], cent_k))
            rank_ratio = list(pd.Series(ratio_k).rank(method='min', ascending=False))
            rank_disc = list(pd.Series(disc_k).rank(method='min', ascending=True))

            #Classify x_test to the class c corresponding to k with the smallest average rank.
            rank_ave = []
            for rank_ratio_k, rank_disc_k in zip(rank_ratio, rank_disc):
                rank_ave.append((rank_ratio_k / max(rank_ratio))*0.5 +(rank_disc_k / max(rank_disc))*0.5)
            index = int(np.argmin(np.array(rank_ave)))
            final_output.append(cent_number_class_list[index][2])
        return final_output

