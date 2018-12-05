import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import DATA_PROCESS, LAPLACIAN, INDICES
from models import Model
import copy

N = 62

def transfer_coors_to_id(coors):
    i, j = coors[0], coors[1]
    id = -0.5 * i**2 + (N-1.5)*i + j -1
    return int(id)

# vertice_adjacencyMatrix = np.array(
#     [[0, 1, 0, 0],
#      [1, 0, 0, 0],
#      [0, 0, 0, 1],
#      [0, 0, 1, 0]])

# vertice_adjacencyMatrix = np.zeros([50, 50])
# vertice_adjacencyMatrix[0,3] = 1
# vertice_adjacencyMatrix[0,5] = 1
# vertice_adjacencyMatrix[1,8] = 1
# vertice_adjacencyMatrix[7,40] = 1
# vertice_adjacencyMatrix[8,12] = 1
# vertice_adjacencyMatrix[8, 20] = 1
# vertice_adjacencyMatrix[7, 30] = 1
def generate_id(adjacency_matrix, indices):
    matrix = copy.deepcopy(adjacency_matrix)
    for i in indices:
        matrix[i[0],i[1]] = 1
    id = DATA_PROCESS(matrix).get_label_id()
    return id
GcM = [8,8,2]
epochs = 200
dir = 'dolphins\out.dolphins'
vertice_adjacencyMatrix = np.zeros([N,N])

# get the indices of training set and probe set
train_positive_indices, train_negative_indices, probe_indices = INDICES(dir).label_indices()

# positive_labelled_id = list(map(transfer_coors_to_id, train_positive_indices))
# negative_labelled_id = list(map(transfer_coors_to_id, train_negative_indices))
# probe_labelled_id = list(map(transfer_coors_to_id, probe_indices))

positive_labelled_id = generate_id(vertice_adjacencyMatrix, train_positive_indices)
negative_labelled_id = generate_id(vertice_adjacencyMatrix, train_negative_indices)
probe_labelled_id = generate_id(vertice_adjacencyMatrix, probe_indices)
#
data = DATA_PROCESS(vertice_adjacencyMatrix)
W = data.get_edge_adjacency_matrix()
positive_labels = np.concatenate((np.ones([len(positive_labelled_id), 1]), np.zeros([len(positive_labelled_id), 1])),axis=1)
negative_labels = np.concatenate((np.zeros([len(negative_labelled_id), 1]), np.ones([len(negative_labelled_id), 1])),axis=1)
y = np.concatenate((positive_labels, negative_labels),axis=0)
# labelled_id = positive_labelled_id + negative_labelled_id
labelled_id = np.concatenate([positive_labelled_id,negative_labelled_id])

L = LAPLACIAN(W).get_laplician()
node_num,_ = np.shape(W)

x = np.eye(node_num)

graph = Model().train(L, GcM, labelled_id)

train_accuracy = []
print('Begin train')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in np.arange(epochs):
        predict_labels = np.zeros(np.shape(vertice_adjacencyMatrix)[0] * (np.shape(vertice_adjacencyMatrix)[0] - 1) // 2)
        print(epoch)
        sess.run(graph['train_op'], feed_dict={graph['x']: x, graph['y']: y})  # train the model
        labels = graph['pred_labels'].eval(feed_dict={graph['x']: x, graph['y']: y})  # evaluate the predicted labels
        predicts = sess.run(graph['y_predict'], feed_dict={graph['x']: x, graph['y']: y})
        for i in range(np.shape(vertice_adjacencyMatrix)[0]*(np.shape(vertice_adjacencyMatrix)[0]-1)//2):
            if predicts[i,0] > 0.5:
                predict_labels[i] = 1
        # if np.sum(np.array(predict_labels)) > node_num // 5 and np.sum(np.array(predict_labels)) < node_num // 4:
        #     break
        print(predict_labels)
        train_accuracy.append(sess.run(graph['loss_op'], feed_dict={graph['x']: x, graph['y']: y}))

# evaluate the model
predicts = predicts[:,0]
ids = sorted(range(len(predicts)), key=lambda k: predicts[k],reverse=True)
edge_num = len(positive_labelled_id) + len(probe_labelled_id)
print(ids)
print(predicts[ids])
ids = ids[0:edge_num]
shoots = [value for value in positive_labelled_id if value in ids]
percent = len(shoots)/len(positive_labelled_id)
print(percent)

plt.figure(1)
x_coor = np.arange(1,len(train_accuracy)+1)
plt.plot(x_coor, train_accuracy)

# plt1 = plt.figure(2)
# plt.hist(predicts[:,0])
plt.show()
print(np.sum(np.array(predict_labels)))
print(predict_labels[positive_labelled_id])
print(predict_labels[negative_labelled_id])

