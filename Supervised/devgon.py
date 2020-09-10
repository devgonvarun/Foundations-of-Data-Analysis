# Varun Devgon
# Fda Lab assignment (supervised)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Task 1 - Visualize the data

def f(s): return 0 if s.decode(
    "utf-8") == "Iris-setosa" else 1 if s.decode("utf-8") == "Iris-versicolor" else 9


data = np.loadtxt('lab_iris_data.csv', delimiter=',', converters={3: f})

fig = plt.figure(figsize=(12, 6))
fig.suptitle('iris_data.csv 3D plots', fontsize=16)
ax1 = fig.add_subplot(121, projection='3d')

x1 = data[:, 0][data[:, 3] == 0]
y1 = data[:, 1][data[:, 3] == 0]
z1 = data[:, 2][data[:, 3] == 0]

x2 = data[:, 0][data[:, 3] == 1]
y2 = data[:, 1][data[:, 3] == 1]
z2 = data[:, 2][data[:, 3] == 1]

ax1.scatter(x2, y2, z2, c='g', label="Iris-versicolor")
ax1.scatter(x1, y1, z1, c='r', label="Iris-setosa")
ax1.legend(loc="lower right")

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Task 1 - Visualize the data')
ax1.view_init(15, -38)

labels = data[:, 3:4]
labels = labels.astype(int)

# Task 2 - Implement and train a regularized logistic regression model using stochastic gradient descent


def hypothesis(x, theta):
    return 1/(np.exp(np.dot(-x, theta))+1)


def gradient(theta, x, y, reg_factor):
    update = np.dot(x.T, (hypothesis(x, theta)-y))+(reg_factor*theta)
    return update


def train_classifier(T, init_learning_rate, k):
    theta = np.zeros((data.shape[1], 1))
    reg_factor = 1
    for t in range(1, T+1):
        n = init_learning_rate/np.sqrt(t)
        smalldata = data[np.random.choice(data.shape[0], k, replace=False), :]
        train_y = smalldata[:, 3:4]
        train_x = smalldata[:, 0:3]
        train_x = np.append(train_x, np.ones([len(train_x), 1]), 1)
        theta = theta - n*(1/k)*gradient(theta, train_x, train_y, reg_factor)
    return theta


theta = train_classifier(100, 0.1, 20)

new_x = data[:, 0:3]
new_x = np.append(new_x, np.ones([len(new_x), 1]), 1)
h = np.sign(hypothesis(new_x, theta) - 0.5) > 0
h = h.astype(int)
accuracy = np.mean(h == labels)
print("accuracy = ", accuracy*100, "%")

# Task 3 - Plot the separating hyperplane

ax2 = fig.add_subplot(122, projection='3d')
plot_x = np.linspace(min(data[:, 0]), max(data[:, 0]))
plot_y = np.linspace(min(data[:, 1]), max(data[:, 1]))
xx, yy = np.meshgrid(plot_x, plot_y)
plot_z = (-1/theta[2])*(theta[0] * plot_x + theta[1] * plot_y)
plot_z = np.expand_dims(plot_z, axis=0)


ax2.plot_surface(xx, yy, plot_z)

ax2.scatter(x2, y2, z2, c='g', label="Iris-versicolor")
ax2.scatter(x1, y1, z1, c='r', label="Iris-setosa")
ax2.legend(loc="lower right")

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Task 3 - Plot the separating hyperplane')
ax2.view_init(15, -38)

plt.tight_layout()
plt.show()

# Task 4 - Further questions (Included in the write-up devgon.pdf)
