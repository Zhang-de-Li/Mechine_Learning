from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split  # 训练集划分
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, precision_score, \
    accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn import tree


# 加载数据集
def loadDataSet():
    iris_dataset = load_iris()
    X = iris_dataset.data
    Y = iris_dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    return X_train, X_test, y_train, y_test


# 训练决策树模型
def trainDT(x_train, y_train):
    # DT生成和训练
    clf = tree.DecisionTreeClassifier(criterion = "gini")
    clf.fit(x_train, y_train)
    return clf


def drawROC(y_one_hot, y_pre_pro):
    # AUC值
    auc = roc_auc_score(y_one_hot, y_pre_pro, average = 'micro')
    # 绘制ROC曲线
    fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(), y_pre_pro.ravel())  # ravel()方法将数组维度拉成一维数组
    plt.plot(fpr, tpr, linewidth = 2, label = 'AUC = %.3f' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1.1, 0, 1.1])
    plt.xlabel('False Postivie Rate')
    plt.ylabel('True Positivie Rate')
    plt.legend()
    plt.show()


# 测试模型
def test(model, x_test, y_test):
    y_one_hot = label_binarize(y_test, classes = np.arange(3))  # 将标签转换为one-hot形式
    y_pre = model.predict(x_test)  # 预测结果
    y_pre_pro = model.predict_proba(x_test)  # 预测结果的概率

    con_matrix = confusion_matrix(y_test, y_pre)  # 混淆矩阵
    print('confusion_matrix:\n', con_matrix)
    print('accuracy:{}'.format(accuracy_score(y_test, y_pre)))  # 准确率
    print('precision:{}'.format(precision_score(y_test, y_pre, average = 'micro')))  # 精度
    print('recall:{}'.format(recall_score(y_test, y_pre, average = 'micro')))  # 召回率
    print('f1-score:{}'.format(f1_score(y_test, y_pre, average = 'micro')))  # F1分数

    # 绘制ROC曲线
    drawROC(y_one_hot, y_pre_pro)


def iris_type(str):
    if str == b'Iris-setosa':
        return 0
    elif str == b'Iris-versicolor':
        return 1
    else:
        return 2


def get_pairs(path):
    data = np.loadtxt(path, dtype = float, delimiter = ',', converters = {4: iris_type})  # 对数据进行预处理
    x_prime, y = np.split(data, (4,), axis = 1)  # 按列切分，将前四列和第五列分开
    feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    feature_name = ['sepal length ', 'sepal width ', 'petal length ', 'petal width ']
    # plt.figure(figsize = (10, 9), facecolor = '#FFFFFF')
    for i, pair in enumerate(feature_pairs):  # enumerate()将一个可遍历的数据对象组合为一个索引序列
        # 准备数据
        x = x_prime[:, pair]  # 从[0, 1]开始，对不同的特征进行组合
        # 决策树学习
        # min_samples_leaf当叶子结点上的样本数小于该参数指定的值时，则该叶子节点及其兄弟节点将被剪枝
        clf = tree.DecisionTreeClassifier(criterion = 'gini', min_samples_leaf = 3)
        dt_clf = clf.fit(x, y)
        y_hat = dt_clf.predict(x)
        y = y.reshape(-1)
        c = np.count_nonzero(y_hat == y)  # 统计预测正确的个数
        print("特征：", feature_name[pair[0]], '+', feature_name[pair[1]])
        print('\t预测正确数目', c)
        print('\t准确率：%.2f%%' % (100 * float(c) / float(len(y))))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = loadDataSet()
    # 训练决策树模型
    model = trainDT(X_train, y_train)
    # 决策树模型的保存
    path = './Iris.data'
    get_pairs(path)
    f = open('./iris_gini.dot', 'w')
    tree.export_graphviz(model, out_file = f)
    test(model, X_test, y_test)
