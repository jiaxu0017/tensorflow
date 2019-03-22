import tensorflow as tf
from numpy.random import RandomState    #通过Numpy工具包生成模拟数据集

batch_size = 8 #定义训练数据batch的大小

# 采用两个输入，一层隐藏层（三个结点），一个输出的网络结构
w1 = tf.Variable(tf.random_normal([2, 3], stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1))

#在 shape 的一个维度上使用 None 可以方便使用不同的 batch 大小。在训练时需要把数据分
#成比较小的 batch ， 但是在测试时，可以一次性使用全部的数据。当数据集比较小时这样比较
#方便测试，但数据集比较大时，将大量数据放入一个 batch 吁能会导致内存溢出。
x = tf.placeholder(tf.float32, shape = (None, 2), name = 'x-input')
y_ = tf.placeholder(tf.float32, shape = (None, 1), name = 'y-input')

# 定义神经网络的前行传播
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#定义损失函数和反向传播算法
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 随机生成一个数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 定义规则来给出样本的标签，在这里所有的x1+x2 <1 的样例都被认为是正样本（比如零件合格）
# 而其他的为负样本（比如零件不合格）。和tensorflow游乐场中的表示法不大一样的地方是，
# 在这里使用0来表示负样本，1来表示正样本。大部分解决分类问题的神经网络都会采用0和1的表示方法。
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# 创建一个会话来运行tensorflow程序
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    #初始化变量
    sess.run(init_op)
    
    print(sess.run(w1))
    print(sess.run(w2))
    
    STEPS = 5000
    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练,start为开始位置，end 为结束位置
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        # 通过选取的样本训练神经网络并更新参数
        #sess.run([train_step, y, y_],feed_dict = {x: X[start:end], y_: Y[start:end]} )
        sess.run([train_step, y, y_], feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            # 每隔一段时间计算在所有数据集上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy,feed_dict = {x:X,y_:Y})
            print ("After %d training step (s), cross entropy on all data is %g " %(i , total_cross_entropy) )
        
    print("\n")
    print(sess.run(w1))
    print(sess.run(w2))
