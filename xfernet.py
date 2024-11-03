import os
import tensorflow as tf
import numpy as np
from PIL import Image
from model.layer_modules import *  # 假设您提供的工具函数都在 layer_modules.py 中
from util import *           # 假设您提供的工具函数都在 util.py 中
import matplotlib.pyplot as plt
import argparse

# 配置参数
batch_size = 32
num_epochs = 50
initial_learning_rate = 0.001
decay_steps = 2000  # 每 2000 步衰减一次
decay_rate = 0.1    # 衰减率
display_step = 10   # 每隔多少步打印一次日志
num_classes_input = 14  # 输入类别数（0到13）
num_classes_output = 34  # 输出类别数（0到33）
image_size = 20     # 图像尺寸

# 数据路径
train_list_file = './dataset/train_real.txt'
val_list_file = './dataset/val_real.txt'
input_directory = './checkpoint/RFINet_front_xferln_160k/eval'
label_directory = './dataset/instruction-complete'
test_list_file = './dataset/test_real.txt'

# 模型保存路径
checkpoint_dir = './checkpoint/xfer_complete_from160k'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# 测试保存路径
test_output_dir = os.path.join(checkpoint_dir, 'eval')
if not os.path.exists(test_output_dir):
    os.makedirs(test_output_dir)

# 日志保存路径
train_log_dir = os.path.join(checkpoint_dir, 'train')
if not os.path.exists(train_log_dir):
    os.makedirs(train_log_dir)

val_log_dir = os.path.join(checkpoint_dir, 'val')
if not os.path.exists(val_log_dir):
    os.makedirs(val_log_dir)


# 保存标签
def save_instr(fname, img):
    img = img[:,:,0].astype(np.uint8)
    img = Image.fromarray(img, mode = 'P')
    img.putpalette([
        255, 0, 16,       # FK
        30, 206, 30,      # BK
        255, 255, 128,    # T
        0, 255, 127,      # H|M
        130, 210, 210,    # M
        34, 139, 34,      # E|V(L)
        0, 191, 255,      # V|HM
        0, 129, 69,       # V(R)
        255, 0, 190,      # V(L)
        255, 164, 4,      # X(R)
        0, 117, 220,      # X(L)
        117, 59, 59,      # S
        179, 179, 179,    # T(F)
        255, 215, 0,      # V|M
        255, 105, 180,    # T(B)
        160, 32, 240,     # M|H(B)
        139, 69, 19,      # E|V(R)
        0, 164, 255,      # V|FK
        255, 30, 30,      # FK, MAK
        230, 230, 110,    # FT, FKMAK
        220, 200, 100,    # FT, MBK
        100, 230, 230,    # M, BK
        110, 220, 220,    # M, FK
        20, 200, 255,     # V, BK
        10, 140, 80,      # VR, FKMAK
        10, 250, 110,     # H, BK
        240, 20, 170,     # VL, FKMAK
        240, 100, 30,     # AO(2)
        250, 110, 40,     # O(5), AK
        200, 100, 70,     # O(5), FKBK
        220, 80, 60,      # BO(2)
        230, 90, 50,      # O(5), BK
        130, 10, 120,     # Y, MATBK
        170, 10, 90       # FO(2)

    ])
    img.save(fname)


# 构建模型
def cnn_model(input_tensor):
    # 第一层卷积
    conv1 = tf.layers.conv2d(inputs=input_tensor, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
    # 第二层卷积
    conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)
    # 输出层
    logits = tf.layers.conv2d(inputs=conv2, filters=num_classes_output, kernel_size=1, padding='same', activation=None)
    return logits


# 构建计算图
def build_graph(global_step):
    # 占位符
    X = tf.placeholder(tf.float32, [None, image_size, image_size, 1], name='input')
    Y = tf.placeholder(tf.int32, [None, image_size, image_size, 1], name='labels')

    # 模型输出
    logits = cnn_model(X)

    # 定义损失函数
    loss_xentropy = tf_MILloss_xentropy(tf.squeeze(Y), logits)
    loss_xentropy *= (1. - 0.5) * 3.
    loss = tf.reduce_mean(loss_xentropy)

    # 定义学习率衰减
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        global_step,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )

    # 定义优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    # 计算准确率
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
    accuracy, accuracy_op = tf_MILloss_accuracy(Y, tf.expand_dims(predictions, axis=3))

    # 计算混淆矩阵
    confusion_matrix = comp_confusionmat(
        tf.expand_dims(predictions, axis=3),
        Y,
        num_classes=num_classes_output,
        normalized_row=True,
        name='real'
    )

    # 定义摘要（TensorBoard）
    # 训练摘要
    train_summary_list = [
        tf.summary.scalar('train_loss', loss),
        tf.summary.scalar('train_accuracy', accuracy),
        tf.summary.scalar('learning_rate', learning_rate),
        tf.summary.image('train_confusion_matrix', tf_summary_confusionmat(confusion_matrix, num_classes_output))
    ]
    train_summary = tf.summary.merge(train_summary_list)

    # 验证摘要
    val_loss_ph = tf.placeholder(tf.float32, name='val_loss')
    val_accuracy_ph = tf.placeholder(tf.float32, name='val_accuracy')
    val_confusion_matrix_ph = tf.placeholder(tf.float32, shape=[num_classes_output, num_classes_output], name='val_confusion_matrix')
    val_learning_rate_ph = tf.placeholder(tf.float32, name='val_learning_rate')  # 新增

    val_summary_list = [
        tf.summary.scalar('val_loss', val_loss_ph),
        tf.summary.scalar('val_accuracy', val_accuracy_ph),
        tf.summary.scalar('learning_rate', val_learning_rate_ph),  # 新增
        tf.summary.image('val_confusion_matrix', tf_summary_confusionmat(val_confusion_matrix_ph, num_classes_output))
    ]
    val_summary = tf.summary.merge(val_summary_list)

    # 初始化变量
    init = tf.global_variables_initializer()

    return X, Y, logits, loss, train_op, accuracy, confusion_matrix, train_summary, val_summary, init, predictions, learning_rate, val_loss_ph, val_accuracy_ph, val_confusion_matrix_ph, val_learning_rate_ph


# 加载数据
def load_data(batch_files):
    batch_X = []
    batch_Y = []
    for file in batch_files:
        input_path = os.path.join(input_directory, file + '.png')
        label_path = os.path.join(label_directory, file + '.png')
        input_img = read_instr(input_path)
        label_img = read_instr(label_path)
        batch_X.append(input_img)
        batch_Y.append(label_img)
    batch_X = np.array(batch_X)
    batch_Y = np.array(batch_Y)
    return batch_X, batch_Y


def load_inputs(batch_files):
    batch_X = []
    for file in batch_files:
        input_path = os.path.join(input_directory, file + '.png')
        input_img = read_instr(input_path)
        batch_X.append(input_img)
    batch_X = np.array(batch_X)
    return batch_X


# 评估模型
def evaluate(sess, val_files, X, Y, loss, accuracy, confusion_matrix, learning_rate):
    num_batches = (len(val_files) + batch_size - 1) // batch_size
    val_loss = 0
    val_acc = 0
    total_confusion_matrix = np.zeros((num_classes_output, num_classes_output), dtype=np.float32)
    val_learning_rate = None  # 新增

    for i in range(num_batches):
        batch_files = val_files[i * batch_size:(i + 1) * batch_size]
        batch_X, batch_Y = load_data(batch_files)

        feed_dict = {X: batch_X, Y: batch_Y}
        loss_value, acc_value, conf_matrix, lr = sess.run(
            [loss, accuracy, confusion_matrix, learning_rate],  # 修改：添加 learning_rate
            feed_dict=feed_dict
        )
        val_loss += loss_value * len(batch_files)
        val_acc += acc_value * len(batch_files)
        total_confusion_matrix += conf_matrix * len(batch_files)
        val_learning_rate = lr  # 获取 learning_rate 的值

    val_loss /= len(val_files)
    val_acc /= len(val_files)
    total_confusion_matrix /= len(val_files)
    return val_loss, val_acc, total_confusion_matrix, val_learning_rate  # 修改：返回 val_learning_rate


# 训练函数
def train():
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # 构建计算图
    X, Y, logits, loss, train_op, accuracy, confusion_matrix, train_summary, val_summary, init, predictions, learning_rate, val_loss_ph, val_accuracy_ph, val_confusion_matrix_ph, val_learning_rate_ph = build_graph(global_step)

    # 创建会话
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init)
        # 设置摘要记录器
        train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        val_summary_writer = tf.summary.FileWriter(val_log_dir)

        # 模型保存器
        saver = tf.train.Saver()

        # 加载数据列表
        with open(train_list_file, 'r') as f:
            train_files = f.read().splitlines()
        with open(val_list_file, 'r') as f:
            val_files = f.read().splitlines()

        # 开始训练
        for epoch in range(num_epochs):
            # 训练阶段
            np.random.shuffle(train_files)
            num_batches = (len(train_files) + batch_size - 1) // batch_size
            for i in range(num_batches):
                batch_files = train_files[i * batch_size:(i + 1) * batch_size]
                batch_X, batch_Y = load_data(batch_files)

                feed_dict = {X: batch_X, Y: batch_Y}
                _, loss_value, acc_value, summary_str, lr = sess.run(
                    [train_op, loss, accuracy, train_summary, learning_rate],
                    feed_dict=feed_dict
                )

                # 写入训练摘要
                global_step_value = sess.run(global_step)
                train_summary_writer.add_summary(summary_str, global_step_value)

                # 打印日志
                if global_step_value % display_step == 0:
                    print('Epoch [{}], Step [{}], Loss: {:.4f}, Accuracy: {:.4f}, Learning Rate: {:.6f}'.format(
                        epoch + 1, global_step_value, loss_value, acc_value, lr))

            # 验证阶段
            val_loss, val_acc, val_conf_matrix, val_lr = evaluate(
                sess, val_files, X, Y, loss, accuracy, confusion_matrix, learning_rate)  # 修改：添加 learning_rate
            print('Validation Loss: {:.4f}, Validation Accuracy: {:.4f}, Learning Rate: {:.6f}'.format(val_loss, val_acc, val_lr))

            # 写入验证摘要
            val_summary_str = sess.run(val_summary, feed_dict={
                val_loss_ph: val_loss,
                val_accuracy_ph: val_acc,
                val_confusion_matrix_ph: val_conf_matrix,
                val_learning_rate_ph: val_lr  # 新增
            })
            val_summary_writer.add_summary(val_summary_str, global_step_value)

            # 保存模型
            saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'), global_step=global_step_value)

        # 关闭摘要记录器
        train_summary_writer.close()
        val_summary_writer.close()

        print('Training Finished.')


# 测试函数
def test():
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # 构建计算图
    X, Y, logits, loss, train_op, accuracy, confusion_matrix, train_summary, val_summary, init, predictions, learning_rate, _, _, _, _ = build_graph(global_step)

    # 创建会话
    with tf.Session() as sess:
        # 模型恢复器
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print('Model restored from:', ckpt)
        else:
            print('No checkpoint found in', checkpoint_dir)
            return

        # 加载测试数据列表
        with open(test_list_file, 'r') as f:
            test_files = f.read().splitlines()

        # 运行测试
        num_batches = (len(test_files) + batch_size - 1) // batch_size
        for i in range(num_batches):
            batch_files = test_files[i * batch_size:(i + 1) * batch_size]
            batch_X = load_inputs(batch_files)

            feed_dict = {X: batch_X}
            preds = sess.run(predictions, feed_dict=feed_dict)

            # 保存输出
            for j, file in enumerate(batch_files):
                pred_img = preds[j]
                output_file = os.path.join(test_output_dir, file + '.png')
                save_instr(output_file, pred_img)
                print('Saved prediction for', file, 'to', output_file)

        print('Testing Finished.')


# 推理函数
def inference(input_file, output_file):
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # 构建计算图
    X, Y, logits, loss, train_op, accuracy, confusion_matrix, train_summary, val_summary, init, predictions, learning_rate, _, _, _, _ = build_graph(global_step)

    with tf.Session() as sess:
        # 模型恢复器
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print('Model restored from:', ckpt)
        else:
            print('No checkpoint found in', checkpoint_dir)
            return

        # 加载输入图像
        input_img = read_instr(input_file)
        input_img = np.expand_dims(input_img, axis=0)  # 添加 batch 维度

        feed_dict = {X: input_img}
        pred = sess.run(predictions, feed_dict=feed_dict)
        pred_img = pred[0]

        save_instr(output_file, pred_img)
        print('Inference result saved to:', output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test', 'inference'], help='Mode to run: train, test, or inference')
    parser.add_argument('--input', help='Input file for inference')
    parser.add_argument('--output', help='Output file for inference')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    elif args.mode == 'inference':
        if not args.input or not args.output:
            print('Please provide --input and --output for inference')
        else:
            inference(args.input, args.output)
    else:
        print('Invalid mode')
