import os
import tensorflow as tf
import numpy as np
from PIL import Image
from model.layer_modules import *  # 请确保这些模块存在并包含必要的函数
from util import *                 # 请确保这些模块存在并包含必要的函数
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

# 默认数据路径和模型保存路径（当未指定参数时）
train_list_file = './dataset/train_real.txt'
val_list_file = './dataset/val_real.txt'
input_directory = './dataset/instruction-front'
label_directory = './dataset/instruction-complete'
test_list_file = './dataset/test_real.txt'
checkpoint_dir = './checkpoint/xfer_complete_fromtrue'

# 日志保存路径将在后面根据 checkpoint_dir 设置
train_log_dir = None
val_log_dir = None

# 保存标签
def save_instr(fname, img):
    img = img[:, :, 0].astype(np.uint8)
    img = Image.fromarray(img, mode='P')
    img.putpalette([
        255, 0, 16,       # FK
        30, 206, 30,      # BK
        255, 255, 128,    # T
        0, 255, 127,      # H,M
        130, 210, 210,    # M
        34, 139, 34,      # E,V(L)
        0, 191, 255,      # V,HM
        0, 129, 69,       # V(R)
        255, 0, 190,      # V(L)
        255, 164, 4,      # X(R)
        0, 117, 220,      # X(L)
        179, 179, 179,    # T(F)
        255, 215, 0,      # V,M
        255, 105, 180,    # T(B)
        160, 32, 240,     # M,H(B)
        139, 69, 19,      # E,V(R)
        0, 164, 255,      # V,FK
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
        170, 10, 90,      # FO(2)
        25, 174, 255      # V,MH(B)
    ])
    img.save(fname)

# 构建模型 - 默认的 CNN 模型
def cnn_model(input_tensor):
    # 第一层卷积
    conv1 = tf.layers.conv2d(inputs=input_tensor, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
    # 第二层卷积
    conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=3, padding='same', activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=3, padding='same', activation=tf.nn.relu)
    conv5 = tf.layers.conv2d(inputs=conv4, filters=512, kernel_size=3, padding='same', activation=tf.nn.relu)
    # 输出层
    logits = tf.layers.conv2d(inputs=conv5, filters=num_classes_output, kernel_size=1, padding='same', activation=None)
    return logits

# 残差块
def residual_block(input_tensor, filters):
    input_channels = input_tensor.get_shape().as_list()[-1]
    shortcut = input_tensor
    x = tf.layers.conv2d(input_tensor, filters, kernel_size=3, padding='same', activation=None)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, filters, kernel_size=3, padding='same', activation=None)
    x = tf.layers.batch_normalization(x)

    if input_channels != filters:
        # 使用 1x1 卷积调整 shortcut 的通道数
        shortcut = tf.layers.conv2d(shortcut, filters, kernel_size=1, padding='same', activation=None)
        shortcut = tf.layers.batch_normalization(shortcut)

    x += shortcut
    x = tf.nn.relu(x)
    return x

# 膨胀卷积层
def dilated_conv_layer(input_tensor, filters, rate):
    return tf.layers.conv2d(inputs=input_tensor, filters=filters, kernel_size=3, padding='same',
                            activation=tf.nn.relu, dilation_rate=rate)

# 构建模型 - Residual 模型
def residual_model(input_tensor):
    net = input_tensor
    filters = 32

    # 编码器部分 - 卷积块 1
    net = tf.layers.conv2d(net, filters, kernel_size=3, padding='same', activation=None)
    net = tf.layers.batch_normalization(net)
    net = tf.nn.relu(net)
    # 残差块 1
    net = residual_block(net, filters)
    # 池化层 1
    pool1 = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='same')
    # 保存跳跃连接
    skip_connection1 = net

    # 编码器部分 - 卷积块 2
    filters *= 2  # filters = 64
    net = pool1
    net = tf.layers.conv2d(net, filters, kernel_size=3, padding='same', activation=None)
    net = tf.layers.batch_normalization(net)
    net = tf.nn.relu(net)
    # 残差块 2
    net = residual_block(net, filters)
    # 池化层 2
    pool2 = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='same')
    # 保存跳跃连接
    skip_connection2 = net

    # 瓶颈层
    filters *= 2  # filters = 128
    net = pool2
    net = tf.layers.conv2d(net, filters, kernel_size=3, padding='same', activation=None)
    net = tf.layers.batch_normalization(net)
    net = tf.nn.relu(net)
    net = residual_block(net, filters)
    # 膨胀卷积层
    net = dilated_conv_layer(net, filters, rate=2)

    # 解码器部分 - 上采样 1
    filters //= 2  # filters = 64
    net = tf.layers.conv2d_transpose(net, filters, kernel_size=3, strides=2, padding='same', activation=None)
    net = tf.layers.batch_normalization(net)
    net = tf.nn.relu(net)
    # 调整尺寸以匹配跳跃连接
    net = tf.image.resize_images(net, tf.shape(skip_connection2)[1:3])
    # 与编码器部分的跳跃连接
    net = tf.concat([net, skip_connection2], axis=-1)
    net = residual_block(net, filters)

    # 解码器部分 - 上采样 2
    filters //= 2  # filters = 32
    net = tf.layers.conv2d_transpose(net, filters, kernel_size=3, strides=2, padding='same', activation=None)
    net = tf.layers.batch_normalization(net)
    net = tf.nn.relu(net)
    # 调整尺寸以匹配跳跃连接
    net = tf.image.resize_images(net, tf.shape(skip_connection1)[1:3])
    # 与编码器部分的跳跃连接
    net = tf.concat([net, skip_connection1], axis=-1)
    net = residual_block(net, filters)

    # Dropout
    net = tf.layers.dropout(net, rate=0.5)

    # 输出层
    logits = tf.layers.conv2d(net, num_classes_output, kernel_size=1, padding='same', activation=None)
    return logits

# 构建模型 - U-Net 模型
def unet_model(input_tensor):
    # 编码器部分
    conv1 = tf.layers.conv2d(input_tensor, 32, kernel_size=3, padding='same', activation=None)
    conv1 = tf.layers.batch_normalization(conv1)
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, padding='same')

    conv2 = tf.layers.conv2d(pool1, 64, kernel_size=3, padding='same', activation=None)
    conv2 = tf.layers.batch_normalization(conv2)
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2, padding='same')

    # 底部层
    conv3 = tf.layers.conv2d(pool2, 128, kernel_size=3, padding='same', activation=None)
    conv3 = tf.layers.batch_normalization(conv3)
    conv3 = tf.nn.relu(conv3)

    # 解码器部分
    up1 = tf.layers.conv2d_transpose(conv3, 64, kernel_size=3, strides=2, padding='same', activation=None)
    up1 = tf.layers.batch_normalization(up1)
    up1 = tf.nn.relu(up1)
    # 调整尺寸以匹配跳跃连接
    up1 = tf.image.resize_images(up1, tf.shape(conv2)[1:3])
    concat1 = tf.concat([up1, conv2], axis=-1)
    conv4 = tf.layers.conv2d(concat1, 64, kernel_size=3, padding='same', activation=None)
    conv4 = tf.layers.batch_normalization(conv4)
    conv4 = tf.nn.relu(conv4)

    up2 = tf.layers.conv2d_transpose(conv4, 32, kernel_size=3, strides=2, padding='same', activation=None)
    up2 = tf.layers.batch_normalization(up2)
    up2 = tf.nn.relu(up2)
    # 调整尺寸以匹配跳跃连接
    up2 = tf.image.resize_images(up2, tf.shape(conv1)[1:3])
    concat2 = tf.concat([up2, conv1], axis=-1)
    conv5 = tf.layers.conv2d(concat2, 32, kernel_size=3, padding='same', activation=None)
    conv5 = tf.layers.batch_normalization(conv5)
    conv5 = tf.nn.relu(conv5)

    # 输出层
    logits = tf.layers.conv2d(conv5, num_classes_output, kernel_size=1, padding='same', activation=None)
    return logits

# 构建计算图
def build_graph(global_step, model_type):
    # 占位符
    X = tf.placeholder(tf.float32, [None, image_size, image_size, 1], name='input')
    Y = tf.placeholder(tf.int32, [None, image_size, image_size, 1], name='labels')

    # 根据模型类型选择模型
    if model_type == 'cnn':
        logits = cnn_model(X)
    elif model_type == 'residual':
        logits = residual_model(X)
    elif model_type == 'unet':
        logits = unet_model(X)
    else:
        raise ValueError("Invalid model_type. Expected 'cnn', 'residual', or 'unet'.")

    # 定义损失函数
    loss_xentropy = tf_loss_xentropy(tf.squeeze(Y, axis=-1), logits)
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
    accuracy, accuracy_op = tf_MILloss_accuracy(Y, tf.expand_dims(predictions, axis=-1))

    # 计算混淆矩阵
    confusion_matrix = comp_confusionmat(
        tf.expand_dims(predictions, axis=-1),
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

        # 确保图像形状为 (height, width, 1)
        if input_img.ndim == 2:
            input_img = np.expand_dims(input_img, axis=-1)
        if label_img.ndim == 2:
            label_img = np.expand_dims(label_img, axis=-1)

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

        # 确保图像形状为 (height, width, 1)
        if input_img.ndim == 2:
            input_img = np.expand_dims(input_img, axis=-1)

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
def train(model_type):
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # 构建计算图
    X, Y, logits, loss, train_op, accuracy, confusion_matrix, train_summary, val_summary, init, predictions, learning_rate, val_loss_ph, val_accuracy_ph, val_confusion_matrix_ph, val_learning_rate_ph = build_graph(global_step, model_type)

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
def test(model_type):
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # 构建计算图
    X, Y, logits, loss, train_op, accuracy, confusion_matrix, train_summary, val_summary, init, predictions, learning_rate, _, _, _, _ = build_graph(global_step, model_type)

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

                # 确保预测结果形状为 (height, width, 1)
                if pred_img.ndim == 2:
                    pred_img = np.expand_dims(pred_img, axis=-1)

                output_file = os.path.join(test_output_dir, file + '.png')
                save_instr(output_file, pred_img)
                print('Saved prediction for', file, 'to', output_file)

        print('Testing Finished.')

# 推理函数
def inference(input_files, output_dir, model_type):
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # 构建计算图
    X, Y, logits, loss, train_op, accuracy, confusion_matrix, train_summary, val_summary, init, predictions, learning_rate, _, _, _, _ = build_graph(global_step, model_type)

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

        input_batch = []
        output_paths = []
        for input_file in input_files:
            # 加载输入图像
            input_img = read_instr(input_file)

            # 确保图像形状为 (height, width, 1)
            if input_img.ndim == 2:
                input_img = np.expand_dims(input_img, axis=-1)
            input_batch.append(input_img)

            # 构造输出路径
            base_name = os.path.basename(input_file)         # e.g., img123.jpg
            name_only = os.path.splitext(base_name)[0]       # e.g., img123
            output_file = os.path.join(output_dir, name_only + '.png')
            output_paths.append(output_file)

        input_batch = np.array(input_batch)  # shape: [batch_size, H, W, 1]

        # 推理
        feed_dict = {X: input_batch}
        preds = sess.run(predictions, feed_dict=feed_dict)

        # 保存输出
        for pred_img, output_file in zip(preds, output_paths):
            if pred_img.ndim == 2:
                pred_img = np.expand_dims(pred_img, axis=-1)

            save_instr(output_file, pred_img)
            print('Inference result saved to:', output_file)

        print('Inference completed for all inputs.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test', 'inference'], help='Mode to run: train, test, or inference')
    parser.add_argument('--dataset', choices=['default', 'sj', 'mj'], default='default', help='Dataset to use: default, sj, or mj')
    parser.add_argument('--input_source', choices=['fromtrue', 'frompred'], default='frompred', help='Input source: fromtrue or frompred')
    parser.add_argument('--checkpoint_dir', help='Checkpoint directory')
    parser.add_argument('--input', nargs='+', help='Input file(s) for inference')
    parser.add_argument('--output', help='Output dir for inference')
    parser.add_argument('--model_type', choices=['cnn', 'residual', 'unet'], default='cnn', help='Model type to use')
    args = parser.parse_args()

    # 根据参数设置数据路径和模型保存路径
    # 默认设置
    if args.dataset == 'default':
        train_list_file = './dataset/train_real.txt'
        val_list_file = './dataset/val_real.txt'
        test_list_file = './dataset/test_real.txt'
    elif args.dataset == 'sj':
        train_list_file = './dataset/train_real_sj.txt'
        val_list_file = './dataset/val_real_sj.txt'
        test_list_file = './dataset/test_real_sj.txt'
    elif args.dataset == 'mj':
        train_list_file = './dataset/train_real_mj.txt'
        val_list_file = './dataset/val_real_mj.txt'
        test_list_file = './dataset/test_real_mj.txt'

    # 设置输入数据路径和模型保存路径
    if args.input_source == 'fromtrue':
        input_directory = './dataset/instruction-front'
    elif args.input_source == 'frompred':
        input_directory = './checkpoint/RFINet_front_xferln_160k/eval'

    # 指定 checkpoint_dir，覆盖
    checkpoint_dir = args.checkpoint_dir

    # 更新日志保存路径
    train_log_dir = os.path.join(checkpoint_dir, 'train')
    val_log_dir = os.path.join(checkpoint_dir, 'val')
    test_output_dir = os.path.join(checkpoint_dir, 'eval')

    # 创建必要的目录
    for directory in [checkpoint_dir, train_log_dir, val_log_dir, test_output_dir]:
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    # 设置全局变量 model_type，供 load_data 等函数使用
    model_type = args.model_type

    if args.mode == 'train':
        if not checkpoint_dir:
            print('Please provide --checkpoint_dir for train mode')
        else:
            train(args.model_type)
    elif args.mode == 'test':
        if not checkpoint_dir:
            print('Please provide --checkpoint_dir for test mode')
        else:
            test(args.model_type)
    elif args.mode == 'inference':
        if not args.input or not args.output:
            print('Please provide --input and --output for inference')
        elif not checkpoint_dir:
            print('Please provide --checkpoint_dir for inference mode')
        else:
            inference(args.input, args.output, args.model_type)
    else:
        print('Invalid mode')
