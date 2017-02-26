import numpy as np
import tensorflow as tf
import initialize_parameters as ip
import loss_funcs as lf
import time
import itertools as its
TRAIN_SIZE = 500000
TEST_SIZE = 100000
NUM_FEATURES = 22
NUM_FEATURES_EMBEDDED = 50
NUM_FEATURES_NN = 152
BATCH_SIZE = 10
NUM_EPOCH = 30
NUM_HEROES = 113
NUM_ITEMS = 265
NUM_USERS = 158360
HIDDEN = 500
SAVED_MODEL_PATH = '.../pairnet.ckpt'
SAVED_MODEL_PATH_TEST = '.../pairnet.ckpt'


def acc(pred, labels):
    s = np.float32(np.sum(pred == labels))
    return s / np.prod(pred.shape)


def main(mode = 'train'):
    idx_item = [17, 18, 19, 20, 21, 22]
    print 'initializing...'

    match_data = np.load('.../data/dota2_576/match_refined.npy')
    match_outcome = np.load('.../data/dota2_576/match.npy')
    match_items = np.load('.../data/dota2_576/match_mat.npy')
    index_items = match_items[:, idx_item]
    train_data = match_data[:, 2 : ]
    gold_sum = np.expand_dims(train_data[:, 0] + train_data[:, 1], 1)
    train_data = np.concatenate((gold_sum, train_data[:, 2:]), 1)
    train_data = train_data * 10 / np.abs(np.max(train_data, 0, keepdims=1))
    index_uh = match_data[:, : 2]
    train_labels = match_outcome[:, 0]
    train_labels = 2* train_labels - 1
    ratio = match_outcome[:, 1]

    train_data_node = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, NUM_FEATURES])
    train_labels_node = tf.placeholder(dtype=tf.float32, shape=[1])
    index_uh_node = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE, 2])
    index_item_node = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE, 6])
    ratio_node = tf.placeholder(dtype=tf.float32, shape=[1])
    loss_curve = None
    # test_data_node = tf.constant(test_data, dtype=tf.float32)

    w_statistics = ip.init_fc_weight_2d([NUM_HEROES, NUM_FEATURES])
    w_item_hero = ip.init_fc_weight_2d([NUM_HEROES, 4])
    w_item_item = ip.init_fc_weight_2d([NUM_ITEMS + 1, 4])
    w_embedded = ip.init_fc_weight_2d([NUM_HEROES, NUM_FEATURES_EMBEDDED])

    b_user_base = ip.init_fc_bias_2d([NUM_USERS + 1, 1])
    b_user_hero = np.zeros([NUM_USERS + 1, NUM_HEROES])
    b_hero = ip.init_fc_bias_2d([NUM_HEROES, 1])

    w_fc1 = ip.init_fc_weight_2d([NUM_FEATURES_NN, HIDDEN])
    b_fc1 = ip.init_fc_bias_2d([HIDDEN])

    w_fc2 = ip.init_fc_weight_2d([HIDDEN, HIDDEN])
    b_fc2 = ip.init_fc_bias_2d([HIDDEN])

    w_fc3 = ip.init_fc_weight_2d([HIDDEN, HIDDEN])
    b_fc3 = ip.init_fc_bias_2d([HIDDEN])

    w_fc4 = ip.init_fc_weight_2d([HIDDEN, 1])

    def model(X, train_mode = 'train'):
        E1_unary, E2_unary, E1_pair, E2_pair = 0, 0, 0, 0
        bh_list = []
        if train_mode == 'train':
            for i in range(10):
                if i < 5:
                    temp = tf.reduce_sum(X[i, :] * tf.gather(w_statistics, index_uh_node[i, 1]))
                    temp += tf.reduce_sum(tf.matmul(tf.gather(w_item_item, index_item_node[i, :]),
                                  tf.expand_dims(tf.gather(w_item_hero, index_uh_node[i, 1]), dim=1)))
                    bh_list.append(temp)
                    E1_unary += temp
                    if index_uh_node[i, 1] != 0:
                        E1_unary += b_hero[index_uh_node[i, 1], :]
                    else:
                        E1_unary += tf.reduce_mean(b_hero)

                    if index_uh_node[i, 0] != 0:
                        E1_unary += b_user_base[index_uh_node[i, 0], :]
                    else:
                        E1_unary += tf.reduce_mean(b_user_base)
                else:
                    temp = tf.reduce_sum(X[i, :] * tf.gather(w_statistics, index_uh_node[i, 1]))
                    temp += tf.reduce_sum(tf.matmul(tf.gather(w_item_item, index_item_node[i, :]),
                                                    tf.expand_dims(tf.gather(w_item_hero, index_uh_node[i, 1]), dim=1)))
                    bh_list.append(temp)
                    E2_unary += temp
                    if index_uh_node[i, 1] != 0:
                        E2_unary += b_hero[index_uh_node[i, 1], :]
                    else:
                        E2_unary += tf.reduce_mean(b_hero)

                    if index_uh_node[i, 0] != 0:
                        E2_unary += b_user_base[index_uh_node[i, 0], :]
                    else:
                        E2_unary += tf.reduce_mean(b_user_base)


        w1_pair, w2_pair = 0, 0
        for tup in its.combinations(range(5), 2):
            idx1 = tf.maximum(index_uh_node[tup[0], 1], index_uh_node[tup[1], 1])
            idx2 = tf.minimum(index_uh_node[tup[0], 1], index_uh_node[tup[1], 1])
            if w1_pair == 0:
                w1_pair = tf.concat(1, [tf.concat(1, [tf.gather(w_statistics, [idx1]),
                                                      tf.gather(w_item_hero, [idx1]),
                                                      tf.gather(w_embedded, [idx1])]),
                                        tf.concat(1, [tf.gather(w_statistics, [idx2]),
                                                      tf.gather(w_item_hero, [idx2]),
                                                      tf.gather(w_embedded, [idx2])])])
            else:
                w1_pair = tf.concat(0, [w1_pair,
                                        tf.concat(1, [tf.concat(1, [tf.gather(w_statistics, [idx1]),
                                                                    tf.gather(w_item_hero, [idx1]),
                                                                    tf.gather(w_embedded, [idx1])]),
                                                      tf.concat(1, [tf.gather(w_statistics, [idx2]),
                                                                    tf.gather(w_item_hero, [idx2]),
                                                                    tf.gather(w_embedded, [idx2])])])
                                        ])

        for tup in its.combinations(range(5, 10), 2):
            idx1 = tf.maximum(index_uh_node[tup[0], 1], index_uh_node[tup[1], 1])
            idx2 = tf.minimum(index_uh_node[tup[0], 1], index_uh_node[tup[1], 1])
            if w2_pair == 0:
                w2_pair = tf.concat(1, [tf.concat(1, [tf.gather(w_statistics, [idx1]),
                                                      tf.gather(w_item_hero, [idx1]),
                                                      tf.gather(w_embedded, [idx1])]),
                                        tf.concat(1, [tf.gather(w_statistics, [idx2]),
                                                      tf.gather(w_item_hero, [idx2]),
                                                      tf.gather(w_embedded, [idx2])])])
            else:
                w2_pair = tf.concat(0, [w2_pair,
                                        tf.concat(1, [tf.concat(1, [tf.gather(w_statistics, [idx1]),
                                                                    tf.gather(w_item_hero, [idx1]),
                                                                    tf.gather(w_embedded, [idx1])]),
                                                      tf.concat(1, [tf.gather(w_statistics, [idx2]),
                                                                    tf.gather(w_item_hero, [idx2]),
                                                                    tf.gather(w_embedded, [idx2])])])
                                        ])

        fc_input = tf.concat(0, [w1_pair, w2_pair])
        hidden = tf.nn.relu(tf.matmul(fc_input, w_fc1) + b_fc1)
        hidden = tf.nn.relu(tf.matmul(hidden, w_fc2) + b_fc2)
        hidden = tf.nn.relu(tf.matmul(hidden, w_fc3) + b_fc3)
        hidden = tf.matmul(hidden, w_fc4)

        E1_pair = tf.reduce_sum(hidden[: 10, : ])
        E2_pair = tf.reduce_sum(hidden[10: , : ])

        if train_mode == 'train':
            return E1_unary, E2_unary, E1_pair, E2_pair, bh_list
        elif train_mode == 'test':
            return E1_pair, E2_pair

        return None


    ret = model(train_data_node)
    reg_hinge = tf.nn.l2_loss(w_statistics) \
                # + tf.nn.l2_loss(w_item_hero)
    reg_nn = tf.nn.l2_loss(w_fc1) + \
              tf.nn.l2_loss(w_fc2) + \
                tf.nn.l2_loss(w_fc3) + \
              tf.nn.l2_loss(w_fc4)


    # loss_hinge = ratio_node * tf.maximum(0.0, -train_labels_node * (ret[0] - ret[1])) + reg_hinge
    #
    # loss_nn = ratio_node * tf.maximum(0.0, -train_labels_node * (ret[2] - ret[3])) + 1e-4 * reg_nn
    #
    # loss = loss_hinge + loss_nn



    loss = ratio_node / 10 * tf.maximum(0.0, -train_labels_node * (ret[0] - ret[1] + ret[2] - ret[3]))\
           + 5e-4 * reg_hinge + 5e-4 * reg_nn

    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.01,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        TRAIN_SIZE / 20,  # Decay step.
        0.995,  # Decay rate.
        staircase=True)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=batch)

    test_pred = model(index_uh_node, 'test')
    saver = tf.train.Saver()
    start_time = time.time()
    if mode == 'train':
        with tf.Session() as training_sess:
            tf.initialize_all_variables().run()
            saver.save(training_sess, SAVED_MODEL_PATH) #test path
            print 'Start training...!!!'
            BATCH_NUM = TRAIN_SIZE / BATCH_SIZE
            MAX_ITERATION = int(NUM_EPOCH * BATCH_NUM)
            loss_epoch = 0
            arr = np.zeros([100, 2])
            count = 0
            tot_acc = 0
            tot_count = 0
            for step in xrange(MAX_ITERATION):
                offset = (step * BATCH_SIZE) % TRAIN_SIZE
                batch_data = train_data[offset:(offset + BATCH_SIZE), :]
                batch_labels = np.expand_dims(train_labels[offset / 10], 0)
                batch_index_uh = index_uh[offset:(offset + BATCH_SIZE), :]
                if batch_index_uh[np.where(batch_index_uh[:, 1] == 0)].size != 0:
                    count += 1
                    continue
                batch_index_items = index_items[offset: (offset + BATCH_SIZE), :]
                feed_dict = {train_data_node: batch_data,
                             train_labels_node: batch_labels,
                             index_uh_node: batch_index_uh,
                             ratio_node: np.expand_dims(ratio[offset / 10], 0),
                             index_item_node: batch_index_items}
                _, l, lr, r = training_sess.run(
                    [optimizer, loss, learning_rate, ret],
                    feed_dict=feed_dict)
                loss_epoch += l
                for i in range(10):
                    if batch_index_uh[i, 0] != 0:
                        b_user_hero[batch_index_uh[i, 0], batch_index_uh[i, 1]] = r[4][i]
                # _, _, l_hinge, l_nn, lr, r = training_sess.run(
                #     [optimizer_hinge, optimizer_nn, loss_hinge, loss_nn, learning_rate, ret],
                #     feed_dict=feed_dict)
                arr[count, 1] = batch_labels
                if r[0] - r[1] + r[2] - r[3] >= 0:
                    arr[count, 0] = 1
                else:
                    arr[count, 0] = -1

                if count == 99:
                    loss_epoch /= 100
                    if loss_curve == None:
                        loss_curve = np.array([[step, loss_epoch]])
                    else:
                        loss_curve = np.concatenate((loss_curve, np.array([[step, loss_epoch]])))
                    accuracy = acc(arr[:, 1] > 0, arr[:, 0] > 0)
                    tot_acc += accuracy
                    tot_count += 1
                    print 'step: %d, learning rate: %.8f, loss: %.8f, acc: %.8f, avg: %.8f' \
                          % (step, lr, loss_epoch, accuracy, tot_acc / tot_count)

                    # print 'step: %d, learning rate: %.8f, loss_hinge: %.8f, loss_nn: %.8f, acc: %.8f' \
                    #       % (step, lr, l_hinge, l_nn, acc(arr[:, 1] > 0, arr[:, 0] > 0))
                    count = -1
                    arr = np.zeros([100, 2])
                    loss_epoch = 0

                # l, predictions = training_sess.run([loss, train_prediction], feed_dict=feed_dict)
                if offset == TRAIN_SIZE - BATCH_SIZE:
                    tot_acc = 0
                    tot_count = 0
                    elapsed_time = time.time() - start_time
                    print 'time elapsed: %.8f sec' % elapsed_time
                    # print 'loss: %0.8f' % \
                    #       (loss_epoch)
                    if step == MAX_ITERATION - 1:
                        elapsed_time = time.time() - start_time
                        print 'time elapsed: %.8f sec' % elapsed_time
                        saver.save(training_sess, SAVED_MODEL_PATH)
                        print 'model saved'
                        return
                    b_user_base_np = b_user_base.eval()
                    b_user_base_np[0, :] = np.mean(b_user_base_np[1:, :])
                    b_hero_np = b_hero.eval()
                    b_hero_np[0, :] = np.mean(b_hero_np[1:, :])
                    np.save('.../saved_models/Dec6/user_hero', b_user_hero)
                    np.save('.../saved_models/Dec6/user_base', b_user_base_np)
                    np.save('.../saved_models/Dec6/hero', b_hero_np)
                    np.save('.../saved_models/Dec6/loss', loss_curve)
                    saver.save(training_sess, SAVED_MODEL_PATH)
                count += 1
    elif mode == 'test':
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            saver.restore(sess, SAVED_MODEL_PATH_TEST)
            print 'Start testing...!!!'
            test_data = np.load('/.../data/dota2_576/test_players.npy')
            test_labels = np.load('.../data/dota2_576/test_labels.npy')
            # b_uh = np.load('.../saved_models/Dec6/user_hero.npy')
            # b_ub = np.load('.../saved_models/Dec6/user_base.npy')
            # b_h = np.load('.../saved_models/Dec6/hero.npy')
            test_size = 5000
            batch_size = 10
            batch_num = test_size / batch_size
            arr = np.zeros([batch_num, 1])
            batch_labels = np.expand_dims(test_labels[:batch_num], 1)
            for step in xrange(batch_num):
                offset = (step * batch_size) % test_size
                batch_data = test_data[offset : (offset + batch_size), :]
                batch_data = np.int32(batch_data)
                b_uh = np.load('.../saved_models/Dec6/user_hero.npy')
                b_ub = np.load('.../saved_models/Dec6/user_base.npy')
                b_h = np.load('.../saved_models/Dec6/hero.npy')
                feed_dict = {index_uh_node: batch_data}
                b1, b2 = 0, 0
                for i in range(10):
                    if i < 5:
                        b1 += b_ub[batch_data[i, 0]]
                        b1 += b_h[batch_data[i, 1]]
                        if batch_data[i, 0] == 0 or b_uh[batch_data[i, 0], batch_data[i, 1]] == 0:
                            nz = np.count_nonzero(b_uh[:, batch_data[i, 1]])
                            if nz != 0:
                                b1 += np.sum(b_uh[:, batch_data[i, 1]]) / nz
                            else:
                                b1 += np.sum(b_uh) / np.count_nonzero(b_uh)
                        else:
                            b1 += b_uh[batch_data[i, 0], batch_data[i, 1]]
                    else:
                        b2 += b_ub[batch_data[i, 0]]
                        b2 += b_h[batch_data[i, 1]]
                        if batch_data[i, 0] == 0 or b_uh[batch_data[i, 0], batch_data[i, 1]] == 0:
                            nz = np.count_nonzero(b_uh[:, batch_data[i, 1]])
                            if nz != 0:
                                b2 += np.sum(b_uh[:, batch_data[i, 1]]) / nz
                            else:
                                b2 += np.sum(b_uh) / np.count_nonzero(b_uh)
                        else:
                            b2 += b_uh[batch_data[i, 0], batch_data[i, 1]]
                ret = sess.run(test_pred, feed_dict=feed_dict)
                if b1 - b2 + ret[0] - ret[1] >= 0:
                    pred = 1
                else:
                    pred = -1
                arr[step, :] = pred
            print acc(arr > 0, batch_labels > 0)


if __name__ == '__main__':
    main('train')
