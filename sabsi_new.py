import datetime as dt
import json
import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import tensorflow as tf
import tensorflow.compat.v1 as tf
from sklearn.preprocessing import MinMaxScaler

# ====================== Loading Data from Alpha Vantage ==================================

api_key = 'GRPY1IDMS1DPDQAL'

# stock market prices of your choice
ticker = "MSI"

# Save data to a csv file
file_to_save = 'stock_market_data-%s.csv' % ticker

# Download the data if it is not available in the file
if not os.path.exists(file_to_save):
    # stock market data from the last 20 years in JSON
    url_string = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={api_key}"
    with urllib.request.urlopen(url_string) as url:
        data = json.loads(url.read().decode())
        # extract stock market data
        data = data['Time Series (Daily)']
        df = pd.DataFrame(columns=['Date', 'Low', 'High', 'Close', 'Open'])
        for k, v in data.items():
            date = dt.datetime.strptime(k, '%Y-%m-%d')
            data_row = [date.date(), float(v['3. low']), float(v['2. high']),
                        float(v['4. close']), float(v['1. open'])]
            df.loc[-1, :] = data_row
            df.index = df.index + 1
    print(f'Data saved to : {file_to_save}')
    df.to_csv(file_to_save)

else:
    print('File already exists. Loading data from CSV')
    df = pd.read_csv(file_to_save)

df = df.sort_values('Date')

plt.figure(figsize=(18, 9))
plt.plot(range(df.shape[0]), (df['Low'] + df['High']) / 2.0)
plt.xticks(range(0, df.shape[0], 500), df['Date'].loc[::500], rotation=45)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Mid Price', fontsize=18)
plt.show()

# First calculate the mid prices from the highest and lowest
high_prices = df.loc[:, 'High'].to_numpy()
low_prices = df.loc[:, 'Low'].to_numpy()
mid_prices = (high_prices + low_prices) / 2.0

train_size = int(len(mid_prices) * 0.9)
all_size = len(mid_prices)
train_data = mid_prices[:train_size]
test_data = mid_prices[train_size:]

# Scale the data to be between -1 and 1
# We normalize both test and train data with respect to training data
scaler = MinMaxScaler()
train_data = train_data.reshape(-1, 1)
test_data = test_data.reshape(-1, 1)

# smoothing the data after training the scaler with train data
smoothing_window_size = 300
for di in range(0, train_size - smoothing_window_size, smoothing_window_size):
    scaler.fit(train_data[di:di + smoothing_window_size, :])
    train_data[di:di + smoothing_window_size, :] = scaler.transform(train_data[di:di + smoothing_window_size, :])

scaler.fit(train_data[di + smoothing_window_size:, :])
train_data[di + smoothing_window_size:, :] = scaler.transform(train_data[di + smoothing_window_size:, :])
test_data = scaler.transform(test_data)

# Reshape the data
train_data = train_data.reshape(-1)
test_data = test_data.reshape(-1)

# Applying exponential moving average smoothing
EMA = 0.0
gamma = 0.1
for ti in range(train_size):
    EMA = gamma * train_data[ti] + (1 - gamma) * EMA
    train_data[ti] = EMA

all_data_norm = np.concatenate([train_data, test_data], axis=0)

window_size = 100
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size, N):

    if pred_idx >= N:
        date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
    else:
        date = df.loc[pred_idx, 'Date']

    std_avg_predictions.append(np.mean(train_data[pred_idx - window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1] - train_data[pred_idx]) ** 2)
    std_avg_x.append(date)

print('MSE error for standard averaging: %.5f' % (0.5 * np.mean(mse_errors)))

plt.figure(figsize=(18, 9))
plt.plot(range(df.shape[0]), all_data_norm, color='b', label='Real')
plt.plot(range(window_size, N), std_avg_predictions, color='orange', label='Averaged')
plt.xticks(range(0, df.shape[0], 50), df['Date'].loc[::50], rotation=45)
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.show()


class DataGeneratorSeq(object):
    def __init__(self, prices, batch_size, num_unroll):
        self._prices = prices
        self._prices_length = len(self._prices) - num_unroll
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._prices_length // self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):
        batch_data = np.zeros((self._batch_size), dtype=np.float32)
        batch_labels = np.zeros((self._batch_size), dtype=np.float32)

        for b in range(self._batch_size):
            if self._cursor[b] + 1 >= self._prices_length:
                # self._cursor[b] = b * self._segments
                self._cursor[b] = np.random.randint(0, (b + 1) * self._segments)

            batch_data[b] = self._prices[self._cursor[b]]
            batch_labels[b] = self._prices[self._cursor[b] + np.random.randint(0, 5)]

            self._cursor[b] = (self._cursor[b] + 1) % self._prices_length

        return batch_data, batch_labels

    def unroll_batches(self):
        unroll_data, unroll_labels = [], []
        init_data, init_label = None, None
        for ui in range(self._num_unroll):
            data, labels = self.next_batch()

            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

    def reset_indices(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(0, min((b + 1) * self._segments, self._prices_length - 1))


num_unrollings = 50  # Number of time steps to look into the future.
batch_size = 500  # Number of samples in a batch
num_nodes = [200, 200, 150]  # Number of hidden nodes in each layer of the deep LSTM stack we're using
n_layers = len(num_nodes)  # number of layers
dropout = 0.2  # dropout amount

tf.compat.v1.reset_default_graph()  # This is important in case you run this multiple times

# Input data.
train_inputs, train_outputs = [], []

tf.compat.v1.disable_eager_execution()
for ui in range(num_unrollings):
    train_inputs.append(tf.compat.v1.placeholder(tf.float32, shape=[batch_size, 1], name='train_inputs_%d' % ui))
    train_outputs.append(tf.compat.v1.placeholder(tf.float32, shape=[batch_size, 1], name='train_outputs_%d' % ui))

lstm_cells = [
    tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=num_nodes[li],
                                      state_is_tuple=True,
                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                  mode="fan_avg",
                                                                                                  distribution="uniform")
                                      )
    for li in range(n_layers)]

drop_lstm_cells = [tf.nn.rnn_cell.DropoutWrapper(
    lstm, input_keep_prob=1.0, output_keep_prob=1.0 - dropout, state_keep_prob=1.0 - dropout
) for lstm in lstm_cells]
drop_multi_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(drop_lstm_cells)
multi_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(lstm_cells)

w = tf.compat.v1.get_variable('w', shape=[num_nodes[-1], 1],
                              initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg",
                                                                                          distribution="uniform"))
b = tf.compat.v1.get_variable('b', initializer=tf.random.uniform([1], -0.1, 0.1))

# cell state and hidden state
c, h = [], []
initial_state = []
for li in range(n_layers):
    c.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
    h.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
    initial_state.append(tf.nn.rnn_cell.LSTMStateTuple(c[li], h[li]))

all_inputs = tf.concat([tf.expand_dims(t, 0) for t in train_inputs], axis=0)

# all_outputs is [seq_length, batch_size, num_nodes]
all_lstm_outputs, state = tf.compat.v1.nn.dynamic_rnn(
    drop_multi_cell, all_inputs, initial_state=tuple(initial_state),
    time_major=True, dtype=tf.float32)

all_lstm_outputs = tf.reshape(all_lstm_outputs, [batch_size * num_unrollings, num_nodes[-1]])

all_outputs = tf.compat.v1.nn.xw_plus_b(all_lstm_outputs, w, b)

split_outputs = tf.split(all_outputs, num_unrollings, axis=0)

print('Defining training Loss')
loss = 0.0
with tf.control_dependencies([tf.compat.v1.assign(c[li], state[li][0]) for li in range(n_layers)] +
                             [tf.compat.v1.assign(h[li], state[li][1]) for li in range(n_layers)]):
    for ui in range(num_unrollings):
        loss += tf.reduce_mean(input_tensor=0.5 * (split_outputs[ui] - train_outputs[ui]) ** 2)

print('Learning rate decay operations')
global_step = tf.Variable(0, trainable=False)
inc_gstep = tf.compat.v1.assign(global_step, global_step + 1)
tf_learning_rate = tf.compat.v1.placeholder(shape=None, dtype=tf.float32)
tf_min_learning_rate = tf.compat.v1.placeholder(shape=None, dtype=tf.float32)

learning_rate = tf.maximum(
    tf.compat.v1.train.exponential_decay(tf_learning_rate, global_step, decay_steps=1, decay_rate=0.5, staircase=True),
    tf_min_learning_rate)

# Optimizer.
print('TF Optimization operations')
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
gradients, v = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
optimizer = optimizer.apply_gradients(
    zip(gradients, v))

print('\tAll done')

print('Defining prediction related TF functions')

sample_inputs = tf.compat.v1.placeholder(tf.float32, shape=[1, 1])

# Maintaining LSTM state for prediction stage
sample_c, sample_h, initial_sample_state = [], [], []
for li in range(n_layers):
    sample_c.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
    sample_h.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
    initial_sample_state.append(tf.nn.rnn_cell.LSTMStateTuple(sample_c[li], sample_h[li]))

reset_sample_states = tf.group(
    *[tf.compat.v1.assign(sample_c[li], tf.zeros([1, num_nodes[li]])) for li in range(n_layers)],
    *[tf.compat.v1.assign(sample_h[li], tf.zeros([1, num_nodes[li]])) for li in range(n_layers)])

sample_outputs, sample_state = tf.compat.v1.nn.dynamic_rnn(multi_cell, tf.expand_dims(sample_inputs, 0),
                                                           initial_state=tuple(initial_sample_state),
                                                           time_major=True,
                                                           dtype=tf.float32)

with tf.control_dependencies([tf.compat.v1.assign(sample_c[li], sample_state[li][0]) for li in range(n_layers)] +
                             [tf.compat.v1.assign(sample_h[li], sample_state[li][1]) for li in range(n_layers)]):
    sample_prediction = tf.compat.v1.nn.xw_plus_b(tf.reshape(sample_outputs, [1, -1]), w, b)

print('\tAll done')

epochs = 30
valid_summary = 1  # predictions interval

n_predict_once = 50  # Number of steps to predict for

train_seq_length = train_data.size

train_mse_ot = []  # train losses array
test_mse_ot = []  # test losses
predictions_over_time = []

session = tf.compat.v1.InteractiveSession()

tf.compat.v1.global_variables_initializer().run()

# Used for decaying learning rate
loss_nondecrease_count = 0
loss_nondecrease_threshold = 2  # If the test error hasn't increased in this many steps, decrease learning rate

print('Initialized')
average_loss = 0

# Define data generator
data_gen = DataGeneratorSeq(train_data, batch_size, num_unrollings)

x_axis_seq = []

# points to begin predictions
test_points_seq = np.arange(train_size, all_size - 50, 50).tolist()
mse_lowest_ind = 0
mse_lowest = 1

for ep in range(epochs):
    # Training
    for step in range(train_seq_length // batch_size):
        u_data, u_labels = data_gen.unroll_batches()

        feed_dict = {}
        for ui, (dat, lbl) in enumerate(zip(u_data, u_labels)):
            feed_dict[train_inputs[ui]] = dat.reshape(-1, 1)
            feed_dict[train_outputs[ui]] = lbl.reshape(-1, 1)

        feed_dict.update({tf_learning_rate: 0.0001, tf_min_learning_rate: 0.000001})

        _, l = session.run([optimizer, loss], feed_dict=feed_dict)

        average_loss += l

    # Validation
    if (ep + 1) % valid_summary == 0:

        average_loss = average_loss / (valid_summary * (train_seq_length // batch_size))

        # The average loss
        if (ep + 1) % valid_summary == 0:
            print('Average loss at step %d: %f' % (ep + 1, average_loss))

        train_mse_ot.append(average_loss)

        average_loss = 0  # reset loss

        predictions_seq = []

        mse_test_loss_seq = []

        # Updating state and making predicitons
        for w_i in test_points_seq:
            mse_test_loss = 0.0
            our_predictions = []

            if (ep + 1) - valid_summary == 0:
                # Only calculate x_axis values in the first validation epoch
                x_axis = []

            # Feed in the recent past behavior of stock prices
            # to make predictions from that point onwards
            for tr_i in range(w_i - num_unrollings + 1, w_i - 1):
                current_price = all_data_norm[tr_i]
                feed_dict[sample_inputs] = np.array(current_price).reshape(1, 1)
                _ = session.run(sample_prediction, feed_dict=feed_dict)

            feed_dict = {}

            current_price = all_data_norm[w_i - 1]

            feed_dict[sample_inputs] = np.array(current_price).reshape(1, 1)

            # Make predictions for this many steps
            # Each prediction uses previous prediciton as it's current input
            for pred_i in range(n_predict_once):

                pred = session.run(sample_prediction, feed_dict=feed_dict)

                our_predictions.append(np.asscalar(pred))

                feed_dict[sample_inputs] = np.asarray(pred).reshape(-1, 1)

                if (ep + 1) - valid_summary == 0:
                    # Only calculate x_axis values in the first validation epoch
                    x_axis.append(w_i + pred_i)

                mse_test_loss += 0.5 * (pred - all_data_norm[w_i + pred_i]) ** 2

            session.run(reset_sample_states)

            predictions_seq.append(np.array(our_predictions))

            mse_test_loss /= n_predict_once
            mse_test_loss_seq.append(mse_test_loss)

            if (ep + 1) - valid_summary == 0:
                x_axis_seq.append(x_axis)

        current_test_mse = np.mean(mse_test_loss_seq)

        # Learning rate decay logic
        if len(test_mse_ot) > 0 and current_test_mse > min(test_mse_ot):
            loss_nondecrease_count += 1
        else:
            loss_nondecrease_count = 0

        if loss_nondecrease_count > loss_nondecrease_threshold:
            session.run(inc_gstep)
            loss_nondecrease_count = 0
            print('\tDecreasing learning rate by 0.5')

        if current_test_mse < mse_lowest:
            mse_lowest = current_test_mse
            mse_lowest_ind = ep
        test_mse_ot.append(current_test_mse)
        print('\tTest MSE: %.5f' % np.mean(mse_test_loss_seq))
        predictions_over_time.append(predictions_seq)
        print('\tFinished Predictions')

best_prediction_epoch = mse_lowest_ind

plt.figure(figsize=(18, 18))
plt.subplot(2, 1, 1)
plt.plot(range(df.shape[0]), all_data_norm, color='b')

# Plotting how the predictions change over time
# Plot older predictions as more transparent
start_alpha = 0.25
alpha = np.arange(start_alpha, 1.1, (1.0 - start_alpha) / len(predictions_over_time[::3]))
for p_i, p in enumerate(predictions_over_time[::3]):
    for xval, yval in zip(x_axis_seq, p):
        plt.plot(xval, yval, color='r', alpha=alpha[p_i])

plt.title('Evolution of Test Predictions Over Time', fontsize=18)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Mid Price', fontsize=18)
plt.xlim(train_size, all_size)

plt.subplot(2, 1, 2)

# Predicting the best test prediction
plt.plot(range(df.shape[0]), all_data_norm, color='b')
for xval, yval in zip(x_axis_seq, predictions_over_time[best_prediction_epoch]):
    plt.plot(xval, yval, color='r')

plt.title('Best Test Predictions Over Time', fontsize=18)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Mid Price', fontsize=18)
plt.xlim(train_size, all_size)
plt.show()

input()
