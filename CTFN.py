import os
import pickle
import time
from datetime import datetime
import sys
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.layers import Add, Lambda
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPool1D
from keras.layers import concatenate
from keras.models import Sequential
import tensorflow as tf
import utils
from datetime import datetime
import time
from sklearn.preprocessing import LabelBinarizer
from keras.layers import Input,Attention, Embedding, MultiHeadAttention, Dense, Flatten, BatchNormalization, Activation, Dropout, LayerNormalization,GlobalAveragePooling1D
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.metrics import confusion_matrix,roc_curve,auc
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
import itertools
from scipy.optimize import brentq
from scipy.interpolate import interp1d
# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 100
BS = 64
LR = 0.00001
decay = LR / EPOCHS
database = "time"
model_name = "combined.h5"


def splits(yy, xx, sig_dims):
    # Binarize the labels
    lbb = LabelBinarizer()
    yy = lbb.fit_transform(yy)

    # Train 70%, test is 30%
    xx_train, xx_test, yy_train, yy_test = train_test_split(
        xx, yy, test_size=0.4, shuffle=True, random_state=42)

    xx_valid, xx_test, yy_valid, yy_test = train_test_split(
        xx_test, yy_test, test_size=0.5, shuffle=True, random_state=42)

    print("X train shape:", xx_train.shape)
    xx_train = xx_train.reshape(xx_train.shape[0], sig_dims[0], sig_dims[1])
    xx_valid = xx_valid.reshape(xx_valid.shape[0], sig_dims[0], sig_dims[1])
    xx_test = xx_test.reshape(xx_test.shape[0], sig_dims[0], sig_dims[1])
    print("X train shape:", xx_train.shape)
    print("X valid shape:", xx_valid.shape)
    print("X test shape:", xx_test.shape, "\n")

    return xx_train, yy_train, xx_valid, yy_valid, xx_test, yy_test, lbb

def plot_history(h):
    acc = h.history['accuracy']
    val_acc = h.history['val_accuracy']

    loss = h.history['loss']
    val_loss = h.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')

    name = "media/plots/transformer_history_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(name, dpi=300, bbox_inches='tight')

    plt.show()

def report(m, x_test, y_test):
    lbb = LabelBinarizer()
    predictions = m.predict(x_test, batch_size=BS, verbose=1)
    y_pred_bool = np.argmax(predictions, axis=1)
    y_pred_bool = lbb.fit_transform(y_pred_bool)
    print(classification_report(y_test, y_pred_bool), "\n")
    return predictions


def plot_predictions(predictions, x_test, y_test, lb):
    up, down = [], []
    for i in predictions:
        pred = max(i)
        if pred >= 0.99:
            up.append(pred)
        else:
            down.append(pred)

    print("Number of predicted Positive Pairs:", len(up))
    print(up, "\n")
    print("Number of predicted Negative Pairs:", len(down))
    print(down, "\n")

    fig = plt.figure(figsize=(64, 54))
    for i, idx in enumerate(np.random.choice(x_test.shape[0], size=225, replace=False)):
        pred_idx = np.argmax(predictions[idx])
        true_idx = np.argmax(y_test[idx])
        prob = max(predictions[pred_idx])
        ax = fig.add_subplot(15, 15, i + 1, xticks=[], yticks=[])
        ax.plot(x_test[idx])
        ax.set_title("T: {} P: {} {:.6f}".format(lb.classes_[true_idx], lb.classes_[pred_idx], prob),
                     color=("green" if pred_idx == true_idx else "red"))

        name = "media/plots/transformer_predictions_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        fig.savefig(name, dpi=300, bbox_inches='tight')
        plt.show()

def positional_encoding(max_sequence_length, embedding_dim):
    position = np.arange(max_sequence_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim))
    pos_enc = np.zeros((max_sequence_length, embedding_dim))
    pos_enc[:, 0::2] = np.sin(position * div_term)
    pos_enc[:, 1::2] = np.cos(position * div_term)
    return pos_enc


def block(m, fs, ks, ps):
    m.add(Conv1D(filters=fs, kernel_size=ks, padding="same"))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPool1D(pool_size=ps, padding='same'))
    return m

def create_cnn_model(sig_dims, output_dim):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=1, padding="same", input_shape=sig_dims))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Blocks
    model = block(model, 32 * 2, 15, 2)
    model = block(model, 32 * 4, 15, 2)
    model = block(model, 32 * 8, 15, 2)
    model = block(model, 32 * 16, 15, 2)
    last = 1 + 2 + 4 + 8 + 16
    model = block(model, 32 * last, 15, 2)

    # Global average pooling layer
    model.add(Flatten())  # Flatten the output
    model.add(Dense(output_dim))  # Output dimension matches Transformer model
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    return model

# Modify train_transformer_with_cnn function
def train_transformer_with_cnn(folder, sig_dims, data,cnn_output_dim,embedding_dim,ffn_dim,num_heads):
    x_train, y_train, x_valid, y_valid, x_test, y_test, lb = data
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Save the label binarizer to disk
    with open(os.path.join(folder, "lb.pickle"), "wb") as f:
        pickle.dump(lb, f)

    # Create CNN model with adjusted output dimension
    # cnn_output_dim = 128  # Adjust this dimension as needed
    cnn_model = create_cnn_model(sig_dims, cnn_output_dim)

    # Define input layer for Transformer
    inputs = Input(shape=sig_dims)

    # Add an embedding layer
    # embedding_dim = 128  # You may adjust this as needed
    # embedding_out = Embedding(input_dim=len(lb.classes_), output_dim=embedding_dim)(inputs)

    # Add positional encoding
    max_sequence_length = x_train.shape[1]  # Assumes sequences are padded to the same length
    pos_encoding = positional_encoding(max_sequence_length, embedding_dim)
    input_with_position = inputs + pos_encoding

    # Add multi-head self-attention layer
    # num_heads = 4  # You may adjust this as needed
    key_dim = embedding_dim // num_heads
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(input_with_position, input_with_position)

    # Add a feed-forward neural network layer
    # ffn_dim = 128  # You may adjust this as needed
    ffn_output = Dense(ffn_dim, activation="relu")(attention_output)

    # Add residual connection and layer normalization
    transformer_output = LayerNormalization(epsilon=1e-6)(input_with_position + ffn_output)

    # Flatten and add fully connected layers
    flattened = Flatten()(transformer_output)
    dense_layer = Dense(cnn_output_dim, activation="relu")(flattened)  # Adjust the dense layer size as needed
    bn_layer = BatchNormalization()(dense_layer)
    activation_layer = Activation('relu')(bn_layer)
    dropout_layer = Dropout(0.25)(activation_layer)

    # Combine CNN and Transformer models
    cnn_output = cnn_model(inputs)  # Use CNN model's output

    # 级联
    # combined_output = tf.concat([dropout_layer, cnn_output], axis=-1)

    # 加权叠加
    alpha = 0.8
    print(alpha)
    weighted_cnn_output = Lambda(lambda x: x * alpha)(cnn_output)
    weighted_transformer_output = Lambda(lambda x: x * (1 - alpha))(dropout_layer)
    combined_output = Add()([weighted_cnn_output, weighted_transformer_output])

    # 注意力机制
    # combined_output = Attention()([cnn_output, dropout_layer])
    
    # Add softmax classifier
    outputs = Dense(len(lb.classes_), activation="softmax")(combined_output)

    # Create the combined model
    combined_model = Model(inputs=inputs, outputs=outputs)

    print(combined_model.summary())

    combined_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    STEPS_PER_EPOCH = len(x_train) // BS
    VAL_STEPS_PER_EPOCH = len(x_valid) // BS

    
    best_model = os.path.join(folder, model_name)
    check_pointer = ModelCheckpoint(filepath=best_model, verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Define the Keras TensorBoard callback.
    log_dir = os.path.join(folder, "logs/fit/", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir)

    # Fit the network
    t = time.time()

    history = combined_model.fit(x_train, y_train, batch_size=BS,
                                  validation_data=(x_valid, y_valid),
                                  steps_per_epoch=STEPS_PER_EPOCH,
                                  validation_steps=VAL_STEPS_PER_EPOCH,
                                  epochs=EPOCHS, verbose=1,
                                  callbacks=[tensorboard_callback, check_pointer, early_stopping])

    print('\nTraining time: ', time.time() - t)

    # Save the model to disk
    combined_model.save(best_model)

    # 进行预测
    y_pred_prob = combined_model.predict(x_valid, batch_size=BS)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_valid = np.argmax(y_valid, axis=1)

    #f1score
    f1 = f1_score(y_valid, y_pred,average='weighted')
    print("F1 Score:", f1)

    return combined_model, history


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for file in self.files:
            file.write(text)

    def flush(self):
        for file in self.files:
            file.flush()




def main():
    # 创建一个txt文件，用于保存完整的输出
    output_file_path = 'output_'+datetime.now().strftime("%Y%m%d-%H%M%S")+'.txt'
    output_file = open(output_file_path, 'w')
    
    # 重定向标准输出到文件和屏幕
    sys.stdout = Tee(sys.stdout, output_file)

    path = 'data/ready/pickles/' + database +'.pickle'
    print(database)
    y, x, people = utils.load_data(path)
    y, x = utils.shuffle(y, x)

    SIG_DIMS = (x.shape[1], 1)
    print("Input Shape:", SIG_DIMS, "\n")

    x_train, y_train, x_valid, y_valid, x_test,y_test, lb = splits(y, x, SIG_DIMS)
    data = x_train, y_train, x_valid, y_valid, x_test,y_test, lb

    model_path = "models/" + database +"/"
    num_heads_list = [4]
    embedding_dim_list = [32]
    # ffn_dim_list = embedding_dim_list  # 与embedding_dim相同
    cnn_output_dim_list = [64]
    
    #创建所有超参数组合的迭代器
    param_combinations = itertools.product(num_heads_list, embedding_dim_list, cnn_output_dim_list)

    # 遍历每个超参数组合
    for num_heads, embedding_dim, cnn_output_dim in param_combinations:
        ffn_dim=embedding_dim
        print(f"Training model with num_heads={num_heads}, embedding_dim={embedding_dim}, ffn_dim={ffn_dim}, cnn_output_dim={cnn_output_dim}")
        model, H = train_transformer_with_cnn(model_path, SIG_DIMS, data,cnn_output_dim,embedding_dim,ffn_dim,num_heads)

    plot_history(H)
    # 加载已保存的模型
    loaded_model = load_model(model_path + model_name)

    # 加载测试数据
    t1= time.time()
    y_pred_prob = loaded_model.predict(x_test,batch_size=BS)
    print(y_pred_prob.shape)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_real= np.argmax(y_test, axis=1)

    segment = len(x_test)//len(lb.classes_)
    sum=0
    for i in range(len(lb.classes_)):
        count=0
        for j in range(segment):
            if(y_real[i*segment+j]==y_pred[i*segment+j]):
                count+=1
        if(count/segment>=0.95):
            print(f"the acc of person {i}: {count/segment*100}%,大于95%,认证成功")
            sum+=1
        else:
            print(f"the acc of person {i}: {count/segment*100}%,小于95%,认证失败")
    print(f"{len(lb.classes_)}名测试者成功认证{sum}人, acc = {sum/len(lb.classes_)*100}%")
    print("time:",time.time()-t1)
    f1 = f1_score(y_real, y_pred, average='weighted')
    print("F1 Score:", f1)
        
    confusion = confusion_matrix(y_real, y_pred, labels=range(len(lb.classes_)))
    # 调整混淆矩阵图的大小以增加间距
    plt.figure(figsize=(20, 20))
            
    sns.heatmap(confusion, annot=False, fmt="d", cmap="Blues", cbar=True, square=True,
                        linewidths=0.5,
                        annot_kws={"size": 10, "weight": "bold"},
                        cbar_kws={"shrink": 0.7, "fraction": 0.05})
    # 清除坐标轴上的刻度标签
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("Predicted", fontsize=24)  # 调整 x 轴标签的字体大小
    plt.ylabel("Actual", fontsize=24)  # 调整 y 轴标签的字体大小
    plt.title(database + " Confusion Matrix",fontsize=48)
    output_directory = os.path.join("media", database)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    plt.savefig(os.path.join(output_directory, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}confusion_matrix.png"))


    # 计算每个类别的 EER
    thresholds = np.linspace(0, 1, 100)
    eer_list = []
    
    for class_label in range(len(lb.classes_)):  # 替换为实际的类别数
        y_class = (y_real == class_label).astype(int)
        y_pred_prob_class = y_pred_prob[:, class_label]
    
        far_values = []
        frr_values = []
    
        for threshold in thresholds:
            y_pred = (y_pred_prob_class >= threshold).astype(int)
            cm = confusion_matrix(y_class, y_pred)
            far = cm[0, 1] / (cm[0, 0] + cm[0, 1] + 1e-10)
            frr = cm[1, 0] / (cm[1, 0] + cm[1, 1] + 1e-10)
            far_values.append(far)
            frr_values.append(frr)
    
        # eer_index = np.argmin(np.abs(np.array(far_values) - np.array(frr_values)))
        # eer_threshold = thresholds[eer_index]
        # eer = 0.5 * (far_values[eer_index] + frr_values[eer_index])
        # eer_list.append(eer)
        eer_threshold = thresholds[np.argmin(np.abs(np.array(far_values) - np.array(frr_values)))]
        eer = 0.5 * (far_values[np.argmin(np.abs(np.array(far_values) - np.array(frr_values)))] +
                        frr_values[np.argmin(np.abs(np.array(far_values) - np.array(frr_values)))])
        eer_list.append(eer)
    
    # 计算平均 EER
    average_eer = np.mean(eer_list)
    print(f"Average Equal Error Rate (EER): {average_eer * 100:.2f}%")
    
    # 绘制 FAR 和 FRR 曲线
    plt.figure()  # 添加这行以创建新的图形
    plt.plot(thresholds, far_values, label='FAR')
    plt.plot(thresholds, frr_values, label='FRR')
    plt.scatter([eer_threshold], [average_eer], c='red', marker='o', label='Average EER Point')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title('FAR and FRR with Average EER Point')
    plt.legend()
    output_directory = os.path.join("media", database)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    plt.savefig(os.path.join(output_directory, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}EER.png"))


    # 恢复标准输出
    sys.stdout = sys.__stdout__
    output_file.close()
    
    print("Output saved to", output_file_path)
