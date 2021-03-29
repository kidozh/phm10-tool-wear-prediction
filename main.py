import keras.backend as K
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from model.yolo import yolo_body
from data import Data

if __name__ == "__main__":
    # input shape
    data = Data()
    x1, y1 = data.get_train_data_by_path(1)
    x4, y4 = data.get_train_data_by_path(4)
    x6, y6 = data.get_train_data_by_path(6)
    print(x1.shape, y1.shape)
    input_shape = (2048, 7)
    signal_input = Input(shape=(2048, 7))
    model = yolo_body(signal_input)

    model.fit(x1, y1, batch_size=16, epochs=100,validation_data=(x4,y4))
    model.save_weights("MODEL.weights")
