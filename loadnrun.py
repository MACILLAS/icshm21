import tensorflow as tf

if __name__ == "__main__":

    model = tf.keras.models.load_model('my_model.h5')
    print(model.summary())