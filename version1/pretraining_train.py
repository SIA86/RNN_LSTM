from one_step_prediction import *

PATH = f'data{os.sep}feed{os.sep}data_25000-30000.csv'
KERAS_MODEL_NAME = f'models{os.sep}model(#0035)'



def main():
    data = ProcessData(PATH, norm_algorythm=NORM, accuracy=ACC, )  #get data from given csv file
    data.add_local_extrema()

    print(data)

    data.plot()
    print(data.data['Close'].describe())

    train_df, _, test_df = map(data.normalization, Utils.split_data(data.data, split=0.99))

    print(train_df.head(10))

    dataset = WindowGenerator(INP_SIZE, LABEL_SIZE, SHIFT, train_df=train_df, val_df=_, test_df=test_df, label_columns=LABEL_NAMES)
    train_ds, test_ds = dataset.train, dataset.test
    
    
    print(f'Data set shape - {list(train_ds.as_numpy_iterator())[0][0].shape}')
    print(f'Lable set shape - {list(train_ds.as_numpy_iterator())[0][1].shape}')

    checkpoint_path = f"models{os.sep}temporary{os.sep}cp.ckpt"
    
    print('Loading trained model')
    
    model = tf.keras.models.load_model(KERAS_MODEL_NAME)
    print(f'Total variables number: {len(model.trainable_variables)}')
    for layers in model.layers[:int(len(model.layers)*0.7)]:
        layers.trainable = False
    print(f'Trainable variables number: {len(model.trainable_variables)}') 
    
    model.compile(loss=LOSS,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=L_RATE/10),
                  metrics=METR)
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        save_weights_only=True,
        monitor='mean_absolute_error',
        mode='min',
        verbose=1
    )
                
    model.fit(train_ds, epochs=20, callbacks=[model_checkpoint_callback]) #fit model
    n_model = create_model()
    n_model.load_weights(checkpoint_path)
    n_model.evaluate(test_ds, batch_size=32)
    n_model.save(KERAS_MODEL_NAME +'_01')

    print(f'Fine fitting ended. Model saved as {KERAS_MODEL_NAME +"_01"}')       
 
if __name__ == "__main__":
    main()
