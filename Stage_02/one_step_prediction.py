from data import *

#data options
ACC = 6
NORM = 'std'
SPLIT = 0.8 #size of training set
VAL = 0.2

#dataset options
INP_SIZE = 5 #length of candles sequence to analize
LABEL_SIZE = 1
SHIFT = 1
LABEL_NAMES = ['Close']

#rnn options
NEURONS = 1
L_RATE = 0.00003
LOSS = 'mean_absolute_error'
METR = 'mean_absolute_error'

#path and filenames
PATH = f'data\SPFB.Si_220511_230607_5min.csv'
KERAS_MODEL_NAME = f'models{os.sep}model(#0001)'
PREDICTION_NAME = f'data{os.sep}predictions{os.sep}#0001_prediction.csv'

def create_uncompiled_model() -> tf.keras.models.Sequential:
    # define a sequential model
    model = tf.keras.models.Sequential([ 
        #tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1, input_shape=[None])),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(NEURONS*16, return_sequences=True)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(NEURONS*8, return_sequences=True)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(NEURONS*4, return_sequences=True)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(NEURONS*2, return_sequences=True)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(NEURONS)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(LABEL_SIZE*len(LABEL_NAMES))])  

    return model


def create_model() -> tf.keras.models.Sequential:    
    tf.random.set_seed(51)
    model = create_uncompiled_model()
    model.compile(loss=LOSS,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=L_RATE),
                  metrics=METR)  
    return model 


def forecasting(model: tf.keras.models.Sequential, data : pd.DataFrame) -> pd.DataFrame:
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.window(INP_SIZE, shift=SHIFT, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(INP_SIZE))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds) #get prediction from given data

    empty_rows = np.array([[np.NAN] for _ in range(INP_SIZE-1)])
    forecast = np.concatenate((empty_rows, forecast)) #massage forecast data in order to have same length
    
    return forecast
    

def plot_loss(model: tf.keras.models.Sequential) -> None:
    acc = model.history[METR]
    val_acc = model.history['val_' + METR]

    loss = model.history['loss']
    val_loss = model.history['val_loss']
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1,2,1)
    
    plt.plot(acc, label='Training acuuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1,2,2)
    
    plt.plot(loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation loss')
    plt.show()


def main():
    data = ProcessData(PATH, accuracy=ACC, )  #get data from given csv file
    print(data)
    data.plot()
   
    train_df, val_df, test_df = Utils.split_data(data.data, 0.8, 0.2)

    ds = WindowGenerator(5, 1, 1, 5, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=['Close'], n_algorythm="std")
    x_train, y_train = ds.train
    x_test, y_test = ds.test
    #x_val, y_val  = ds.val 
    val_ds = (x_val, y_val) = ds.val
    
    
    print(f'Inputs type: {type(x_train).__name__} of {type(x_train[0]).__name__}, shape: {x_train[0].shape}')
    print(f'Labels type: {type(y_train).__name__} of {type(y_train[0]).__name__}, shape: {y_train[0].shape}')      

    #checkpoint_path = KERAS_MODEL_NAME +f"{os.sep}cp.ckpt"
    checkpoint_path = KERAS_MODEL_NAME
    #checkpoint_dir = os.path.dirname(checkpoint_path)
    #latest = tf.train.latest_checkpoint(checkpoint_dir) #if there are several weights saved
    try:
        print('Trying to load predictions')
        df = pd.read_csv(PREDICTION_NAME, index_col=0, parse_dates=True )
    except Exception:
        print('No predictions found')
         
        try:
            print('Loading trained model')
            n_model = create_model()
            n_model.load_weights(checkpoint_path)
            n_model.evaluate(x_test, y_test, batch_size=50)
            n_model.save('model#35')
            print(f'Weights loaded successfully')          
        except Exception:
            print('No saved model')
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True
                save_best_only=True,
                monitor='val_mean_absolute_error',
                mode='min',
                verbose=1
            )
            
            model = create_model() #create model
            history = model.fit(x_train, y_train, epochs=20, validation_data = val_ds, callbacks=[model_checkpoint_callback]) #fit model
            #model.summary()
            plot_loss(history)
            n_model = create_model()
            n_model.load_weights(checkpoint_path)
            #n_model = tf.keras.models.load_model(KERAS_MODEL_NAME)
            print('Evaluate model test part')
            n_model.evaluate(x_test, y_test, batch_size=50)

        forecast = forecasting(n_model, test_df)
        df = test_df.copy() #copy data to get datetime index
        if test_df.shape[0] > 3:
            df = df.iloc[:,:4] #drop all except "OHLC"
        if NORM in ['std']:
            forecast = forecast * data.std['Close'] + data.mean['Close']
            for col in df.columns.to_list():             
                df[col] = df[col] *  data.std[col] + data.mean[col]
        elif NORM in ['minmax']:
            forecast = forecast*(data.max.loc['Close'] - data.min.loc['Close']) + data.min.loc['Close']
            for col in df.columns.to_list():
                df[col] = df[col]*(data.max[col] - data.min[col]) + data.min[col]
        df['Predicted_close'] = forecast #paste forecast array to dataframe
        df.to_csv(PREDICTION_NAME)
        


    analyzer = Analyzer(PREDICTION_NAME)
    analyzer.filtration()
    print(analyzer)

    print('plotting...')
    apdict = mpf.make_addplot((df['Predicted_close']))
    mpf.plot(df.iloc[:,:-1],type='candle', volume=False, addplot=apdict)
    mpf.show()

if __name__ == "__main__":
    main()

