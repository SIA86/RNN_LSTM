from data import *

#data options
ACC = 5
NORM = 'std'
SPLIT = 0.8 #size of training set
VAL = 0.2

#dataset options
INP_SIZE = 10 #length of candles sequence to analize
LABEL_SIZE = 1 #length of target 
SHIFT = 1 # distance between analyzing data and target
WARM_UP = 100
LABEL_NAMES = ['Close']

#rnn options
NEURONS = 8
L_RATE = 0.0001
EPOCH = 50
LOSS = 'poisson'
METR = 'mean_absolute_error'

#path and filenames
PATH = f'data\SPFB.Si_220511_230607_5min.csv'
KERAS_MODEL_NAME = f'models{os.sep}model(#0009)'
PREDICTION_NAME = f'data{os.sep}predictions{os.sep}#0009_prediction.csv'

def create_uncompiled_model() -> tf.keras.models.Sequential:
    # define a sequential model
    model = tf.keras.models.Sequential([ 
        #tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
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
                  metrics=METR       
                  )  
    return model 
   

def plot_loss(model: tf.keras.models.Sequential) -> None:
    acc = model.history[METR]
    val_acc = model.history['val_' + METR]
   
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1,2,1)
    
    plt.plot(acc, label=f'Training acuuracy')
    plt.plot(val_acc, label=f'Validation Accuracy')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1,2,2)
    
    plt.plot(loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation loss')
    plt.show()


def main():
    print('Loading data...')
    data = ProcessData(PATH, accuracy=ACC, )  #get data from given csv file
    print(data)
    #data.plot()
   
    train_df, val_df, test_df = Utils.split_data(data.data, 0.8, 0.2)

    ds = TrainingWindowGenerator(
        INP_SIZE, LABEL_SIZE, SHIFT, WARM_UP,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=LABEL_NAMES, n_algorythm=NORM)
    print(ds)

    #processing data
    print('Working out datasets...')
    train_ds = ds.train
    print('...train set ready...')
    test_ds = ds.test
    print('...test set ready...')
    val_ds = ds.val
    print('...val set ready')
    
    print(f'Data set shape - {list(train_ds.as_numpy_iterator())[0][0].shape}')
    print(f'Lable set shape - {list(train_ds.as_numpy_iterator())[0][1].shape}')      

    checkpoint_path = KERAS_MODEL_NAME +f"{os.sep}cp.ckpt"
    #checkpoint_path = KERAS_MODEL_NAME
    #checkpoint_dir = os.path.dirname(checkpoint_path)
    #latest = tf.train.latest_checkpoint(checkpoint_dir) #if there are several weights saved
    try:
        print('Trying to load predictions')
        prediction = pd.read_csv(PREDICTION_NAME, index_col=0, parse_dates=True )
    except Exception:
        print('No predictions found')
        try:
            print('Loading trained model')
            n_model = create_model()
            n_model.load_weights(checkpoint_path)
            n_model.evaluate(test_ds)
            #n_model.save('model#35')
            print(f'Weights loaded successfully')          
        except Exception:
            print('No saved model')
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,
                save_best_only=True,
                monitor='val_mean_absolute_error',
                mode='min',
                verbose=1)           
            
            model = create_model() #create model
            history = model.fit(train_ds, epochs=EPOCH, shuffle=True, validation_data = val_ds, callbacks=[model_checkpoint_callback]) #fit model
            #model.summary()
            plot_loss(history)
            n_model = create_model()
            n_model.load_weights(checkpoint_path)
            #n_model = tf.keras.models.load_model(KERAS_MODEL_NAME)
            print('Evaluate model test part')
            n_model.evaluate(test_ds)

        forecasting = ForecastingWindowGenerator(
            INP_SIZE, LABEL_SIZE, SHIFT, WARM_UP,
            forecasting_data=test_df,
            label_columns=LABEL_NAMES, n_algorythm=NORM)
        
        prediction = forecasting.get_prediction(n_model)
        prediction.to_csv(PREDICTION_NAME) 

    analyzer = Analyzer(PREDICTION_NAME, SHIFT)
    analyzer.filtration()
    print(analyzer)

    print('plotting...')
    apdict = mpf.make_addplot((prediction['Pred_Close']))
    mpf.plot(prediction.iloc[:,:-1], type='candle', volume=False, addplot=apdict)
    mpf.show()

if __name__ == "__main__":
    main()

