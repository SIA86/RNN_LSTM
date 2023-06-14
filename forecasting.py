from one_step_prediction import *

#PATH_CONTR = f'data{os.sep}feed{os.sep}control_part.csv'
PATH_CONTR = f'data{os.sep}SPFB.Si_220511_230607_1d.csv'
#PATH_CONTR = f'data{os.sep}feed{os.sep}data_35000-40000.csv'


KERAS_MODEL_NAME = f'models{os.sep}model(#0036)'
#CONT_PREDICTION_NAME = f'data{os.sep}predictions{os.sep}#0035_contr_part_01.csv'
CONT_PREDICTION_NAME = f'data{os.sep}predictions{os.sep}#0036_SPFB.Si_220511_230607_1d.csv'
#CONT_PREDICTION_NAME = f'data{os.sep}predictions{os.sep}#0035_data_35000-40000+.csv'



def main():
    control_data = ProcessData(PATH_CONTR, norm_algorythm=NORM, accuracy=ACC)
    control_data.add_local_extrema()

    print(control_data)
    control_data.plot()
    print(control_data.data['Close'].describe())

    control_df = control_data.normalization(control_data.data)
    print(control_df.tail(10))

    """ model = create_model()
    model.load_weights(KERAS_MODEL_NAME + f'{os.sep}cp+.ckpt')   """  
    print('Loading model...')    
    model = tf.keras.models.load_model(KERAS_MODEL_NAME)  
    print('Model loaded!') 
            
    forecast = forecasting(model, control_df)
    df = control_df.copy() #copy data to get datetime index
    if control_df.shape[0] > 3:
        df = df.iloc[:,:4] #drop all except "OHLC"
    if NORM in ['std']:
        forecast = forecast * control_data.std['Close'] + control_data.mean['Close']
        for col in df.columns.to_list():             
            df[col] = df[col] *  control_data.std[col] + control_data.mean[col]
    elif NORM in ['minmax']:
        forecast = forecast*(control_data.max.loc['Close'] - control_data.min.loc['Close']) + control_data.min.loc['Close']
        for col in df.columns.to_list():
            df[col] = df[col]*(control_data.max[col] - control_data.min[col]) + control_data.min[col]
    df['Predicted_close'] = forecast #paste forecast array to dataframe
    df.to_csv(CONT_PREDICTION_NAME)

    analyzer = Analyzer(CONT_PREDICTION_NAME)
    analyzer.filtration()
    print(analyzer)

    print('plotting...')
    apdict = mpf.make_addplot((df['Predicted_close']))
    mpf.plot(df.iloc[:,:-1],type='candle', volume=False, addplot=apdict)
    mpf.show()


if __name__ == "__main__":
    main()