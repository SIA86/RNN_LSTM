from one_step_prediction import *

PATH = f'data{os.sep}control_data'
PATH_LIST = os.listdir(PATH)

KERAS_MODEL_NAME = f'models{os.sep}model(#0036)'


def main():
    print('Loading model...')    
    model = tf.keras.models.load_model(KERAS_MODEL_NAME)  
    print('Model loaded!')
    result = ''
    for num, file_name in enumerate(PATH_LIST):

        control_data = ProcessData(os.path.join(PATH, file_name), accuracy=ACC)

        print(control_data)
        #control_data.plot()
        forecasting = ForecastingWindowGenerator(
                INP_SIZE, LABEL_SIZE, SHIFT, WARM_UP,
                forecasting_data=control_data.data,
                label_columns=LABEL_NAMES, n_algorythm=NORM)
        print(forecasting)
        
        print(f'Get prediction for {num} control data set')
        prediction = forecasting.get_prediction(model)
        pred_name = os.path.join(f'data{os.sep}predictions', 'pred_model#0036_' + file_name)
        prediction.to_csv(pred_name)
        analyzer = Analyzer(pred_name, SHIFT)
        result = result.join(f'\n{num}. {file_name} Matching ratio: {int(analyzer.data.value_counts("Match_PC", normalize=True)[1]*100)}%')


    print(result)

if __name__ == "__main__":
    main()