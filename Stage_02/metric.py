import tensorflow as tf
import numpy as np
import pandas as pd


class MatchingDirection(tf.keras.metrics.Metric):
    def __init__(self):
        super().__init__()
        self.match = self.add_weight(name = 'tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """ pred_diff = tf.subtract(y_pred, y_true)
        self.match = pred_diff """


        y_true_np = y_true.numpy().squeeze()
        y_pred_np = y_pred.numpy().squeeze()


        df = pd.DataFrame({'Close' : y_true_np,
                           'Pred_Close' :y_pred_np})
        
        df['Pred_Close_diff'] = df['Pred_Close'] - df['Close'] 
        df['Fact_diff'] = df['Close'].diff().shift(-1)
        df = df.dropna()
        
        df.loc[((df['Pred_Close_diff'] >= 0) & (df['Fact_diff'] >= 0)) | ((df['Pred_Close_diff'] < 0) & (df['Fact_diff'] < 0)), 'Match_PC'] = 1 
        df.loc[((df['Pred_Close_diff'] >= 0) & (df['Fact_diff'] < 0)) | ((df['Pred_Close_diff'] < 0) & (df['Fact_diff'] >= 0)), 'Match_PC'] = 0
        
        rate = df.value_counts('Match_PC', normalize=True)[1]*100
        self.match = tf.constant(rate)
        
    def result(self):
        return self.match
