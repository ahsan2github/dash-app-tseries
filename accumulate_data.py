import numpy as np
from sklearn.preprocessing import MinMaxScaler

class tseries_data:
    def __init__(self, source, source_dates):
        self.dates = np.array(source_dates)
        self.dates = self.dates.flatten()
        self.scaler = MinMaxScaler()
        self.scaler.fit(np.reshape(source.as_matrix(), [-1,1]));
        self.tseries = self.scaler.transform(np.reshape(source.as_matrix(), [-1,1]))
        self.tseries = self.tseries.flatten()
        self.nrow = self.tseries.shape
        self.nrow = self.nrow[0]
        
    def get_series(self, series_len, lag, date_yes = False):
        if series_len > self.nrow:
            print("Requested series is longer than existing series")
            return
        else:
            begin = np.random.rand()
            begin = np.int(begin * (self.nrow - series_len - lag -1))
            x = self.tseries[begin+lag:begin+series_len+lag]
            y = self.tseries[begin:begin+series_len]
            date_series = self.dates[begin:begin+series_len+lag]
            if date_yes:
                return np.reshape(x, [-1, series_len, 1]), np.reshape(y, [-1, series_len, 1]), np.reshape(date_series, [-1, series_len+lag, 1])
            else:
                return np.reshape(x, [-1, series_len, 1]), np.reshape(y, [-1, series_len, 1]) 
        
    def get_latest_series(self, series_len,lag,date_yes = False):
        if series_len > self.nrow:
            print("Requested series is longer than existing series")
            return
        else:
            x = self.tseries[lag:series_len+lag]
            y = self.tseries[0:series_len]
            date_series = self.dates[0:series_len+lag]
        if date_yes:
            return np.reshape(x, [-1, series_len, 1]), np.reshape(y, [-1, series_len, 1]), np.reshape(date_series, [-1, series_len+lag, 1])
        else:
            return np.reshape(x, [-1, series_len, 1]), np.reshape(y, [-1, series_len, 1])
        
    def get_batch(self, batch_size, series_len, lag, date_yes = False):
        if series_len > self.nrow:
            print("Requested series is longer than existing series")
            return
        else:
            batch_x = np.zeros([series_len, batch_size])
            batch_y = np.zeros([series_len, batch_size])
            batch_date = np.empty([series_len+lag, batch_size], dtype=np.object_)
            for ii in range(batch_size):
                x, y, d = self.get_series(series_len, lag, 1)
                batch_x[:, ii], batch_y[:,ii], batch_date[:, ii] = x.flatten(), y.flatten(), d.flatten()
            if date_yes:   
                return np.reshape(batch_x, [-1, series_len, 1]), np.reshape(batch_y, [-1, series_len, 1]), np.reshape(batch_date, [-1, series_len+lag, 1]) 
            else:
                return np.reshape(batch_x, [-1, series_len, 1]), np.reshape(batch_y, [-1, series_len, 1])  
