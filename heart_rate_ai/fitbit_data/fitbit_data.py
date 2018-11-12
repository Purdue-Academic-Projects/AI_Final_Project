import pandas as pd
import os
import fitbit
import datetime
import heart_rate_ai.fitbit_data.fitbit_authentication as Oauth2
from dateutil import parser
from definitions import ROOT_DIR

CLIENT_ID = '22D56P'
CLIENT_SECRET = '9aefa27740d00cd57d1b06beb43992ac'
START_DATE = '2018-11-10' # '2016-09-28' ACTUAL DATE, JUST FOR TESTING
DATA_DIRECTORY = '/Downloaded_Data/'
FILE_EXT = '.csv'


class FitbitDataScraper:
    def __init__(self):
        self.server = None
        self.authorization_client = None
        self._project_dir = ROOT_DIR

    def authentication_process(self):
        self.server = Oauth2.OAuth2Server(CLIENT_ID, CLIENT_SECRET)
        self.server.browser_authorize()
        retrieved_access_token = str(self.server.oauth.session.token['access_token'])
        retrieved_refresh_token = str(self.server.oauth.session.token['refresh_token'])
        self.authorization_client = fitbit.Fitbit(CLIENT_ID, CLIENT_SECRET, oauth2=True,
                                                  access_token=retrieved_access_token,
                                                  refresh_token=retrieved_refresh_token)

    def _dataframe_extraction(self,data_type):

    def _process_heart_rate_data(self, date):
        heart_rate_stats = self.authorization_client.intraday_time_series('activities/heart',
                                                                          base_date=date, detail_level='1sec')
        time_list = []
        val_list = []
        for i in heart_rate_stats['activities-heart-intraday']['dataset']:
            val_list.append(i['value'])
            time_list.append(i['time'])

        heart_df = pd.DataFrame({'Time': time_list, 'Heart Rate': val_list})
        filename = self._project_dir + DATA_DIRECTORY + date + FILE_EXT
        heart_df.to_csv(filename, columns=['Time', 'Heart Rate'], header=True, index=False)

    def _process_distance_data(self, date):
        distance_stats = self.authorization_client.intraday_time_series('activities/distance',
                                                                        base_date=date, detail_level='1min')

    def scrape_activities(self, date):
        self._process_heart_rate_data(date)
        self._process_distance_data(date)

    def scrape_activities_over_dates(self):
        # date = str((datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
        for current_date in self._daterange(parser.parse(START_DATE), datetime.datetime.now()):
            if self._check_if_data_exists(current_date.strftime("%Y-%m-%d")):
                print('Data already exists')
            else:
                print('scraping...')
                self.scrape_activities(current_date)

    def _check_if_data_exists(self, date):
        filename = self._project_dir + DATA_DIRECTORY + date + FILE_EXT
        does_file_exist = os.path.isfile(filename)
        return does_file_exist

    def _daterange(self, start_date, end_date):
        return (start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1))


def main():
    data_scraper = FitbitDataScraper()
    data_scraper.authentication_process()
    data_scraper.scrape_activities_over_dates()


if __name__ == '__main__':
    main()
