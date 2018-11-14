import pandas as pd
import fitbit
import datetime
import heart_rate_ai.fitbit_data.fitbit_authentication as Oauth2
from dateutil import parser
from enum import Enum
from definitions import *
from heart_rate_ai.utilities.data_frame_support import *


class FitbitActivity(Enum):
    STEPS = 0
    SLEEP = 1
    CALORIES = 2
    DISTANCE = 3
    FLOORS = 4
    ELEVATION = 5
    HEART = 6


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

    def _write_data_to_file(self, date, data_frame, activity):
        filename = self._project_dir + RAW_DATA_DIRECTORY + date.strftime("%Y-%m-%d") + FILE_EXT
        DataFrameIO.append_df_to_excel(filename, data_frame, activity.name, index=False)

    @staticmethod
    def _data_frame_conversion(activity_data, activity):
        # Ensure data exists before processing
        if not activity_data['activities-' + activity.name.lower() + '-intraday']:
            return pd.DataFrame()
        else:
            time_list = []
            val_list = []
            for entry in activity_data['activities-' + activity.name.lower() + '-intraday']['dataset']:
                val_list.append(entry['value'])
                time_list.append(entry['time'])
            return pd.DataFrame({'Time': time_list, activity.name: val_list})

    @staticmethod
    def _sleep_data_frame_conversion(activity_data, activity):
        # Ensure data exists before processing
        if not activity_data[activity.name.lower()]:
            return pd.DataFrame()
        else:
            time_list = []
            val_list = []
            for entry in activity_data[activity.name.lower()][0]['minuteData']:
                val_list.append(int(entry['value']))
                time_list.append(entry['dateTime'])
            return pd.DataFrame({'Time': time_list, activity.name: val_list})

    def scrape_activities(self, date):
        # Iterate through each activity type
        for activity in FitbitActivity:
            # Reflect the current enumerated value into its string equivalent
            activity_type = 'activities/' + activity.name.lower()

            # Determine the data detail-level
            if activity == FitbitActivity.HEART:
                sampling_interval = '1sec'
            else:
                sampling_interval = '1min'

            # Request the specified activity type from the Fitbit API
            activity_data_frame = None
            if activity != FitbitActivity.SLEEP:
                activity_data = self.authorization_client.intraday_time_series(activity_type,
                                                                               base_date=date,
                                                                               detail_level=sampling_interval)
                # Process the data into a DataFrame
                activity_data_frame = FitbitDataScraper._data_frame_conversion(activity_data, activity)
            else:
                activity_data = self.authorization_client.get_sleep(date)
                # Process the data into a DataFrame
                activity_data_frame = FitbitDataScraper._sleep_data_frame_conversion(activity_data, activity)

            # Save Data to a file
            if not activity_data_frame.empty:
                self._write_data_to_file(date, activity_data_frame, activity)

    def scrape_activities_over_dates(self, overwrite_old_data):
        yesterday = datetime.datetime.now() - datetime.timedelta(1)  # Use yesterday so we have a full day's data
        for current_date in FitbitDataScraper._generate_date_range(parser.parse(START_DATE), yesterday):
            if self._check_if_data_exists(current_date.strftime("%Y-%m-%d")):
                if overwrite_old_data:
                    print('Overwriting existing data')
                    self._remove_file(current_date.strftime("%Y-%m-%d"))
                    self.scrape_activities(current_date)
                else:
                    print('Data already exists, overwriting not enabled')
            else:
                print('scraping...')
                self.scrape_activities(current_date)

    def _check_if_data_exists(self, date):
        filename = self._project_dir + RAW_DATA_DIRECTORY + date + FILE_EXT
        does_file_exist = os.path.isfile(filename)
        return does_file_exist

    def _remove_file(self, date):
        filename = self._project_dir + RAW_DATA_DIRECTORY + date + FILE_EXT
        os.remove(filename)

    @staticmethod
    def _generate_date_range(start_date, end_date):
        return (start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1))


def main():
    data_scraper = FitbitDataScraper()
    data_scraper.authentication_process()
    data_scraper.scrape_activities_over_dates(True)


if __name__ == '__main__':
    main()
