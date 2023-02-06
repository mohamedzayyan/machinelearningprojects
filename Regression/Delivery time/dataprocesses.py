import pandas as pd
import numpy as np

#Class to process data for delivery time prediction
class data_processer(self, dataset):

    def haversine(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance in kilometers between two points 
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
        return c * r

    #Remove any samples with missing latitude/longitude values as this is necessary for the Haversine function
    dataset.dropna(subset=['Restaurant_latitude', 'Restaurant_longitude', 
                            'Delivery_location_latitude','Delivery_location_longitude'], axis=0,inplace=True)
    
    #Drop any samples missing the 'Time_Orderd' and/or 'Time_Order_picked' features
    dataset.dropna(subset=['Time_Orderd', 'Time_Order_picked'], axis=0, inplace=True)
    
    #Strip appropriate info from weather column
    dataset['Weatherconditions'] = dataset['Weatherconditions'].map(lambda x: str(x)[11:])

    #If any missing values represented by the string 'NaN', replace with np.nan
    for i in dataset.columns:
        dataset[i].loc[dataset[i] == 'NaN '] = np.nan

    #Fill any missing samples with their forward value
    dataset = dataset.fillna(method='ffill')

    #Convert non-string samples to float
    features = ['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries', 'Time_taken(min)']
    for i in features:
        try:
            dataset[i] = dataset[i].astype(str).astype(float)
    
    #Use Haversine function to convert latitude/longitude values to distance traveled (in KM)
    dataset['Distance(km)'] = dataset.apply(lambda x: haversine(x['Restaurant_latitude'], x['Restaurant_longitude'],
                               x['Delivery_location_latitude'], x['Delivery_location_longitude']), axis=1)

    

    

