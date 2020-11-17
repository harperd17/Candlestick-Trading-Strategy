# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:03:37 2020

@author: david
"""

#from tradingFunctions import *
#import statsmodels.stats.contingency_tables as sm_table
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.stats.contingency_tables as sm_table
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
"""------------------------------------------------------------------------------------------------------"""

def onlyNormalHours(data, normal_hours):
    """
    This function takes in the data, and only spits out the data that falls under normal trading hours.

    Parameters
    ----------
    data : DataFrame
        dataframe of the stock data.
    normal_hours : list
        list containing the opening hour and closing hour of the market in datetime.

    Returns
    -------
    data : DataFrame
        the dataframe passed in, but without after hours data.

    """
    discard_pile = []
    for r in range(data.shape[0]):
        #rewriting the normal hours so it is current with the day of the current row in the dataframe
        normal_hours[0] = normal_hours[0].replace(year=data['Date'][r].year, month=data['Date'][r].month,
                                               day = data['Date'][r].day)
        normal_hours[1] = normal_hours[1].replace(year=data['Date'][r].year, month=data['Date'][r].month,
                                               day = data['Date'][r].day)

        #comparing the current row to see if it's within normal trading hours - if so then keep, otherwise add the index to a discard pile
        if not(normal_hours[0] <= data['Date'][r] <= normal_hours[1]):
            discard_pile.append(r)
    #now that the non trading hours rows have been recorded, drop them all
    data = data.drop(discard_pile)
    #reindex the dataframe so the indeces are sequential again
    data = data.reset_index(drop=True)
    
    return data

"""------------------------------------------------------------------------------------------------------------------"""

def findHangingMans(data, w_b_ratio1, w_b_ratio2):
    """
    This function finds occurences of the hanging man candlestick pattern
    Parameters:
    data: dataframe
        Is the dataframe that is being searched for hanging man occurrences
    w_b_ratio1: float
        Is the minimum ratio of the tall top/bottom wick length to body length. This is the top wick for green candles and bottom wick for red candles.
    w_b_ratio2: float
        Is the maximum ratio of the short top/bottom wik length to body length. This is the bottom wick for green candles and top wick for red candles.
        
    Returns:
        list of indeces for hanging man occurences - 1 for occurrence, 0 for non occurence
    """
    
    body_lengths = data['Close']-data['Open']
    occurrences = []
    for i in range(data.shape[0]):
        top_wick = data['High'][i] - max(data['Close'][i],data['Open'][i])
        bottom_wick = min(data['Close'][i],data['Open'][i]) - data['Low'][i]
        #check to see if candle is neutral
        if body_lengths[i] == 0:
            #if it's a neutral candle, then treat either the top or bottom wick as the body to compare to the other wick
            if top_wick > w_b_ratio1*bottom_wick or bottom_wick > w_b_ratio1*top_wick:
                occurrences.append(1)
            else:
                occurrences.append(0)
        else:
            if (bottom_wick >= w_b_ratio1*abs(body_lengths[i]) and top_wick <= w_b_ratio2*abs(body_lengths[i]) and body_lengths[i] < 0) or (top_wick >= w_b_ratio1*abs(body_lengths[i]) and bottom_wick <= w_b_ratio2*abs(body_lengths[i]) and body_lengths[i]>0):
                occurrences.append(1)
            else:
                occurrences.append(0)    
    return occurrences

"""-------------------------------------------------------------------------------------------------------------"""
def hangingManSuccess(data, variable, ratio, fill):
    """
    This function tests whether each hanging man occurrence was a win (1) or a lose (0)
    Parameters:
    data: dataframe
        Is the dataframe with the hanging man occurrences
    variable: string
        The name of the column with the hanging man occurrences.
    ratio: float
        Ratio of reward to risk.
    fill: float
        Ratio of the tall wick that we are looking to get filled.
        
    Returns:
        list of successes of failures for the hanging man data
    """
    
    looking = False
    successes = []
    directions = []
    for r in range(data.shape[0]):
        if data[variable][r] == 1:
            #start looping through the data until either the stop loss or target has been reached
            #first find out if it is bullish or bearish move
            #if bullish
            if (data['High'][r] - max(data['Close'][r],data['Open'][r])) >= (min(data['Close'][r],data['Open'][r]) - data['Low'][r]):
                target = data['Close'][r] + (data['High'][r] - data['Close'][r])*fill
                stop = data['Close'][r] - (target - data['Close'][r])/ratio
                directions.append(1)
            #bearish
            else:
                target = data['Close'][r] - (data['Close'][r] - data['Low'][r])*fill
                stop = data['Close'][r] + (data['Close'][r]-target)/ratio
                directions.append(-1)
            
            looking = True
            i = r
            while looking:
                i = i + 1
                #if the end of dataframe has been reached, then exit - else continue the search
                if i == data.shape[0]-1:
                    looking = False
                    successes.append(0)
                else:
                    #check to see if its a bullish or bearish candle
                    #if bullish
                    if target > stop:
                        #if the stop is reached
                        if data['Low'][i] <= stop:
                            successes.append(0)
                            looking = False
                        #if target is reached
                        elif data['High'][i] >= target:
                            successes.append(1)
                            looking = False
                    #bearish
                    else:
                        #if the stop is reached
                        if data['High'][i] >= stop:
                            successes.append(0)
                            looking = False
                        #if target is reached
                        elif data['Low'][i] <= target:
                            successes.append(1)
                            looking = False
        else:
            successes.append(np.nan)
            directions.append(np.nan)
            
    return successes, directions

"""--------------------------------------------------------------------------------------------------------------"""
def plotResults(data, addings, adding_types, addings_colors, sizes, within_data,legend,names):
    if len(addings) == len(adding_types) == len(addings_colors):
        apdict = []
        #if within_data is True, then the addings will be a list of column names to add to the plot
        if within_data:
            for i in range(len(addings)):
                apdict.append(mpf.make_addplot(data[addings[i]],type= adding_types[i], color=addings_colors[i],markersize=sizes[i]))
        #if within_data is False, then the addings will be a list of dataframes to add to the plot
        else:
            for i in range(len(addings)):
                apdict.append(mpf.make_addplot(addings[i], type=adding_types[i],color=addings_colors[i],markersize=sizes[i]))
              

        if legend:
            fig, ax = plt.subplots(1)
            fig, ax = mpf.plot(data,type='candle',block=False,addplot=apdict,returnfig=True)
            ax.legend(names)
            fig.show()
            #return ax
        else:
            mpf.plot(data,type='candle',block=False,addplot=apdict)



"""--------------------------------------------------------------------------------------------------------------"""
def getOtherTimeFramePivots(data1, data2, data2_time_frame, zone_percent, look_back, HAs, dist_away, smoothing_period, degree):
    #getting the pivots
    data2 = findPivots(data2.copy(), HAs, dist_away, smoothing_period, degree)
    #filling in the gaps of the pivots
    """mins, maxes = fillPivotGaps(data2.copy(),'Min Pivot','Max Pivot')
    data2['Min Pivot'] = mins
    data2['Max Pivot'] = maxes"""
    print("filling gaps")
    data2 = fillPivotGaps(data2.copy(),'Min Pivot','Max Pivot')
    print("gaps filled")
    #creating the zones 
    zone_dates = []
    zones = []
    #for zone_type, 1 = max, -1 = min
    zone_type = []
    #keep track of the most recent pivot direction (min = -1 or max = 1)
    last_pivots = []
    useable_dates = []
    for row in range(data2.shape[0]):
        if str(data2['Min Pivot'][row]) != 'nan':
            zone_type.append(-1)
            zone_dates.append(data2['Usable Min Date'][row])
            zones.append([data2['Min Pivot'][row]*(1-zone_percent),data2['Min Pivot'][row]*(1+zone_percent)])
        if str(data2['Max Pivot'][row]) !=  'nan':
            zone_type.append(1)
            zone_dates.append(data2['Usable Max Date'][row])
            zones.append([data2['Max Pivot'][row]*(1-zone_percent),data2['Max Pivot'][row]*(1+zone_percent)])
    #print(zone_dates)
    #print('\n\n')
    #print(zones)
    #now to loop through the time frame dataframe of interest and mark each candlestick as either being in the zone, or not
    zone_matches = []
    types_of_zone = []
    min_max_zones = []
    zone_index = 0
    searching_for_first_row = True
    first_row = 0
    while searching_for_first_row:
        if data1.index[first_row] >= zone_dates[zone_index]:
            searching_for_first_row = False
        else:
            first_row += 1
            zone_matches.append(0)
            types_of_zone.append(0)
            last_pivots.append(0)
            min_max_zones.append([np.nan])
    print("Found first row at "+str(zone_index))
    #now that the first row has been found, loop through and see if either the high, or the low has dipped into a zone
    for i in range(first_row,data1.shape[0]):
        #if the zone_index is already the last of the zones, then no need to look for an update in current last zone
        in_zone = False
        if zone_index == len(zone_dates)-1:
            zone_index = zone_index
        else:
            if data1.index[i] >= zone_dates[zone_index+1]:
                zone_index += 1
            if zone_index == len(zone_dates)-1:
                zone_index = zone_index
            else:
                if data1.index[i] >= zone_dates[zone_index+1]:
                    zone_index += 1
        zone_direction = 0
        for j in range(max(0,zone_index-look_back),zone_index):
            #check to see if either the max or min has into a zone
            if zones[j][0] <= data1['High'][i] <= zones[j][1] or zones[j][0] <= data1['Low'][i] <= zones[j][1]:
                in_zone = True
                zone_direction = zone_type[j]
        if in_zone:
            zone_matches.append(1)
            types_of_zone.append(zone_direction)  
        else:
            zone_matches.append(0)
            types_of_zone.append(0)
        #last_pivots.append(types_of_zone[zone_index])
        last_pivots.append(zone_type[zone_index])
        zone_line = []
        for z in range(min(look_back,zone_index+1)):
            zone_line.append(zones[zone_index-z][0])
            zone_line.append(zones[zone_index-z][1])
        min_max_zones.append(zone_line)
        
    data1[data2_time_frame+' Zones'] = zone_matches
    data1[data2_time_frame+' Direction'] = types_of_zone
    data1['Last '+data2_time_frame+' Pivot'] = last_pivots
    
    return data1, min_max_zones
"""--------------------------------------------------------------------------------------------------------------"""

"""--------------------------------------------------------------------------------------------------------------"""
def createData(data_file_path, tickers, time_frames, minutes, normal_hours, emas,hanging_man_params, success_params, pivot_params):
    dataframes = []
    for symbol in tickers:
        line_of_dfs = []
        for time in time_frames:
            file_location = data_file_path+'/'+symbol+time+"Data.csv"
            new_data = pd.read_csv(file_location)
            new_data['Date'] = pd.to_datetime(new_data['Date'],format='%Y-%m-%d %H:%M:%S')
            line_of_dfs.append(new_data)
        dataframes.append(line_of_dfs)
    
    
    total_count = 0
    min_max_zone_collection = []
    collection_names = []
    look_back_period = 8
    for i in range(len(tickers)):
        for j in range(len(time_frames)):
            #taking out all rows that in normal trading hours
            dataframes[i][j] = onlyNormalHours(dataframes[i][j].copy(),normal_hours)   
            #setting the the index to the datetime (needed for plotting later on)
            dataframes[i][j].index = dataframes[i][j]['Date']
            dataframes[i][j]['dates'] = dataframes[i][j]['Date']
            del dataframes[i][j]['Date']     
            
        #finding the occurrences of a hanging man
        j = 0
        dataframes[i][j]['Hanging Man'] = findHangingMans(dataframes[i][j].copy(), hanging_man_params[0], hanging_man_params[1])
        total_count = total_count + sum(dataframes[i][j]['Hanging Man'])
        #getting the exponential moving averages
        for ema in emas:
            dataframes[i][j]['Distance from '+str(ema)+"EMA"] = dataframes[i][j]['Close'] - getEMA(dataframes[i][j]['Close'],ema)
        #getting the moving average for the candle sizes (high-low, not open-close)
        dataframes[i][j]['Size'] = dataframes[i][j]['High'] - dataframes[i][j]['Low']
        dataframes[i][j]['Moving Average Size'] = getEMA(dataframes[i][j]['Size'],21)
        dataframes[i][j]['Moving Average Volume'] = getEMA(dataframes[i][j]['Volume'],21)
        dataframes[i][j]['Distance from Moving Size'] = dataframes[i][j]['Size'] - dataframes[i][j]['Moving Average Size']
        dataframes[i][j]['Distance from Moving Volume'] = dataframes[i][j]['Volume'] - dataframes[i][j]['Moving Average Volume']
        #getting the successes 
        dataframes[i][j]['Successes'], dataframes[i][j]['Directions'] = hangingManSuccess(dataframes[i][j].copy(), 'Hanging Man', success_params[0], success_params[1])
        dataframes[i][j]['Success Closes'] = dataframes[i][j]['Close']*dataframes[i][j]['Successes']
        dataframes[i][j]['Success Closes'] = dataframes[i][j]['Success Closes'].replace(0,np.nan)
        dataframes[i][j]['Failure Closes'] = dataframes[i][j]['Close']*np.absolute(dataframes[i][j]['Successes']-1)
        dataframes[i][j]['Failure Closes'] = dataframes[i][j]['Failure Closes'].replace(0,np.nan)
        #finding the pivot points for different time frames
        time_zone_totals = [0]*dataframes[0][0].shape[0]
        for j in range(len(time_frames)):
            dataframes[i][0], min_max = getOtherTimeFramePivots(dataframes[i][0].copy(), dataframes[i][j].copy(), time_frames[j], pivot_params[0], pivot_params[1], pivot_params[2], pivot_params[3]*minutes[j], pivot_params[4], pivot_params[5])
            dataframes[i][0][time_frames[j]+" Zone Definitions"] = min_max
            #min_max_zone_collection.append(min_max)
            #collection_names.append(tickers[i]+" "+time_frames[j])
            time_zone_totals = time_zone_totals + dataframes[i][0][time_frames[j]+" Zones"]
        dataframes[i][0]['Zone Totals'] = time_zone_totals
    
    #now to get the data that only has hanging man occurrence candlesticks!
    subset_dataframes = []
    for i in range(len(tickers)):
        subset_dataframes.append(dataframes[i][0][dataframes[i][0]['Hanging Man']==1])  
    
    return dataframes, subset_dataframes

"""-----------------------------------------------------------------------------------------------------"""
def returnRates(X,y):
    total_target_count = sum(y)
    
    unique_p_values = np.unique(X)
    
    rates = []
    counts = []
    temp = pd.concat((y,X),axis=1)
    count = 0
    for value in unique_p_values:
        value_subset = temp[temp[X.name]==value]
        rate = sum(value_subset[y.name])/len(value_subset[y.name])
        count += sum(value_subset[y.name])
        rates.append(rate)
        counts.append(len(value_subset[y.name]))
    if count == total_target_count:
        return list(unique_p_values), rates, counts
    else:
        return "Totals Didn't Match"
    
"""----------------------------------------------------------------------------------------------------"""
def getOddsRatios(data,alpha):
    ORs = []
    lower_CIs = []
    upper_CIs = []
    for i in range(data.shape[0]):
        row = data.loc[data.index[i],:]
        table = np.array([[int(round(row['Rate 2']*row['Count 2'],0)),int(round((1-row['Rate 2'])*row['Count 2'],0))],
                       [int(round(row['Rate 1']*row['Count 1'],0)),int(round((1-row['Rate 1'])*row['Count 1'],0))]])
        t = sm_table.Table2x2(table)
        ORs.append(t.oddsratio)
        interval = t.oddsratio_confint(alpha)
        lower_CIs.append(interval[0])
        upper_CIs.append(interval[1])
    return ORs, lower_CIs, upper_CIs

"""-----------------------------------------------------------------------------------------------------"""
def performOddsRatioTesting(interactions_data, y, alpha):
    all_uniques = []
    all_rates = []
    all_counts = []
    for item in list(interactions_data.columns):
        uniques, rates, counts = returnRates(interactions_data[item],y)
        all_uniques.append(uniques)
        all_rates.append(rates)
        all_counts.append(counts)
    total_rates_df = pd.DataFrame({'Unique Values':all_uniques,'Success Rates':all_rates,'Group Counts':all_counts})
    total_rates_df.index = list(interactions_data.columns)
    
    unique_len = []
    for i in range(total_rates_df.shape[0]):
        unique_len.append(len(total_rates_df['Unique Values'][i]))


    u1 = []
    u2 = []
    r1 = []
    r2 = []
    c1 = []
    c2 = []
    feature_names = []
    for i in range(total_rates_df.shape[0]):
        if unique_len[i] == 2:
            u1.append(total_rates_df['Unique Values'][i][0])
            u2.append(total_rates_df['Unique Values'][i][1])
            r1.append(total_rates_df['Success Rates'][i][0])
            r2.append(total_rates_df['Success Rates'][i][1])
            c1.append(total_rates_df['Group Counts'][i][0])
            c2.append(total_rates_df['Group Counts'][i][1])
            feature_names.append(total_rates_df.index[i])
    two_valued_feature_rates_df = pd.DataFrame({'Value 1':u1,'Value 2':u2,'Rate 1':r1,'Rate 2':r2,'Count 1':c1,'Count 2':c2})
    two_valued_feature_rates_df.index = feature_names
    
    ors, lower_cis, upper_cis = getOddsRatios(two_valued_feature_rates_df,alpha)
    
    two_valued_feature_rates_df['Odds Ratios'] = ors
    two_valued_feature_rates_df['Odds Lower Interval'] = lower_cis
    two_valued_feature_rates_df['Odds Upper Interval'] = upper_cis
    
    return two_valued_feature_rates_df


"""--------------------------------------------------------------------------------------------------------"""
def getRates(data,p_variable,t_variable,ax):
    """This function takes in a dataframe, the name of the predictor variable to group on, and the target variable
    to create the rates on. It returns a list of [0 value rate, 1 value rate]. Also the axis is passed through
    in case of wanting to handle subplots. Lastly, a mapping dictionary variable is passed through which gives the definitions of
    each numerical value in the p_variable"""
    total_target_count = sum(data[t_variable])
    
    unique_p_values = np.unique(data[p_variable])
    
    names = []
    rates = []
    counts = []
    
    for value in unique_p_values:
        value_subset = data[data[p_variable]==value]
        rate = sum(value_subset[t_variable])/len(value_subset[t_variable])
        counts.append(len(value_subset[t_variable]))
        rates.append(rate)
    
    ax.bar(unique_p_values, rates, align='center', alpha=0.5)
    ax.set_xlabel(p_variable)
    ax.set_xticks(unique_p_values)
    ax.set_xticklabels(unique_p_values)
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate by '+p_variable)
    
    return ax, counts

"""--------------------------------------------------------------------------------------------------------"""
def plotBins(data, p_variable, t_variable, ax, bins):
    min_value = np.nanmin(data[p_variable]) - 0.001
    max_value = np.nanmax(data[p_variable]) + 0.001
    
    increment = (max_value - min_value)/bins
    bins = []
    bin_centers = []
    bin_names = []
    rates = []
    total = 0
    for i in np.arange(min_value, max_value, increment):
        bins.append([i,i+increment])
        bin_centers.append((i+i+increment)/2)
        bin_names.append(str("%.2f" % bins[-1][0])+" - "+str("%.2f" % bins[-1][1]))
        data_subset = data[(data[p_variable]>i) & (data[p_variable] <= i+increment)]
        rates.append(sum(data_subset[t_variable])/max(1,data_subset.shape[0]))
        total += data_subset.shape[0]

    ax.bar(bin_centers, rates, align='center', alpha=0.5,width=increment*3/4)
    ax.set_xlabel(p_variable)
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(bin_names)
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate by '+p_variable)
    
    ax.xaxis.set_visible(True)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    return ax, bins

"""----------------------------------------------------------------------------------------------"""
def plotHist(data, p_variable, ax, bins):
    bin_centers = []
    bin_names = []
    frequencies = []
    for b in bins:
        bin_centers.append((b[0]+b[1])/2)
        bin_names.append(str("%.2f" % b[0])+" - "+str("%.2f" % b[1]))
        data_subset = data[(data[p_variable]>b[0]) & (data[p_variable] <= b[1])]
        frequencies.append(data_subset.shape[0])
        
    ax.bar(bin_centers, frequencies, align='center', alpha=0.5,width=(b[1]-b[0])*3/4)
    ax.set_xlabel(p_variable)
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(bin_names)
    ax.set_ylabel('Frequency')
    ax.set_title('Frequency by '+p_variable)
    
    ax.xaxis.set_visible(True)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        
    return ax


def createCandlestick(data, time_frame, new_time_frame,trading_hours):
    #!!!! Note that this function doesn't yet take into account days where the trading hours are shorter than normal!!!!
    
    #now to format the trading_hours values
    trading_hours[0] = datetime.strptime(trading_hours[0],'%H:%M')
    trading_hours[1] = datetime.strptime(trading_hours[1],'%H:%M')
    
    last_time = list(data['Date'])[-1]
    time = int(time_frame[:len(time_frame)-1])
    #now to get the number of minutes in the existing time frame
    if time_frame[-1] == 'h':
        time = time * 60
    #now to get the number of minutes in the new time frame
    new_time = int(new_time_frame[:len(time_frame)-1])
    if new_time_frame[-1] == 'h':
        new_time = new_time * 60
    if new_time%time != 0:
        return "New time frame isn't divisible by existing time frame"
    
    else:
        #now to do the actual candlestick creations
        new_dates = []
        new_opens = []
        new_highs = []
        new_lows = []
        new_closes = []
        new_volumes = []
        starting_date = data['Date'][0]
        ending_date = starting_date + timedelta(minutes=new_time)
        new_open = data['Open'][0]
        new_high = data['High'][0]
        new_low = data['Low'][0]
        new_close = data['Close'][0]
        new_volume = data['Volume'][0]
        for i in range(data.shape[0]-1):
            new_close = data['Close'][i]
            new_high = max(new_high,data['High'][i])
            new_low = min(new_low,data['Low'][i])
            new_volume += data['Volume'][i]
            #if the next date is greater or equal to the ending date, then we have finished the current candlestick to append
            #to the lists and restart the candlestick calculations
            if data['Date'][i+1] >= ending_date:
                new_dates.append(starting_date)
                new_opens.append(new_open)
                new_highs.append(new_high)
                new_lows.append(new_low)
                new_closes.append(new_close)
                new_volumes.append(new_volume)
                
                new_open = data['Open'][i+1]
                new_high = data['High'][i+1]
                new_low = data['Low'][i+1]
                new_close = data['Close'][i+1]
                new_volume = data['Volume'][i+1]
                
                #need something to test if there is enough time left in the data for another candlestick
                if (last_time - data['Date'][i+1]).total_seconds()/60 >= new_time:
                
                    end_of_day = datetime(year=starting_date.year, month=starting_date.month, day=starting_date.day, 
                                         hour = trading_hours[1].hour, minute = trading_hours[0].minute)
                    minutes_left_in_trading = (end_of_day-starting_date).total_seconds()/60

                    #if the ending date is the end of trading hours, then start at next day, otherwise make starting_date prior ending_date
                    if ending_date == end_of_day:
                        #since the next trading day may not be the next day (because of holidays and weekends) I must look forward to the 
                        #next day in the dataframe
                        starting_date = datetime(year = data['Date'][i+1].year, month = data['Date'][i+1].month, day = data['Date'][i+1].day,
                                                hour = trading_hours[0].hour, minute = trading_hours[0].minute)
                    else:
                        starting_date = ending_date
                    #if there are enough minutes left in the trading hours, then no worries, otherwise, carry it over to next day
                    if new_time <= minutes_left_in_trading:
                        ending_date = starting_date + timedelta(minutes=new_time)
                    else:
                        next_day = data['Date'][i+int(minutes_left_in_trading/time)+1]
                        ending_date = datetime(year=next_day.year, month=next_day.month, day=next_day.day, 
                                               hour=trading_hours[0].hour, minute = trading_hours[0].minute) + timedelta(minutes=new_time-minutes_left_in_trading)
                else:
                    ending_date = last_time + timedelta(minutes = 1)
                
        new_data = pd.DataFrame({'Date':new_dates,'Open':new_opens,'High':new_highs,'Low':new_lows,'Close':new_closes,'Volume':new_volumes})
        return new_data
    
    
def performValidations(X,y,variable_sets,folds,model):
    """Function takes in the X and y arrays and also takes in the stepwise models at each step of forward stepwise
    selection. Also takes in how many folds to do with K-fold and which model data should be fitted on"""
    cv = KFold(n_splits=folds, random_state=1, shuffle=True)
    #loop through each set and perform k-fold cross validation for each  model, then report the average classification rate
    all_metrics = []
    for sett in variable_sets:
        predictions = cross_val_predict(model,X[sett],y,cv=cv,n_jobs=-1)
        cm = metrics.confusion_matrix(y,predictions)
        all_metrics.append(cm)
    return all_metrics


def getMetricSummaries(metrics):
    """This function takes in a list of confusion metric summaries (usually from cross validation over a range of factors) and
    then computes the listing of precision, recall, and accuracy for each factor then returns precision, recall and accuracy in that order."""
    precision = []
    recall = []
    acc = []
    for i in metrics:
        #the max(...,1) is in the case of 0 being in the denominator so precision shows up as 0 instead of error
        precision.append(i[1][1]/max((i[0][1]+i[1][1]),1))
        recall.append(i[1][1]/max(sum(i[1]),1))
        acc.append((i[0][0]+i[1][1])/max(1,(sum(i[0])+sum(i[1]))))
    
    return precision, recall, acc

def getSMA(data,period):
    sma = data.rolling(window=period).mean()
    return sma

def removeOutliers(q1, q2,series,name):
    q1 = series[name].quantile(q1)
    q2 = series[name].quantile(q2)
    output = series[(series[name]>q1)&(series[name]<q2)]
    return output