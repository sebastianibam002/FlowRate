
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from textwrap import wrap
from scipy import stats

plt.style.use('ggplot')



# retrieve all the data from the timelines
TIMES_FILE = "emission_rates_times.xlsx"
OUT = "Out/"
DATA_LOCATION = "Data"


"""
Given a slope in ppm/minutes convert it into mg/h
"""
def convert_units(slope_ppm_s, volume, molar_mass=16040):
    #molar_mass = 16040  # mg/mol
    
    # Ideal gas law conversion
    P = 1  # atm
    R = 0.0821  # atm·L/(mol·K)
    T = 295  # K
    dc_dt_mg_L_s = slope_ppm_s * (1 / (R * T)) * 10**(-6) * molar_mass
    
    # Calculate volume in liters
    volume_L = volume / 1000  # Convert to liters
    # Flow rate in mg/s
    F_mg_s = dc_dt_mg_L_s * volume_L
    
    # Convert to mg/h
    F_mg_h = F_mg_s * 3600
    
    return F_mg_h

    #molar_mass = 16040
    #dc_dt = slope * (1/60) * (1/(0.0821*295)) * 10**(-6) * molar_mass
    #return dc_dt * (volume/1000) * 3600

"""
This function would plot the linear regression of df, from start, time to end_time
and add it the line to the plot

"""
def get_linear_regression(df, start_time, end_time, ax, is_pmd, volume, col_name="CH4 (ppm)"):
    time_str = "Time Stamp"
    ch4_str = col_name
    if is_pmd:
        time_str = 'DATE TIME'
        ch4_str = ' PPM'
    if start_time == end_time:
        return 0, 0, 1

    
    subset_analysis = df[(df[time_str] > start_time) & (df[time_str] <= end_time)].copy()

    if subset_analysis.empty:
        return 0, 0, 1
    # Convert DATE TIME to minutes from the start
    subset_analysis.loc[:, 'Seconds'] = (subset_analysis[time_str] - subset_analysis[time_str].min()).dt.total_seconds()

    # Simple linear regression
    x= subset_analysis['Seconds'].values
    y = subset_analysis[ch4_str].values
    slope, intercept, r, p, _ = stats.linregress(x, y)

    #model.fit(X, y)
    # Predict values
    y_pred = x*slope + intercept
    # Add text to plot with the slope value
    # Plot regression line

    flow_rate = 0
    if col_name != 'CH4 (ppm)':
        flow_rate = convert_units(slope, volume, molar_mass=30070)
    else:
        flow_rate = convert_units(slope, volume)
    if ax is not None:
        ax.plot(subset_analysis[time_str], y_pred, color="yellow", label='Regression Line')
        ax.text(x=start_time, y=df[ch4_str].max(), s=f'm: {slope:.4f}\nEm: {flow_rate:.2e} mg/h\np-val: {p}', fontsize=12, color='blue')
        ax.axvline(x = start_time, color = 'b', label = 'Start Analysis')
        ax.axvline(x = end_time, color = 'b', label = 'End Analysis')
    
    return slope, flow_rate, p


"""
This function will compute the emission rate increasing the end_time by step_size and would return the different emission
reates per each iteration. Notice that ste_size is in seconds

"""



def compute_emissions_iteratively(df, start_time, end_time, ax, is_pmd, volume, col_name="CH4 (ppm)", step_size=1, filter_window=None):
    possible_times = pd.date_range(start=start_time, end=end_time, freq=f"{step_size}s")
    all_slopes, all_flows, all_pval = np.zeros(len(possible_times)), np.zeros(len(possible_times)), np.zeros(len(possible_times))
    time_str = "Time Stamp"
    ch4_str = col_name
    if is_pmd:
        time_str = 'DATE TIME'
        ch4_str = ' PPM'

    if is_pmd:
        time_str = 'DATE TIME'
    i = 0
    for dummy_endtime in possible_times:
        subset = df[(df[time_str] > start_time) & (df[time_str] <= dummy_endtime)].copy()
        if filter_window:
            subset[ch4_str] = subset[ch4_str].rolling(window=filter_window, min_periods=1, center=True).mean()
        slope, flow_rate, pval = get_linear_regression(subset, start_time, dummy_endtime, None, is_pmd, volume, col_name)
        all_slopes[i] = slope
        all_flows[i] = flow_rate
        all_pval[i] = pval
        i += 1
    return all_slopes, all_flows, all_pval


"""
Function to plot filtered emissions
"""
def plot_filtered_emissions(df, start_time, end_time, ax, is_pmd, volume, col_name="CH4 (ppm)", step_size=1, filter_window=10, mID=0):
    slopes, flow_rates, pvals = compute_emissions_iteratively(df, start_time, end_time, ax, is_pmd, volume, col_name, step_size, filter_window)
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(flow_rates)), flow_rates, label="Filtered Emission Rates")
    plt.xlabel("Time (s)")
    plt.ylabel("Emission Rate (mg/h)")
    plt.title("Filtered Emission Rates over Time")
    plt.legend()
    plt.savefig(f"{OUT}/{mID[:1]}/plot_flow_rate_filtered_{mID}.png")
    plt.close()

    return slopes, flow_rates, pvals


    
def plot_concentration(data, start_time: str, end_time: str, mID, ethane, note, pmd_dat, volume, iteratively:bool, filter_data:int, verbose:bool):
    # EDIT: 14 - 05 if there is no data from the AERIS but there is a valid initial time and end time
    # then I will look for the PMD dataset and make the plot with that data, maybe add extra layer of
    # checking by looking for the word PMD in the Notes section
    # I will subset the data from the specific start to the end
    #     tup = plot_concentration(data, start_time, end_time, mID, ethane, note, pmd_dat=pmd_dat,
    #                        times_df=time_df, volume=volume, iteratively=iteratively, filter_data=window, verbose=verbose)

    subset = data[(data['Time Stamp'] > start_time) & (data['Time Stamp'] <= end_time)]
    is_pmd = False
    flow_rate_methane, flow_rate_ethane = 0, 0
    wrapped_note = ""
    # add the times lines of the sample selected
    if not isinstance(note, float):
        is_pmd = ("pmd" in note.lower()) and pmd_dat is not None
        wrapped_note = "\n".join(wrap(note, width=100))
    
    date_str = 'Time Stamp'
    ch4_str = 'CH4 (ppm)'
    if is_pmd:
        date_str = 'DATE TIME'
        ch4_str = ' PPM'
        subset = pmd_dat[(pmd_dat[date_str] > start_time) & (pmd_dat[date_str] <= end_time)].copy()
        subset['C2H6 (ppb)'] = 0
    


    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(13, 8))


    SMALL_SIZE = 15
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    plt.suptitle(f"{mID}")
    # plt.figure()
    ax.scatter(subset[date_str], subset[ch4_str], label="$CH_4$", marker=".")
    slope_methane, flow_rate_methane, _ = get_linear_regression(subset, start_time, end_time, ax, is_pmd, volume)
    if verbose:
        ax.set_xlabel(f"Time\n\nNote: {wrapped_note}")
    ax.set_ylabel("$CH_4 [ppm]$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/{mID[:1]}\\plot_concentration{mID}.png")
    plt.close()
        

    if filter_data is not None:
        #all_slope_methane, all_flow_rate_methane = compute_emissions_iteratively(subset, start_time, second_line, None, is_pmd, volume, step_size=step_size)
        plt.figure(figsize=(10, 5))
        df_filter = subset[ch4_str].rolling(filter_data).mean()
        plt.scatter(np.arange(0, len(df_filter), step=1), df_filter)
        plt.xlabel(f"Time [window: {filter_data}]")
        plt.ylabel("Average $CH_4$ [ppm]")
        plt.tight_layout()
        plt.savefig(f"{OUT}/{mID[:1]}/plot_smooth_concentration{mID}.png")
        plt.close()
       

    if iteratively:
        step_size = 1
        all_slope_methane, all_flow_rate_methane, list_pval = compute_emissions_iteratively(subset, start_time, end_time, None, is_pmd, volume, step_size=step_size)
        plt.figure(figsize=(10, 5))
        plt.scatter(np.arange(0, len(all_flow_rate_methane), step_size), all_flow_rate_methane)
        plt.xlabel("Time [s]")
        plt.ylabel("Flow Rate [mg/hr]")
        #slope_methane = all_slope_methane
        #flow_rate_methane = all_flow_rate_methane
        plt.tight_layout()
        plt.savefig(f"{OUT}/{mID[:1]}/plot_flow_rate_time{mID}.png")
        plt.close()

        if filter_data is not None:
            slope, list_methane_rate, list_pval = plot_filtered_emissions(subset, start_time, end_time, None, is_pmd, volume, filter_window=filter_data, mID=mID)
        
    
    return (subset[date_str], subset[ch4_str], subset['C2H6 (ppb)'], flow_rate_methane, flow_rate_ethane, list_methane_rate, list_pval)




def plot_environment(df, start:str, end:str, id):
    """
    Plots the environment variables from the df at the specified time
    """

    # I will subset the data from the specific start to the end
    subset = df[(df['Time Stamp'] > start) & (df['Time Stamp'] <= end)]

    f, ((ax1, ax2, ax3)) = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))
    ax1.scatter(subset['Time Stamp'], subset['T0 (degC)'], marker=".", label="T0")
    ax1.scatter(subset['Time Stamp'], subset['T5 (degC)'], marker=".", label="T5")
    ax1.scatter(subset['Time Stamp'], subset['Tgas(degC)'], marker=".", label="Tgas")

    ax2.scatter(subset['Time Stamp'], subset['P (mbars)'], marker=".", label="Pressure")

    ax3.scatter(subset['Time Stamp'], subset['H2O (ppm)'])

    plt.xlabel("Time [minutes]")
    ax1.set_ylabel("Temperature [degC]")
    ax2.set_ylabel("Pressure [mBar]")
    ax3.set_ylabel("$H_2O$ [ppm]")
    ax1.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/{id[:1]}/plot_environment{id}.png")
    plt.close()


def box_plot_comparison(pico_df, first_tup, second_tup, id, dat):
    no_app = pico_df[(pico_df['Time Stamp'] > first_tup[0]) & (pico_df['Time Stamp'] < first_tup[1])]
    app = pico_df[(pico_df['Time Stamp'] > second_tup[0]) & (pico_df['Time Stamp'] < second_tup[1])]

    data_noapp = no_app['CH4 (ppm)']
    data_app = app['CH4 (ppm)']

    # making boxplot of the data
    plt.figure(figsize=(10, 8))
    plt.boxplot([data_noapp, data_app], labels=['No appliances', 'Appliances'])
    plt.ylabel("$CH_4$ [ppm]")
    plt.title(f"{first_tup[0]}")
    plt.savefig(f"{OUT}/{id[:3]}/plot_comparison{id}.png")
    plt.close()
    if dat:
        return [data_noapp, data_app]

def summarize_day(data, day, start_time, end_time, mID, pressure: bool, water: bool, ethane: bool,
                   pmd_dat, iteratively, window:int, verbose: bool, note: str, volume: float) -> None:
    """
    plots the timeseries of the pico_df from the id given, also showing a plot
    for pressure and water if these are set to true
    Assumes that the pico_df drift of 6 hrs before March 10 and 7 after
    Is already corrected

    """
    data_dic = {}
  
    tup = plot_concentration(data, start_time, end_time, mID, ethane, note, pmd_dat=pmd_dat,
                             volume=volume, iteratively=iteratively, filter_data=window, verbose=verbose)
    data_dic['time_stamp_conc'] = tup[0]
    data_dic['ch4_conc'] = tup[1]
    data_dic['c2h6_conc'] = tup[2]
    data_dic['flow_rate_methane'] = tup[3]
    data_dic['flow_rate_ethane'] = tup[4]
    data_dic['list_methane_rate'] = tup[5]
    data_dic['list_p_val'] = tup[6]
    # for the environment variables, I will just get across the whole, time
    tup = plot_environment(data, start_time, end_time, mID)
    data_dic['environment'] = tup

    return data_dic


def load_start_end(id_str:str, df):
    day_str = df[df['ID'] == id_str]
    return (day_str['Start Time'].iloc[0], day_str['End Time'].iloc[0])


def load_day_pmd(id_str: str, sample_df):
    """
    Loads the particular day and adjust the time using the rule that
    before the March 10, the drift is 6 hours but after is 7h
    """
    day_str = sample_df[sample_df['Measurement ID'] == id_str]['Date']
    day = pd.to_datetime(day_str.iloc[0])
    folder = f"{day.month_name()} {day.day} {day.year}"
    expected_name = f"{(day.strftime('%d%b%y')).upper()}.CSV"
    full_name = f"{DATA_LOCATION}{folder}/{expected_name}"
    if os.path.isfile(full_name):
        df = pd.read_csv(full_name, skiprows=2)
        # 03MAR2024 11:54:57
        df['DATE TIME'] = pd.to_datetime(df['DATE TIME'], format="%d%b%Y %H:%M:%S")
        return df
    return None


def load_day(day: str, correct_time:bool=True):
    """
    Loads the particular day and adjust the time using the rule that
    before the March 10, the drift is 6 hours but after is 7h
    """

    # get a list of all the aeris data file that are available, I want those files of the day of the mesurement´
    # one dy before an one day after if these are available
    AERISLOC = f"{DATA_LOCATION}/Aeris/"
    ls = os.listdir(AERISLOC)
    ls_df = []

    # I will use regular expression to get the files that have the following format
    pprev =  rf"Pico\d+_{(day + pd.Timedelta(1, 'day')).strftime("%y%m%d")}_\d+\.txt"
    pday = rf"Pico\d+_{day.strftime("%y%m%d")}_\d+\.txt"
    pnex =  rf"Pico\d+_{(day - pd.Timedelta(1, 'day')).strftime("%y%m%d")}_\d+\.txt"

    for element in ls:
        if re.match(pprev, element) or re.match(pday, element) or re.match(pnex, element) :
            # sometimes Aeris gives an extra column, so just take the first 16
            df = pd.read_csv(f"{AERISLOC}/{element}")
            
            if df.shape[1] > 16:
                # file is broken, take first 16
                df = pd.read_csv(f"{AERISLOC}/{element}", usecols=list(range(1, 17)))
            else:
                df.set_index('Time Stamp', inplace=True)
            df['Time Stamp'] = pd.to_datetime(df.index)
            if correct_time:
                if df['Time Stamp'].iloc[0] <= pd.to_datetime("March 10, 2024"):
                    df['Time Stamp'] = df['Time Stamp'] + pd.Timedelta(6, 'h')
                else:
                    df['Time Stamp'] = df['Time Stamp'] + pd.Timedelta(7, 'h')
            ls_df.append(df)
    return pd.concat(ls_df, axis=0, ignore_index=True)


import pandas as pd
import os

# Helper function to handle Aeris and PMD files in a directory
def process_files_for_flow_rate(directory, df_times, pmd_dat=None):
    """
    Processes Aeris and PMD files in the given directory to compute flow rates.
    """
    all_results = []
    
    for index, row in df_times.iterrows():
        mID = row['mID']
        date = pd.to_datetime(row['Day']) 
        chamber_start = pd.to_datetime(row['Day'].strftime('%Y%m%d') + ' ' + row['Start Time'].strftime('%X'))
        chamber_end = pd.to_datetime(row['Day'].strftime('%Y%m%d') + ' ' + row['End Time'].strftime('%X'))
        # volume in ml
        volume = row['Volume Chamber']
        notes = row['Notes']
        
        pmd_data, aeris_data = None, None
        if pmd_dat:
            pmd_data = load_day_pmd(mID, df_times)
        else:
            aeris_data = load_day(date, correct_time=True)

        if aeris_data is not None:
  
            # Perform analysis
            data_dic = summarize_day(
                data=aeris_data, day=date, start_time=chamber_start, end_time=chamber_end, mID=mID, pressure=False, water=False,
                ethane=False, iteratively=True, window=10, verbose=False, pmd_dat=pmd_dat, note=notes, volume=volume)
            
            result = {
                "mID": mID,
                "Flow Rate CH4": data_dic.get("flow_rate_methane"),
                "Flow Rate C2H6": data_dic.get("flow_rate_ethane"),
                "Start Time": chamber_start,
                "End Time": chamber_end,
                "Notes": notes,
            }
            all_results.append(result)
    
    return pd.DataFrame(all_results)

if __name__ == "__main__":


    # Given a folder with two subfolders of Aeris and PMD, this program will compute the flow-rate
    # of different static chambers, given an extra file containing the columns
    #   mID	Day	Start Time	End Time	Volume Chamber	Notes
    # end time and location id

    # Read the main Excel file with the times and volume information
    df_times = pd.read_excel("emission_rates.xlsx")

    # Define directories where Aeris and PMD files are stored
    aeris_directory = f"{DATA_LOCATION}/Aeris/"
    pmd_directory = f"{DATA_LOCATION}/PMD/"
    
    # Process all files in the Aeris and PMD directories
    result_df = process_files_for_flow_rate(aeris_directory, df_times)
    
    # Save the result to a CSV file for further analysis
    result_df.to_csv(f"{OUT}/computed_flow_rates.csv", index=False)
    print("Flow rate computation completed. Results saved to computed_flow_rates.csv")





    

