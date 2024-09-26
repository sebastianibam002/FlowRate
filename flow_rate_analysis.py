
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from scipy import stats

# Style used for the graphs, you can pick any other styles available
plt.style.use('ggplot')

# get the system os

WINDOWS = (os.name == 'nt')


# Files of excel name (relative path)
TIMES_FILE = "emission_rates.xlsx"
# Where all the figures are going out
OUT = "Out/"
# Where the Aeris data is located
DATA_LOCATION = "Data"
AERISLOC = f"{DATA_LOCATION}/Aeris/"

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

"""
This function would plot the linear regression of df, from start, time to end_time
and add it the line to the plot

"""
def get_linear_regression(df, start_time, end_time, ax, volume, col_name="CH4 (ppm)"):
    time_str = "Time Stamp"
    ch4_str = col_name
    if start_time == end_time:
        return 0, 0, 1
    subset_analysis = df[(df[time_str] > start_time) & (df[time_str] <= end_time)].copy()
    if subset_analysis.empty:
        return 0, 0, 1
    subset_analysis.loc[:, 'Seconds'] = (subset_analysis[time_str] - subset_analysis[time_str].min()).dt.total_seconds()
    # Simple linear regression
    x= subset_analysis['Seconds'].values
    y = subset_analysis[ch4_str].values
    slope, intercept, r, p, _ = stats.linregress(x, y)
    # Predict values
    y_pred = x*slope + intercept
    # Add text to plot with the slope value
    # Plot regression line
    flow_rate = 0
    if col_name != 'CH4 (ppm)':
        # Ethane
        flow_rate = convert_units(slope, volume, molar_mass=30070)
    else:
        flow_rate = convert_units(slope, volume)
    if ax is not None:
        ax.plot(subset_analysis[time_str], y_pred, color="yellow", label='Regression Line')
        ax.text(x=start_time, y=df[ch4_str].max(), s=f'm: {slope:.4f}\nEm: {flow_rate:.2e} mg/h\np-val: {p}', fontsize=12, color='blue')
        ax.axvline(x = start_time, color = 'b', label = 'Start Analysis')
        ax.axvline(x = end_time, color = 'b', label = 'End Analysis')
    
    return slope, flow_rate, p


    
def plot_concentration(data, start_time: str, end_time: str, mID, ethane, volume):

    subset = data[(data['Time Stamp'] > start_time) & (data['Time Stamp'] <= end_time)]
    flow_rate_methane, flow_rate_ethane = 0, 0
    
    date_str = 'Time Stamp'
    ch4_str = 'CH4 (ppm)'

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
    ax.scatter(subset[date_str], subset[ch4_str], label="$CH_4$", marker=".")
    slope_methane, flow_rate_methane, _ = get_linear_regression(subset, start_time, end_time, ax, volume)
    ax.set_ylabel("$CH_4 [ppm]$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}{mID[:1]}/plot_concentration{mID}.png")
    plt.close()
        
    
    return (subset[date_str], subset[ch4_str], subset['C2H6 (ppb)'], flow_rate_methane, flow_rate_ethane)




def summarize_day(data, day, start_time, end_time, mID, ethane: bool, volume: float) -> None:
    """
    plots the timeseries of the pico_df from the id given, also showing a plot
    for pressure and water if these are set to true
    Assumes that the pico_df drift of 6 hrs before March 10 and 7 after
    Is already corrected

    """
    data_dic = {}
  
    tup = plot_concentration(data, start_time, end_time, mID, ethane, volume=volume)
    data_dic['time_stamp_conc'] = tup[0]
    data_dic['ch4_conc'] = tup[1]
    data_dic['c2h6_conc'] = tup[2]
    data_dic['flow_rate_methane'] = tup[3]
    data_dic['flow_rate_ethane'] = tup[4]
    return data_dic


def load_start_end(id_str:str, df):
    day_str = df[df['ID'] == id_str]
    return (day_str['Start Time'].iloc[0], day_str['End Time'].iloc[0])



def load_day(day: str, correct_time:bool=True):
    """
    Loads the particular day and adjust the time using the rule that
    before the March 10, the drift is 6 hours but after is 7h
    """

    # List all the files in the aeris directory
    ls = os.listdir(AERISLOC)
    ls_df = []

    # I will use regular expression to get the files that have the following format
    # One day after
    pprev =  rf"Pico\d+_{(day + pd.Timedelta(1, 'day')).strftime('%y%m%d')}_\d+\.txt"
    # The day of the measurements
    pday = rf"Pico\d+_{day.strftime('%y%m%d')}_\d+\.txt"
    # The day before
    pnex =  rf"Pico\d+_{(day - pd.Timedelta(1, 'day')).strftime('%y%m%d')}_\d+\.txt"

    # The reason behind, UTC time is the standard for the Aeris files

    for element in ls:
        if re.match(pprev, element) or re.match(pday, element) or re.match(pnex, element) :
            # sometimes Aeris gives an extra column, so just take the first 16
            df = pd.read_csv(f"{AERISLOC}/{element}")
            # print(df['Time Stamp'].iloc[0])
            if "Time Stamp" in df.columns and df['Time Stamp'].iloc[2] == 0:
                # print(f"check: {element}")
                # file is broken, take first 16
                df = pd.read_csv(f"{AERISLOC}/{element}", usecols=list(range(1, 17)))
                # df['Time Stamp'] = pd.to_datetime(df.index)
            else:
                # print(f"Otro {element}")
                df.set_index('Time Stamp', inplace=True)
            df['Time Stamp'] = pd.to_datetime(df.index)

            if correct_time:
                if df['Time Stamp'].iloc[0] <= pd.to_datetime("March 10, 2024"):
                    df['Time Stamp'] = df['Time Stamp'] + pd.Timedelta(6, 'h')
                else:
                    df['Time Stamp'] = df['Time Stamp'] + pd.Timedelta(7, 'h')
            ls_df.append(df)
    val_res = pd.concat(ls_df, axis=0, ignore_index=True)
    return val_res



# Helper function to handle Aeris and PMD files in a directory
def process_files_for_flow_rate(df_times):
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
        
        aeris_data = load_day(date, correct_time=True)

        if aeris_data is not None:
            # Perform analysis
            data_dic = summarize_day(
                data=aeris_data, day=date, start_time=chamber_start, end_time=chamber_end, mID=mID, ethane=False, volume=volume)
            
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

    # Read the main Excel file with the times and volume information
    df_times = pd.read_excel(f"{TIMES_FILE}")
    # Process all files in the Aeris and PMD directories
    result_df = process_files_for_flow_rate(df_times)
    
    # Save the result to a CSV file for further analysis
    result_df.to_csv(f"{OUT}computed_flow_rates.csv", index=False)
    print("Flow rate computation completed. Results saved to computed_flow_rates.csv")
    #print(load_day(pd.to_datetime("2024-08-26  00:00:00"), correct_time=True))




    

