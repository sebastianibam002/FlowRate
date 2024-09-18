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
