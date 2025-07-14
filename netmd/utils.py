import pandas as pd


def extract_contacts(file_path, intervals):
    """
    Extracts contacts from a given file and returns a DataFrame.
    
    Parameters:
        file_path (str): Path to the file containing contact data.
        
    Returns:
        pd.DataFrame: DataFrame containing contacts with columns 'Frame', 'Interaction', 'Atom1', 'Atom2'.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

        data = []

        for line in lines: 
            if not line.startswith('#') and not line.startswith('\n'):
                data.append(line.strip().split("\t")[:4])

    df = pd.DataFrame(data)
    df.fillna(0, inplace=True)
    df = df.iloc[:, :4]
    df.columns = ["Frame", "Interaction", "Atom1", "Atom2"]

    # Extract Res1 and Res2
    df['Res1'] = df['Atom1'].str.extract(r':(\d+):')
    df['Res2'] = df['Atom2'].str.extract(r':(\d+):')
    df[["Res1", "Res2"]] = df[["Res1", "Res2"]].astype(int)
    df.loc[df["Res2"] < df["Res1"], ["Res1", "Res2"]] = df.loc[df["Res2"] < df["Res1"], ["Res2", "Res1"]].values
    df = (df.drop_duplicates(subset=['Frame', 'Res1', 'Res2']))[["Frame", "Res1", "Res2"]]

    if intervals[0] < df['Frame'].astype(int).min() or intervals[1] > df['Frame'].astype(int).max():
        print(f"Warning: The specified intervals {intervals} are outside the range of available frames ({df['Frame'].astype(int).min()}–{df['Frame'].astype(int).max()}).")

    print(f"Interval of interest: {intervals[0]}–{intervals[1]} (inclusive)")

    # Filter frames in the interval 0–111 (inclusive)
    interval = df[(df['Frame'].astype(int) >= intervals[0]) & (df['Frame'].astype(int) <= intervals[1])]

    # Determine total number of unique frames in the interval
    total_frames = interval['Frame'].nunique()

    # Compute frequency for each contact
    interval_freq = interval.groupby(['Res1', 'Res2']).size().reset_index(name='Count')
    interval_freq['Frequency'] = interval_freq['Count'] / total_frames
    interval_freq = interval_freq.sort_values('Frequency', ascending=False)

    return interval_freq



def compare_contact_frequencies(frequencies_1, frequencies_2, title_1='WT', title_2='R333Q'):

    # Merge on Res1 and Res2 to find common and exclusive contacts
    merged = pd.merge(
        frequencies_1, frequencies_2,
        on=['Res1', 'Res2'],
        how='outer',
        suffixes=('_' + title_1.lower(), '_' + title_2.lower())
    )

    # Rename columns for clarity
    

    # Fill NaN frequencies with 0 for contacts that are exclusive to one set
    merged['Frequency_' + title_1.lower()] = merged['Frequency_' + title_1.lower()].fillna(0)
    merged['Frequency_' + title_2.lower()] = merged['Frequency_' + title_2.lower()].fillna(0)

    # Exclusive to WT
    exclusive_wt = merged[
        (merged['Frequency_' + title_1.lower()] > 0) & (merged['Frequency_' + title_2.lower()] == 0)
    ][['Res1', 'Res2', 'Frequency_' + title_1.lower()]]

    # Exclusive to R333q
    exclusive_r333q = merged[
        (merged['Frequency_' + title_2.lower()] > 0) & (merged['Frequency_' + title_1.lower()] == 0)
    ][['Res1', 'Res2', 'Frequency_' + title_2.lower()]]

    # Common contacts
    common = merged[
        (merged['Frequency_' + title_1.lower()] > 0) & (merged['Frequency_' + title_2.lower()] > 0)
    ][['Res1', 'Res2', 'Frequency_' + title_1.lower(), 'Frequency_' + title_2.lower()]]

    # Compute odds ratio (add small epsilon to avoid division by zero, if needed)
    common['Odds_Ratio'] = (common['Frequency_' + title_1.lower()] + 1e-6) / (common['Frequency_' + title_2.lower()] + 1e-6)

    # Print results
    print("Exclusive contacts in " + title_1 + ":")
    print(exclusive_wt.head())

    print("\nExclusive contacts in " + title_2 + ":")
    print(exclusive_r333q.head())

    print("\nCommon contacts with odds ratio (" + title_1 + "/" + title_2 + ") - " + title_1 + " favored:")
    print(common.sort_values('Odds_Ratio', ascending=False).head(20))

    print("\nCommon contacts with odds ratio (" + title_1 + "/" + title_2 + ") - " + title_2 + " favored:")
    print(common.sort_values('Odds_Ratio', ascending=True).head(20))

    return common


def moving_average(ts, window_size):
    """
    Applies a moving average to the time series using np.convolve.
    
    Parameters:
        ts (array-like): Input time series.
        window_size (int): Size of the moving average window. Must be odd.
        
    Returns:
        np.ndarray: Smoothed time series with corrected borders.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd for symmetric smoothing.")

    # Create a uniform kernel
    kernel = np.ones(window_size) / window_size

    # Pad the time series to handle border effects
    pad_size = window_size // 2
    padded_ts = np.pad(ts, pad_width=pad_size, mode='edge')

    # Apply convolution
    smoothed_ts = np.convolve(padded_ts, kernel, mode='valid')
    
    return smoothed_ts