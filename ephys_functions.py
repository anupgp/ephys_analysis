import pandas as pd

# contains all the accessory functions for ephys data analysis
# ---------------------------------------------------------
# joins values from multiple columns into a single column
# input: dataframe and the columns to combine
# return: a single column with joined values from the other columns

def combine_columns(df,columns):
    converted = pd.DataFrame()
    for i in range(len(columns)):
        if(df[columns[i]].dtype.kind in 'biufc'):
            # print(columns[i],' is ', 'not string')
            converted[columns[i]] = df[columns[i]].astype(int).astype(str)
        else:
            # print(columns[i],' is ', 'string')
            converted[columns[i]] = df[columns[i]]
    joined = converted[columns].agg('_'.join,axis=1)
    joined=joined.rename("joined")
    return(joined)
