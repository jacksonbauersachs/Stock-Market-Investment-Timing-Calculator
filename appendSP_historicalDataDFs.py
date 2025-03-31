import pandas as pd
import os


def appendSP_historicalDataDFs(dfMain, dfsToAppendList, outputDF):

    dfMain = pd.read_csv(dfMain)

    #Loop through DFs in list and append each
    for df in dfsToAppendList:
        dfToAppend = pd.read_csv(df)
        dfMain = pd.concat([dfMain, dfToAppend], ignore_index=True)

    outputDir = os.path.dirname(outputDF)

    main_df.to_csv(outputDF, index=False)

if __name__ == "__main__":

    dfMain = "S&P 500 Data Sets/S&P 500 Data (7_18_2019 to 3_14_2025).csv"

    fileToAppend1 = "S&P 500 Data Sets/S&P 500 Data (10_01_1999 to 07_17_2019).csv"
    fileToAppend2 = "S&P 500 Data Sets/S&P 500 Data (12_26_1979 to 9_30_1999).csv"


    dfsToAppendList = [fileToAppend1, fileToAppend2]

    outputFolder = "S&P 500 Data Sets"
    outputDF = 'S&P 500 Data (12_26_1979 to 3_14_2025)'
    outputDF = os.path.join(outputFolder, outputDF)

    appendSP_historicalDataDFs(dfMain, dfsToAppendList, outputDF)