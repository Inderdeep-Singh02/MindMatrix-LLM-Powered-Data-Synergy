def remove_outliers(df, threshold=5.0):
    columns_to_remove = []
    for column in df.columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[column] <= q1 - 1.5 * iqr) | (df[column] >= q3 + 1.5 * iqr)]
        perc = len(outliers) * 100.0 / len(df)
        if perc >= threshold:
            columns_to_remove.append(column)
    df = df[~df.isin(columns_to_remove)]
    
    return df

