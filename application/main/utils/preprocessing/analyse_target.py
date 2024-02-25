
def analyse_target(df ):
    target_column = df.iloc[:, -1]
    num_unique_values = target_column.nunique()
    threshold = 10
    if num_unique_values > threshold:
        return "Regression"
    else:
        return "Classification"