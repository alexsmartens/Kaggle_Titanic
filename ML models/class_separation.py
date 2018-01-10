def class_separation(df_X):
    df_X_1_2 = df_X.copy().loc[df_X['Pclass'] != 3]
    df_X_3 = df_X.copy().loc[df_X['Pclass'] == 3]
    return [df_X_1_2, df_X_3, df_X_1_2.index, df_X_3.index]