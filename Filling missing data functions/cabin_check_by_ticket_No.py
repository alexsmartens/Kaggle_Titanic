def cabin_check_by_ticket_No(df_X):
    import pandas as pd

    # Creating a data base of tickets with known cabin numbers
    ticket_w_cabin = df_X.copy().loc[pd.notnull(df_X['Cabin']), ['Ticket','Cabin']]
    df_ticket_cabin = pd.DataFrame(columns=['Cabin'])
    for ticket_No in ticket_w_cabin['Ticket']:
        cabins_by_ticket = ticket_w_cabin.loc[ticket_w_cabin['Ticket']==ticket_No,'Cabin'].unique().tolist()
        df_ticket_cabin.loc[ticket_No,'Cabin'] = cabins_by_ticket

    # Assigning cabin numbers to passengers holding tickets from the ticket_w_cabin data base
    for PassengerID in df_X.index:
        if pd.isnull(df_X.loc[PassengerID, 'Cabin']):
            if df_X.loc[PassengerID, 'Ticket'] in df_ticket_cabin.index:
                if len(df_ticket_cabin.loc[ df_X.loc[PassengerID, 'Ticket'], 'Cabin' ])==1:
                    df_X.set_value(PassengerID, 'Cabin', df_ticket_cabin.loc[df_X.loc[PassengerID, 'Ticket'], 'Cabin'][0])
                else:
                    # combining cabins in one unit
                    new_cabin_No =' '.join( df_ticket_cabin.loc[df_X.loc[PassengerID, 'Ticket'], 'Cabin'])
                    df_X.set_value(PassengerID, 'Cabin',new_cabin_No)
                    # updating cabin numbers with the same tickets
                    for cabin_temp in df_ticket_cabin.loc[df_X.loc[PassengerID, 'Ticket'], 'Cabin']:
                        df_X.loc[df_X['Cabin']==cabin_temp, 'Cabin'] = new_cabin_No
    return df_X



