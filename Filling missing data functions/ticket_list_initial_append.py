def ticket_list_initial_append(row, ticket_list):
    # selecting passengers with defined cabin number
    if type(row['Cabin']) is str:
        # discarding ambiguous passengers with known only deck level but not cabin number
        if (len(row['Cabin'])>1) | (row['Cabin']=='T'):
            # appending ticket_list
            if str(row['Ticket']) not in ticket_list.index:
                ticket_list.loc[str(row['Ticket'])] = [row['Cabin'], False]
            else:
                if row['Cabin'] not in ticket_list.loc[str(row['Ticket']), 'Cabin']:
                    ticket_list.loc[str(row['Ticket']), 'Multiple_cabins'] = True
                    ticket_list.loc[str(row['Ticket']), 'Cabin'] = [ticket_list.loc[str(row['Ticket']), 'Cabin'],
                                                                    row['Cabin']]