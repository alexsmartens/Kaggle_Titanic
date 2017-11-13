def cabin_loc_allocation(row, ticket_list, cabin_loc, cabin_loc_availability, np):
    # assigning cabin numbers by tickets where this is possible
    if type(row['Cabin']) is not str:
        if str(row['Ticket']) in ticket_list.index:
            if isinstance(ticket_list.loc[str(row['Ticket']), 'Cabin'], list):
                row.loc['Cabin'] = ticket_list.loc[row.loc['Ticket'], 'Cabin'][0]
            else:
                row.loc['Cabin'] = ticket_list.loc[row.loc['Ticket'], 'Cabin']

    # selecting passengers with defined cabin number
    if type(row['Cabin']) is str:
        # discarding ambiguous passengers with known only deck level but not cabin number
        if (len(row['Cabin'])>1) | (row['Cabin']=='T'):
            # passengers occupying only one cabin
            if row['Cabin'] in cabin_loc.index.tolist():
                row.set_value('Room center longitude(X)', cabin_loc.loc[row['Cabin'],'Room center longitude(X)'] )
                row.set_value('Room center latitude(Y)', cabin_loc.loc[row['Cabin'], 'Room center latitude(Y)'] )
                row.set_value('Deck level', cabin_loc.loc[row['Cabin'], 'Deck level'] )
                # adding cabin availability info
                if cabin_loc_availability.loc[row['Cabin'],'Available'] == True:
                    cabin_loc_availability.loc[row['Cabin'],'Available'] = False
                    cabin_loc_availability.loc[row['Cabin'],'Occupied_by_passengers'] = [row.name]
                    cabin_loc_availability.loc[row['Cabin'],'Ticket'] = [str(row['Ticket'])]
                else:
                    cabin_loc_availability.loc[row['Cabin'], 'Occupied_by_passengers'].append(row.name)
                    if str(row['Ticket']) not in cabin_loc_availability.loc[row['Cabin'],'Ticket']:
                        cabin_loc_availability.loc[row['Cabin'],'Ticket'].append(str(row['Ticket']))
            else:
                # passengers occupying multiple cabins
                if row['Cabin'].split(' ')[0] in cabin_loc.index.tolist():
                    # assigning coordinates of "multiple cabin" units
                    mean_loc = np.mean(cabin_loc.loc[row['Cabin'].split(' ')])
                    row.set_value('Room center longitude(X)', mean_loc['Room center longitude(X)'])
                    row.set_value('Room center latitude(Y)', mean_loc['Room center latitude(Y)'] )
                    row.set_value('Deck level', mean_loc['Deck level'])
                    # adding cabin availability info for each unit number
                    for cab_no in row['Cabin'].split(' '):
                        if cabin_loc_availability.loc[cab_no, 'Available'] == True:
                            cabin_loc_availability.loc[cab_no, 'Available'] = False
                            cabin_loc_availability.loc[cab_no, 'Occupied_by_passengers'] = [row.name]
                            cabin_loc_availability.loc[cab_no, 'Ticket'] = [str(row['Ticket'])]
                        else:
                            cabin_loc_availability.loc[cab_no, 'Occupied_by_passengers'].append(row.name)
                            if str(row['Ticket']) not in cabin_loc_availability.loc[cab_no, 'Ticket']:
                                cabin_loc_availability.loc[cab_no, 'Ticket'].append(str(row['Ticket']))
                        cabin_loc_availability.loc[cab_no, 'Occupies_multiple_cabins'] = True
                        cabin_loc_availability.loc[cab_no, 'Multiple_units_No'] = row['Cabin']
    return row