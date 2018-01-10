def ticket_standardization(row, global_check=False):
    ticket_splited = row.loc['Ticket'].split(' ')
    # Assigning ticket number
    row.set_value('Ticket_No', int(ticket_splited[-1]))

    # Assigning ticket series where applicable
    if global_check== False:
    # getting rid from very probable spelling mistakes
            if len(ticket_splited) == 2:
                # Correcting some misspellings of ticket series
                if (ticket_splited[0] == 'A/4') | (ticket_splited[0] == 'A4.') | (
                        ticket_splited[0] == 'A/4.'):
                    row.set_value('Ticket_Series', 'A/4')
                elif (ticket_splited[0] == 'SC/A4') | (ticket_splited[0] == 'S.C./A.4.'):
                    row.set_value('Ticket_Series', 'SC/A4')
                elif (ticket_splited[0] == 'A./5.') | (ticket_splited[0] == 'A.5.') | (ticket_splited[0] == 'A/5.') | (
                            ticket_splited[0] == 'A/S'):
                    row.set_value('Ticket_Series', 'A/5')
                elif (ticket_splited[0] == 'CA') | (ticket_splited[0] == 'C.A./SOTON') | (ticket_splited[0] == 'CA.') | (
                    ticket_splited[0] == 'C.A.') :
                    row.set_value('Ticket_Series', 'CA')
                elif (ticket_splited[0] == 'SC/PARIS') | (ticket_splited[0] == 'SC/Paris') | (
                    ticket_splited[0] == 'S.C./PARIS'):
                    row.set_value('Ticket_Series', 'SC/PARIS')
                elif (ticket_splited[0] == 'W/C') | (ticket_splited[0] == 'W./C.'):
                    row.set_value('Ticket_Series', 'W/C')
                elif (ticket_splited[0] == 'WE/P') |  (ticket_splited[0] == 'W.E.P.'):
                    row.set_value('Ticket_Series', 'WE/P')
                elif (ticket_splited[0] == 'SOTON/O.Q.') | (ticket_splited[0] == 'SOTON/OQ') | (
                            ticket_splited[0] == 'STON/OQ.'):
                    row.set_value('Ticket_Series', 'SOTON/OQ')
                elif (ticket_splited[0] == 'SOTON/O2') | (ticket_splited[0] == 'STON/O2.'):
                    row.set_value('Ticket_Series', 'SOTON/O2')
                elif (ticket_splited[0] == 'S.O.C.') | (ticket_splited[0] == 'SO/C'):
                    row.set_value('Ticket_Series', 'S.O.C.')
                elif (ticket_splited[0] == 'SW/PP') | (ticket_splited[0] == 'S.W./PP'):
                    row.set_value('Ticket_Series', 'SW/PP')
                elif (ticket_splited[0] == 'W.E.P.') | (ticket_splited[0] == 'WE/P'):
                    row.set_value('Ticket_Series', 'W.E.P.')
                else:
                    row.set_value('Ticket_Series', ticket_splited[0])
            elif ticket_splited[0] == 'STON/O':
                row.set_value('Ticket_Series', 'SOTON/O2')
            else:
                if ''.join(ticket_splited[0:-1]) == 'A.2.':
                    row.set_value('Ticket_Series', 'A/2')
                elif ''.join(ticket_splited[0:-1]) == 'SC/AHBasle':
                    row.set_value('Ticket_Series', 'SC/AH')
                else:
                    row.set_value('Ticket_Series', ''.join(ticket_splited[0:-1]))

    # Global (broad) ticket series standardization to only 5 series
    else:
        if len(ticket_splited) == 1:
            # tickets initially without series
            row.set_value('Ticket_Series', 'Orig')
        elif len(ticket_splited) == 2:
            # Correcting some misspellings of ticket series
            if (ticket_splited[0] == 'A/4') | (ticket_splited[0] == 'A4.') | (ticket_splited[0] == 'AQ/3.') | (
                        ticket_splited[0] == 'S.C./A.4.') | (ticket_splited[0] == 'SC/A4') | \
                    (ticket_splited[0] == 'SC/A4') | (ticket_splited[0] == 'A/4.'):
                # row.set_value('Ticket_Series', 'A/4')
                row.set_value('Ticket_Series', 'A/5')
            elif (ticket_splited[0] == 'A./5.') | (ticket_splited[0] == 'A.5.') | (ticket_splited[0] == 'A/5.') | (
                        ticket_splited[0] == 'A/S') | (ticket_splited[0] == 'AQ/4') | (
                ticket_splited[0] == 'SC/A.3') | (
                        ticket_splited[0] == 'AQ/5') | (ticket_splited[0] == 'F.C.') | (ticket_splited[0] == 'C') | (
                        ticket_splited[0] == 'P/PP') | (ticket_splited[0] == 'PP') | (ticket_splited[0] == 'SW/PP') | (
                        ticket_splited[0] == 'S.W./PP') | (ticket_splited[0] == 'S.P.'):
                row.set_value('Ticket_Series', 'A/5')
            elif (ticket_splited[0] == 'C.A.') | (ticket_splited[0] == 'C.A./SOTON') | (ticket_splited[0] == 'CA.') | (
                ticket_splited[0] == 'PP'):
                row.set_value('Ticket_Series', 'CA')
            elif (ticket_splited[0] == 'SC/AH') | (ticket_splited[0] == 'SCO/W') | (ticket_splited[0] == 'SO/C') | (
                        ticket_splited[0] == 'S.O./P.P.') | (ticket_splited[0] == 'LP') | (
                ticket_splited[0] == 'S.O.C.') | (
                        ticket_splited[0] == 'S.O.P.') | (ticket_splited[0] == 'SC'):
                # row.set_value('Ticket_Series', 'SC/AH')
                row.set_value('Ticket_Series', 'CA')
            elif (ticket_splited[0] == 'SC/PARIS') | (ticket_splited[0] == 'SC/Paris') | (
                ticket_splited[0] == 'S.C./PARIS'):
                # row.set_value('Ticket_Series', 'SC/PARIS')
                row.set_value('Ticket_Series', 'A/5')
            elif (ticket_splited[0] == 'W/C') | (ticket_splited[0] == 'W./C.') | (ticket_splited[0] == 'WE/P') | (
                        ticket_splited[0] == 'F.C.C.') | (ticket_splited[0] == 'W.E.P.'):
                # row.set_value('Ticket_Series', 'W/C')
                row.set_value('Ticket_Series', 'Orig')
            elif (ticket_splited[0] == 'SOTON/O.Q.') | (ticket_splited[0] == 'SOTON/O2') | (
                ticket_splited[0] == 'SOTON/OQ') \
                    | (ticket_splited[0] == 'STON/O2.') | (ticket_splited[0] == 'STON/OQ.'):
                row.set_value('Ticket_Series', 'STON/O')
            elif (ticket_splited[0] == 'Fa'):
                # row.set_value('Ticket_Series', '')
                row.set_value('Ticket_Series', 'Orig')
            else:
                row.set_value('Ticket_Series', ticket_splited[0])
        elif ticket_splited[0] == 'STON/O':
            row.set_value('Ticket_Series', 'STON/O')
        else:
            if ''.join(ticket_splited[0:-1]) == 'A.2.':
                # row.set_value('Ticket_Series', 'A/4')
                row.set_value('Ticket_Series', 'A/5')
            elif ''.join(ticket_splited[0:-1]) == 'SC/AHBasle':
                # row.set_value('Ticket_Series', 'SC/AH')
                row.set_value('Ticket_Series', 'CA')
            else:
                row.set_value('Ticket_Series', ''.join(ticket_splited[0:-1]))

    return row




