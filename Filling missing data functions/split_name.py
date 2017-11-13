def split_name(row):
    name_list1 = row['Name'].split(', ')
    row['Name_last'] = name_list1[0]

    name_list2 = name_list1[1].split('. ')
    row['Name_title'] = name_list2[0]
    row['Name_other'] = name_list2[1]
    return row