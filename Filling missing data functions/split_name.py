def split_name(row):
    name_list1 = row['Name'].split(', ')
    row['Name_last'] = name_list1[0]

    name_list2 = name_list1[1].split('. ')
    row['Name_title'] = name_list2[0]

    name_list3 = name_list2[1].split(' ')
    row['Name_first'] = name_list3[0].replace("(", "")
    if len(name_list3)>1:
        row['Name_other'] = ' '.join(name_list3[1:]).replace("(", "").replace(")", "")
    return row