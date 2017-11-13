def plot3d_loc_and_price(train,plt,Axes3D,np):
    colors_dict = {'S':'red', 'C':'#FFA500', 'Q':'green',np.nan:'black'}
    fig = plt.figure()
    ax = Axes3D(fig)
    S_index = train.loc[train['Embarked']=='S', 'Room center longitude(X)'].dropna().index
    S_series = ax.scatter(train.loc[S_index, 'Room center longitude(X)'],
               train.loc[S_index, 'Room center latitude(Y)'],
               train.loc[S_index, 'Deck level'],
               c='red',
               s=train.loc[S_index, 'Fare'])
    C_index = train.loc[train['Embarked']=='C', 'Room center longitude(X)'].dropna().index
    C_series = ax.scatter(train.loc[C_index, 'Room center longitude(X)'],
               train.loc[C_index, 'Room center latitude(Y)'],
               train.loc[C_index, 'Deck level'],
               c='#FFA500',
               s=train.loc[C_index, 'Fare'])
    Q_index = train.loc[train['Embarked']=='Q', 'Room center longitude(X)'].dropna().index
    Q_series = ax.scatter(train.loc[Q_index, 'Room center longitude(X)'],
               train.loc[Q_index, 'Room center latitude(Y)'],
               train.loc[Q_index, 'Deck level'],
               c='green',
               s=train.loc[Q_index, 'Fare'])
    ax.set_xlabel('Room center longitude(X)')
    ax.set_ylabel('Room center latitude(Y)')
    ax.set_zlabel('Deck level')
    ax.set_title('Price by cabin location and by port of embarkation')
    plt.legend([S_series, C_series, Q_series],
               ['S - Southampton', 'C - Cherbourg', 'Q - Queenstown'],
               title='Embarked at',
               loc=6)
    plt.show()