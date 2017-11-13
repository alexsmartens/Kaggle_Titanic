def plot3d_surv_by_loc(train, plt, Axes3D):
    fig = plt.figure()
    ax = Axes3D(fig)
    Y_series = ax.scatter(train.loc[train['Survived']==1, 'Room center longitude(X)'],
                          train.loc[train['Survived']==1, 'Room center latitude(Y)'],
                          train.loc[train['Survived']==1, 'Deck level'],
                          c='green')
    N_series = ax.scatter(train.loc[train['Survived'] == 0, 'Room center longitude(X)'],
                          train.loc[train['Survived'] == 0, 'Room center latitude(Y)'],
                          train.loc[train['Survived'] == 0, 'Deck level'],
                          c='red')
    ax.set_xlabel('Room center longitude(X)')
    ax.set_ylabel('Room center latitude(Y)')
    ax.set_zlabel('Deck level')
    ax.set_title('Survival by cabin location')
    plt.legend([Y_series, N_series],
               ['Survived', 'Not survived'],
               title='Legend',
               loc=6)
    plt.show()