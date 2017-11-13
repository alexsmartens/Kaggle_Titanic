def plot_scores (scores):
    import matplotlib.pyplot as plt
    plt.figure()
    train_series = plt.plot(scores.index,
                            scores.loc[:, 'Avg_train_score'],
                            c='black',
                            label='Train')
    test_series = plt.plot(scores.index,
                            scores.loc[:, 'Avg_test_score'],
                            c='green',
                            label='Test')
    plt.xlabel('Model parameter')
    plt.ylabel('Classifier score')
    plt.title('Selection of model parameters')
    plt.legend(title='Legend',
               loc=6)
    plt.show()