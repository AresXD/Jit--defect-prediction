
import pickle
import os
qt_train='data/qt_data/qt_train_changed.pkl'
qt_test='data/qt_data/qt_test_changed.pkl'

op_train='data/op_data/op_train_changed.pkl'
op_test='data/op_data/op_test_changed.pkl'

qt_cross='qt_train.pkl'
op_cross='op_train.pkl'

def combnation():
    op_data = pickle.load(open(qt_train, 'rb'))
    # print(op_data[0])

    ids, labels, msgs, codes = op_data
    # print(len(ids))
    #
    op_data_test=pickle.load(open(qt_test,'rb'))
    test_ids, test_labels, test_msgs, test_codes = op_data_test
    ids+=test_ids
    print(len(ids))
    # labels+=labels
    # msgs+=test_msgs
    # codes+=test_codes
    # data=(ids,labels,msgs,codes)
    # with open('op_train.pkl', 'wb') as f:
    #     pickle.dump(data, f)
    #
    #     print("successful!")
    # op_data = pickle.load(open('op_train.pkl', 'rb'))
    #
    # ids, labels, msgs, codes = op_data
    # print(len(ids))


def test_data():
    op_data = pickle.load(open('op_train.pkl', 'rb'))

    ids, labels, msgs, codes = op_data
    print(len(ids))
    dict_data=pickle.load(open('op_dict.pkl','rb'))
    ids, labels, msgs, codes = op_data
    print(len(ids))

def memory_test():
    import torch
    print(torch.__version__)

memory_test()