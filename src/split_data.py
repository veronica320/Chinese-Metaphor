import random

def test_split(infile, split, out_tr, out_te):
    with open(infile, 'r') as inf:
        all_s = inf.readlines()
    print(len(all_s))

    test = random.sample(all_s, round(len(all_s)*split))
    train = set(all_s) - set(test)
    print(len(train), len(test))

    with open(out_tr, 'w') as trf:
        for i in train:
            trf.write(i)
    with open(out_te, 'w') as tef:
        for i in test:
            tef.write(i)

split = 0.1
infile = 'data/verb.txt'
out_tr = 'data/verb_train.txt'
out_te = 'data/verb_test.txt'

test_split(infile, split, out_tr, out_te)