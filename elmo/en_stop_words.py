import os
from nltk.corpus import stopwords


def english():
    return stopwords.words('english')


if __name__ == '__main__':
    words = english()
    words = [w.strip() for w in words]
    print(type(words))
    print(words[0])
    print(words[1])
    # print len(words)
    # for sw in words:
    #     print sw
    with open('stop_words_en.txt', 'w') as f:
        f.write('\n'.join(words))