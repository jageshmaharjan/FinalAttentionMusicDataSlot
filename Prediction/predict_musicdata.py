import argparse
import re, math
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
# from Prediction.attention_lstm_prediction import get_attention_label

def getTFIDF(document):
    tfiddf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfiddf_vectorizer.fit_transform(document)
    return tfidf_matrix


def getJaccardSimillarity(key, query):
    intersection = len(set.intersection(*[set(key), set(query)]))
    union = len(set.union(*[set(key), set(query)]))
    return intersection/float(union)

def jaccard_label():
    maxScore = 0.0
    label = None
    if (label == None):
        for i,key in enumerate(whitelist_dict):
            score = getJaccardSimillarity(key, query)
            if score >= maxScore:
                maxScore = score
                label = whitelist_dict.get(key)
    return label

def cosine_label():
    maxScore = 0.0
    label = None
    if (label == None):
        x.append(query)
        tfidf_matrix = getTFIDF(x)
        for i, key in enumerate(whitelist_dict):
            matrix_len = tfidf_matrix.shape[0] - 1
            score = cosine_similarity(tfidf_matrix[matrix_len], tfidf_matrix[i])
            if score >= maxScore:
                maxScore = score
                label = whitelist_dict.get(key)
    return label


def get_xy():
    whitelist_X = '/home/jugs/PycharmProjects/FinalAttentionMusicDataSlot/Database/whilelist_X'
    whitelist_Y = '/home/jugs/PycharmProjects/FinalAttentionMusicDataSlot/Database/whitelist_y'

    with open(whitelist_X) as fx:
        x = fx.readlines()
        x = [lst.lower().rstrip() for i, lst in enumerate(x)]

    with open(whitelist_Y) as fy:
        y = fy.readlines()
        y = [lst.lower().rstrip() for i, lst in enumerate(y)]

    return x,y


# def attention_label():
#     attn_lbl = get_attention_label(query)
#     return "Work in progress..."  #attn_lbl


def blueprint_label():
    blueprint = "/home/jugs/PycharmProjects/FinalAttentionMusicDataSlot/master/Template.txt"
    templateMap = dict()
    with open(blueprint) as f:
        template = f.readlines()
        max_common = 0
        min_difference = 100
        maxScore = 0.0
        for line in template:
            linereplacement = line.replace('{song}',"#").replace('{artist}','*').replace('{genre}','@')
            templateMap[line] = linereplacement
            commons = len([x for x in linereplacement.split() if x in query.split()])
            difference = len([x for x in linereplacement.split() if x not in query.split()])
            if commons > max_common:
                max_common = commons
                min_difference = difference
                maxScore = getJaccardSimillarity(query, linereplacement.strip())
                v = linereplacement
            elif (commons == max_common) & (difference < min_difference):
                if (len(v.split()) - len(query.split())) > (len(linereplacement.split()) - len(query.split())):   # might be un-necessary
                    min_difference = difference
                    score = getJaccardSimillarity(query, linereplacement.strip())
                    if score > maxScore:
                        maxScore = score
                        v = linereplacement
    q_labels = get_label_from_blueprint(v, query)
    return q_labels


def get_label_from_blueprint(v, query):
    v = v.split()
    query = query.split()
    lbl_string =''
    if v.__contains__('#') & v.__contains__('*'):
        if v.__contains__('by') & str(query).__contains__('by'):
            idx = query.index('by')
            for i,val in enumerate(query):
                if i > idx:
                    lbl_string += 'artist '
                else:
                    if v.__contains__(val):
                        lbl_string += 'o '
                    else:
                        lbl_string += 'music '
    elif v.__contains__('*') :
        artist = [x for x in query if x not in v]
        common = [a for a in v if a in query]
        for i, value in enumerate(query):
            if common.__contains__(value):
                lbl_string += 'o '
            if artist.__contains__(value):
                lbl_string += 'artist '
    elif v.__contains__('#'):
        song_title = [x for x in query if x not in v]
        common = [a for a in v if a in query]
        for i, value in enumerate(query):
            if common.__contains__(value):
                lbl_string += 'o '
            if song_title.__contains__(value):
                lbl_string += 'music '
    elif v.__contains__('@'):
        genre = [x for x in query if x not in v]
        common = [a for a in v if a in query]
        for i, value in enumerate(query):
            if common.__contains__(value):
                lbl_string += 'o '
            if genre.__contains__(value):
                lbl_string += 'genre '
    return lbl_string

def find_best_match(matchlist):
    matchlist = sorted(matchlist, key=len)[::-1]
    lbl_line = []
    for line in matchlist:
        if 'by' not in query:
            if line.__contains__('{song}'):
                matcher = line.split('{song}')[0]
                if query.__contains__(matcher):
                    for i, val in enumerate(query.split()):
                        if val not in matcher.split():
                            lbl_line.insert(i, 'song')
                        else:
                            lbl_line.insert(i, 'o')
                    return lbl_line
            elif line.__contains__('{genre}') & query.__contains__('genre'):
                matcher = line.split('{genre}')[0]
                tag = line.split('{genre}')[1]
                if query.__contains__(matcher):
                    for i, val in enumerate(query.split()):
                        if (val not in matcher.split()) & (val != 'genre'):
                            lbl_line.insert(i, 'genre')
                        else:
                            lbl_line.insert(i, 'o')
                    return lbl_line
        else:
            if (line.__contains__('{artist}')) & ('{song}' not in line):
                matcher = line.split('{artist}')[0]
                query_match = query.split('by')[0]
                if matcher == query_match:
                    for i, val in enumerate(query.split()):
                        if val not in matcher.split():
                            lbl_line.insert(i,'artist')
                        else:
                            lbl_line.insert(i, 'o')
                    return lbl_line
            else:
                matcher = line.split('{song}')[0]
                if query.__contains__(matcher):
                    artist_val = (' '.join(reversed(query.split()))).split('by')[0] # query.split('by')[1]
                    song_value = query.split(matcher)[1].split('by')[0]
                    for i, val in enumerate(query.split()):
                        if val in song_value.split():
                            lbl_line.insert(i, 'music')
                        elif val in artist_val.split():
                            lbl_line.insert(i, 'artist')
                        else:
                            lbl_line.insert(i, 'o ')
                    return lbl_line


def blueprint_pattern():
    blueprint = "/home/jugs/PycharmProjects/FinalAttentionMusicDataSlot/master/Template.txt"
    templateMap = dict()
    matcher = ''
    matchlist = []
    with open(blueprint) as f:
        template = f.readlines()
        lbl_line = list()
        for line in template:
            line = line.strip()
            if line.__contains__('{artist}') & line.__contains__('{song}'):
                if query.__contains__('by'):
                    artist_val = query.split('by')[1]
                    matcher = line.split('{song}')[0]
                    if query.__contains__(matcher):
                        matchlist.append(line)
                        # song_value = query.split(matcher)[1].split('by')[0]
                        # for i, val in enumerate(query.split()):
                        #     if val in song_value:
                        #         lbl_line.insert(i, 'music')
                        #     elif val in artist_val:
                        #         lbl_line.insert(i, 'artist')
                        #     else:
                        #         lbl_line.insert(i, 'o ')
                        # return lbl_line

            elif line.__contains__('{song}') & ('{artist}' not in line):
                matcher = line.split('{song}')[0]
                if (query.__contains__(matcher)) & ('by' not in query):
                    matchlist.append(line)
                    # for i, val in enumerate(query.split()):
                    #     if val not in matcher:
                    #         lbl_line.insert(i, 'song')
                    #     else:
                    #         lbl_line.insert(i, 'o')
                    # return lbl_line

            elif line.__contains__('{artist}') & ('{song}' not in line):
                matcher = line.split('by')[0]
                if query.__contains__(matcher) & ('by' in query):
                    matchlist.append(line)
                    # for i, val in enumerate(query.split()):
                    #     if val not in matcher:
                    #         lbl_line.insert(i,'artist')
                    #     else:
                    #         lbl_line.insert(i, 'o')
                    # return lbl_line

            elif line.__contains__('{genre}') & ('{song}' not in line) & ('{artist}' not in line):
                matcher = line.split('{genre}')[0]
                if (query.__contains__(matcher)) & ('by' not in query) & ('genre' in query):
                    matchlist.append(line)
                    # for i, val in enumerate(query.split()):
                    #     if val not in matcher:
                    #         lbl_line.insert(i, 'genre')
                    #     else:
                    #         lbl_line.insert(i,'o')
                    # return lbl_line
        if len(matchlist):
            return find_best_match(matchlist)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--q', help='input your query', type=str)
    args = parser.parse_args()
    query = args.q
    query = query.lower()
    # query = "i would like to listen bad romance by lady gaga"
    # query = query.lower()

    x, y = get_xy()
    whitelist_dict = dict(zip(x, y))

    label = None
    label = whitelist_dict.get(query)

    print("Input query: " + query)
    if label == None:
        jac_label = jaccard_label()
        print("Using jaccard Simillarity: " + jac_label)
        cos_sim_label = cosine_label()
        print("Using Cosine Simillarity: " + cos_sim_label)
        # rule_based_label = blueprint_label()
        # print("using rule_based_NLP: " + rule_based_label)
        pattern_rule_label = blueprint_pattern()
        print("using Pattern: \n" + str(pattern_rule_label) + '\n')
        # attn_label = attention_label()
        # print("Attention based label: " + attn_label)
    else:
        print("Output: " + label)
