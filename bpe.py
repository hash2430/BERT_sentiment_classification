from typing import List, Dict, Set
from itertools import chain

### You can import any Python standard libraries here.
### Do not import external library such as numpy, torchtext, etc.
from collections import Counter, defaultdict
import re
### END YOUR LIBRARIES

def build_bpe(
    corpus: List[str],
    max_vocab_size: int
) -> List[int]:
    """ BPE Vocabulary Builder
    Implement vocabulary builder for byte pair encoding.
    Please sort your idx2word by subword length in decsending manner.

    Hint: Counter in collection library would be helpful

    Note: If you convert sentences list to word frequence dictionary,
          building speed is enhanced significantly because duplicated words are preprocessed together

    Arguments:
    corpus -- List of words to build vocab
    max_vocab_size -- The maximum size of vocab

    Return:
    idx2word -- Subword list
    """
    # Special tokens
    PAD = BytePairEncoding.PAD_token # Index of <PAD> must be 0
    UNK = BytePairEncoding.UNK_token # Index of <UNK> must be 1
    CLS = BytePairEncoding.CLS_token # Index of <CLS> must be 2
    SEP = BytePairEncoding.SEP_token # Index of <SEP> must be 3
    MSK = BytePairEncoding.MSK_token # Index of <MSK> must be 4
    SPECIAL = [PAD, UNK, CLS, SEP, MSK]

    WORD_END = BytePairEncoding.WORD_END # Use this token as the end of a word

    ### YOUR CODE HERE (~22 lines)
    idx2word: List[str] = SPECIAL
    _corpus = [word + WORD_END for word in corpus]
    vocab = []
    for word in _corpus:
        vocab+=list(word)
    vocab = set(vocab)
    corpus =[]
    for word in _corpus:
        letter = list(word)
        word_ = ' '.join(letter)
        corpus.append(word_)
    corpus = Counter(corpus)

    '''
    Arguments:
        vocab: Counter dict
    Return:
        pairs: dictionary with key as tuple of symbols and value as its count in corpus
    '''

    def get_stats(corpus):
        pairs = defaultdict(int)
        for word, cnt in corpus.items():
            symbols = word.split(' ')
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += cnt

        return pairs

    def merge_vocab(pair, c_in):
        c_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in c_in:
            w_out = p.sub(''.join(pair), word)
            c_out[w_out] = c_in[word]
        return c_out

    # pair_count_dict: dictionary with key as tuple of symbols and value as its count in corpus
    while(len(vocab)+len(SPECIAL) < max_vocab_size):
        pair_count_dict = get_stats(corpus)
        if len(pair_count_dict) == 0:
            break
        best = max(pair_count_dict, key=pair_count_dict.get)
        vocab.add(best[0]+best[1])
        corpus = merge_vocab(best, corpus)
    ### END YOUR CODE
    new_idx2word = list(vocab)
    new_idx2word.sort(reverse=True, key=len)
    idx2word += new_idx2word
    return idx2word

def encode(
    sentence: List[str],
    idx2word: List[str]
) -> List[int]:
    """ BPE encoder
    Implement byte pair encoder which takes a sentence and gives the encoded tokens

    Arguments:
    sentence -- The list of words which need to be encoded.
    idx2word -- The vocab that you have made on the above build_bpe function.
    
    Return:
    tokens -- The list of the encoded tokens
    """
    WORD_END = BytePairEncoding.WORD_END

    ### YOUR CODE HERE (~10 lines)
    def tokenize_word(word, vocab, bias):
        UNK = BytePairEncoding.UNK_token
        # Define base case to stop recursion
        if word == '':
            return []
        if len(vocab) == 0:
            return [UNK]

        # i: index for current vocab
        # bias + i: index for global vocab
        # skip tokens until we find the token that satisfies the pattern
        # word = [substring + token]*n + remainder
        #  => substrings and remainder are fed  recursively to this function with partial vocab
        for i in range(len(vocab)):
            token = vocab[i]
            pattern = re.escape(token)
            matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(pattern, word)]

            if len(matched_positions) == 0:
                continue

            substring_end_positions = [matched_position[0] for matched_position in matched_positions]
            substring_start_position = 0

            # substring start ~ substring end: recursively tokenize with vocabulary of idx2word[i+1:]
            # substring end ~ token[i] length: tokenize into token[i]
            # update substring start to be substring end + len(token[i])
            # iterate until there is no more matched token in the string
            tokens = []
            for substring_end_position in substring_end_positions:
                substring = word[substring_start_position:substring_end_position]
                tokens += tokenize_word(substring, vocab[i + 1:], i + bias + 1)
                tokens.append(i + bias)
                substring_start_position = substring_end_position + len(token)

            # Take care of the remaining substring by recursive call with vocab idex2word[i+1:] then break
            remaining_substring = word[substring_start_position:]
            tokens += tokenize_word(remaining_substring, vocab[i + 1:], i + bias + 1)
            break
        return tokens
    tokens = []
    WORD_END = BytePairEncoding.WORD_END
    for word in sentence:
        word_tokenized = tokenize_word(word + WORD_END, idx2word, 0)
        tokens += word_tokenized
        ### END YOUR CODE

    return tokens

def decode(
    tokens: List[int],
    idx2word: List[str]
) -> List[str]:
    """ BPE decoder
    Implement byte pair decoder which takes tokens and gives the decoded sentence.

    Arguments:
    tokens -- The list of tokens which need to be decoded
    idx2word -- the vocab that you have made on the above build_bpe function.

    Return:
    sentence  -- The list of the decoded words
    """
    WORD_END = BytePairEncoding.WORD_END

    ### YOUR CODE HERE (~1 lines)
    sentence: List[str] = None
    sentence = [idx2word[token] for token in tokens]
    sentence2 = ""
    for i in range(len(sentence)):
        sentence2 += sentence[i]
    sentence = sentence2.split(WORD_END)
    if sentence[-1] == "":
        sentence.pop(-1)
    ### END YOUR CODE
    return sentence


#############################################
# Helper functions below. DO NOT MODIFY!    #
#############################################

class BytePairEncoding(object):
    """ Byte Pair Encoding class
    We aren't gonna use this class for encoding. Because it is too slow......
    We will use sentence piece Google have made.
    Thus, this class is just for special token index reference.
    """
    PAD_token = '<pad>'
    PAD_token_idx = 0
    UNK_token = '<unk>'
    UNK_token_idx = 1
    CLS_token = '<cls>'
    CLS_token_idx = 2
    SEP_token = '<sep>'
    SEP_token_idx = 3
    MSK_token = '<msk>'
    MSK_token_idx = 4

    WORD_END = '_'

    def __init__(self, corpus: List[List[str]], max_vocab_size: int) -> None:
        self.idx2word = build_bpe(corpus, max_vocab_size)

    def encode(self, sentence: List[str]) -> List[int]:
        return encode(sentence, self.idx2word)

    def decoder(self, tokens: List[int]) -> List[str]:
        return decode(tokens, self.idx2word)
    
#############################################
# Testing functions below.                  #
#############################################

def build_bpe_test():
    print ("======Building BPE Vocab Test Case======")
    PAD = BytePairEncoding.PAD_token
    UNK = BytePairEncoding.UNK_token
    CLS = BytePairEncoding.CLS_token
    SEP = BytePairEncoding.SEP_token
    MSK = BytePairEncoding.MSK_token
    WORD_END = BytePairEncoding.WORD_END

    # First test
    corpus = ['abcde']
    vocab = build_bpe(corpus, max_vocab_size=15)
    assert vocab[:5] == [PAD, UNK, CLS, SEP, MSK], \
        "Please insert the special tokens properly"
    print("The first test passed!")

    # Second test
    assert sorted(vocab[5:], key=len, reverse=True) == vocab[5:], \
        "Please sort your idx2word by subword length in decsending manner."
    print("The second test passed!")

    # Third test
    corpus = ['low'] * 5 + ['lower'] * 2 + ['newest'] * 6 + ['widest'] * 3
    vocab = set(build_bpe(corpus, max_vocab_size=24))
    assert vocab > {PAD, UNK, CLS, SEP, MSK, 'est_', 'low', 'newest_', \
                    'i', 'e', 'n', 't', 'd', 's', 'o', 'l', 'r', 'w', WORD_END} and \
           "low_" not in vocab and "wi" not in vocab and "id" not in vocab, \
           "Your bpe result does not match expected result"
    print("The third test passed!")

    # forth test
    corpus = ['aaaaaaaaaaaa', 'abababab']
    vocab = set(build_bpe(corpus, max_vocab_size=13))
    assert vocab == {PAD, UNK, CLS, SEP, MSK, 'aaaaaaaa', 'aaaa', 'abab', 'aa', 'ab', 'a', 'b', WORD_END}, \
           "Your bpe result does not match expected result"
    print("The forth test passed!")

    # fifth test
    corpus = ['abc', 'bcd']
    vocab = build_bpe(corpus, max_vocab_size=10000)
    assert len(vocab) == 15, \
           "Your bpe result does not match expected result"
    print("The fifth test passed!")

    print("All 5 tests passed!")

def encoding_test():
    print ("======Encoding Test Case======")
    PAD = BytePairEncoding.PAD_token
    UNK = BytePairEncoding.UNK_token
    CLS = BytePairEncoding.CLS_token
    SEP = BytePairEncoding.SEP_token
    MSK = BytePairEncoding.MSK_token
    SPECIAL = [PAD, UNK, CLS, SEP, MSK]
    WORD_END = BytePairEncoding.WORD_END

    # First test
    vocab = SPECIAL + ['bcc', 'bb', 'bc', 'a', 'b', 'c', WORD_END]
    assert encode(['abbccc'], vocab) == [8, 9, 5, 10, 11], \
           "Your bpe encoding does not math expected result"
    print("The first test passed!")

    # Second test
    vocab = SPECIAL + ['aaaa', 'aa', 'a', WORD_END]
    assert len(encode(['aaaaaaaa', 'aaaaaaa'], vocab)) == 7, \
           "Your bpe encoding does not math expected result"
    print("The second test passed!")

    print("All 2 tests passed!")

def decoding_test():
    print ("======Decoding Test Case======")
    PAD = BytePairEncoding.PAD_token
    UNK = BytePairEncoding.UNK_token
    CLS = BytePairEncoding.CLS_token
    SEP = BytePairEncoding.SEP_token
    MSK = BytePairEncoding.MSK_token
    SPECIAL = [PAD, UNK, CLS, SEP, MSK]
    WORD_END = BytePairEncoding.WORD_END

    # First test
    vocab = SPECIAL + ['bcc', 'bb', 'bc', 'a', 'b', 'c', WORD_END]
    assert decode([8, 9, 5, 10, 11], vocab) == ['abbccc'], \
           "Your bpe decoding does not math expected result"
    print("The first test passed!")

    # Second test
    vocab = SPECIAL + ['aaaa', 'aa', 'a', WORD_END]
    assert decode([5, 5, 8, 5, 6, 7, 8], vocab) == ['aaaaaaaa', 'aaaaaaa'], \
           "Your BPE decoding does not math expected result"
    print("The second test passed!")

def consistency_test():
    print ("======Consistency Test Case======")
    corpus = ['this is test corpus .',
              'we will check the consistency of your byte pairing encoding .', 
              'you have to pass this test to get full scores .',
              'we hope you to pass tests wihtout any problem .',
              'good luck .']

    vocab = build_bpe(chain.from_iterable(sentence.split() for sentence in corpus), 80)
    
    sentence = 'this is another sentence to test encoding and decoding .'.split()

    assert decode(encode(sentence, vocab), vocab) == sentence, \
            "Your BPE does not show consistency."
    print("The consistency test passed!")

if __name__ == "__main__":
    build_bpe_test()
    encoding_test()
    decoding_test()
    consistency_test()
    