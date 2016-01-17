import string

list1 = ['da"!!!n', 'job??', 'dan#][ny']


def punctuationWord(x):
    # Remove Punctuation from Word
    for letter in string.punctuation:
        x = x.replace(letter, '')
    return str(x)

for word in list1:
    print punctuationWord(word)
