###################################################################
# How to Write a Spelling Corrector
# http://norvig.com/spell-correct.html
###################################################################

###################################################################
# Import Modules
import re
import collections
from time import time

###################################################################
# Define Functions


def words(text): return re.findall('[a-z]+', text.lower())


def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model


def edits1(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    inserts = [a + c + b for a, b in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)


def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)


def known(words): return set(w for w in words if w in NWORDS)


def correct(word):
    candidates = known(
        [word]) or known(
        edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get).capitalize()


def spelltest(tests, bias=None, verbose=False):
    n, bad, unknown, start = time(), 0, 0, time()
    if bias:
        for target in tests:
            NWORDS[target] += bias
    for target, wrongs in tests.items():
        for wrong in wrongs.split():
            n += 1
            w = correct(wrong)
            if w != target:
                bad += 1
                unknown += (target not in NWORDS)
                if verbose:
                    print '%r => %r (%d); expected %r (%d)' % (
                        wrong, w, NWORDS[w], target, NWORDS[target])
    return dict(bad=bad, n=n, bias=bias, pct=float(100. - 100. * bad / n),
                unknown=unknown, secs=float(time() - start))
###################################################################
# Create Vector of Words
NWORDS = train(words(file('/home/dan/Spark_Files/Web/big.txt').read()))
# Define the Alphabet
alphabet = 'abcdefghijklmnopqrstuvwxyz'

###################################################################
tests1 = {'access': 'acess', 'accessing': 'accesing', 'accommodation':
          'accomodation acommodation acomodation', 'account': 'acount'}

tests2 = {'forbidden': 'forbiden', 'decisions': 'deciscions descisions',
          'supposedly': 'supposidly', 'embellishing': 'embelishing'}

print spelltest(tests1)
print spelltest(tests2)  # only do this after everything is debugged

###################################################################
# Facebook
test_text = 'For all my friends, whether close or casual, just because. This is \
             one of the longest posts I will ever make, and one of the most real \
             too. Everyone will go through some hard times at some point. Life \
             isn''t easy. Just something to think about. Did you know the people \
             that are the strongest are usually the most sensitive? Did you know \
             the people who exhibit the most kindness are the first to get \
             mistreated? Did you know the ones who take care of others all the time \
             are usually the ones who need it the most? Did you know the three \
             hardest things to say are I love you, I''m sorry, and help me? \
             Sometimes just because a person looks happy, you have to look past \
             their smile to see how much pain they may be in. To all my friends \
             who are going through some issues right now--let''s start an intentional\
             avalanche. We all need positive intentions right now. If I don''t see \
             your name, I''ll understand. May I ask my friends wherever you might \
             be, to kindly copy and paste this status for one hour to give a moment \
             of support to all of those who have family problems, health struggles, \
             job issues, worries of any kind and just needs to know that someone cares. \
             Do it for all of us, for nobody is immune. I hope to see this on the walls \
             of all my friends just for moral support. I know some will!!! I did \
             it for a friend and you can too. You have to copy and paste this one, \
             NO SHARING... I will leave it in the comments so it is easier \
             for you to copy.'
# Split into Words
test_text = words(test_text)
# Loop through
for word in test_text:
    if correct(word) == word.capitalize():
        print word.capitalize()
    else:
        y = 'CORRECTION TO'
        print word.capitalize(), y, correct(word)
##################################################################
# Import Data
Dilberate = words(
    file('/home/dan/Spark_Files/Web/Spellings_and_Errors.txt').read())
# Create a Subset of 1000 Words
Dilberate2 = Dilberate[0:100]
# Loop through
n = 0
p = 0
list_corrections = []
for word in Dilberate2:
    start_time = time()
    correction = correct(word)
    if correction == word.capitalize():
        p += 1
    else:
        y = 'CORRECTION TO'
        list_corrections.append(
            [word.capitalize(), y, correction, time() - start_time])
        n += 1
print('Number of Corrections: %i   Number of Correctly Spelt Words: %i') % (n, p)
