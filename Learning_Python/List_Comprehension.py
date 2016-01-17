# Start
sentence = "the quick brown fox jumps over the lazy dog"
words = sentence.split()
word_lengths = []
for word in words:
    if word != "the":
        word_lengths.append(len(word))

# Final
sentence = "the quick brown fox jumps over the lazy dog"
words = sentence.split()
print [len(x) for x in words if x != "the"]
