############################################
# Daniel Dixey
# Datagenic - Interest in Sentiment Analysis
# 28/4/15
############################################

# Import Modules
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import time
from pattern.web import Twitter


def Harry_Potter_Review():
    # Start Stop Watch
    t1 = time.time()

    # Demo Text
    text = '''
    Harry Potter and the Philosopher's stone is a book about a boy called Harry and when he is a baby something terrible happened to his parents. This very evil wizard called Voldemort killed his mum and dad however he tried to kill Harry but somehow he could not. Therefore Harry had to go to live with his aunt and uncle. Eleven years later he had a letter saying he is invited to go to Hogwarts. Harry travelled there on a scarlet red steam engine. At Hogwarts Harry Ron and Hermione, Harry's friends, were caught out of school and their punishment was to collect unicorn blood from the dark woods. Then Harry and his friends go on a big adventure!\n\n
    I think the book is very exciting! My favourite part is when Harry and his friends go on a very exciting adventure!\n\n
    I think this book is suitable for eight and above. Eleven out of eleven people from Bancffosfelen school said they loved the book! My mark out of ten is nine!\n\n
    '''

    # Process the Text using the NLTK through Textblob
    blob = TextBlob(text)

    # Check the Text has been imported correctly
    # print(text)

    # Iterate Through Each Sentence and calculate the Sentiment
    for sentence in blob.sentences:
        print(sentence)
        print('Using Naive Bayes - The Sentence Above is: %s') % \
            (TextBlob(sentence.string,
                      analyzer=NaiveBayesAnalyzer()).sentiment[0].upper())

    # Translate Text to French
    # print(blob.translate(to="fr") )

    return time.time() - t1


def Pattern_Module_Twitter_Stream():

     # Start Stop Watch
    t1 = time.time()

    # Create a list to Store the Data
    List = Twitter().stream('#Fail')

    # For 10 Instances
    for second in range(10):
        # Get Stream Data
        value = List.update(bytes=1024)
        # Add Value to List if not Empty
        if len(value) == 0:
            # Pass
            continue
        else:
            # Storing Results
            List.append()
            # Print Tweet
            print('Tweet: %s') % (value.text)
            # Get Sentiment
            print('Sentiment Analysis of Tweet: %s') % (TextBlob(str(value.text),
                                                                 analyzer=NaiveBayesAnalyzer()).sentiment[0].upper())
        # Wait 3 Seconds between queries - Do not want to get blocked
        time.sleep(3)

    return time.time() - t1

# Define the Main Function
if __name__ == "__main__":

    # Execute the Harry Potter Function
    timeF = Harry_Potter_Review()

    print('Time to Complete: %.3f Seconds') % (timeF)

    # Connect and Retrieve a Twitter Stream
    timeF1 = Pattern_Module_Twitter_Stream()

    print('Time to Complete: %.3f Seconds') % (timeF1)
