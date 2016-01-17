__author__ = 'dan'

"""SimpleApp.py"""
from pyspark import SparkContext

# Should be some file on your system
logFile = "/usr/share/spark-1.1.0/README.md"
sc = SparkContext("local", "Simple App")
logData = sc.textFile(logFile).cache()

numAs = logData.filter(lambda s: 'a' in s).count()
numBs = logData.filter(lambda s: 'b' in s).count()

print "Lines with a: %i, lines with b: %i" % (numAs, numBs)
