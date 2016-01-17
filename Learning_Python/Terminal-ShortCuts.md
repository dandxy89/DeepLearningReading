#### Shortcuts

# Updated: 17/1/2015

# Spark Directory
cd /usr/share/spark-1.1.0/

# Running a Spark Job - Boots and loads server on the fly.
./bin/spark-submit /home/dan/Spark_Files/wordcount.py /home/dan/Spark_Files/emma.txt

# Start IPython Notebook from Terminal
ipython notebook

# Connectecting to a Server via SSH
ssh linux.example.com -l user_id

# Connecting to the Uni Data Centre
ssh lewes.example.com -l user_id

# Start IPython Notebook in the Spark Java Virtual Machine
IPYTHON_OPTS="notebook" ./bin/pyspark

# Start IPython in the Spark Java Virtual Machine
IPYTHON=1 ./bin/pyspark

# Removing Files from Git
git ls-files --deleted -z | xargs -0 git rm 

