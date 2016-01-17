##########################################################
#
# Spark SQL and Parquet Testing Script
# (C) 2014 Daniel Dixey
#
################# Import Libraries #######################
# Import Spark API Modules
from pyspark import SparkContext
from pyspark.sql import SQLContext
# from pyspark.sql import *
from pyspark.sql import Row, StructField, StructType, StringType, IntegerType
# Import Time for Measuring Processing Times
from time import time
# Import OS Operation
import os.path

############### Initiate Spark Task #####################
# /usr/share/spark-1.1.0/bin/spark-submit /home/dan/Spark_Files/SparkSQL.py > /home/dan/Desktop/output.txt

# Create Table 1 - People Age


def table1():
    #####################################################
    # Import/Create First Schema
    #
    # RDD is created from a list of rows
    some_rdd = sc.parallelize([Row(Person_First_Name="John", age=19),
                               Row(Person_First_Name="Smith", age=23),
                               Row(Person_First_Name="Sarah", age=18)])
    # Infer schema from the first row, create a SchemaRDD and print the schema
    some_schemardd = sqlCtx.inferSchema(some_rdd)
    # Print Schema on Screen
    print('Print the First Schema - People_Age\n')
    some_schemardd.printSchema()
    # Register this SchemaRDD as a table.
    some_schemardd.registerAsTable("People_Age")
    #####################################################
    # Save Data Above as a Parqet File
    # SchemaRDDs can be saved as Parquet files, maintaining the schema
    # information.
    some_schemardd.saveAsParquetFile("/home/dan/Desktop/People_Age.parquet")
    # Return Data
    return some_schemardd.registerAsTable("People_Age")


def table2():
    #####################################################
    # Import/Create Second Schema
    #
    # Another RDD is created from a list of tuples
    another_rdd = sc.parallelize(
        [("John", "England", 120), ("Jenny", "Spain", 45), ("Sarah", "Japan", 55)])
    # Schema with two fields - person_name and person_age
    schema = StructType([StructField("Person_First_Name", StringType(), False),
                         StructField(
        "Person_Location_Country", StringType(), False),
        StructField("Person_Avg_Spend", IntegerType(), False)])
    # Create a SchemaRDD by applying the schema to the RDD and print the schema
    another_schemardd = sqlCtx.applySchema(another_rdd, schema)
    # Print Schema on Screen
    print('Print the Second Schema - People_Details\n')
    another_schemardd.printSchema()
    #####################################################
    # Save Data Above as a Parqet File
    # SchemaRDDs can be saved as Parquet files, maintaining the schema
    # information.
    another_schemardd.saveAsParquetFile(
        "/home/dan/Desktop/People_Details.parquet")
    # Register this SchemaRDD as a table.
    return another_schemardd.registerAsTable("People_Details")

################# Algorithm Development #################
if __name__ == "__main__":
    # Time of the Process
    start_time_overall = time()
    # Initiate Spark
    sc = SparkContext(appName="PythonSQL")
    # Deploy the SQL Module of Spark
    sqlCtx = SQLContext(sc)

    print('Spark SQL and Parquet Testing Script\nTest Script to Build and Test how Spark SQL and Parquet Files can be used\nCreate Date: 17/1/2015\n')

    # Check if Data is Pre-Saved as a Parquet File - People Age
    if os.path.isdir('/home/dan/Desktop/People_Age.parquet') == False:
        # Print Statement
        print('Reading in and creating a Parquet File - People Age')
        # Initiate Command
        table1()
    else:
        # Print Statement
        print('Reading in Pre-made Parquet File')
        # Read in the Parquet file created above.
        parquetFile = sqlCtx.parquetFile(
            "/home/dan/Desktop/People_Age.parquet")
        # Parquet files can also be registered as tables and then used in SQL
        # statements.
        parquetFile.registerTempTable("People_Age")

    # Check if Data is Pre-Saved as a Parquet File - People Details
    if os.path.isdir('/home/dan/Desktop/People_Details.parquet') == False:
        # Print Statement
        print('Reading in and creating a Parquet File - People Details')
        # Initiate Command
        table2()
    else:
        # Print Statement
        print('Reading in Pre-made Parquet File')
        # Read in the Parquet file created above.
        parquetFile = sqlCtx.parquetFile(
            "/home/dan/Desktop/People_Details.parquet")
        # Parquet files can also be registered as tables and then used in SQL
        # statements.
        parquetFile.registerTempTable("People_Details")

    #####################################################
    # Extract First SQL Statement in Spark SQL
    teenagers = sqlCtx.sql(
        "SELECT Person_First_Name, age FROM People_Age WHERE age <= 20").collect()
    # Print First Extract
    print('\n My First SQL Extract - Where Schema has been infered\n')
    # For Loop to Iterate throuch Query line by line
    for r_values in teenagers:
        # Print Statement
        print('Name of Person = %s AND Age = %i') % (r_values[0], r_values[1])
    # Extract First SQL Statement in Spark SQL - check if Second Method Worked
    print('\n My Second SQL Extract - Where Schema has been predefined\n')
    query2 = sqlCtx.sql(
        "SELECT Person_First_Name FROM People_Details").collect()
    # For Loop to Iterate throuch Query line by line
    for names in query2:
        # Print Statement
        print('Name of Person = %s') % (names[0])
    # Inner Join
    print('\nInner Join Example\n')
    inner_join = sqlCtx.sql("SELECT pd.Person_First_Name, pd.Person_Location_Country, pd.Person_Avg_Spend, pa.age \
                                    FROM People_Details pd \
                                    INNER JOIN People_Age pa \
                                    ON pd.Person_First_Name=pa.Person_First_Name").collect()
    # For Loop to Iterate throuch Query line by line
    for row in inner_join:
        # Print Statement
        print('Name = %s, Country = %s, Avg Spend = %i, Age = %i') % (
            row[0], row[1], row[2], row[3])
    # Left Join
    print('\nLeft Join Example\n')
    left_join = sqlCtx.sql("SELECT pd.Person_First_Name, pd.Person_Location_Country, pd.Person_Avg_Spend, pa.age \
                                    FROM People_Details pd \
                                    LEFT OUTER JOIN People_Age pa \
                                    ON pd.Person_First_Name=pa.Person_First_Name").collect()
    # For Loop to Iterate throuch Query line by line
    print('First Name, Country, Avg_Spend, Age')
    for row in left_join:
        # Print Statement
        print row[0], row[1], row[2], row[3]
    # Right Join
    print('\nRight Join Example\n')
    right_join = sqlCtx.sql("SELECT pd.Person_First_Name, pd.Person_Location_Country, pd.Person_Avg_Spend, pa.age \
                                    FROM People_Details pd \
                                    RIGHT OUTER JOIN People_Age pa \
                                    ON pd.Person_First_Name=pa.Person_First_Name").collect()
    # For Loop to Iterate throuch Query line by line
    print('First Name, Country, Avg_Spend, Age')
    for row in right_join:
        # Print Statement
        print row[0], row[1], row[2], row[3]
    # Full Outer Join
    print('\nFull Outer Join Example\n')
    full_outer = sqlCtx.sql("SELECT pd.Person_First_Name, pd.Person_Location_Country, pd.Person_Avg_Spend, pa.age \
                                    FROM People_Details pd \
                                    FULL OUTER JOIN People_Age pa \
                                    ON pd.Person_First_Name=pa.Person_First_Name").collect()
    # For Loop to Iterate throuch Query line by line
    print('First Name, Country, Avg_Spend, Age')
    for row in full_outer:
        # Print Statement
        print row[0], row[1], row[2], row[3]
    #####################################################
    # Wall Clock Time
    print('\nWall Clock Time %.4f') % (time() - start_time_overall)
    # Wall Clock Time 9.1677 - Not using Parquet Files
    # Wall Clock Time 7.7217 - Using Parquet Files

    # Stop Spark
    sc.stop()
