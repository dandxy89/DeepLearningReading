#
# Python to MySQL Test Script - Getting Table Names and Columns within.
# (C) 2014 Daniel Dixey
#

# Import Modules
import MySQLdb as mdb
import pandas as pd

# Function to Extract the Version of MySQL in Use


def mysql_version():
    # Determine the Version of MySQL in Use
    cur.execute("SELECT VERSION()")
    # Collect Query
    ver = cur.fetchone()
    # Print Version on Screen
    print("\nDatabase version : %s \n") % (ver)

# Function to Extract the Table and Columns Pairs


def show_tables_col(Database_Name, DB_List):
    # Test the Server which Database you want to use
    cur.execute('use ' + Database_Name)
    # Query the Database to get Names of Tables
    database_tables = cur.execute('show tables')
    # Collect Data
    database_tables = cur.fetchall()
    # Look Through Each Table and Obtain the Pairs
    for tuples in database_tables:
        for table in tuples:
            # Query the Database to Obtain the Names of the Colummns in the
            # Tables
            columns = cur.execute('show columns in ' + str(table) + '')
            # Collect Data
            columns = cur.fetchall()
            # Append the Name of Table and the Column name into a List
            for col_name in columns:
                # Appending of List
                DB_List.append([Database_Name, table, col_name[0]])
    # Return the Output of the List
    return DB_List

# Function to Extract the Names of the Databases on the Server


def Databases_in_Server():
    # Query the Server to Obtain the Names of the Databases
    db = cur.execute('show databases')
    # Get all the Names
    db = cur.fetchall()
    # Create a List of the Database Name, Table Name and Column Names
    DB_List = []
    # Look Through Each Database and get Names of Tables and Columns
    for db_name in db:
        DB_List = show_tables_col(db_name[0], DB_List)
    # Return the Data
    return DB_List

# Start Algorithm:
if __name__ == "__main__":
    # Define Connection inputs
    Host = 'localhost'
    User = 'root'
    Password = 'asd1'
    # Connect to MySQL Server
    con = mdb.connect(Host, User, Password)
    # The cursor is used to traverse the records from the result set.
    cur = con.cursor()
    # Determine the Version of MySQL in Use
    mysql_version()
    # Obtain the List of DB Name, Table Name and Col Name
    DB_List = Databases_in_Server()
    # Import in Dataframe for Distribution
    Table_Col_DF = pd.DataFrame(DB_List)
    # Change Column Names
    Table_Col_DF.columns = ['Database_Name', 'Table_Name', 'Column_Name']
    # Show First 10 Rows
    print Table_Col_DF.head(10)
    # Obtain the Size of Table
    print(
        '\nNumber of Combinations of Database, Table and Columns: %i\n') % Table_Col_DF.shape[0]
    # Close Connection
    con.close()

# Example Output

# Database version : 5.5.41-0ubuntu0.14.10.1
#
#        Database_Name      Table_Name           Column_Name
# 0  information_schema  CHARACTER_SETS    CHARACTER_SET_NAME
# 1  information_schema  CHARACTER_SETS  DEFAULT_COLLATE_NAME
# 2  information_schema  CHARACTER_SETS           DESCRIPTION
# 3  information_schema  CHARACTER_SETS                MAXLEN
# 4  information_schema      COLLATIONS        COLLATION_NAME
# 5  information_schema      COLLATIONS    CHARACTER_SET_NAME
# 6  information_schema      COLLATIONS                    ID
# 7  information_schema      COLLATIONS            IS_DEFAULT
# 8  information_schema      COLLATIONS           IS_COMPILED
# 9  information_schema      COLLATIONS               SORTLEN
#
# Number of Combinations of Database, Table and Columns: 811
