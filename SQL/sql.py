import sqlite3
import os


connection= sqlite3.connect("student.db")
cursor= connection.cursor()

table_info = """
CREATE TABLE STUDENT(NAME VARCHAR(25), CLASS VARCHAR(25), SECTION VARCHAR(25), MARKS INT);
"""
cursor.execute(table_info)
#Create 10 entries with Indian names like Rahul Varun etc give them marks from zero to 100 classes from first to 12th section from A to E
cursor.execute('''INSERT INTO STUDENT VALUES('Rahul', '10', 'A', '90')''')
cursor.execute('''INSERT INTO STUDENT VALUES('Varun', '10', 'B', '85')''')
cursor.execute('''INSERT INTO STUDENT VALUES('Riya', '9', 'C', '92')''')
cursor.execute('''INSERT INTO STUDENT VALUES('Aman', '8', 'D', '78')''')
cursor.execute('''INSERT INTO STUDENT VALUES('Neha', '7', 'E', '95')''')
cursor.execute('''INSERT INTO STUDENT VALUES('Sachin', '6', 'A', '80')''')
cursor.execute('''INSERT INTO STUDENT VALUES('Priya', '5', 'B', '88')''')
cursor.execute('''INSERT INTO STUDENT VALUES('Karan', '4', 'C', '75')''')
cursor.execute('''INSERT INTO STUDENT VALUES('Anjali', '3', 'D', '98')''')
cursor.execute('''INSERT INTO STUDENT VALUES('Raj', '2', 'E', '82')''')
cursor.execute('''INSERT INTO STUDENT VALUES('Pooja', '1', 'A', '93')''')

print("Records")
data=cursor.execute("SELECT * FROM STUDENT")

for row in data:
    print(row)



    # Commit the changes to the database
connection.commit()

    # Close the database connection
connection.close()



