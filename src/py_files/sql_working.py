

import pyodbc

# SQL server connection details
server = r"DESKTOP-6SIQQDV\INSTANCE2022"
database = "ABC_Company"
username = "sa"
password = "Sagar@12"

# Connection string for SQL Authentication
conn_str = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password};"
    f"Encrypt=no;"
)

try:
    # connect to database
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    # execute query
    cursor.execute("SELECT TOP 5 * FROM EMPLOYEES")
    # ferch rows
    rows = cursor.fetchall()

    print("first 5 records: \n")
    for row in rows:
        print(row)

except Exception as e:
    print(f" error found: {e}")

finally:
    try:
        cursor.close()
        conn.close()
    except:
        pass 
