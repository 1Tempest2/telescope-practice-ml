import sqlite3
import pandas as pd

connection = sqlite3.connect("FINAL_ASSIGNMENT.db")
cursor_obj = connection.cursor()
df = pd.read_csv("Data/ChicagoCensusData.csv")
df.to_sql("SOCIOECONOMIC", connection, if_exists="replace", index=False)
df = pd.read_csv("Data/ChicagoPublicSchools.csv")
df.to_sql("PUBLIC_SCHOOLS", connection, if_exists="replace", index=False)
df = pd.read_csv("Data/ChicagoCrimeData.csv")
df.to_sql("CRIME", connection, if_exists="replace", index=False)

cursor_obj.execute("SELECT COUNT(*) FROM CRIME")
total_num_of_crimes = cursor_obj.fetchone()[0]
cursor_obj.execute("SELECT COMMUNITY_AREA_NUMBER ,COMMUNITY_AREA_NAME FROM SOCIOECONOMIC WHERE PER_CAPITA_INCOME < 11000")
poor_areas = cursor_obj.fetchall()
cursor_obj.execute("SELECT CASE_NUMBER FROM CRIME WHERE DESCRIPTION LIKE '%minor%' ")
pedos = cursor_obj.fetchall()
cursor_obj.execute("SELECT * FROM CRIME WHERE DESCRIPTION LIKE '%CHILD ABDUCTION%' ")
abductions = cursor_obj.fetchall()
cursor_obj.execute("SELECT DISTINCT PRIMARY_TYPE FROM CRIME WHERE LOCATION_DESCRIPTION LIKE '%SCHOOL%'")
school_problems = cursor_obj.fetchall()
cursor_obj.execute("SELECT SCHOOL_TYPE, AVG(SAFETY_SCORE) FROM PUBLIC_SCHOOLS GROUP BY SCHOOL_TYPE")
average_safety = cursor_obj.fetchall()
cursor_obj.execute("SELECT COMMUNITY_AREA_NAME FROM SOCIOECONOMIC ORDER BY PERCENT_HOUSEHOLDS_BELOW_POVERTY desc LIMIT 5")
below_poverty = cursor_obj.fetchall()
cursor_obj.execute("SELECT COMMUNITY_AREA_NUMBER AS TOTAL_CRIMES FROM CRIME GROUP BY COMMUNITY_AREA_NUMBER ORDER BY COUNT(*) DESC LIMIT 1")
prone_to_crime = cursor_obj.fetchall()
print(prone_to_crime)