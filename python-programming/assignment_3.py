import pymysql

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm


db = pymysql.connect("localhost", "root", "Anly560Anly560!", "sakila")
cursor = db.cursor()
sql = """select distinct sf.title, sf.description, sa.first_name, sa.last_name from sakila.film as sf
left join sakila.film_actor as sfa on sf.film_id = sfa.film_id
left join sakila.actor as sa on sa.actor_id = sfa.actor_id 
where sf.title like 'BL%'"""

try:
    cursor.execute(sql)
    results = cursor.fetchall()

    print("title\tdescription\tfirst_name\tlast_name")

    for column in results:
        title = column[0]
        description = column[1]
        first_name = column[2]
        last_name = column[3]
        data_array = np.array([title, description, first_name, last_name])

        print("%s\t%s\t%s\t%s" % \
              (title, description, first_name, last_name))
except:
    print("ERROR: unable to fetch data")

db.close()