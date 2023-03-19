import argparse
import time

from classes.PathProvider import PathProvider
from classes.SearchUser import SearchUser
from mysql.connector import connect, Error


def worker(path_provider):
    count = get_count_user()[0]
    while (True):
        value = select_queue()
        if value is not None:
            img_path = value[1]
            queue_id = value[0]
            newCount = get_count_user()[0]
            if count != newCount:
                path_provider = PathProvider(select_users())
                count = newCount
            temp = SearchUser(path_provider).recognition(img_path)
            id = temp.get_id_user()
            if id is None:
                id = 0
            insert_queue_user(id, queue_id, temp.percent)
        time.sleep(3)

def insert_queue_user(user_id, queue_id, percent):
    try:
        with connect(
                host="localhost",
                user='home',
                password='home',
                database="hakaton2023",
        ) as connection:
            insert = f"update queue set user_id = {user_id}, percent = {percent} where id = {queue_id}"
            with connection.cursor() as cursor:
                cursor.execute(insert)
                connection.commit()
            # for i in other_users:
            #     insert = f"insert into other_users () user_id = {user_id} where id = {queue_id}"
            #     with connection.cursor() as cursor:
            #         cursor.execute(insert)
            #         connection.commit()

    except Error as e:
        print(e)

def get_count_user():
    try:
        with connect(
                host="localhost",
                user='home',
                password='home',
                database="hakaton2023",
        ) as connection:
            select_movies_query = "SELECT count(*) FROM users"
            with connection.cursor() as cursor:
                cursor.execute(select_movies_query)
                return cursor.fetchone()

    except Error as e:
        print(e)

def select_queue():
    try:
        with connect(
                host="localhost",
                user='home',
                password='home',
                database="hakaton2023",
        ) as connection:
            select_movies_query = "SELECT * FROM queue where user_id is null order by id asc LIMIT 1"
            with connection.cursor() as cursor:
                cursor.execute(select_movies_query)
                return cursor.fetchone()

    except Error as e:
        print(e)

def select_users():
    try:
        with connect(
                host="localhost",
                user='home',
                password='home',
                database="hakaton2023",
        ) as connection:
            select_movies_query = "SELECT id FROM users"
            with connection.cursor() as cursor:
                cursor.execute(select_movies_query)
                ids = []
                for row in cursor.fetchall():
                   ids.append(row[0])
                return ids

    except Error as e:
        print(e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    path_provider = PathProvider(select_users())
    worker(path_provider)


    # temp = search_user.recognition('noname.jpg', 'unknown/').get_result_image()
    # cv2.imshow('image', temp)
    # cv2.waitKey(0)

    # create_movies_table_query = f"""
    #             insert into dataset (img_path, user_id) values ("{temp}", {args.userId})
    #         """
    # with connection.cursor() as cursor:
    #     cursor.execute(create_movies_table_query)
    #     connection.commit()



    # cv2.imshow('image', search_user.recognition('1.jpg', 'unknown/').get_result_image())
    # cv2.waitKey(0)
