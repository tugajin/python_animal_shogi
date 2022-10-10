import sqlite3
import json
from game import State
DB_NAME = "./data/history.db"

def create_db():
    # すでに存在していれば、それにアスセスする。
    conn = sqlite3.connect(DB_NAME)
    # sqliteを操作するカーソルオブジェクトを作成
    cur = conn.cursor()

    cur.execute("CREATE TABLE history(id INTEGER PRIMARY KEY AUTOINCREMENT, pos STRING, result REAL)")

    # データベースへコミット。これで変更が反映される。
    conn.commit()
    conn.close()

def insert(pos, result):
    # すでに存在していれば、それにアスセスする。
    conn = sqlite3.connect(DB_NAME)
    # sqliteを操作するカーソルオブジェクトを作成
    cur = conn.cursor()

    j = json.dumps(pos)
    cur.execute(f"insert into history(pos, result) values('{j}',{result})")
    # データベースへコミット。これで変更が反映される。
    conn.commit()
    conn.close()

def create_conn():
 # すでに存在していれば、それにアスセスする。
    conn = sqlite3.connect(DB_NAME)
    # sqliteを操作するカーソルオブジェクトを作成
    cur = conn.cursor()
    return conn, cur

def insert2(cur, pos, result):
    j = json.dumps(pos)
    cur.execute(f"insert into history(pos, result) values('{j}',{result})")

def close_conn(conn):
    # データベースへコミット。これで変更が反映される。
    conn.commit()
    conn.close()

def select(id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    res = cur.execute(f"select pos, result from history where id = {id}")
    result = res.fetchone()
    conn.close()
    return result

def select2(cur, id):
    res = cur.execute(f"select pos, result from history where id = {id}")
    result = res.fetchone()
    return result

def select_all():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    res = cur.execute(f"select pos, result from history")
    result = res.fetchall()
    conn.close()
    return result

def select_count_all():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    res = cur.execute(f"select count(1) from history")
    cnt = res.fetchone()
    conn.close()
    return cnt[0]


def delete_all():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    res = cur.execute(f"drop table history")
    conn.commit()
    conn.execute("VACUUM")
    conn.close()

if __name__ == "__main__":
    s = State()
    create_db()
    insert(s.pieces_array(), 1)
    insert(s.pieces_array(), 0)
    insert(s.pieces_array(), -1)
    insert(s.pieces_array(), 0.4)
    print("all:",select_all())
    print("one:",select(1))
    print("cnt:",select_count_all())
    delete_all()