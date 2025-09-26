# File: project/db.py  (NEW)
import sqlite3, json
from pathlib import Path
DB_PATH = Path(__file__).resolve().parent / "app.db"

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY,
        ts TEXT, annot_id TEXT, h5_index INTEGER,
        prob REAL, pred INTEGER, threshold REAL,
        agg_method TEXT, meta_json TEXT
    )""")
    con.commit(); con.close()

def log_prediction(ts, annot_id, h5_index, prob, pred, threshold, agg_method, meta: dict):
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("INSERT INTO predictions(ts,annot_id,h5_index,prob,pred,threshold,agg_method,meta_json) VALUES (?,?,?,?,?,?,?,?)",
                (ts, annot_id, int(h5_index) if h5_index is not None else None,
                 float(prob), int(pred), float(threshold), agg_method, json.dumps(meta or {})))
    con.commit(); con.close()
