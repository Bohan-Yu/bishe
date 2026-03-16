"""SQLite 数据库操作模块
表结构: id(PK自增) | news_text | image_path | is_real(0-10) | reason | evidence_url
"""

import os
import sqlite3

import pandas as pd

from app.config import DB_PATH

# CSV 数据集与图片目录（外部，不复制）
CSV_PATH = r"E:\mcfend\news.csv"
EXT_IMG_DIR = r"E:\mcfend\img"


def get_db():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库表"""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            news_text   TEXT    NOT NULL,
            image_path  TEXT    DEFAULT NULL,
            is_real     REAL    DEFAULT NULL,
            reason      TEXT    DEFAULT NULL,
            evidence_url TEXT   DEFAULT NULL
        )
    """)
    conn.commit()
    conn.close()


def insert_news(news_text, image_path=None, is_real=None, reason=None, evidence_url=None):
    """插入一条新闻记录，返回新 ID"""
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO news (news_text, image_path, is_real, reason, evidence_url) VALUES (?,?,?,?,?)",
        (news_text, image_path, is_real, reason, evidence_url),
    )
    new_id = cur.lastrowid
    conn.commit()
    conn.close()
    return new_id


def search_news(query=None, news_id=None, limit=None):
    """按 ID 或关键字搜索新闻"""
    conn = get_db()
    if news_id is not None:
        rows = conn.execute("SELECT * FROM news WHERE id = ?", (int(news_id),)).fetchall()
    elif query:
        sql = "SELECT * FROM news WHERE news_text LIKE ? ORDER BY id DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql, (f"%{query}%",)).fetchall()
    else:
        sql = "SELECT * FROM news ORDER BY id DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_news():
    """获取所有新闻（按 ID 降序）"""
    return search_news()


def get_news_count():
    """获取新闻总数"""
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    conn.close()
    return count


def clear_db():
    """清空新闻表"""
    conn = get_db()
    conn.execute("DELETE FROM news")
    conn.execute("DELETE FROM sqlite_sequence WHERE name='news'")
    conn.commit()
    conn.close()


# ════════════════════════════════════════════════════
#  CSV 导入建库
# ════════════════════════════════════════════════════
def import_csv_to_db():
    """
    从 E:\\mcfend\\news.csv 导入数据建库。
    - content 列为空则用 title 列
    - label='事实' → is_real=10; label='谣言' → is_real=0
    - 跳过 '尚无定论' 行
    - 图片: E:\\mcfend\\img\\{news_id}_0.jpg，存在则记录，路径前缀 ext_img/
    返回 dict: { success, count, skipped, error? }
    """
    if not os.path.exists(CSV_PATH):
        return {"success": False, "count": 0, "error": f"CSV 文件不存在: {CSV_PATH}"}

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        return {"success": False, "count": 0, "error": f"读取 CSV 失败: {e}"}

    # 先清空旧数据
    clear_db()
    init_db()

    conn = get_db()
    count = 0
    skipped = 0

    for _, row in df.iterrows():
        label = str(row.get("label", "")).strip()
        if label == "尚无定论" or label not in ("事实", "谣言"):
            skipped += 1
            continue

        # 文本
        content = row.get("content")
        title = row.get("title", "")
        if pd.isna(content) or str(content).strip() == "":
            news_text = str(title).strip()
        else:
            news_text = str(content).strip()
        if not news_text:
            skipped += 1
            continue

        # 真假得分
        is_real = 10.0 if label == "事实" else 0.0

        # 图片路径
        news_id = str(row.get("news_id", "")).strip()
        pic_url = row.get("pic_url")
        image_path = None
        if not pd.isna(pic_url) and str(pic_url).strip():
            img_file = f"{news_id}_0.jpg"
            if os.path.exists(os.path.join(EXT_IMG_DIR, img_file)):
                image_path = f"ext_img/{img_file}"

        # 证据链接
        url = row.get("url")
        evidence_url = str(url).strip() if not pd.isna(url) else None

        # 原因
        reason = f"数据集标注: {label}"

        conn.execute(
            "INSERT INTO news (news_text, image_path, is_real, reason, evidence_url) VALUES (?,?,?,?,?)",
            (news_text, image_path, is_real, reason, evidence_url),
        )
        count += 1

    conn.commit()
    conn.close()
    return {"success": True, "count": count, "skipped": skipped}


def update_news(news_id, news_text=None, image_path=None, is_real=None, reason=None, evidence_url=None):
    """更新一条新闻记录"""
    conn = get_db()
    fields = []
    values = []
    if news_text is not None:
        fields.append("news_text=?")
        values.append(news_text)
    if image_path is not None:
        fields.append("image_path=?")
        values.append(image_path if image_path else None)
    if is_real is not None:
        fields.append("is_real=?")
        values.append(float(is_real))
    if reason is not None:
        fields.append("reason=?")
        values.append(reason)
    if evidence_url is not None:
        fields.append("evidence_url=?")
        values.append(evidence_url if evidence_url else None)
    if not fields:
        conn.close()
        return False
    values.append(int(news_id))
    conn.execute(f"UPDATE news SET {','.join(fields)} WHERE id=?", values)
    conn.commit()
    conn.close()
    return True


def delete_news(news_id):
    """删除一条新闻记录"""
    conn = get_db()
    conn.execute("DELETE FROM news WHERE id=?", (int(news_id),))
    conn.commit()
    conn.close()
    return True


def search_news_paginated(page=1, per_page=50, query=None):
    """分页查询新闻，返回 (rows, total)"""
    conn = get_db()
    offset = (page - 1) * per_page
    if query:
        total = conn.execute(
            "SELECT COUNT(*) FROM news WHERE news_text LIKE ?", (f"%{query}%",)
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM news WHERE news_text LIKE ? ORDER BY id DESC LIMIT ? OFFSET ?",
            (f"%{query}%", per_page, offset),
        ).fetchall()
    else:
        total = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM news ORDER BY id DESC LIMIT ? OFFSET ?",
            (per_page, offset),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows], total
