"""
Flask 后端入口 —— 提供聊天 API / SSE 流式 API / 数据库查询 API / CSV导入 API 以及前端页面
"""

import os
import base64
import uuid
import json

from flask import Flask, request, jsonify, render_template, send_from_directory, Response

from app.config import UPLOAD_FOLDER, PROJECT_ROOT
from app.database import (
    init_db, search_news, import_csv_to_db, get_news_count,
    EXT_IMG_DIR, insert_news, update_news, delete_news, clear_db,
    search_news_paginated,
)
from app.agent import fact_check_pipeline, fact_check_pipeline_stream
from app.source_credibility import list_source_classifications, update_source_classification

# ────────────────────── Flask 应用 ──────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


# ────────────────────── 页面路由 ──────────────────────
@app.route("/")
def chat_page():
    """聊天查证页面"""
    return render_template("chat.html")


@app.route("/database")
def database_page():
    """数据库展示页面"""
    return render_template("db_view.html")


@app.route("/source-credibility")
def source_credibility_page():
    """信源分类知识库页面"""
    return render_template("source_credibility.html")


# ────────────────────── API 路由 ──────────────────────
@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    聊天查证接口
    接收: form-data  text=... & image=文件(可选)
    返回: JSON { steps, final_result, db_updated, new_id }
    """
    news_text = request.form.get("text", "").strip()
    if not news_text:
        return jsonify({"error": "请输入新闻文本"}), 400

    use_db = request.form.get("use_db", "true").lower() in ("true", "1", "yes")

    img_base64 = None
    image_path = None
    image_file = request.files.get("image")

    if image_file and image_file.filename:
        # 保存上传图片
        ext = image_file.filename.rsplit(".", 1)[-1] if "." in image_file.filename else "jpg"
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(UPLOAD_FOLDER, unique_name)
        image_file.save(save_path)
        image_path = f"uploads/{unique_name}"

        with open(save_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")

    result = fact_check_pipeline(news_text, img_base64, image_path, use_db=use_db)
    return jsonify(result)


@app.route("/api/chat_stream", methods=["POST"])
def api_chat_stream():
    """
    SSE 流式查证接口
    接收: form-data  text=... & image=文件(可选)
    返回: text/event-stream，每行为一个 JSON 事件
    """
    news_text = request.form.get("text", "").strip()
    if not news_text:
        return jsonify({"error": "请输入新闻文本"}), 400

    use_db = request.form.get("use_db", "true").lower() in ("true", "1", "yes")

    img_base64 = None
    image_path = None
    image_file = request.files.get("image")

    if image_file and image_file.filename:
        ext = image_file.filename.rsplit(".", 1)[-1] if "." in image_file.filename else "jpg"
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(UPLOAD_FOLDER, unique_name)
        image_file.save(save_path)
        image_path = f"uploads/{unique_name}"
        with open(save_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")

    def generate():
        try:
            for evt in fact_check_pipeline_stream(news_text, img_base64, image_path, use_db=use_db):
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        except Exception as exc:
            error_evt = {
                "type": "final",
                "source": "main_agent",
                "message": "📊 最终判断结果",
                "data": {
                    "classification": None,
                    "reason": f"流式查证失败: {exc}",
                    "evidence_url": "",
                    "claim_verdicts": [],
                    "has_contradiction": False,
                    "cross_verify_score": 5.0,
                },
            }
            yield f"data: {json.dumps(error_evt, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/news", methods=["GET"])
def api_news():
    """
    数据库分页查询接口
    参数: ?id=xxx  或  ?q=关键字  ?page=1  ?per_page=50
    """
    news_id = request.args.get("id")
    query = request.args.get("q", "").strip()
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)

    if news_id:
        rows = search_news(news_id=news_id)
        return jsonify({"news": rows, "total": len(rows), "page": 1, "per_page": per_page})

    rows, total = search_news_paginated(page=page, per_page=per_page, query=query or None)
    return jsonify({"news": rows, "total": total, "page": page, "per_page": per_page})


@app.route("/api/news", methods=["POST"])
def api_add_news():
    """手动添加一条新闻"""
    data = request.get_json(force=True)
    news_text = (data.get("news_text") or "").strip()
    if not news_text:
        return jsonify({"error": "新闻文本不能为空"}), 400
    image_path = (data.get("image_path") or "").strip() or None
    is_real = data.get("is_real")
    if is_real is not None and is_real != "":
        is_real = float(is_real)
    else:
        is_real = None
    reason = (data.get("reason") or "").strip() or None
    evidence_url = (data.get("evidence_url") or "").strip() or None
    new_id = insert_news(news_text, image_path, is_real, reason, evidence_url)
    return jsonify({"success": True, "new_id": new_id})


@app.route("/api/news/<int:nid>", methods=["PUT"])
def api_update_news(nid):
    """编辑一条新闻"""
    data = request.get_json(force=True)
    update_news(
        nid,
        news_text=data.get("news_text"),
        image_path=data.get("image_path"),
        is_real=data.get("is_real"),
        reason=data.get("reason"),
        evidence_url=data.get("evidence_url"),
    )
    return jsonify({"success": True})


@app.route("/api/news/<int:nid>", methods=["DELETE"])
def api_delete_news(nid):
    """删除一条新闻"""
    delete_news(nid)
    return jsonify({"success": True})


@app.route("/api/news/all", methods=["DELETE"])
def api_delete_all_news():
    """清空所有新闻"""
    clear_db()
    init_db()
    return jsonify({"success": True})


@app.route("/api/import_csv", methods=["POST"])
def api_import_csv():
    """从 CSV 数据集导入数据建库"""
    result = import_csv_to_db()
    return jsonify(result)


@app.route("/api/db_count")
def api_db_count():
    """获取数据库记录总数"""
    return jsonify({"count": get_news_count()})


@app.route("/api/source-credibility", methods=["GET"])
def api_source_credibility():
    """获取信源分类知识库，支持搜索与按层级筛选。"""
    query = request.args.get("q", "").strip()
    tier = request.args.get("tier", "").strip().lower()
    scope = request.args.get("scope", "all").strip().lower()
    country = request.args.get("country", "").strip()
    return jsonify(list_source_classifications(query=query, tier=tier, scope=scope, country=country))


@app.route("/api/source-credibility/<path:domain>", methods=["PUT"])
def api_update_source_credibility(domain):
    """更新某一条信源分类记录。"""
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    tier = (data.get("tier") or "unknown").strip().lower()
    reason = (data.get("reason") or "").strip()
    country = (data.get("country") or "").strip()
    if not name:
        return jsonify({"error": "来源名不能为空"}), 400

    updated = update_source_classification(
        domain=domain,
        name=name,
        tier=tier,
        reason=reason,
        country=country,
    )
    if not updated:
        return jsonify({"error": "更新失败，请检查域名或层级"}), 400
    return jsonify({"success": True, "item": updated})


# 提供上传图片的静态访问
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# 提供外部图片目录 (E:\mcfend\img) 的静态访问
@app.route("/ext_img/<path:filename>")
def ext_img_file(filename):
    return send_from_directory(EXT_IMG_DIR, filename)


# ────────────────────── 启动 ──────────────────────
if __name__ == "__main__":
    init_db()
    print("=" * 50)
    print("  假新闻检测系统已启动")
    print("  聊天页面: http://127.0.0.1:5000/")
    print("  数据库页面: http://127.0.0.1:5000/database")
    print("  信源知识库: http://127.0.0.1:5000/source-credibility")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)
