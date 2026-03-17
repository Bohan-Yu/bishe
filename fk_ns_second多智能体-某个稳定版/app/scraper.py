"""
爬虫模块 —— 从中文辟谣平台爬取假新闻数据建库

目标网站: 中国互联网联合辟谣平台 (piyao.org.cn)
备选方案: 使用 Tavily 搜索中文假新闻/辟谣文章
"""

import os
import hashlib
import requests
from bs4 import BeautifulSoup

from app.config import UPLOAD_FOLDER, TAVILY_API_KEY
from app.database import insert_news

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def _download_image(url: str) -> str | None:
    """下载图片到 static/uploads，返回相对路径"""
    try:
        if not url.startswith("http"):
            return None
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200 or len(resp.content) < 1024:
            return None
        ext = url.split(".")[-1].split("?")[0][:5]
        if ext.lower() not in ("jpg", "jpeg", "png", "gif", "webp"):
            ext = "jpg"
        filename = hashlib.md5(url.encode()).hexdigest() + "." + ext
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, "wb") as f:
            f.write(resp.content)
        return f"uploads/{filename}"
    except Exception:
        return None


# ═══════════════ 方案 1: 爬取辟谣平台 ═══════════════
def scrape_piyao(max_articles: int = 10) -> dict:
    """
    爬取中国互联网联合辟谣平台 (piyao.org.cn) 的「今日辟谣」栏目。
    辟谣平台中发布的内容均为已证实的谣言，因此 is_real 标记为低分(假新闻)。
    """
    base_url = "https://www.piyao.org.cn"
    # 尝试多个可能的列表页
    list_urls = [
        f"{base_url}/jrpy/index.htm",
        f"{base_url}/jrpy/index_1.htm",
        f"{base_url}/yybgt/index.htm",
    ]

    article_links = []
    for list_url in list_urls:
        try:
            resp = requests.get(list_url, headers=HEADERS, timeout=15)
            resp.encoding = "utf-8"
            soup = BeautifulSoup(resp.text, "html.parser")

            for a_tag in soup.find_all("a", href=True):
                href = a_tag.get("href", "")
                title = a_tag.get_text(strip=True)
                # 过滤有效的文章链接
                if (
                    title
                    and len(title) > 8
                    and (".shtml" in href or "/c/" in href or "/t/" in href)
                ):
                    full_url = href if href.startswith("http") else base_url + href
                    article_links.append({"title": title, "url": full_url})
        except Exception:
            continue

    # 去重
    seen = set()
    unique_articles = []
    for art in article_links:
        if art["url"] not in seen:
            seen.add(art["url"])
            unique_articles.append(art)
    article_links = unique_articles[:max_articles]

    count = 0
    for article in article_links:
        try:
            art_resp = requests.get(article["url"], headers=HEADERS, timeout=15)
            art_resp.encoding = "utf-8"
            art_soup = BeautifulSoup(art_resp.text, "html.parser")

            # 提取正文
            content_div = (
                art_soup.find("div", class_="content")
                or art_soup.find("div", class_="TRS_Editor")
                or art_soup.find("div", class_="article-content")
                or art_soup.find("article")
                or art_soup.find("div", id="content")
            )
            content = ""
            if content_div:
                content = content_div.get_text(strip=True)[:1000]
            if not content:
                content = article["title"]

            news_text = f"{article['title']}\n{content}"

            # 提取第一张有效图片
            img_path = None
            if content_div:
                for img_tag in content_div.find_all("img", src=True):
                    img_url = img_tag["src"]
                    if not img_url.startswith("http"):
                        img_url = base_url + img_url
                    downloaded = _download_image(img_url)
                    if downloaded:
                        img_path = downloaded
                        break

            # 辟谣平台文章 = 已证实为谣言 → is_real 设为 2.0 (低分=假)
            insert_news(
                news_text=news_text,
                image_path=img_path,
                is_real=2.0,
                reason="来源：中国互联网联合辟谣平台，该内容已被官方辟谣，属于谣言/假新闻",
                evidence_url=article["url"],
            )
            count += 1
        except Exception as e:
            print(f"[scraper] 文章爬取失败 {article['url']}: {e}")
            continue

    return {"success": count > 0, "count": count, "method": "piyao"}


# ═══════════════ 方案 2: Tavily 搜索建库 ═══════════════
def scrape_with_tavily(max_articles: int = 10) -> dict:
    """
    使用 Tavily 搜索中文辟谣/假新闻文章来建库。
    搜索词涵盖典型的辟谣关键词，返回的文章大多为假新闻或辟谣。
    """
    from tavily import TavilyClient

    client = TavilyClient(api_key=TAVILY_API_KEY)

    queries = [
        "中文假新闻 辟谣 谣言",
        "网传谣言 事实核查 虚假信息",
        "辟谣 不实消息 假消息",
    ]

    count = 0
    seen_urls = set()
    for query in queries:
        if count >= max_articles:
            break
        try:
            results = client.search(
                query=query,
                max_results=5,
                include_raw_content=False,
                topic="news",
            )
            for r in results.get("results", []):
                if count >= max_articles:
                    break
                url = r.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                title = r.get("title", "")
                content = r.get("content", "")
                news_text = f"{title}\n{content}"

                insert_news(
                    news_text=news_text,
                    image_path=None,
                    is_real=2.0,
                    reason="来源：网络搜索辟谣文章，该内容为已辟谣的虚假信息",
                    evidence_url=url,
                )
                count += 1
        except Exception as e:
            print(f"[scraper-tavily] 搜索失败: {e}")
            continue

    return {"success": count > 0, "count": count, "method": "tavily"}


# ═══════════════ 统一入口 ═══════════════
def run_scraper(max_articles: int = 10) -> dict:
    """
    统一爬虫入口：先尝试爬取辟谣平台，若失败回退到 Tavily 搜索。
    """
    result = scrape_piyao(max_articles)
    if result["count"] > 0:
        return result

    # 辟谣平台爬取失败，使用 Tavily
    print("[scraper] 辟谣平台爬取失败，切换到 Tavily 搜索模式")
    return scrape_with_tavily(max_articles)
