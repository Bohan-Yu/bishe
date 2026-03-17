import argparse
import os
import sys
from typing import Any

import requests
from dotenv import load_dotenv


DEFAULT_QUERY = "【伊朗目前仅剩约100个可正常使用的导弹发射器。】这个新闻是真的还是假的？请给我你具体多个搜索语句，以及怎么判断的具体思考流程。"
ALLOWED_SEARCH_ENGINES = {
    "search_std",
    "search_pro",
    "search_pro_quark",
    "search_pro_sogou",
}


def build_payload(query: str, search_engine: str) -> dict[str, Any]:
    return {
        "model": os.getenv("LLM_MODEL2", "glm-4.5-air"),
        "messages": [
            {"role": "user", "content": query},
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
        "stream": False,
        "tools": [
            {
                "type": "web_search",
                "web_search": {
                    "enable": "True",
                    "search_engine": search_engine,
                    "search_result": "True",
                    "search_prompt": (
                        "Please answer in Chinese using the network search results {search_result}. "
                        "Summarize the key findings and cite the source in the answer."
                    ),
                    "count": "5",
                    "search_recency_filter": "noLimit",
                    "content_size": "medium",
                },
            }
        ],
    }


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Call glm-4.5-air with Zhipu built-in web search."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help="Search question for the model.",
    )
    parser.add_argument(
        "--search-engine",
        default="search_std",
        choices=sorted(ALLOWED_SEARCH_ENGINES),
        help="Built-in web search engine code.",
    )
    parser.add_argument(
        "--print-raw",
        action="store_true",
        help="Print the full JSON response for debugging.",
    )
    args = parser.parse_args()

    api_key = os.getenv("SF_API_KEY")
    base_url = os.getenv("SF_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").rstrip("/")

    if not api_key:
        print("Missing SF_API_KEY in .env", file=sys.stderr)
        return 1

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=build_payload(args.query, args.search_engine),
        timeout=120,
    )

    if response.status_code != 200:
        print(f"Request failed: HTTP {response.status_code}", file=sys.stderr)
        print(response.text, file=sys.stderr)
        return 1

    data = response.json()
    if args.print_raw:
        print("=== Raw JSON ===")
        print(response.text)
        print()

    choice = data["choices"][0]
    message = choice.get("message", {})
    answer = message.get("content") or ""
    web_search = data.get("web_search") or []

    print("=== Model ===")
    print(data.get("model"))
    print()
    print("=== Answer ===")
    print(answer.strip() or "<empty>")
    print()
    print("=== Search Results Used ===")
    if not web_search:
        print("No web_search results were returned.")
        return 0

    for index, item in enumerate(web_search, start=1):
        title = item.get("title") or "<no title>"
        link = item.get("link") or "<no link>"
        media = item.get("media") or "<unknown media>"
        publish_date = item.get("publish_date") or "<unknown date>"
        refer = item.get("refer") or str(index)
        content = (item.get("content") or "").strip()

        print(f"[{refer}] {title}")
        print(f"    media: {media}")
        print(f"    date : {publish_date}")
        print(f"    link : {link}")
        if content:
            print(f"    note : {content}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())