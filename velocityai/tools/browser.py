import re
import sys
import urllib.request
import urllib.parse
from html.parser import HTMLParser
from typing import List, Dict, Optional

class HTMLTextExtractor(HTMLParser):
    """Clean HTML text extractor that ignores scripts, styles, and extracts readable text."""
    def __init__(self):
        super().__init__()
        self.result = []
        self.skip_tags = {'script', 'style', 'head', 'title', 'meta', '[document]', 'noscript'}
        self.in_skip_tag = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.skip_tags:
            self.in_skip_tag = True
        elif tag.lower() in ('p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'br', 'tr'):
            self.result.append('\n')

    def handle_endtag(self, tag):
        if tag.lower() in self.skip_tags:
            self.in_skip_tag = False
        elif tag.lower() in ('p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'tr'):
            self.result.append('\n')

    def handle_data(self, data):
        if not self.in_skip_tag:
            text = data.strip()
            if text:
                self.result.append(text + ' ')

    def get_text(self) -> str:
        raw = ''.join(self.result)
        # Normalize multiple spaces and multiple blank lines
        raw = re.sub(r'[ \t]+', ' ', raw)
        raw = re.sub(r'\n\s*\n', '\n\n', raw)
        return raw.strip()


class BrowserTool:
    """
    Permission-Gated Web Browser and Documentation Search Tool.
    Zero external dependencies (uses standard library urllib and html.parser).
    Guarantees 100% offline privacy unless explicitly granted permission by user.
    """
    def __init__(self, allow_internet: bool = False):
        self.allow_internet = allow_internet
        self.user_agent = "VelocityAI-Agent/1.0 (Apple Silicon; Native Inference)"

    def check_or_request_permission(self, intent: str = "search documentation") -> bool:
        """Checks if internet is allowed; prompts user if running in interactive terminal."""
        if self.allow_internet:
            return True

        if sys.stdin.isatty():
            print(f"\n\033[33m\033[1m[🔒 Permission Request]\033[0m Researcher Agent requests internet access to {intent}.")
            try:
                ans = input("Allow internet connection for this action? [y/N]: ").strip().lower()
                if ans in ('y', 'yes'):
                    self.allow_internet = True
                    print("\033[32m[✓] Internet access granted.\033[0m\n")
                    return True
                else:
                    print("\033[31m[✗] Internet access denied. Operating 100% offline.\033[0m\n")
                    return False
            except Exception:
                return False
        return False

    def fetch_page(self, url: str, max_chars: int = 3500) -> str:
        """Fetches page content, strips HTML/CSS/scripts, and returns clean readable text."""
        if not self.check_or_request_permission(f"browse {url}"):
            return "[Error: Internet access is disabled. Run with --web or grant permission to fetch live web pages.]"

        try:
            req = urllib.request.Request(url, headers={'User-Agent': self.user_agent})
            with urllib.request.urlopen(req, timeout=6.0) as response:
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' not in content_type and 'text/plain' not in content_type:
                    return f"[Non-text content type: {content_type}]"
                html_bytes = response.read(150000) # Read up to 150KB
                html_text = html_bytes.decode('utf-8', errors='ignore')

            extractor = HTMLTextExtractor()
            extractor.feed(html_text)
            text = extractor.get_text()
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n... [Truncated after {max_chars} characters]"
            return text if text else "[Empty webpage content]"
        except Exception as e:
            return f"[Error fetching URL {url}: {e}]"

    def search(self, query: str, max_results: int = 4) -> List[Dict[str, str]]:
        """
        Performs a web search via DuckDuckGo Lite.
        Returns a list of dicts: [{'title': ..., 'url': ..., 'snippet': ...}]
        """
        if not self.check_or_request_permission(f"search query: '{query}'"):
            return [{"title": "Internet Disabled", "url": "", "snippet": "Internet access is disabled. Enable via --web or in chat via /web on."}]

        try:
            encoded_query = urllib.parse.urlencode({'q': query})
            url = f"https://lite.duckduckgo.com/lite/"
            data = encoded_query.encode('utf-8')
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    'User-Agent': self.user_agent,
                    'Content-Type': 'application/x-www-form-urlencoded'
                }
            )
            with urllib.request.urlopen(req, timeout=6.0) as response:
                html_bytes = response.read(100000)
                html_text = html_bytes.decode('utf-8', errors='ignore')

            # Parse DuckDuckGo Lite results table
            results = []
            # Match result links: <a class="result-link" href="...">...</a>
            link_pattern = re.compile(r'<a[^>]*class=["\']result-link["\'][^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
            snippet_pattern = re.compile(r'<td[^>]*class=["\']result-snippet["\'][^>]*>(.*?)</td>', re.IGNORECASE | re.DOTALL)

            links = link_pattern.findall(html_text)
            snippets = snippet_pattern.findall(html_text)

            for i in range(min(len(links), max_results)):
                href, title_html = links[i]
                title = re.sub(r'<[^>]+>', '', title_html).strip()
                snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ""
                # Clean DuckDuckGo redirect URL if present
                if "duckduckgo.com/l/?uddg=" in href:
                    match = re.search(r'uddg=([^&]+)', href)
                    if match:
                        href = urllib.parse.unquote(match.group(1))
                results.append({
                    "title": title,
                    "url": href,
                    "snippet": snippet
                })

            if not results:
                # Fallback: extract any links if result-link class was changed
                generic_links = re.findall(r'<a[^>]*href=["\'](https?://[^"\']+)["\'][^>]*>(.*?)</a>', html_text)
                for href, title_html in generic_links[:max_results]:
                    if "duckduckgo" not in href:
                        title = re.sub(r'<[^>]+>', '', title_html).strip()
                        if title:
                            results.append({"title": title, "url": href, "snippet": ""})

            return results
        except Exception as e:
            return [{"title": f"Search Error: {e}", "url": "", "snippet": "Failed to connect to search service."}]
