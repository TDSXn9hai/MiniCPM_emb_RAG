from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 960})
        console_errors = []
        page.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error"
            else None,
        )

        page.goto("http://127.0.0.1:8011")
        page.wait_for_load_state("networkidle")
        assert "A2A Knowledge Lab" in page.title()

        page.locator("#connectButton").click()
        page.wait_for_function(
            "() => document.querySelector('#connectionState').textContent.includes('已连接')",
            timeout=180000,
        )
        page.locator("#textQuery").fill("上海出差住宿能报多少")
        page.locator("#sendQuery").click()
        page.wait_for_function(
            "() => document.querySelector('#assistantStream')"
            ".textContent.includes('六百')",
            timeout=120000,
        )

        assert "差旅报销制度" in page.locator("#evidenceList").inner_text()
        screenshot = ROOT / "tests" / "browser" / "a2a-real-ui.png"
        page.screenshot(path=str(screenshot), full_page=True)
        assert not console_errors, console_errors
        browser.close()
        print(f"screenshot: {screenshot}")


if __name__ == "__main__":
    main()
