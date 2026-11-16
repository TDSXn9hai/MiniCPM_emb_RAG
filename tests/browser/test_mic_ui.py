from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT.parent
    / "minicpm_demo"
    / "tests"
    / "cases"
    / "common"
    / "user_audio"
    / "当出现植物大战僵尸的时候提醒我.wav"
)
FAKE_MIC = ROOT / "tests" / "browser" / "fake-mic.wav"


def prepare_fake_mic() -> None:
    audio, _ = librosa.load(SOURCE, sr=16000, mono=True)
    audio = np.concatenate([audio, np.zeros(16000 * 10, dtype=np.float32)])
    sf.write(FAKE_MIC, audio, 16000, subtype="PCM_16")


def main() -> None:
    prepare_fake_mic()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={FAKE_MIC}",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        context = browser.new_context(
            viewport={"width": 1440, "height": 960},
            permissions=["microphone"],
        )
        page = context.new_page()
        console_errors = []
        page.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error"
            else None,
        )

        page.goto("http://127.0.0.1:8012")
        page.wait_for_load_state("networkidle")
        page.locator("#connectButton").click()
        page.wait_for_function(
            "() => document.querySelector('#connectionState').textContent.includes('已连接')"
        )
        page.locator("#micButton").click()
        page.wait_for_function(
            "() => document.querySelector('#userStream').textContent.includes('上海住宿能报多少')",
            timeout=30000,
        )
        page.wait_for_function(
            "() => document.querySelector('#assistantStream').textContent.includes('mock 双工回答')",
            timeout=30000,
        )
        assert page.locator("#retrievalMs").inner_text() != "—"
        assert page.locator("#firstTextMs").inner_text() != "0 ms"
        page.locator("#micButton").click()
        screenshot = ROOT / "tests" / "browser" / "a2a-mic-ui.png"
        page.screenshot(path=str(screenshot), full_page=True)
        assert not console_errors, console_errors
        browser.close()
        print(f"screenshot: {screenshot}")


if __name__ == "__main__":
    main()
