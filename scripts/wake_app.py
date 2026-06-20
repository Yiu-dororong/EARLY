import os
import sys
import time

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


# 1. Resolve target environment parameter
STREAMLIT_URL = os.getenv("STREAMLIT_APP_URL")

if not STREAMLIT_URL:
    print("❌ Error: STREAMLIT_APP_URL environment variable is missing.")
    sys.exit(1)

print(f"🌐 Target deployed endpoint: {STREAMLIT_URL}")

# 2. Configure sandboxed headless Chrome options for GitHub runner
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36")

driver = webdriver.Chrome(options=chrome_options)

try:
    print("⏳ Dispatching request to target URL...")
    driver.get(STREAMLIT_URL)

    # Allow asynchronous JS layout elements to fully settle
    time.sleep(10)

    # Hybrid Selector: Matches by Test ID OR by Text Content
    wake_button_xpath = (
        "//button[contains(text(), 'Yes, get this app back up!')] "
        "| //button[@data-testid='wakeup-button-owner']"
    )

    print("🔍 Evaluating page state for hibernation targets...")
    try:
        # Check for button availability for up to 15 seconds
        wait = WebDriverWait(driver, 15)
        button = wait.until(EC.element_to_be_clickable((By.XPATH, wake_button_xpath)))

        print("💤 App hibernation detected. Simulating wake button click selection...")
        button.click()
        print("⚡ Click action dispatched successfully.")

        print("⏳ Holding connection for 30 seconds to allow application spin-up...")
        time.sleep(30)
        print("✅ Recovery sequence completed successfully.")

    except TimeoutException:
        # Graceful fallback:
        # A normal page load automatically updates the Streamlit activity timer
        print("🎉 Wake button absent. Application state is active. "
              "Idle countdown timer reset.")

except Exception as e:
    print(f"❌ Automation runtime exception encountered: {e}")
    driver.quit()
    sys.exit(1)

finally:
    driver.quit()
    print("🏁 Browser automation footprint cleared cleanly.")
