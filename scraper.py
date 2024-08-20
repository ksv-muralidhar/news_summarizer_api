from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import FirefoxOptions
import re
import logging
import os


def get_text(url, n_words=15):
    try:
        driver = None
        logging.warning(f"Initiated Scraping {url}")
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        opts = FirefoxOptions()
        opts.add_argument("--headless")
        opts.add_argument(f"user-agent={user_agent}")
        driver = webdriver.Firefox(options=opts)
        driver.set_page_load_timeout(30)
        driver.get(url)
        elem = driver.find_element(By.TAG_NAME, "body").text
        sents = elem.split("\n")
        sentence_list = []
        for sent in sents:
            sent = sent.strip()
            if (len(sent.split()) >= n_words) and (len(re.findall(r"^\w.+[^\w\)\s]$", sent))>0):
                sentence_list.append(sent)
        driver.close()
        driver.quit()
        logging.warning("Closed Webdriver")
        logging.warning("Successfully scraped text")
        if len(sentence_list) < 3:
            raise Exception("Found nothing to scrape.")
        return "\n".join(sentence_list), ""
    except Exception as e: 
        logging.warning(str(e))
        if driver:
            driver.close()
            driver.quit()
            logging.warning("Closed Webdriver")
        err_msg = str(e).split('\n')[0]
        return "", err_msg


def scrape_text(url, n_words=15,max_retries=2):
    scraped_text = ""
    scrape_error = ""
    try:
        n_tries = 1
        while (n_tries <= max_retries) and (scraped_text == ""):
            scraped_text, scrape_error = get_text(url=url, n_words=n_words)
            n_tries += 1
        return scraped_text, scrape_error
    except Exception as e:
        err_msg = str(e).split('\n')[0]
        return "", err_msg
