# 爬虫
import csv
import time
from lxml import etree
import random
import requests
import os

# 爬取页面并保存
def getdata(i,comUrl):
    revUrl = f"/ref=cm_cr_arp_d_paging_btm_next_{i}?ie=UTF8&reviewerType=all_reviews&pageNumber={i}"
    url=comUrl+revUrl
    # print(url)
    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36 Edg/96.0.1054.29",
        "cookie": 'session-id-time=2082787201l; i18n-prefs=USD; sp-cdn="L5Z9:HK"; session-id=142-8009421-0821632; ubid-main=133-0047950-5521365; session-token="0Ap62C1BnXfiaxDeS363H+D2koePVtwykgtnyDoP5NLmIyLiSNd607mWXMf52jwy1NMlVOVDt/ClkaEX3Ben8yfHPiLi8RAACifdrqCIPX1LIvX0C5VNuHrCqJ8Tos2GBV2JXEFVRgjsnEnRkz/br4LBiESWm4V/NphHKii0bF7hzUAwuiQX6+TGXd559ROjN1jXzmqTxkSzyg1iUIel4g=="; csm-hit=tb:23T9J88SV02BXQTD535R+s-264PB9FH012S7TWGQVHS|1651406357005&t:1651406357005&adb:adblk_yes'
    }
    r = requests.get(url=url, headers=header)

    with open("tmp.html", 'w', encoding='utf-8') as f:
        f.write(r.text)

# 从爬取的页面获取评论并存入csv
def anadata():
    parser = etree.HTMLParser(encoding="utf-8")
    tree = etree.parse('tmp.html', parser=parser)
    comments = tree.xpath('//span[@class="a-size-base review-text review-text-content"]/span/text()')
    with open("comments.csv", mode="a", encoding='UTF-8', newline="") as f:
        csvpencil = csv.writer(f)
        for comment in comments:
            comment = comment.replace("\n","").replace("\r","").replace('"',"")
            print(comment)
            csvpencil.writerow([comment])
            f.flush()#清空缓冲区


def spider(comUrl):
    if(os.path.exists("comments.csv")):
       os.remove("comments.csv")
    # 爬十页
    for i in range(9):
        getdata(i + 1,comUrl)
        # time.sleep(random.randint(5, 10))
        anadata()
