import scrapy
import re
import csv
import os
import pandas as pd
from scrapy.crawler import CrawlerProcess

data = pd.DataFrame()

path = os.path.dirname(os.path.abspath(__file__))
output_filename = os.path.join(path, 'pracujpl_data.csv')

class PracujPlSpider(scrapy.Spider):
    name = "pracuj.pl_spider"
    start_urls = ['https://archiwum.pracuj.pl/archive/offers?Year=2015&Month=1&PageNumber=1']

    def parse_single_page(self, response):
        content_selector = ".//div[@id='description']"
        content = response.xpath(content_selector).extract_first()
        result = {
            'year': response.meta['year'],
            'month': response.meta['month'],
            'title': re.sub('<[^>]+>', '', str(response.meta['job_title'])).strip(),
            'location': re.sub('<[^>]+>', '', str(response.meta['job_location'])).strip(),
            'content': re.sub('<[^>]+>', '', str(content)).strip(),
        }
        global data
        data = data.append(pd.Series(result), ignore_index=True)
        yield {}  # result

    def parse(self, response):
        list_item_selector = ".//div[@class='offers_item']"
        for list_item in response.xpath(list_item_selector):
            details_page_link_selector = './/a/@href'
            job_title_selector = ".//div[@class='offers_item_link_cnt']/span"
            job_location_selector = ".//p[@class='offers_item_desc']/span[1]"
            desc_page_link = list_item.xpath(details_page_link_selector).extract_first()
            job_title = list_item.xpath(job_title_selector).extract_first()
            job_location = list_item.xpath(job_location_selector).extract_first()
            request = response.follow(desc_page_link, callback=self.parse_single_page)
            request.meta['job_title'] = job_title
            request.meta['job_location'] = job_location
            current_url = response.request.url
            request.meta['year'] = 0
            request.meta['month'] = 0
            z = re.match(".*Year=([0-9]+)&Month=([0-9]+)&PageNumber.*", current_url)
            if z:
                request.meta['year'] = z.group(1)
                request.meta['month'] = z.group(2)
            yield request

        next_page_selector = ".//a[@class='offers_nav_next']/@href"
        # self.job_title = ''
        # self.job_location = ''
        # next_page = response.xpath(next_page_selector).extract_first()
        # if next_page:
        #     yield scrapy.Request(
        #         response.urljoin(next_page),
        #         callback=self.parse
        #     )


def main():
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
    })

    process.crawl(PracujPlSpider)
    process.start()
    print(data)
    columns = ['year', 'month', 'title', 'location', 'content']
    data[columns].to_csv(output_filename, sep=';', encoding='utf-8', mode='w', quotechar='"', line_terminator='\n')
    print(data)


if __name__ == '__main__':
    main()
