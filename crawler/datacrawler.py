import scrapy
import re
import csv
import os
import pandas as pd
from scrapy.crawler import CrawlerProcess
import logging

logger = logging.getLogger('datacrawler')
data = pd.DataFrame()

path = os.path.dirname(os.path.abspath(__file__))
output_filename = os.path.join(path, 'pracujpl_data.csv')

class PracujPlSpider(scrapy.Spider):
    name = "pracuj.pl_spider"
    start_urls = ['https://archiwum.pracuj.pl/archive/offers?Year=2015&Month=1&PageNumber=1']
    year_month_to_skip = [('2015', '9'), ('2015', '10'), ('2015', '11')]

    def __init__(self, *args, **kwargs):
        http_error_logger = logging.getLogger('scrapy.spidermiddlewares.httperror')
        http_error_logger.setLevel(logging.WARNING)
        scrapy_core_logger = logging.getLogger('scrapy.core.scraper')
        scrapy_engine_logger = logging.getLogger('scrapy.core.engine')
        scrapy_downloader_logger = logging.getLogger('scrapy.downloadermiddlewares.redirect')
        scrapy_core_logger.setLevel(logging.INFO)
        scrapy_engine_logger.setLevel(logging.INFO)
        scrapy_downloader_logger.setLevel(logging.INFO)
        super().__init__(*args, **kwargs)

    def parse_single_page(self, response):
        content_selector = ".//div[@id='description']"
        content = response.xpath(content_selector).extract_first()
        result = {
            'year': response.meta['year'],
            'month': response.meta['month'],
            'title': re.sub('[\\s]+', ' ', re.sub('<[^>]+>', '', str(response.meta['job_title'])).strip()),
            'location': re.sub('<[^>]+>', '', str(response.meta['job_location'])).strip(),
            'content': re.sub('<[^>]+>', '', str(content)).strip(),
        }
        global data
        if result['content'] not in 'None':
            data = data.append(pd.Series(result), ignore_index=True)
            if data.size % 10000 == 0:
                columns = ['year', 'month', 'title', 'location', 'content']
                data[columns].to_csv(output_filename.replace('.csv', '_{}.csv'.format(data.size)), sep=';', encoding='utf-8', mode='w', quotechar='"', line_terminator='\n')
        yield {}  # result

    def parse(self, response):
        # parsing single page of data
        list_item_selector = ".//div[@class='offers_item']"
        request = {}
        current_url = response.request.url
        z = re.match(".*Year=([0-9]+)&Month=([0-9]+)&PageNumber=([0-9]+).*", current_url)
        year = 0
        month = 0
        page_number = 0
        if z:
            year = z.group(1)
            month = z.group(2)
            page_number = z.group(3)
        if not (year, month) in self.year_month_to_skip:
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
                request.meta['year'] = 0
                request.meta['month'] = 0
                request.meta['year'] = year
                request.meta['month'] = month
                yield request

            # save data to file after each page processed
            if type(request) is scrapy.http.request.Request:
                logger.info("Year: {0}, month: {1}, page_no: {2}".format(request.meta['year'], request.meta['month'], page_number))


            # moving to next pages of data
            next_page_selector = ".//a[@class='offers_nav_next']/@href"
            next_page = response.xpath(next_page_selector).extract_first()
            if next_page:# and int(page_number) < 2:
                yield scrapy.Request(
                    response.urljoin(next_page),
                    callback=self.parse
                )
        else:
            logger.info("Skipping [1] {}".format(current_url))

        # jumping to next month in archive
        next_month_selector = ".//a[@class='date_item_cnt_link']/@href"
        next_months = response.xpath(next_month_selector).extract()
        if next_months:
            for next_month in next_months:
                z = re.match(".*Year=([0-9]+)&Month=([0-9]+).*", next_month)
                if z:
                    year = z.group(1)
                    month = z.group(2)
                if (year, month) in self.year_month_to_skip:
                    logger.info("Skipping [2] {}".format(next_month))
                    continue
                yield scrapy.Request(
                    response.urljoin(next_month),
                    callback=self.parse
                )


def main():
    try:
        process = CrawlerProcess({
            'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
        })

        process.crawl(PracujPlSpider)
        process.start()
    finally:
        print("Finishing up")
        columns = ['year', 'month', 'title', 'location', 'content']
        data[columns].to_csv(output_filename, sep=';', encoding='utf-8', mode='w', quotechar='"', line_terminator='\n')
        print("Data saved to file")


if __name__ == '__main__':
    main()
