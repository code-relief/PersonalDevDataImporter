import scrapy
import re
import csv
import os
import pandas as pd
from scrapy.crawler import CrawlerProcess
import logging
import json
import time

logger = logging.getLogger('datacrawler')
data = pd.DataFrame()

path = os.path.dirname(os.path.abspath(__file__))
output_filename = os.path.join(path, 'pracujpl_data.csv')


class PracujPlSpider(scrapy.Spider):
    batch_size = 200000
    name = "pracuj.pl_spider"
    start_urls = ['https://archiwum.pracuj.pl/archive/offers?Year=2015&Month=1&PageNumber=1']
    year_month_to_skip = []
    replacements = {'&#260;': 'Ą', '&#261;': 'ą', '&#262;': 'Ć', '&#263;': 'ć', '&#280;': 'Ę', '&#281;': 'ę',
                    '&#321;': 'Ł', '&#322;': 'ł', '&#323;': 'Ń', '&#324;': 'ń', '&#211;': 'Ó', '&#243;': 'ó',
                    '&#$3;': 'ó', '&#346;': 'Ś', '&#347;': 'ś', '&#377;': 'Ź', '&#378;': 'ź', '&#379;': 'Ż',
                    '&#380;': 'ż', '\\x3a': ':', '\\x2f': '\\\\', '\\x28': '(', '\\x29': ')', '\\x22': '', '\\x2b': ',',
                    '\\x23': '#', '\\u00f3': 'ó'}

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
        result = self.getMetedata(response)
        content_selector = ".//div[@id='description']"
        content = response.xpath(content_selector).extract_first()
        result['year'] = response.meta['year']
        result['month'] = response.meta['month']
        result['title'] = re.sub('[\\s]+', ' ', re.sub('<[^>]+>', '', str(response.meta['job_title'])).strip())
        result['location'] = re.sub('<[^>]+>', '', str(response.meta['job_location'])).strip()
        result['content'] = re.sub('<[^>]+>', '', str(content)).strip()
        global data
        if result['content'] not in 'None':
            data = data.append(pd.Series(result), ignore_index=True)
            if data.size % self.batch_size == 0:
                columns = result.keys()
                columns = ['offerData_id', 'offerData_commonOfferId', 'offerData_jobTitle', 'offerData_categoryNames', 'offerData_countryName', 'offerData_regionName', 'offerData_appType', 'offerData_appUrl', 'offerData_recommendations', 'gtmData_name', 'gtmData_id', 'gtmData_price', 'gtmData_brand', 'gtmData_category', 'gtmData_variant', 'gtmData_list', 'gtmData_position', 'gtmData_dimension6', 'gtmData_dimension7', 'gtmData_dimension8', 'gtmData_dimension9', 'gtmData_dimension10', 'socProduct_identifier', 'socProduct_fn', 'socProduct_category', 'socProduct_description', 'socProduct_brand', 'socProduct_price', 'socProduct_amount', 'socProduct_currency', 'socProduct_url', 'socProduct_valid', 'socProduct_photo', 'dataLayer_level', 'dataLayer_ekosystem', 'dataLayer_receiver', 'year', 'month', 'title', 'location', 'content']
                data[columns].to_csv(output_filename.replace('.csv', '_{0}_{1}_{2}.csv'.format(self.batch_size, data.size, time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()))), sep=';',
                                     encoding='utf-8', mode='w', quotechar='"', line_terminator='\n')
                # clean in-memory data
                logger.info("Data dumped to file")
                data = pd.DataFrame()
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
                logger.info("Year: {0}, month: {1}, page_no: {2}".format(request.meta['year'], request.meta['month'],
                                                                         page_number))

            # moving to next pages of data
            next_page_selector = ".//a[@class='offers_nav_next']/@href"
            next_page = response.xpath(next_page_selector).extract_first()
            if next_page:  # and int(page_number) < 2:
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

    def getMetedata(self, response):
        result = {}
        # https://blog.michaelyin.info/scrapy-tutorial-11-how-to-extract-data-from-native-javascript-statement/
        response_body = response.body.decode('utf-8','ignore')
        self.extract_json_data("var offerData =([^\}]+\});?", response_body, result, 'offerData')
        self.extract_json_data("window.gtmData.offer =([^\}]+\});?", response_body, result, 'gtmData')
        self.extract_json_data("var soc_product =([^\}]+\});?", response_body, result, 'socProduct')
        self.extract_value("dataLayer\[0\]\['Oferta poziom'\] = '(\w+)';", response_body, result, 'dataLayer_level')
        self.extract_value("dataLayer\[0\]\['Typ ekosystem'\] = '(\w+)';", response_body, result, 'dataLayer_ekosystem')
        self.extract_value("dataLayer\[0\]\['Typ odbiorca'\] = '(\w+)';", response_body, result, 'dataLayer_receiver')

        for key, val in result.items():
            if type(val) == list:
                result[key] = ', '.join(val)

        return result

    def extract_json_data(self, regex, response_body, result, prefix):
        data = re.findall(regex, response_body, re.DOTALL)
        json_data = None
        if data:
            json_string = data[0]
            keys = re.findall("([\w]+):", json_string)
            for key in keys:
                json_string = json_string.replace(key, '"{}"'.format(key))
            json_string = json_string.replace("'", '"')

            for repl_key, repl_val in self.replacements.items():
                json_string = json_string.replace(repl_key, repl_val)
            try:
                json_data = json.loads(json_string)
            except json.decoder.JSONDecodeError:
                json_string = re.sub('\\\\x[0-9abcdef]{2}', '', json_string)
                values = re.findall(':\\s+"(.*?)",', json_string)
                if values:
                    for val in values:
                        if '"' in val:
                            json_string = json_string.replace(val, val.replace('"', ''))
                num_values = re.findall(':\\s+([0-9"]{3,}),', json_string)
                if num_values:
                    for val in num_values:
                        if '"' in val:
                            json_string = json_string.replace(val, val.replace('"', ''))
                json_string = re.sub('\\\\x[0-9abcdef]{2}', '', json_string)
                try:
                    json_data = json.loads(json_string)
                except json.decoder.JSONDecodeError:
                    json_string = re.sub("\\\\u[0-9abcdef]+", '', json_string)
                    try:
                        json_data = json.loads(json_string)
                    except json.decoder.JSONDecodeError:
                        pass

            if json_data:
                for key, value in json_data.items():
                    result['{}_{}'.format(prefix, key)] = value
    def extract_value(self, regex, response_body, result, param_name):
        data = re.findall(regex, response_body)
        if data:
            result[param_name] = data[0]


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
