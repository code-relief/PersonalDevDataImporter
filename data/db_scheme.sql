create table pracujpl
(
  id                        BIGINT,
  "Unnamed: 0"              BIGINT,
  offerData_id              FLOAT,
  offerData_commonOfferId   FLOAT,
  offerData_jobTitle        TEXT,
  offerData_categoryNames   TEXT,
  offerData_countryName     TEXT,
  offerData_regionName      TEXT,
  offerData_appType         FLOAT,
  offerData_appUrl          TEXT,
  offerData_recommendations BOOLEAN,
  gtmData_name              TEXT,
  gtmData_id                FLOAT,
  gtmData_price             FLOAT,
  gtmData_brand             TEXT,
  gtmData_category          TEXT,
  gtmData_variant           TEXT,
  gtmData_list              TEXT,
  gtmData_position          FLOAT,
  gtmData_dimension6        FLOAT,
  gtmData_dimension7        FLOAT,
  gtmData_dimension8        FLOAT,
  gtmData_dimension9        TEXT,
  gtmData_dimension10       FLOAT,
  socProduct_identifier     FLOAT,
  socProduct_fn             TEXT,
  socProduct_category       TEXT,
  socProduct_description    TEXT,
  socProduct_brand          TEXT,
  socProduct_price          FLOAT,
  socProduct_amount         FLOAT,
  socProduct_currency       TEXT,
  socProduct_url            TEXT,
  socProduct_valid          FLOAT,
  socProduct_photo          TEXT,
  dataLayer_level           TEXT,
  dataLayer_ekosystem       TEXT,
  dataLayer_receiver        TEXT,
  year                      BIGINT,
  month                     BIGINT,
  title                     TEXT,
  location                  TEXT,
  content                   TEXT,
  check ("offerData_recommendations" IN (0, 1))
);

create index ix_pracujpl_id
  on pracujpl (id);

alter table pracujpl add column stem TEXT;
alter table pracujpl add column lang TEXT;
alter table pracujpl add column stem_phrases TEXT;
alter table pracujpl add column content_fixed BOOLEAN;
