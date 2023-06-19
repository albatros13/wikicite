import json
from urllib.request import Request, urlopen
from urllib.parse import quote
from crossref.restful import Works
import Levenshtein
import pandas as pd
import os
from google.cloud import storage
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def find_google_books(title, threshold: float = 0.75):
    if not title:
        return None
    book_data = query_google_books(title)
    if "items" in book_data:
        items = [item for item in book_data["items"] if "volumeInfo" in item]
        max_ratio = 0
        best_match = None
        for res in items:
            volume_info = res["volumeInfo"]
            ratio = Levenshtein.ratio(title, volume_info["title"])
            if ratio >= threshold:
                if ratio > max_ratio:
                    max_ratio = ratio
                    best_match = res
            else:
                break
        if best_match is not None:
            return google_book_ids(best_match)
    return None


def find_crossref(title, threshold: float = 0.75):
    if not title:
        return None
    res = query_crossref_pub(title)
    if res is not None and "title" in res:
        ratio = 0
        try:
            ratio = Levenshtein.ratio(title, res["title"][0])
        except Exception as e:
            print(e)
        if ratio >= threshold:
            return crossref_ids(res)
    return None


def crossref_ids(res):
    identifiers = []
    if "DOI" in res:
        identifiers.append(['DOI', res["DOI"]])
    if "ISBN" in res:
        identifiers.append(['ISBN', res["ISBN"]])
    if "PMID" in res:
        identifiers.append(['PMID', res["PMID"]])
    if "PMC" in res:
        identifiers.append(['PMC', res["PMC"]])
    return identifiers


def google_book_ids(res):
    identifiers = []
    volume_info = res["volumeInfo"]
    if "industryIdentifiers" in volume_info:
        for obj in volume_info["industryIdentifiers"]:
            identifiers.append([obj["type"].lower(), obj["identifier"]])
    return identifiers


def query_google_books(ref):
    url = "https://www.googleapis.com/books/v1/volumes?q=" + quote(ref) + "&intitle:" + quote(ref)
    res = urlopen(url)
    book_data = json.load(res)
    return book_data


def query_crossref_pub(ref):
    works = Works()
    res = works.query(bibliographic=ref)
    if res:
        for item in res:
           return item


def disambiguate(row):
    labels = ['cite book', 'cite journal', 'cite encyclopedia', 'cite proceedings']
    ids = ''
    try:
        if row['actual_label'] == 'other' and row['type_of_citation'] in labels:
            ids = find_crossref(row['Title'])
            if not ids:
                ids = find_google_books(row['Title'])
    except Exception as e:
        print("Failed to disambiguate: ", row)
        print(e)
    if ids is None:
        ids = ''
    return str(ids)


def is_same(row):
    for id_ in row['acquired_ID_list']:
        if len(id_) == 2:
            res = id_[1] in row['ID_list']
            if res:
                print(id_[1], row['ID_list'])
                return True
    return False


def is_google_book(row):
    return row['URL'].startswith('https://books.google.com/books?id=')


PROJECT_HOME = 'c:///users/natal/PycharmProjects/cite-classifications-wiki/'
ext = "en_"
INPUT_DIR = "data/parts/dumps"
OUTPUT_DIR = 'data/parts/'
BOOK_JOURNAL_DIR = OUTPUT_DIR + 'book_journal/'
BOOK_JOURNAL_DIR_EXT = OUTPUT_DIR + 'book_journal_ext/'
NEWS_DIR = OUTPUT_DIR + 'news/'
NEWS_FEATURE_DIR = OUTPUT_DIR + 'news/features/'


def get_files_from_bucket():
    BUCKET_NAME = os.getenv("BUCKET_NAME", "wikicite-1")
    BUCKET_PATH = os.getenv("BUCKET_PATH", INPUT_DIR)
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    content_list = list(bucket.list_blobs(prefix=f"{BUCKET_PATH}/"))
    file_paths = []
    extensions = []
    for index__, b in enumerate(content_list):
        if b.name.endswith('.bz2'):
            file_paths.append(PROJECT_HOME + b.name)
            suffix = b.name[b.name.rfind('-a'): b.name.rfind('.')]
            extensions.append(suffix)
    return file_paths, extensions


def get_files_from_disk():
    content_list = os.listdir(PROJECT_HOME + INPUT_DIR)
    file_paths = []
    extensions = []
    for index__, f_name in enumerate(content_list):
        if f_name.endswith('.bz2'):
            file_paths.append(PROJECT_HOME + INPUT_DIR + '/' + f_name)
            suffix = f_name[f_name.rfind('-a'): f_name.rfind('.')]
            extensions.append(suffix)
    return file_paths, extensions


def get_part_names_from_disk(file_in, file_out):
    parts = os.listdir(file_in)
    parts_in = []
    parts_out = []
    for index__, f_name in enumerate(parts):
        if not f_name.startswith('part'):
            continue
        parts_in.append('{}/{}'.format(file_in, f_name))
        parts_out.append('{}/{}'.format(file_out, f_name))
    return parts_in, parts_out


def get_part_names_from_bucket(file_in, file_out):
    BUCKET_NAME = os.getenv("BUCKET_NAME", "wikicite-1")
    BUCKET_PATH = os.getenv("BUCKET_PATH", file_in)
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    parts = list(bucket.list_blobs(prefix=f"{BUCKET_PATH}/"))
    parts_in = []
    parts_out = []
    for index__, b in enumerate(parts):
        if not b.name.startswith('part'):
            continue
        part_name = b.name[b.name.rfind('\\'):]
        part_in = '{}/{}'.format(file_in, part_name)
        part_out = '{}/{}'.format(file_out, part_name)
        parts_in.append(part_in)
        parts_out.append(part_out)
    return parts_in, parts_out


def disambiguate_files_dump():
    file_paths, extensions = get_files_from_disk()
    for index__, f_in in enumerate(file_paths):
        suffix = extensions[index__]
        if suffix and index__ == 0:
            f_book_journal = PROJECT_HOME + BOOK_JOURNAL_DIR + ext + 'book_journal_citations' + suffix + '.parquet'
            f_book_journal_ext = PROJECT_HOME + BOOK_JOURNAL_DIR_EXT + ext + 'book_journal_citations' + suffix + '.parquet'
            if not os.path.exists(f_book_journal_ext):
                os.mkdir(f_book_journal_ext)

            parts_in, parts_out = get_part_names_from_disk(f_book_journal, f_book_journal_ext)
            print(parts_in)
            for i, part_in in enumerate(parts_in):
                print("Processing: ", part_in)
                all_examples = pd.read_parquet(part_in, engine='pyarrow').head(100)
                labelled = all_examples[all_examples['actual_label'] != 'other']
                unlabelled = all_examples[all_examples['actual_label'] == 'other']
                labels = ['cite book', 'cite journal', 'cite encyclopedia', 'cite proceedings']
                likely_books_or_journals = unlabelled[unlabelled['type_of_citation'].isin(labels)]
                print("All: ", all_examples.shape)
                print("Books and journals: ", labelled.shape)
                print("Could be books and journals: ", likely_books_or_journals.shape)

                # Can run for a long time, hence better to use batches (see example in disambiguate_bucket_dump)
                selected_fields = ['type_of_citation', 'page_title', 'Title', 'URL', 'Authors', 'ID_list', 'citations', 'actual_label']
                all_examples = all_examples[selected_fields]
                all_examples['acquired_ID_list'] = all_examples.apply(lambda x: disambiguate(x), axis=1)
                all_examples.to_parquet(parts_out[i])
                print("Saved output: ", parts_out[i])


disambiguate_files_dump()


def disambiguate_bucket_dump():
    file_paths, extensions = get_files_from_bucket()

    for index__, f_in in enumerate(file_paths):
        suffix = extensions[index__]

        if suffix:
            f_book_journal = BOOK_JOURNAL_DIR + ext + 'book_journal_citations' + suffix + '.parquet'
            f_book_journal_ext = BOOK_JOURNAL_DIR_EXT + ext + 'book_journal_citations' + suffix + '.parquet'
            parts_in, parts_out = get_part_names_from_bucket(f_book_journal, f_book_journal_ext)
            for i, part_in in enumerate(parts_in):
                print("Processing part: ", i)
                file_in = PROJECT_HOME + part_in
                file_out = parts_out[i]

                all_examples = pd.read_parquet(file_in, engine='pyarrow')
                selected_fields = ['type_of_citation', 'page_title', 'Title', 'URL', 'Authors', 'ID_list', 'citations',
                                   'actual_label']
                all_examples = all_examples[selected_fields]

                # Write in small batches because it can fail at any moment
                for j, df in enumerate(np.array_split(all_examples, 1000)):
                    print("Processing batch: ", j)
                    df['acquired_ID_list'] = df[selected_fields].apply(lambda x: disambiguate(x), axis=1)
                    table = pa.Table.from_pandas(df)
                    if j == 0:
                        pqwriter = pq.ParquetWriter(file_out, table.schema)
                    pqwriter.write_table(table)

                if pqwriter:
                    pqwriter.close()

            # stats
            all_examples = pd.read_parquet(file_out, engine='pyarrow')
            added = all_examples[all_examples['acquired_ID_list'] != '']
            print("Disambiguated: ", added.shape[0])


TEST_FILE = PROJECT_HOME + "data/downloads/" + \
            "en_book_journal_citations-articles.xml.parquet_part-00000-0696ec2a-5c19-475a-abe6-8e8b660dae00-c000.snappy.parquet"

OUT_FILE = PROJECT_HOME + "data/downloads/" + \
           "en_book_journal_citations_ext-articles.xml.parquet_part-00000-0696ec2a-5c19-475a-abe6-8e8b660dae00-c000.snappy.parquet"


def run_test():
    all_examples = pd.read_parquet(TEST_FILE, engine='pyarrow').head(1000)
    selected_fields = ['type_of_citation', 'page_title', 'Title', 'URL', 'Authors', 'ID_list', 'citations', 'actual_label']
    all_examples = all_examples[selected_fields]
    all_examples['acquired_ID_list'] = all_examples.apply(lambda x: disambiguate(x), axis=1)
    all_examples.to_parquet(OUT_FILE)


def run_stats():
    all_examples = pd.read_parquet(TEST_FILE, engine='pyarrow')
    print(all_examples.shape)

    google_books = all_examples[all_examples.apply(lambda x: is_google_book(x), axis=1)]
    print(google_books.shape)

    google_books_no_label = google_books[google_books['actual_label'] == 'other']
    print(google_books_no_label.shape)

    unlabelled = all_examples[all_examples['actual_label'] == 'other']
    labels = ['cite book', 'cite journal', 'cite encyclopedia', 'cite proceedings']
    likely_books_or_journals = unlabelled[unlabelled['type_of_citation'].isin(labels)]
    print(likely_books_or_journals.shape)

    added = all_examples[all_examples['acquired_ID_list'] != '']
    print(added.shape[0])

    labelled = all_examples[all_examples['actual_label'] != 'other']
    print(labelled.shape[0])

    # Use for evaluating precision when disambiguation is applied to citations with IDs
    # same = disambiguated[['ID_list', 'acquired_ID_list']].apply(lambda x: is_same(x), axis=1)
    # print("Same:", same.shape)
