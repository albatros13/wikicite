"""
Get citations content only for certain set of titles which were based on argument file.
"""


def filter_content(sql_context, file_in1, file_in2, file_out):
    citations_content = sql_context.read.parquet(file_in1)
    titles = sql_context.read.parquet(file_in2)

    # Join and get only the titles which are present in the dataset
    filtered = citations_content.join(titles, citations_content.page_title == titles.titles)

    # Select only titles and content and not id and the duplicated column
    filtered = filtered.select('page_title', 'content')
    filtered.write.mode('overwrite').parquet(file_out)

