from CorpusBuilder import Parser3, build_corpus_stat, build_corpus_clustering, get_keys, get_values, build_keys_corpus_stat, build_values_corpus_stat, build_keys_clustering, build_values_clustering
from Statis import stat_function
from get_osm_files import find_files_with_extension


def Generate():
    base_folder = "C:\\Users\\Omar Ghannou\\Downloads\\stests"

    osm_files = find_files_with_extension(base_folder, ".osm")

    #Building corpus
    print("Found .osm files:")
    for osm_file in osm_files:
        tags = Parser3(osm_file)
        build_corpus_clustering(tags)
        #build_keys_clustering(tags)
        #build_values_clustering(tags)
        #values = get_values(tags)
        #build_values_corpus_stat(values)
        #keys = get_keys(tags)
        #build_keys_corpus_stat(keys)
        #build_corpus_stat(tags)


corpus = "corpus.crps"
report_file = "report.txt"
Generate()
#stat_function(corpus, report_file)