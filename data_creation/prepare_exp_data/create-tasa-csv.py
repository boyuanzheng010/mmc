#!/usr/bin/env python

import json
import time

from nltk.tokenize import word_tokenize
import unicodecsv

# use the current TASA version.
VERSION = {
    'MAJOR': 1,  
    'MINOR': 1,  
    'PATCH': 2,  
}

def main():
    # source sentence
    source_sent = "the findings are being published today in the Annals of Internal Medicine ."

    # target language
    target_sent = "the findings are published in the July 1st issue of the Annals of Internal Medicine ."

    # output CSV filename
    csv_file = 'tasa_example.csv'

    # In this example, we only have 1 source-target 
    # pair. You may append more rows to this csv file.
    csv_fh = open(csv_file, 'wb')
    fieldnames = [
        'src_tokens',
        'tar_tokens',
        'config_obj',
    ]
    writer = unicodecsv.DictWriter(csv_fh, fieldnames, lineterminator='\n',
                                   quoting=unicodecsv.QUOTE_ALL)
    writer.writeheader()

    # tokenize the sentence with NLTK. 
    source_tokens = word_tokenize(source_sent)
    target_tokens = word_tokenize(target_sent)
    
    # un-comment if initial alignment is present. 
    # the line below is the default value.
    #alignment = [[False] * len(target_tokens)] * len(source_tokens)

    # the configuration object. If you use the default value, 
    # you could omit the corresponding key in `config_obj`. 
    config_obj = {
        'version': VERSION,
        #'alignment': alignment,
        #'src_enable_retokenize': False,
        #'tar_enable_retokenize': False,
        #'src_spans': token_source_spans,
        #'tar_spans': [],
        #'src_head_inds': src_head_inds,
        #'tar_head_inds': tar_head_inds,
    }

    # write row to CSV file.
    writer.writerow({
        'src_tokens': json.dumps(source_tokens),
        'tar_tokens': json.dumps(target_tokens),
        'config_obj': json.dumps(config_obj),
    })

if __name__ == "__main__":
    main()
