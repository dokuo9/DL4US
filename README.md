
# Install
    $ brew install mecab mecab-ipadic
    $ mkvirtualenv --python=/usr/bin/python3 [NAME]

# train
    $ python parse --in data/raw --out data/formatted
    $ python train --input-x data/formatted/data_x --input-y data/formatted/data_y

# predict
    $ python predict --model /path/to/model --text /path/to/text

# get text
    $ curl http://www.aozorahack.net/api/v0.1/books/{book_id}/content?format=txt -o {output_file_name}
    $ nkf -w --overwrite {output_file_name}
