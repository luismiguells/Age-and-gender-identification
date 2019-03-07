#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 16:46:20 2018

@author: luismiguells
"""

import re
from nltk.tokenize.casual import EMOTICON_RE as emo_re
import emoji

def clean(line,hashs,ats,links):
    for link in links:
        line = line.replace(link,'')
    for has_h in hashs:
        line = line.replace('#'+has_h,'')
        line = line.replace('＃'+has_h,'')
    for at in ats:
        line = line.replace('@'+at,'')
        line = line.replace('＠'+at,'')
    return line

#Regular expression for URLs
URLS = r"""			# Capture 1: entire matched URL
  (?:
  https?:				# URL protocol and colon
    (?:
      /{1,3}				# 1-3 slashes
      |					#   or
      [a-z0-9%]				# Single letter or digit or '%'
                                       # (Trying not to match e.g. "URI::Escape")
    )
    |					#   or
                                       # looks like domain name followed by a slash:
    [a-z0-9.\-]+[.]
    (?:[a-z]{2,13})
    /
  )
  (?:					# One or more:
    [^\s()<>{}\[\]]+			# Run of non-space, non-()<>{}[]
    |					#   or
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)				# balanced parens, non-recursive: (...)
  )+
  (?:					# End with:
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)				# balanced parens, non-recursive: (...)
    |					#   or
    [^\s`!()\[\]{};:'".,<>?«»“”‘’]	# not a space or one of these punct chars
  )
  |					# OR, the following to match naked domains:
  (?:
  	(?<!@)			        # not preceded by a @, avoid matching foo@_gmail.com_
    [a-z0-9]+
    (?:[.\-][a-z0-9]+)*
    [.]
    (?:[a-z]{2,13})
    \b
    /?
    (?!@)			        # not succeeded by a @,
                            # avoid matching "foo.na" in "foo.na@example.com"
  )
"""

main_dir = 'C:/Users/luismiguells/Dropbox/TesisG/Data/'
text_file = main_dir+'all_dataset_text.txt'
words_file = main_dir+'words_file.txt'
emo_file = main_dir+'emo_file.txt'
hash_file = main_dir+'hash_file.txt'
at_file = main_dir+'ats_file.txt'
link_file = main_dir+'links_file.txt'

i = 0
#Finding the features with the regular expressions of: URLs, hastags and mentions
url_re = re.compile(URLS, re.VERBOSE | re.I | re.UNICODE) 
hashtag_re = re.compile('(?:^|\s)[＃#]{1}(\w+)', re.UNICODE)
mention_re = re.compile('(?:^|\s)[＠@]{1}(\w+)', re.UNICODE)
                       
#Iterate over each pin to extract the features
with open(text_file,'r', encoding = 'utf-8') as text_reader, open(words_file,'w', encoding='utf-8') as words_writer, open(emo_file, 'w',encoding='utf-8') as emo_writer, open(hash_file,'w',encoding='utf-8') as hash_writer, open(at_file,'w',encoding='utf-8') as at_writer, open(link_file,'w',encoding='utf-8') as link_writer:
    for line in text_reader:
        line = line.rstrip().lower()
        hashs = hashtag_re.findall(line)
        ats = mention_re.findall(line)
        links = url_re.findall(line)
        line = clean(line,hashs,ats,links)
        emoticons = emo_re.findall(line)
        emojis = [w for w in line if w in emoji.UNICODE_EMOJI]
        words = re.findall('[a-záéíóúñ_-]+',line) #Revisar para remover ats, hashs y links
        
        words_writer.write(' '.join(w for w in words)+'\n')
        emo_writer.write(' '.join(w for w in emoticons+emojis)+'\n')
        hash_writer.write(' '.join(w for w in hashs)+'\n')
        at_writer.write(' '.join(w for w in ats)+'\n')
        link_writer.write(' '.join(w for w in links)+'\n')
        i += 1
        print(i)
