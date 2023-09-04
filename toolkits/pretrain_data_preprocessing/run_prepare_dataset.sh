#! /bin/bash

#clean
python clean_raw_text.py -i WuDaoCorpus2.0_base_200G  -o cleaned_wudao_dataset -p 32

#merge
find cleaned_wudao_dataset -name "*.json" -exec cat {} + > /cpfs01/user/paigpt/wudao/merged_wudao_cleaned.json

#build zst
split -l 6000000 --numeric-suffixes --additional-suffix=.jsonl /cpfs01/user/paigpt/wudao/merged_wudao_cleaned.json /cpfs01/user/paigpt/wudao/
zstd -z /cpfs01/user/paigpt/wudao/00.jsonl -o /cpfs01/user/paigpt/wudao/00.jsonl.zst &
zstd -z /cpfs01/user/paigpt/wudao/01.jsonl -o /cpfs01/user/paigpt/wudao/01.jsonl.zst &
zstd -z /cpfs01/user/paigpt/wudao/02.jsonl -o /cpfs01/user/paigpt/wudao/02.jsonl.zst &
zstd -z /cpfs01/user/paigpt/wudao/03.jsonl -o /cpfs01/user/paigpt/wudao/03.jsonl.zst &
zstd -z /cpfs01/user/paigpt/wudao/04.jsonl -o /cpfs01/user/paigpt/wudao/04.jsonl.zst &
zstd -z /cpfs01/user/paigpt/wudao/05.jsonl -o /cpfs01/user/paigpt/wudao/05.jsonl.zst &
zstd -z /cpfs01/user/paigpt/wudao/06.jsonl -o /cpfs01/user/paigpt/wudao/06.jsonl.zst &
zstd -z /cpfs01/user/paigpt/wudao/07.jsonl -o /cpfs01/user/paigpt/wudao/07.jsonl.zst &
zstd -z /cpfs01/user/paigpt/wudao/08.jsonl -o /cpfs01/user/paigpt/wudao/08.jsonl.zst &
zstd -z /cpfs01/user/paigpt/wudao/09.jsonl -o /cpfs01/user/paigpt/wudao/09.jsonl.zst &
