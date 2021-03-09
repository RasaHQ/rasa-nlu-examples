This name list is created using the [public PESEL registry][1] of all the names of all the living,
registered citizens of Poland as of 31 January 2021. The lists published in CSV files were
cleaned up using the script below. The list contains names of Polish citizens, not only
names of Polish origin. The list contains names of living citizens, so may be biased against
historical names. The preprocessing did not account for the multi-word names like "da Cruz".

```shell
$ ls Wykaz_imion_* \                # take all the files
	| xargs -n 1 tail -n +2 \       # remove headers
	| awk -F ',' '{ print $1 }' \   # take only the first column of the csv file
	| sed 's/ /\n/g' \              # split by space (in case of two names)
	| sed 's/\.//g' \               # remove dots (in case of initials e.g. "SŁAWOMIR M.")
	| sed -nr '/^.{2,}$/p' \        # filter-out all the lines shorter than two chars
	| sort | uniq \                 # sort, keep only unique rows
	| sed -e 's/./\L\0/g' \         # transform unicode uppercase letters to lowercase
	> names.txt
```

```shell
$ wc -l names.txt 
40966 names.txt
```

 [1]: https://dane.gov.pl/pl/dataset/1667,lista-imion-wystepujacych-w-rejestrze-pesel-osoby-zyjace
