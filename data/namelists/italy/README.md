# Italian first names

The source is the github repo: https://github.com/napolux/paroleitaliane/blob/master/paroleitaliane/9000_nomi_propri.txt (MIT license).
I just corrected some wovels accents.


```bash
# first names
wget https://raw.githubusercontent.com/napolux/paroleitaliane/master/paroleitaliane/9000_nomi_propri.txt
 
cat 9000_nomi_propri.txt | sort -u | sed 's/ë/è/g' 9000_nomi_propri.txt > names.txt

```
