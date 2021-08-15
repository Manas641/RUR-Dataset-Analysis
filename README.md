# RUR-Dataset-Analysis

ReadMe2 file has details of the dataset used in this analysis.

A brief account of column modifications and creations.
1) stime and utime were in clock ticks (AMD Opteron 6274 with a base clock of 2.2 GHz) and was converted to seconds using this value.
2) job time was calcuated from the time delta (start time - end time) in seconds.  
3) 'ttime' refers to the total time in the cpu = stime + utime.
4) 'gpu_summem' and 'gpu_maxmem' were calculated to be in bytes and was converted to GB.
5) 'gpu_use_status' is equal to one if 'gpu_summem' is non-zero.
6) 'max_rss' was estimated to be in KB* and was converted to GB. (*This value could have been KiB, hence a conversion from GB to GiB could be necessary)
7) 'sorf' is the success or failure indicator, which is 1 if alps_exit is equal to 0, else is 0.
8) Certain rows had missing values, these were very few in number and was dropped from the tables.
9) 'gpu_usg' column refers to the ratio of time spent on gpus to the job time.
10) 'app_cpu_usg' column refers to the ratio of total time spent on cpu* to job time. (*A base of clock of 2.2 GHz was used, though a max boost of 3.1 GHz)
11) 'file_system' column refers to the file system used and can take the values 'atlas', 'atlas1', 'atlas2', 'atlas1_thin' and 'NA'.This is obtained from the 
    'command' column.
12) 'Area' column refers to the project area, e.g, nanoscience,etc and 'Science' column refers to the Science domain, e.g, biology, chemistry, etc, obtained by merging
    the project ID obtained from the 'command' column and the table from projectAreas.csv.
13) Each of the RUR dataset(2015.csv - 2019.csv) was partitioned into four datasets, tx_y, where x = {25,50,75,100} and y = {15,16,17,18,19},
    this partition is based on 0-25, 25-50, 50-75 and 75-100 percentile of job times each year (y).
14) 'r/w' column is the ratio of rchar to wchar, provided they are not zero.
15) 'u/s' column is the ratio of utime to stime,provided they are not zero.
16) 'Date' column is obtained from the files (one for each year) which were created by extracting the date from start time for each job for convinience.
