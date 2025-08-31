# WALDEN

Thanks for all of you for your interest in our work.


We propose a novel Workload-Aware
Learned Tree Index (WALDEN) which sets weights to keys and
arranges important keys into the shallower levels of the tree index.

This project contains the code of WALDEN and welcomes contributions or suggestions.


## Getting Started
- use cmake to build the project

      mkdir build
      cd build
      cmake ..
      make

- run simple example of walden
  
      ./build/example
      ./build/example_bulkload
  


## Run Benchmark
To run the benchmark

      ./build/benchmark \
      --keys_file=../data/[download location] \
      --keys_file_type=binary \
      --init_num_keys=100000000 \
      --total_num_keys=200000000 \
      --batch_size=20000000 \
      --insert_frac=0.5 \
      --lookup_distribution=uniform \
      --print_batch_stats=true \
      --kweight=10 \
      --weight_file_type=1
You can also run this benchmark on your own dataset. Your keys will need to be in either binary format or text format (one key per line). You will need to modify `#define KEY_TYPE double` to `#define KEY_TYPE [your data type]` in `src/benchmark/main.cpp`.      
You can compute the distribution of keyset with GMM that we provide  or replace it with other methods.

## Datasets
The datasets we used in paper are publicly available (all in binary format):
- [LONGLAT](https://registry.opendata.aws/osm/)
- [LONGITUDES](https://drive.google.com/file/d/1zc90sD6Pze8UM_XYDmNjzPLqmKly8jKl/view?usp=sharing)
- [OSM](https://s2geometry.io/)
- [BOOK](https://www.kaggle.com/ucffool/amazon-sales-rank-data-for-print-and-kindle-books)
- [GOWALLA](https://snap.stanford.edu/data/loc-gowalla.html)

|          | LLT   | GOW   | LTD   | OSM    | BOOK   |
|----------|-------|-------|-------|--------|--------|
| Num Keys | 200M  | 6.4M  | 200M  | 400M   | 800M   |
| Key Type | double| double| double| unint  | unint  |
| Payload Size | 8byte | 8byte | 8byte | 8byte | 8byte |

*Table: Statistics of datasets*


## Publication

## Acknowledgements

- Our implementation is based on the code of LIPP.

- The benchmark used to evaluate the WALDEN is ALEX.
