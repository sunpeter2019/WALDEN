// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*
 * Simple benchmark that runs a mixture of point lookups and inserts on WALDEN.
 */


#include <walden.h>


#include <iomanip>
#include <unordered_set>
#include<functional>
#include <chrono>
#include <random>
#include <fstream>
#include <iostream>
#include "flags.h"
#include "utils.h"

#include <tuple>
#include <unordered_set>

// Modify these if running your own workload
#define KEY_TYPE double                          // need modifie
//#define KEY_TYPE uint64_t 


#define PAYLOAD_TYPE double
#define WEIGHT_TYPE double

/*
 * Required flags:
 * --keys_file              path to the file that contains keys
 * --keys_file_type         file type of keys_file (options: binary or text)
 * --init_num_keys          number of keys to bulk load with
 * --total_num_keys         total number of keys in the keys file
 * --batch_size             number of operations (lookup or insert) per batch
 *
 * Optional flags:
 * --insert_frac            fraction of operations that are inserts (instead of
 * lookups)
 * --lookup_distribution    lookup keys distribution (options: uniform or zipf)
 * --time_limit             time limit, in minutes
 * --print_batch_stats      whether to output stats for each batch
 */

int main(int argc, char* argv[]) {
  auto flags = parse_flags(argc, argv);
  std::string keys_file_path = get_required(flags, "keys_file");
  std::string keys_file_type = get_required(flags, "keys_file_type");
  auto init_num_keys = stoi(get_required(flags, "init_num_keys"));
  auto total_num_keys = stoi(get_required(flags, "total_num_keys"));
  auto batch_size = stoi(get_required(flags, "batch_size"));
  auto insert_frac = stod(get_with_default(flags, "insert_frac", "0.5"));
  std::string lookup_distribution =get_with_default(flags, "lookup_distribution", "zipf");
  auto time_limit = stod(get_with_default(flags, "time_limit", "0.5")); 
  bool print_batch_stats = get_boolean_flag(flags, "print_batch_stats");
  auto kweight = stoi(get_required(flags, "kweight"));
  auto weight_file_type = stoi(get_required(flags, "weight_file_type"));
  



  // Read keys from file
  auto keys = new KEY_TYPE[total_num_keys];
  if (keys_file_type == "binary") {
    load_binary_data(keys, total_num_keys, keys_file_path);
  } else if (keys_file_type == "text") {
    load_text_data(keys, total_num_keys, keys_file_path);
  } else {
    std::cerr << "--keys_file_type must be either 'binary' or 'text'"
              << std::endl;
    return 1;
  }

  //tracking keys(unique)
  std::unordered_set<KEY_TYPE> ab(keys,keys+ init_num_keys);
 
  /*
  compute key distribution by GMM;  enable below-->     
  */
//   double* gmm_para = nullptr;                 
//   gmm_para = gen_fit(keys, init_num_keys );


  auto values = new std::tuple<KEY_TYPE, PAYLOAD_TYPE, WEIGHT_TYPE>[init_num_keys];  
  std::mt19937_64 gen_payload(std::random_device{}());

  int act_num = 0;

  // generate weight automatically
  auto auto_weight = false; 
  if(auto_weight == true){      //test weight
 
    KEY_TYPE* temp_lookup_keys = nullptr;                 //
    temp_lookup_keys = get_search_keys(keys,0, total_num_keys, batch_size, ab);
    double  temp_time = 0.0;
    int temp_weight;
    
    for(int autow=1; autow <=11; autow +=3){
        int temp_act_num = 0;
        double batch_lookup_time = 0.0;
        if( autow ==1 ){
            for (std::unordered_set<KEY_TYPE>::iterator it = ab.begin();it != ab.end();it++) {  
                std::get<0>(values[temp_act_num]) = *it;
                std::get<1>(values[temp_act_num]) = static_cast<PAYLOAD_TYPE>(gen_payload());
                std::get<2>(values[temp_act_num]) = static_cast<WEIGHT_TYPE>(1 + autow * gen_weight_data(*it, weight_file_type)  );   
                temp_act_num++;      
            }
        }
        else{
            for (std::unordered_set<KEY_TYPE>::iterator it = ab.begin();it != ab.end();it++) {  
                //std::get<0>(values[act_num]) = *it;
                //std::get<1>(values[act_num]) = static_cast<PAYLOAD_TYPE>(gen_payload());
                std::get<2>(values[temp_act_num]) = static_cast<WEIGHT_TYPE>(1 + autow * gen_weight_data(*it, weight_file_type)  );   
                temp_act_num++;      
            }
        }
        WALDEN<KEY_TYPE, double, double> index;
        std::sort(values, values + temp_act_num,
                [](auto const& a, auto const& b) { return std::get<0>(a)  < std::get<0>(b) ; });   //return a.first < b.first 
        index.bulk_load(values, temp_act_num);
        
        auto lookups_start_time = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < batch_size; j++) {

            KEY_TYPE key = temp_lookup_keys[j];          
            PAYLOAD_TYPE payload = index.at(key);
            
        }
        auto lookups_end_time = std::chrono::high_resolution_clock::now();
        batch_lookup_time = std::chrono::duration_cast<std::chrono::nanoseconds>(lookups_end_time - lookups_start_time).count();
        
        if (autow == 1){
            temp_time = batch_lookup_time;
            temp_weight = autow;
        }
        else{
            if(batch_lookup_time < temp_time){
                temp_time = batch_lookup_time;
                temp_weight = autow;
            }
        }
    }
    
    kweight = temp_weight;    
    std::cout << "kweight: \n";
    std::cout << kweight << std::endl;
  }                

  // formal bulkload phase
  for (std::unordered_set<KEY_TYPE>::iterator it = ab.begin();it != ab.end();it++) {  
      std::get<0>(values[act_num]) = *it;
      std::get<1>(values[act_num]) = static_cast<PAYLOAD_TYPE>(gen_payload());
      std::get<2>(values[act_num]) = static_cast<WEIGHT_TYPE>(1 + kweight * gen_weight_data(*it, weight_file_type) );   //gen_weight_data_p(*it, gmm_para)       
      act_num++;
  }

  // Create WALDEN and bulk load    
  WALDEN<KEY_TYPE, double, double> index;


  std::sort(values, values + act_num,
            [](auto const& a, auto const& b) { return std::get<0>(a)  < std::get<0>(b) ; });   //return a.first < b.first  

  auto bulk_load_start = std::chrono::high_resolution_clock::now();
                                             
  index.bulk_load(values, act_num);

  auto bulk_load_end = std::chrono::high_resolution_clock::now();
  double bulk_load_index_time = std::chrono::duration_cast<std::chrono::nanoseconds>(bulk_load_end - bulk_load_start).count();
  std::cout << "bulk_load_index_time " << bulk_load_index_time << std::endl;



  // Run workload
  int i = init_num_keys;
  long long cumulative_inserts = 0;
  long long cumulative_lookups = 0;
  int num_inserts_per_batch = static_cast<int>(batch_size * insert_frac);
  int num_lookups_per_batch = batch_size - num_inserts_per_batch;
  double cumulative_insert_time = 0;
  double cumulative_lookup_time = 0;

  auto workload_start_time = std::chrono::high_resolution_clock::now();
  int batch_no = 0;
  PAYLOAD_TYPE sum = 0;
  
  auto dis_shift_mark = 0;
  double* gmm_para2 = nullptr;                 //for other workload

  std::cout << std::scientific;
  std::cout << std::setprecision(3);
  while (true) {
    batch_no++;

    // Do lookups
    double batch_lookup_time = 0.0;
    if (i > 0) {
                      
        KEY_TYPE* lookup_keys = nullptr;                 
        
        if (lookup_distribution == "uniform") {
            lookup_keys = get_search_keys(keys, 0, total_num_keys , num_lookups_per_batch, ab); //get_search_keys(keys, i , num_lookups_per_batch); 
        }
        else if (lookup_distribution == "zipf") {
            lookup_keys = get_search_keys_zipf(keys, i, num_lookups_per_batch);
        }
        else if (lookup_distribution == "normal") {

            lookup_keys = get_search_keys_normal(keys, i, num_lookups_per_batch);                        
        }
        else if (lookup_distribution == "real") {
            /*
                load from file
            */

            // auto lookupkeys = new KEY_TYPE[20000000];
            // load_binary_data(lookup_keys, 20000000, "../data/testdata2-50.bin");                 
        }
        else {
            std::cerr << "--lookup_distribution must be either 'uniform' or 'zipf'"
                << std::endl;
            return 1;
        }

    
      auto lookups_start_time = std::chrono::high_resolution_clock::now();
      for (int j = 0; j < num_lookups_per_batch; j++) {
          KEY_TYPE key = lookup_keys[j];     
          PAYLOAD_TYPE payload = index.at(key);
          if (payload) {
              sum += payload;
          }
      }

      auto lookups_end_time = std::chrono::high_resolution_clock::now();
      batch_lookup_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              lookups_end_time - lookups_start_time)
                              .count();
      cumulative_lookup_time += batch_lookup_time;
      cumulative_lookups += num_lookups_per_batch;
      delete[] lookup_keys;
    }



    // Do inserts
    int num_actual_inserts =
        std::min(num_inserts_per_batch, total_num_keys - i);
    int num_keys_after_batch = i + num_actual_inserts;


    auto temp = new std::pair<KEY_TYPE, double>[num_actual_inserts];
    int count = 0;
    auto keys2 = new KEY_TYPE[num_actual_inserts];
    int count2 = 0;

    for (; i < num_keys_after_batch; i++) {
        keys2[count2++] =   keys[i];
        if(ab.find(keys[i] ) == ab.end() ){
            ab.emplace(keys[i]);

            temp[count].first = keys[i];
            //std::get<0>(values[act_num]) = keys[i];
            //std::get<1>(values[act_num]) = static_cast<PAYLOAD_TYPE>(gen_payload());  
            temp[count].second = 1 + kweight *gen_weight_data(keys[i], weight_file_type);     //gen_weight_data(keys[i], weight_file_type)
        
            // weight[count]=1 + kweight *gen_weight_data(keys[i], dataset);
            count++;
            //act_num++;
        }
    }

    auto inserts_start_time = std::chrono::high_resolution_clock::now();    

    
    for (int i=0; i<count; i++) {
         
      index.insert( temp[i].first , static_cast<PAYLOAD_TYPE>(gen_payload()) ,temp[i].second  );   
                     
    }

    auto inserts_end_time = std::chrono::high_resolution_clock::now();
    double batch_insert_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(inserts_end_time -
                                                             inserts_start_time)
            .count();
    cumulative_insert_time += batch_insert_time;
    cumulative_inserts += num_actual_inserts;

    //test for distribution shift
    double rebuild_time = 0;
    auto dis_shift = false; 
    



    if (print_batch_stats) {
      int num_batch_operations = num_lookups_per_batch + num_actual_inserts;
      double batch_time = batch_lookup_time + batch_insert_time;
      long long cumulative_operations = cumulative_lookups + cumulative_inserts;
      double cumulative_time = cumulative_lookup_time + cumulative_insert_time ;
      std::cout << "Batch " << batch_no
                << ", cumulative ops: " << cumulative_operations
                << "\n\tbatch throughput:\t"
                << num_lookups_per_batch / batch_lookup_time * 1e9
                << " lookups/sec,\t"
                << num_actual_inserts / batch_insert_time * 1e9
                << " inserts/sec,\t" << num_batch_operations / batch_time * 1e9
                << " ops/sec"
                << "\n\tcumulative throughput:\t"
                << cumulative_lookups / cumulative_lookup_time * 1e9
                << " lookups/sec,\t"
                << cumulative_inserts / cumulative_insert_time * 1e9
                << " inserts/sec,\t"
                << cumulative_operations / cumulative_time * 1e9 << " ops/sec"
                << std::endl;
    }

    
    // Check for workload end conditions
    if (num_actual_inserts < num_inserts_per_batch) {
      // End if we have inserted all keys in a workload with inserts
      break;
    }

    double workload_elapsed_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - workload_start_time)
            .count();
    if (workload_elapsed_time > time_limit * 1e9 * 60 * 15) {                       //control exit time 
      break;
    }
    if (batch_no > 4) {                                                             //batch_no
        break;
    }
    
  }


  long long cumulative_operations = cumulative_lookups + cumulative_inserts;
  double cumulative_time = cumulative_lookup_time + cumulative_insert_time;
  std::cout << "Cumulative stats: " << batch_no << " batches, "
            << cumulative_operations << " ops (" << cumulative_lookups
            << " lookups, " << cumulative_inserts << " inserts)"
            << "\n\tcumulative throughput:\t"
            << cumulative_lookups / cumulative_lookup_time * 1e9
            << " lookups/sec,\t"
            << cumulative_inserts / cumulative_insert_time * 1e9
            << " inserts/sec,\t"
            << cumulative_operations / cumulative_time * 1e9 << " ops/sec"
            << std::endl;

  

  delete[] keys;
  delete[] values;

  // show tree structure
  //index.show();
  index.print_stats();
  index.print_depth();
  std::cout << "model_size " << index.model_size() << std::endl;
  std::cout << "index_size " << index.index_size() << std::endl;
}
