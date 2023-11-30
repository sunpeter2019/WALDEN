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
#include <thread>
#include <tuple>
#include <unordered_set>

// Modify these if running your own workload
#define KEY_TYPE double                            // need modifie
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

void thread_2(  WALDEN<KEY_TYPE, double, double> & index, std::tuple<KEY_TYPE, PAYLOAD_TYPE, WEIGHT_TYPE>* values,int act_num, bool & rebuild_finish)
{
    index.re_load(values, act_num);
    rebuild_finish =true;
}

void cal_insert( KEY_TYPE keys2[], int count2,double *& gmm_para , double *& gmm_para2, int & dis_shift_mark ,KEY_TYPE keys[], int i)
{
        std::cout << " cal distribution for insert key: \n";
        auto calinsert_start_time = std::chrono::high_resolution_clock::now();   

        gmm_para2 = gen_fit(keys2, count2/10 );

        auto calinsert_end_time = std::chrono::high_resolution_clock::now();
        double calinsert_time = std::chrono::duration_cast<std::chrono::nanoseconds> (calinsert_end_time - calinsert_start_time).count();
        std::cout << "calinsert_time " << calinsert_time << std::endl;
    
        double KL =cal_divergence(keys2, gmm_para, gmm_para2, count2 );
        std::cout << "KL_divergence:  "<< KL << std::endl ; 

        if( KL >1){
            std::cout << " distribution shift for all key \n";
            dis_shift_mark ++;
        }
        if(dis_shift_mark>=2){
            std::cout << "retrain distribution for all key \n";
            vector<KEY_TYPE> key_retrain( keys + i - 6000000 ,keys + i);

            auto retrain_start_time = std::chrono::high_resolution_clock::now();   

            gmm_para = gen_fit(key_retrain, 6000000 );

            auto retrain_end_time = std::chrono::high_resolution_clock::now();
            double retrain_time = std::chrono::duration_cast<std::chrono::nanoseconds> (retrain_end_time - retrain_start_time).count();
            std::cout << "retrain_time " << retrain_time << std::endl;
        }
    
}

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

  std::unordered_set<KEY_TYPE> ab(keys,keys+ init_num_keys);


  /*
  compute key distribution by GMM;  enable below-->     
  */
   double* gmm_para = nullptr;                 //other workload
//    double para[13] = {4,0.554173,0.0827103,0.2382,0.124917,-15697,9317.6,1932.77,-21178.6,3.67237e+06,2.37463e+08,2.88089e+06,794161};
//    gmm_para = para;
   gmm_para = gen_fit(keys, init_num_keys );

  std::cout << " print para \n";
  for(int a=0;a< gmm_para[0]*3+1;a++)
  {
    std::cout << gmm_para[a] << "," ;
  }                                                
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
  std::cout << "\n cal distribution for initial key \n";

  auto values = new std::tuple<KEY_TYPE, PAYLOAD_TYPE, WEIGHT_TYPE>[total_num_keys]; 
  std::mt19937_64 gen_payload(std::random_device{}());

  int act_num = 0;

  
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
      std::get<2>(values[act_num]) = static_cast<WEIGHT_TYPE>(1 + kweight * gen_weight_data_p(*it, gmm_para)   );   
      act_num++;

  }

  
  // Create WALDEN and bulk load    
  WALDEN<KEY_TYPE, double, double> index;
  WALDEN<KEY_TYPE, double, double> *p;
  WALDEN<KEY_TYPE, double, double> index_re;


  std::sort(values, values + act_num,
            [](auto const& a, auto const& b) { return std::get<0>(a)  < std::get<0>(b) ; });   //return a.first < b.first  
  index.bulk_load(values, act_num);
  
  p = &index;

  std::cout << "check point bulkload \n";

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
  auto rebuild_start = false;
  auto rebuild_finish = false;
  double* gmm_para2 = nullptr;                 //other workload
  thread second;
  thread first;
  auto index_tag = true;

  std::cout << std::scientific;
  std::cout << std::setprecision(3);
  while (true) {
    batch_no++;
    

    // Do inserts
    int num_actual_inserts =
        std::min(num_inserts_per_batch, total_num_keys - i);
    int num_keys_after_batch = i + num_actual_inserts;


    auto temp = new std::pair<KEY_TYPE, double>[num_actual_inserts];
    int count = 0;
    auto keys2 = new KEY_TYPE[num_actual_inserts];
    int count2 = 0;

    auto gen_insert_start_time = std::chrono::high_resolution_clock::now();

    for (; i < num_keys_after_batch; i++) {
        keys2[count2++] =   keys[i];
        if(ab.find(keys[i] ) == ab.end() ){
            ab.emplace(keys[i]);

            temp[count].first = keys[i];
            std::get<0>(values[act_num]) = keys[i];
            std::get<1>(values[act_num]) = static_cast<PAYLOAD_TYPE>(gen_payload());
        
            if(dis_shift_mark==0){
            temp[count].second = 1 + kweight *gen_weight_data_p(keys[i], gmm_para);    
            std::get<2>(values[act_num]) = static_cast<WEIGHT_TYPE>(1 + kweight * gen_weight_data_p(keys[i], gmm_para)   );
            }
            else{
            temp[count].second = 1 + kweight * (0.5*gen_weight_data_p(keys[i], gmm_para)+ 0.5*gen_weight_data_p(keys[i], gmm_para2)); 
            std::get<2>(values[act_num]) = static_cast<WEIGHT_TYPE>(1 + kweight * gen_weight_data_p(keys[i], gmm_para)   );
            }

            count++;
            act_num++;
        }
    }

    auto gen_insert_end_time = std::chrono::high_resolution_clock::now();
    double gen_insert_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              gen_insert_end_time - gen_insert_start_time)
                              .count();
    // std::cout << "gen_insert_time " << gen_insert_time << std::endl;    

    first=thread(cal_insert, keys2, count2, ref(gmm_para), ref(gmm_para2), ref(dis_shift_mark) ,keys, i);
    first.join();
    
    auto inserts_start_time = std::chrono::high_resolution_clock::now();    


    
    for (int i=0; i<count; i++) {
        
      p->insert( temp[i].first , static_cast<PAYLOAD_TYPE>(gen_payload()) ,temp[i].second  );   
                             
    }

    auto inserts_end_time = std::chrono::high_resolution_clock::now();


    double batch_insert_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(inserts_end_time -
                                                             inserts_start_time)
            .count();
    cumulative_insert_time += batch_insert_time;
    cumulative_inserts += num_actual_inserts;


    // Do lookups
    double batch_lookup_time = 0.0;
    

    auto gen_lookups_start_time = std::chrono::high_resolution_clock::now();
               
           
        KEY_TYPE* lookup_keys = nullptr;                
        
        if (lookup_distribution == "uniform") {
            lookup_keys = get_search_keys(keys, init_num_keys, total_num_keys, num_lookups_per_batch, ab); 
        }
        else if (lookup_distribution == "zipf") {
            lookup_keys = get_search_keys_zipf(keys, i, num_lookups_per_batch);
        }
        else if (lookup_distribution == "normal") {
            //std::cout << "check point fuck1 \n";
            lookup_keys = get_search_keys_normal(keys, i, num_lookups_per_batch);                      
        }
        else if (lookup_distribution == "real") {
            //auto lookupkeys = new KEY_TYPE[20000000];
            load_binary_data(lookup_keys, 20000000, "../data/testdata2-50.bin");                   
        }
        else {
            std::cerr << "--lookup_distribution must be either 'uniform' or 'zipf'"
                << std::endl;
            return 1;
        }

    auto gen_lookups_end_time = std::chrono::high_resolution_clock::now();
    double gen_lookup_time  = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              gen_lookups_end_time - gen_lookups_start_time)
                              .count();
    std::cout << "gen_lookup_time " << gen_lookup_time << std::endl;


      auto lookups_start_time = std::chrono::high_resolution_clock::now();
      for (int j = 0; j < num_lookups_per_batch; j++) {
          KEY_TYPE key = lookup_keys[j];   

          PAYLOAD_TYPE payload = p->at(key);

      }
      auto lookups_end_time = std::chrono::high_resolution_clock::now();

      batch_lookup_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              lookups_end_time - lookups_start_time)
                              .count();
      cumulative_lookup_time += batch_lookup_time;
      cumulative_lookups += num_lookups_per_batch;



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


    //test for distribution shift
    double rebuild_time = 0;

    auto dis_shift = true; 
    if(dis_shift == true){
        //double* gmm_para2 = nullptr;                 //other workload
        // std::cout << " cal distribution for insert key: \n";

        // auto calinsert_start_time = std::chrono::high_resolution_clock::now();   

        // gmm_para2 = gen_fit(keys2, count2/10 );

        // auto calinsert_end_time = std::chrono::high_resolution_clock::now();
        // double calinsert_time = std::chrono::duration_cast<std::chrono::nanoseconds> (calinsert_end_time - calinsert_start_time).count();
        // std::cout << "calinsert_time " << calinsert_time << std::endl;

        // //gmm_para2 = gen_fit(lookup_keys, num_lookups_per_batch );
        // double KL =cal_divergence(keys2, gmm_para, gmm_para2, count2 );
        // //double KL =cal_divergence(lookup_keys, gmm_para, gmm_para2, num_lookups_per_batch );
        // std::cout << "KL_divergence:  "<< KL << std::endl ; 

        // if( KL >1){
        //     std::cout << " distribution shift for all key \n";
        //     //gmm_para = gen_fit(keys, i );
        //     dis_shift_mark ++;
        // }

        // first=thread(cal_insert, keys2, count2, ref(gmm_para), ref(gmm_para2), ref(dis_shift_mark) ,keys, i);
        // first.join();

        if(dis_shift_mark>=2){
            // std::cout << "retrain distribution for all key \n";
            // vector<KEY_TYPE> key_retrain( keys + i - 6000000 ,keys + i);

            // auto retrain_start_time = std::chrono::high_resolution_clock::now();   

            // gmm_para = gen_fit(key_retrain, 6000000 );

            // auto retrain_end_time = std::chrono::high_resolution_clock::now();
            // double retrain_time = std::chrono::duration_cast<std::chrono::nanoseconds> (retrain_end_time - retrain_start_time).count();
            // std::cout << "retrain_time " << retrain_time << std::endl;

            dis_shift_mark =0;

            for(int i=0; i<act_num; i++ ){
                std::get<2>(values[i]) = static_cast<WEIGHT_TYPE>(1 + kweight * gen_weight_data_p(std::get<0>(values[i]), gmm_para)   );
            }
            std::sort(values, values + act_num,
            [](auto const& a, auto const& b) { return std::get<0>(a)  < std::get<0>(b) ; });   //return a.first < b.first  
            
            // auto temp = new std::tuple<KEY_TYPE, PAYLOAD_TYPE, WEIGHT_TYPE>[1];
            // std::tuple<double, double,double> fifth((double)1,(double)1,(double)1); 
            // temp[0] = fifth;

            // index.bulk_load( temp,1);
            // delete[] temp;
            // WALDEN<KEY_TYPE, double, double> index_re;
            if(index_tag==true){

            rebuild_start=true;

            auto temp = new std::tuple<KEY_TYPE, PAYLOAD_TYPE, WEIGHT_TYPE>[1];
            std::tuple<double, double,double> fifth((double)1,(double)1,(double)1); 
            temp[0] = fifth;
            index_re.bulk_load( temp,1);    

            second=thread(thread_2, ref(index_re) ,values,act_num,ref(rebuild_finish));
            // index.re_load(values, act_num);
            index_tag=false;
            }
            else{
            rebuild_start=true;

            auto temp = new std::tuple<KEY_TYPE, PAYLOAD_TYPE, WEIGHT_TYPE>[1];
            std::tuple<double, double,double> fifth((double)1,(double)1,(double)1); 
            temp[0] = fifth;
            index.bulk_load( temp,1); 

            second=thread(thread_2, ref(index) ,values,act_num,ref(rebuild_finish));    
            index_tag=true;
            }

            continue;

        }
    }

    if (rebuild_finish==true)  //
    {
        // second.join(); 
        if(index_tag==false){
        p = &index_re;
        }
        else{
        p = &index;    
        }
        auto rebuild_inserts_start_time = std::chrono::high_resolution_clock::now();    
 
        for (int i=0; i<count; i++) {   
            p->insert( temp[i].first , static_cast<PAYLOAD_TYPE>(gen_payload()) ,temp[i].second  );                             
        }
        
        auto rebuild_inserts_end_time = std::chrono::high_resolution_clock::now();        
        double rebuild_batch_insert_time = std::chrono::duration_cast<std::chrono::nanoseconds>(inserts_end_time -
                                                             inserts_start_time).count();
        std::cout << "rebuild_batch_insert_time " << rebuild_batch_insert_time << std::endl;
        rebuild_finish= false;
    }
    

    // Check for workload end conditions
    if (num_actual_inserts < num_inserts_per_batch) {
      // End if we have inserted all keys in a workload with inserts
      break;
    }
    //std::cout << "check point 4 \n";
    double workload_elapsed_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - workload_start_time)
            .count();
    // if (workload_elapsed_time > time_limit * 1e9 * 60 * 150) {               //control exit time  *10
    //   break;
    // }
    if (batch_no > 9) {                                                            //batch_no
        break;
    }

  }


  second.join(); 

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
  delete gmm_para;
  delete gmm_para2;

  // show tree structure
  //index.show();
  p->print_stats();
  p->print_depth();
  
}


