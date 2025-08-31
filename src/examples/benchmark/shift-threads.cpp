
#include <walden.h>

#include <iomanip>
#include <unordered_set>
#include <functional>
#include <chrono>
#include <random>
#include <fstream>
#include <iostream>
#include <thread>
#include <tuple>
#include <mutex>
#include <queue>
#include <atomic>
#include <condition_variable>
#include <vector>
#include "flags.h"
#include "utils.h"

// 数据类型定义
#define KEY_TYPE double
#define PAYLOAD_TYPE double
#define WEIGHT_TYPE double


WALDEN<KEY_TYPE, PAYLOAD_TYPE, WEIGHT_TYPE> g_index;
WALDEN<KEY_TYPE, PAYLOAD_TYPE, WEIGHT_TYPE> g_index_re;

WALDEN<KEY_TYPE, PAYLOAD_TYPE, WEIGHT_TYPE>* g_active_index = &g_index;

std::atomic<bool> g_index_tag{true};

std::tuple<KEY_TYPE, PAYLOAD_TYPE, WEIGHT_TYPE>* g_values = nullptr;

std::atomic<int> g_act_num{0};

int g_kweight = 0;

double* g_gmm_para = nullptr;
double* g_gmm_para2 = nullptr;

// 线程同步相关
std::mutex g_values_mutex;       
std::mutex g_index_mutex;       
std::mutex g_queue_mutex;       
std::mutex g_rebuild_mutex;    
std::condition_variable g_rebuild_cv; 

// 重建状态标记
std::atomic<bool> g_rebuild_needed{false};   
std::atomic<bool> g_rebuild_in_progress{false};
std::atomic<int> g_dis_shift_mark{0};         
std::atomic<bool> g_stop_worker{false};       


using InsertBatch = std::vector<std::pair<KEY_TYPE, std::pair<PAYLOAD_TYPE, WEIGHT_TYPE>>>;
std::queue<InsertBatch> g_insert_queue;


void background_worker() {
    std::cout << "[Background] Worker started." << std::endl;
    while (!g_stop_worker) {
       
        std::unique_lock<std::mutex> rebuild_lock(g_rebuild_mutex);
        g_rebuild_cv.wait(rebuild_lock, []() { 
            return g_rebuild_needed || g_stop_worker; 
        });

       
        if (g_stop_worker) break;

       
        if (g_rebuild_needed && !g_rebuild_in_progress) {
            g_rebuild_in_progress = true;
            g_rebuild_needed = false; 
            rebuild_lock.unlock();   

            try {
                std::cout << "[Background] Start index rebuild..." << std::endl;
                auto rebuild_start = std::chrono::high_resolution_clock::now();

             
                std::vector<std::tuple<KEY_TYPE, PAYLOAD_TYPE, WEIGHT_TYPE>> temp_data;
                int current_act_num = 0;
                {
                    std::lock_guard<std::mutex> val_lock(g_values_mutex);
                    current_act_num = g_act_num;
                    temp_data.resize(current_act_num);
                   
                    for (int i = 0; i < current_act_num; ++i) {
                        temp_data[i] = g_values[i];
                     
                        std::get<2>(temp_data[i]) = static_cast<WEIGHT_TYPE>(
                            1 + g_kweight * gen_weight_data_p(std::get<0>(temp_data[i]), g_gmm_para)
                        );
                    }
                    // 按key排序（bulk_load要求）
                    std::sort(temp_data.begin(), temp_data.end(),
                        [](const auto& a, const auto& b) { 
                            return std::get<0>(a) < std::get<0>(b); 
                        });
                }

    
                bool current_tag = g_index_tag;
                WALDEN<KEY_TYPE, PAYLOAD_TYPE, WEIGHT_TYPE>& new_index = 
                    current_tag ? g_index_re : g_index;

       
                auto dummy = new std::tuple<KEY_TYPE, PAYLOAD_TYPE, WEIGHT_TYPE>[1];
                *dummy = std::make_tuple(static_cast<KEY_TYPE>(1), 
                                         static_cast<PAYLOAD_TYPE>(1), 
                                         static_cast<WEIGHT_TYPE>(1));
                new_index.bulk_load(dummy, 1);
                delete[] dummy;

                new_index.re_load(temp_data.data(), current_act_num);


                std::cout << "[Background] Start reinserting queued data..." << std::endl;
                InsertBatch all_inserts;
                {
                    std::lock_guard<std::mutex> q_lock(g_queue_mutex);
                    while (!g_insert_queue.empty()) {
                        auto& batch = g_insert_queue.front();
                        all_inserts.insert(all_inserts.end(), 
                                          std::make_move_iterator(batch.begin()), 
                                          std::make_move_iterator(batch.end()));
                        g_insert_queue.pop();
                    }
                }

                for (const auto& item : all_inserts) {
                    const auto& key = item.first;
                    const auto& [payload, weight] = item.second;
                    new_index.insert(key, payload, weight);
                }
                std::cout << "[Background] Reinserted " << all_inserts.size() << " entries." << std::endl;


                {
                    std::lock_guard<std::mutex> idx_lock(g_index_mutex);
                    g_index_tag = !current_tag;       
                    g_active_index = &new_index;      
                }

                g_rebuild_in_progress = false;
                g_dis_shift_mark = 0; 
                auto rebuild_end = std::chrono::high_resolution_clock::now();
                double rebuild_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    rebuild_end - rebuild_start
                ).count();
                std::cout << "[Background] Rebuild completed (time: " << rebuild_time << "ms). Switched to new index." << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[Background] Rebuild failed: " << e.what() << std::endl;
                g_rebuild_in_progress = false; 
            }
        } else {
            rebuild_lock.unlock();
        }

        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    std::cout << "[Background] Worker stopped." << std::endl;
}


void cal_insert(KEY_TYPE keys2[], int count2) {
    std::cout << "[Main] Calculate distribution for new keys..." << std::endl;
    auto cal_start = std::chrono::high_resolution_clock::now();

    g_gmm_para2 = gen_fit(keys2, count2 / 10);

    double KL = cal_divergence(keys2, g_gmm_para, g_gmm_para2, count2);
    auto cal_end = std::chrono::high_resolution_clock::now();
    double cal_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cal_end - cal_start
    ).count();

    std::cout << "[Main] KL divergence: " << KL << " (calc time: " << cal_time << "ms)" << std::endl;
    if (KL > 1) {
        std::cout << "[Main] Distribution shift detected. Mark count: " << ++g_dis_shift_mark << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // 1. 解析命令行参数
    auto flags = parse_flags(argc, argv);
    std::string keys_file_path = get_required(flags, "keys_file");
    std::string keys_file_type = get_required(flags, "keys_file_type");
    int init_num_keys = stoi(get_required(flags, "init_num_keys"));
    int total_num_keys = stoi(get_required(flags, "total_num_keys"));
    int batch_size = stoi(get_required(flags, "batch_size"));
    double insert_frac = stod(get_with_default(flags, "insert_frac", "0.5"));
    std::string lookup_distribution = get_with_default(flags, "lookup_distribution", "zipf");
    double time_limit = stod(get_with_default(flags, "time_limit", "0.5"));
    bool print_batch_stats = get_boolean_flag(flags, "print_batch_stats");
    g_kweight = stoi(get_required(flags, "kweight"));
    int weight_file_type = stoi(get_required(flags, "weight_file_type"));

    // 2. 读取keys
    auto keys = new KEY_TYPE[total_num_keys];
    if (keys_file_type == "binary") {
        load_binary_data(keys, total_num_keys, keys_file_path);
    } else if (keys_file_type == "text") {
        load_text_data(keys, total_num_keys, keys_file_path);
    } else {
        std::cerr << "--keys_file_type must be 'binary' or 'text'" << std::endl;
        return 1;
    }
    std::unordered_set<KEY_TYPE> existing_keys(keys, keys + init_num_keys);

    g_gmm_para = gen_fit(keys, init_num_keys);
    g_values = new std::tuple<KEY_TYPE, PAYLOAD_TYPE, WEIGHT_TYPE>[total_num_keys];
    std::mt19937_64 gen_payload(std::random_device{}());
）
    {
        std::lock_guard<std::mutex> val_lock(g_values_mutex);
        for (const auto& key : existing_keys) {
            std::get<0>(g_values[g_act_num]) = key;
            std::get<1>(g_values[g_act_num]) = static_cast<PAYLOAD_TYPE>(gen_payload());
            std::get<2>(g_values[g_act_num]) = static_cast<WEIGHT_TYPE>(
                1 + g_kweight * gen_weight_data_p(key, g_gmm_para)
            );
            g_act_num++;
        }
    }

    g_index.bulk_load(g_values, g_act_num);
    std::cout << "[Main] Initial bulk load completed. Total keys: " << g_act_num << std::endl;

    std::thread bg_worker(background_worker);
    bg_worker.detach(); // 分离线程，主线程不等待其退出


    int i = init_num_keys;
    long long cumulative_inserts = 0;
    long long cumulative_lookups = 0;
    int num_inserts_per_batch = static_cast<int>(batch_size * insert_frac);
    int num_lookups_per_batch = batch_size - num_inserts_per_batch;
    double cumulative_insert_time = 0;
    double cumulative_lookup_time = 0;
    auto workload_start = std::chrono::high_resolution_clock::now();
    int batch_no = 0;
    PAYLOAD_TYPE sum_payload = 0;

    std::cout << "[Main] Start workload. Batch size: " << batch_size 
              << " (inserts: " << num_inserts_per_batch << ", lookups: " << num_lookups_per_batch << ")" << std::endl;
    std::cout << std::scientific << std::setprecision(3);

    while (true) {
        batch_no++;
        auto batch_start = std::chrono::high_resolution_clock::now();

        int num_actual_inserts = std::min(num_inserts_per_batch, total_num_keys - i);
        int num_keys_after_batch = i + num_actual_inserts;
        auto temp_inserts = new std::pair<KEY_TYPE, WEIGHT_TYPE>[num_actual_inserts];
        auto new_keys = new KEY_TYPE[num_actual_inserts];
        int insert_count = 0;
        int new_key_count = 0;

        auto gen_insert_start = std::chrono::high_resolution_clock::now();
        {
            std::lock_guard<std::mutex> val_lock(g_values_mutex);
            for (; i < num_keys_after_batch; i++) {
                KEY_TYPE key = keys[i];
                new_keys[new_key_count++] = key;

                if (existing_keys.count(key) != 0) continue;
                existing_keys.emplace(key);

                WEIGHT_TYPE weight;
                if (g_dis_shift_mark == 0) {
                    weight = 1 + g_kweight * gen_weight_data_p(key, g_gmm_para);
                } else {
                    weight = 1 + g_kweight * (
                        0.5 * gen_weight_data_p(key, g_gmm_para) + 
                        0.5 * gen_weight_data_p(key, g_gmm_para2)
                    );
                }
                temp_inserts[insert_count++] = {key, weight};

                std::get<0>(g_values[g_act_num]) = key;
                std::get<1>(g_values[g_act_num]) = static_cast<PAYLOAD_TYPE>(gen_payload());
                std::get<2>(g_values[g_act_num]) = weight;
                g_act_num++;
            }
        }
        auto gen_insert_end = std::chrono::high_resolution_clock::now();
        double gen_insert_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            gen_insert_end - gen_insert_start
        ).count();

        InsertBatch current_batch;
        auto insert_start = std::chrono::high_resolution_clock::now();
        {
            std::lock_guard<std::mutex> idx_lock(g_index_mutex); // 保护活跃索引访问
            for (int j = 0; j < insert_count; j++) {
                auto [key, weight] = temp_inserts[j];
                PAYLOAD_TYPE payload = static_cast<PAYLOAD_TYPE>(gen_payload());

                g_active_index->insert(key, payload, weight);

                if (g_rebuild_in_progress) {
                    current_batch.emplace_back(key, std::make_pair(payload, weight));
                }
            }
        }

        if (!current_batch.empty()) {
            std::lock_guard<std::mutex> q_lock(g_queue_mutex);
            g_insert_queue.push(std::move(current_batch));
        }
        auto insert_end = std::chrono::high_resolution_clock::now();
        double batch_insert_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            insert_end - insert_start
        ).count();
        cumulative_insert_time += batch_insert_time;
        cumulative_inserts += insert_count;

        double batch_lookup_time = 0;
        KEY_TYPE* lookup_keys = nullptr;
        auto gen_lookup_start = std::chrono::high_resolution_clock::now();

        if (lookup_distribution == "uniform") {
            lookup_keys = get_search_keys(keys, init_num_keys, total_num_keys, num_lookups_per_batch, existing_keys);
        } else if (lookup_distribution == "zipf") {
            lookup_keys = get_search_keys_zipf(keys, i, num_lookups_per_batch);
        } else if (lookup_distribution == "normal") {
            lookup_keys = get_search_keys_normal(keys, i, num_lookups_per_batch);
        } else if (lookup_distribution == "real") {
            lookup_keys = new KEY_TYPE[20000000];
            load_binary_data(lookup_keys, 20000000, "../data/testdata2-50.bin");
        } else {
            std::cerr << "--lookup_distribution must be 'uniform'/'zipf'/'normal'/'real'" << std::endl;
            return 1;
        }
        auto gen_lookup_end = std::chrono::high_resolution_clock::now();
        double gen_lookup_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            gen_lookup_end - gen_lookup_start
        ).count();

        auto lookup_start = std::chrono::high_resolution_clock::now();
        {
            std::lock_guard<std::mutex> idx_lock(g_index_mutex);
            for (int j = 0; j < num_lookups_per_batch; j++) {
                KEY_TYPE key = lookup_keys[j];
                PAYLOAD_TYPE payload = g_active_index->at(key);
                if (payload != 0) sum_payload += payload;
            }
        }
        auto lookup_end = std::chrono::high_resolution_clock::now();
        batch_lookup_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            lookup_end - lookup_start
        ).count();
        cumulative_lookup_time += batch_lookup_time;
        cumulative_lookups += num_lookups_per_batch;

        if (new_key_count > 0) {
            std::thread cal_thread(cal_insert, new_keys, new_key_count);
            cal_thread.detach(); 
        }

        if (g_dis_shift_mark >= 2 && !g_rebuild_in_progress) {
            std::cout << "[Main] Distribution shift count >=2. Request background rebuild..." << std::endl;
            std::lock_guard<std::mutex> rebuild_lock(g_rebuild_mutex);
            g_rebuild_needed = true;
            g_rebuild_cv.notify_one(); // 唤醒后台线程
        }

        if (print_batch_stats) {
            int total_batch_ops = num_lookups_per_batch + insert_count;
            double batch_time = batch_insert_time + batch_lookup_time;
            long long total_cum_ops = cumulative_lookups + cumulative_inserts;
            double total_cum_time = cumulative_lookup_time + cumulative_insert_time;

            std::cout << "Batch " << batch_no
                      << " | Cumulative Ops: " << total_cum_ops
                      << " | Batch Lookup Throughput: " << (num_lookups_per_batch / batch_lookup_time) * 1e9 << " ops/sec"
                      << " | Batch Insert Throughput: " << (insert_count / batch_insert_time) * 1e9 << " ops/sec"
                      << " | Cumulative Throughput: " << (total_cum_ops / total_cum_time) * 1e9 << " ops/sec" << std::endl;
        }

        delete[] temp_inserts;
        delete[] new_keys;
        if (lookup_distribution == "real") delete[] lookup_keys;

        if (num_actual_inserts < num_inserts_per_batch) {
            std::cout << "[Main] All keys inserted. Exit workload." << std::endl;
            break;
        }
        double workload_elapsed = std::chrono::duration_cast<std::chrono::minutes>(
            std::chrono::high_resolution_clock::now() - workload_start
        ).count();
        if (workload_elapsed > time_limit) {
            std::cout << "[Main] Time limit reached. Exit workload." << std::endl;
            break;
        }
        if (batch_no > 9) {
            std::cout << "[Main] Batch limit reached. Exit workload." << std::endl;
            break;
        }
    }

    g_stop_worker = true;
    {
        std::lock_guard<std::mutex> rebuild_lock(g_rebuild_mutex);
        g_rebuild_cv.notify_one(); 
    }
    std::this_thread::sleep_for(std::chrono::seconds(2));

    long long total_cum_ops = cumulative_lookups + cumulative_inserts;
    double total_cum_time = cumulative_lookup_time + cumulative_insert_time;
    std::cout << "\n==================== Cumulative Stats ====================" << std::endl;
    std::cout << "Total Batches: " << batch_no
              << " | Total Ops: " << total_cum_ops
              << " (Lookups: " << cumulative_lookups << ", Inserts: " << cumulative_inserts << ")" << std::endl;
    std::cout << "Lookup Throughput: " << (cumulative_lookups / cumulative_lookup_time) * 1e9 << " ops/sec" << std::endl;
    std::cout << "Insert Throughput: " << (cumulative_inserts / cumulative_insert_time) * 1e9 << " ops/sec" << std::endl;
    std::cout << "Overall Throughput: " << (total_cum_ops / total_cum_time) * 1e9 << " ops/sec" << std::endl;

    delete[] keys;
    delete[] g_values;
    delete g_gmm_para;
    delete g_gmm_para2;

    std::lock_guard<std::mutex> idx_lock(g_index_mutex);
    g_active_index->print_stats();
    g_active_index->print_depth();

    return 0;
}
