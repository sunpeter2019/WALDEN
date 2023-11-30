#ifndef __WALDEN_H__
#define __WALDEN_H__

#include "walden_base.h"
#include <stdint.h>
#include <math.h>
#include <limits>
#include <cstdio>
#include <stack>
#include <vector>
#include <cstring>
#include <sstream>
#include <tuple>

typedef uint8_t bitmap_t;
#define BITMAP_WIDTH (sizeof(bitmap_t) * 8)
#define BITMAP_SIZE(num_items) (((num_items) + BITMAP_WIDTH - 1) / BITMAP_WIDTH)
#define BITMAP_GET(bitmap, pos) (((bitmap)[(pos) / BITMAP_WIDTH] >> ((pos) % BITMAP_WIDTH)) & 1)
#define BITMAP_SET(bitmap, pos) ((bitmap)[(pos) / BITMAP_WIDTH] |= 1 << ((pos) % BITMAP_WIDTH))
#define BITMAP_CLEAR(bitmap, pos) ((bitmap)[(pos) / BITMAP_WIDTH] &= ~bitmap_t(1 << ((pos) % BITMAP_WIDTH)))
#define BITMAP_NEXT_1(bitmap_item) __builtin_ctz((bitmap_item))

// runtime assert
#define RT_ASSERT(expr) \
{ \
    if (!(expr)) { \
        fprintf(stderr, "RT_ASSERT Error at %s:%d, `%s`\n", __FILE__, __LINE__, #expr); \
        exit(0); \
    } \
}

#define COLLECT_TIME 0

#if COLLECT_TIME
#include <chrono>
#endif

template<class T, class P,class W, bool USE_WMCD = true>
class WALDEN
{
    static_assert(std::is_arithmetic<T>::value, "WALDEN key type must be numeric.");

    inline int compute_gap_count(int size) {
        if (size >= 1000000) return 1;
        if (size >= 100000) return 2;
        return 5;
    }

    struct Node;
    inline int PREDICT_POS(Node* node, T key) const {   
        double v = node->model.predict_double(key);
        if (v > std::numeric_limits<int>::max() / 2) {
            return node->num_items - 1;
        }
        if (v < 0) {
            return 0;
        }
        return std::min(node->num_items - 1, static_cast<int>(v));
    }

    static void remove_last_bit(bitmap_t& bitmap_item) {
        bitmap_item -= 1 << BITMAP_NEXT_1(bitmap_item);
    }

    const double BUILD_LR_REMAIN;
    const bool QUIET;

    struct {
        long long wmcd_success_times = 0;
        long long wmcd_broken_times = 0;
        #if COLLECT_TIME
        double time_scan_and_destory_tree = 0;
        double time_build_tree_bulk = 0;
        #endif
    } stats;

public:
    typedef std::tuple<T, P, W> V;

    WALDEN(double BUILD_LR_REMAIN = 0, bool QUIET = true)
        : BUILD_LR_REMAIN(BUILD_LR_REMAIN), QUIET(QUIET) {
        {
            std::vector<Node*> nodes;
            for (int _ = 0; _ < 1e7; _ ++) {
                Node* node = build_tree_two(T(0), P(), W(), T(1), P(), W());
                nodes.push_back(node);
            }
            for (auto node : nodes) {
                destroy_tree(node);
            }
            if (!QUIET) {
                printf("initial memory pool size = %lu\n", pending_two.size());
            }
        }
        if (USE_WMCD && !QUIET) {
            printf("enable WMCD\n");
        }

        root = build_tree_none();
    }
    ~WALDEN() {
        destroy_tree(root);
        root = NULL;
        destory_pending();  
    }

    void insert(const V& v) {
        //insert(v.first, v.second, v.third);
        insert(std::get<0>(v), std::get<1>(v), std::get<2>(v));
    }
    void insert(const T& key, const P& value, const W& weight) {
        root = insert_tree(root, key, value, weight);
    }
    void rebuild() {
        root = rebuild_tree(root);
    }

    P at(const T& key, bool skip_existence_check = true) const {
        Node* node = root;

        while (true) {
            int pos = PREDICT_POS(node, key);
            if (BITMAP_GET(node->child_bitmap, pos) == 1) {
                node = node->items[pos].comp.child;
            } else {
                if (skip_existence_check) {
                    return node->items[pos].comp.data.value;
                } else {
                    if (BITMAP_GET(node->none_bitmap, pos) == 1) {
                        RT_ASSERT(false);
                    } else if (BITMAP_GET(node->child_bitmap, pos) == 0) {
                        RT_ASSERT(node->items[pos].comp.data.key == key);
                        return node->items[pos].comp.data.value;
                    }
                }
            }
        }
    }

    bool exists(const T& key) const {
        Node* node = root;
        while (true) {
            int pos = PREDICT_POS(node, key);
            if (BITMAP_GET(node->none_bitmap, pos) == 1) {
                return false;
            } else if (BITMAP_GET(node->child_bitmap, pos) == 0) {
                return node->items[pos].comp.data.key == key;
            } else {
                node = node->items[pos].comp.child;
            }
        }
    }

    void bulk_load(const V* vs, int num_keys) {
        if (num_keys == 0) {
            destroy_tree(root);
            root = build_tree_none();
            return;
        }
        if (num_keys == 1) {
            destroy_tree(root);
            root = build_tree_none();
            insert(vs[0]);
            return;
        }
        if (num_keys == 2) {
            destroy_tree(root);
            //root = build_tree_two(vs[0].first, vs[0].second, vs[0].third, vs[1].first, vs[1].second, vs[1].third);
            root = build_tree_two(std::get<0>(vs[0]), std::get<1>(vs[0]), std::get<2>(vs[0]), std::get<0>(vs[1]), std::get<1>(vs[1]), std::get<2>(vs[1]));
            return;
        }

        RT_ASSERT(num_keys > 2);
        for (int i = 1; i < num_keys; i ++) {
            //RT_ASSERT(vs[i].first > vs[i-1].first);
            RT_ASSERT( std::get<0>(vs[i])>  std::get<0>(vs[i-1]));
            
        }

        T* keys = new T[num_keys];
        P* values = new P[num_keys];
        W* weights = new P[num_keys];
        for (int i = 0; i < num_keys; i ++) {
            keys[i] = std::get<0>(vs[i]);
            values[i] = std::get<1>(vs[i]);
            weights[i] = std::get<2>(vs[i]);
            //keys[i] = vs[i].first;
            //values[i] = vs[i].second;
            //weights[i] = vs[i].third;
        }
        destroy_tree(root);
        root = build_tree_bulk(keys, values, weights, num_keys);
        delete[] keys;
        delete[] values;
        delete[] weights;
    }

    void re_load(const V* vs, int num_keys) {


        T* keys = new T[num_keys];
        P* values = new P[num_keys];
        W* weights = new P[num_keys];
        for (int i = 0; i < num_keys; i ++) {
            keys[i] = std::get<0>(vs[i]);
            values[i] = std::get<1>(vs[i]);
            weights[i] = std::get<2>(vs[i]);
            //keys[i] = vs[i].first;
            //values[i] = vs[i].second;
            //weights[i] = vs[i].third;
        }
        destroy_tree(root);
        root = build_tree_bulk(keys, values, weights, num_keys);
        delete[] keys;
        delete[] values;
        delete[] weights;
    }    

    void show() const {
        
        // FILE* fp = fopen("dayin.txt", "w");
               
        // fprintf(fp, "============= SHOW WALDEN ================\n");
        printf("============= SHOW WALDEN ================\n");

        std::stack<Node*> s;
        s.push(root);
        while (!s.empty()) {
            Node* node = s.top(); s.pop();
            
            // fprintf(fp, "Node(%p, a = %lf, b = %lf, num_items = %d)", node, node->model.a, node->model.b, node->num_items);
            printf("Node(%p, a = %lf, b = %lf, num_items = %d)", node, node->model.a, node->model.b, node->num_items);
            // fprintf(fp, "[");
            printf("[");
            int first = 1;
            for (int i = 0; i < node->num_items; i ++) {
                if (!first) {
                    // fprintf(fp,", ");
                    printf(", ");
                }
                first = 0;
                if (BITMAP_GET(node->none_bitmap, i) == 1) {
                    // fprintf(fp,"None");
                    printf("None");
                } else if (BITMAP_GET(node->child_bitmap, i) == 0) {
                    std::stringstream s;
                    s << node->items[i].comp.data.key;
                    // fprintf(fp,"Key(%s)", s.str().c_str());
                    printf("Key(%s)", s.str().c_str());
                } else {
                    // fprintf(fp,"Child(%p)", node->items[i].comp.child);
                    printf("Child(%p)", node->items[i].comp.child);
                    // s.push(node->items[i].comp.child);
                }
            }
            // fprintf(fp,"]\n");
            printf("]\n");
        }
        // fclose(fp);
        print_depth();
    }

    void print_depth() const {
        std::stack<Node*> s;
        std::stack<int> d;
        s.push(root);
        d.push(1);

        int max_depth = 1;
        int sum_depth = 0, sum_nodes = 0;
        while (!s.empty()) {
            Node* node = s.top(); s.pop();
            int depth = d.top(); d.pop();
            for (int i = 0; i < node->num_items; i ++) {
                if (BITMAP_GET(node->child_bitmap, i) == 1) {
                    s.push(node->items[i].comp.child);
                    d.push(depth + 1);
                } else if (BITMAP_GET(node->none_bitmap, i) != 1) {
                    max_depth = std::max(max_depth, depth);
                    sum_depth += depth;
                    sum_nodes ++;
                }
            }
        }

        printf("max_depth = %d, avg_depth = %.2lf\n", max_depth, double(sum_depth) / double(sum_nodes));
    }
    void verify() const {
        std::stack<Node*> s;
        s.push(root);

        while (!s.empty()) {
            Node* node = s.top(); s.pop();
            int sum_size = 0;
            for (int i = 0; i < node->num_items; i ++) {
                if (BITMAP_GET(node->child_bitmap, i) == 1) {
                    s.push(node->items[i].comp.child);
                    sum_size += node->items[i].comp.child->size;
                } else if (BITMAP_GET(node->none_bitmap, i) != 1) {
                    sum_size ++;
                }
            }
            RT_ASSERT(sum_size == node->size);
        }
    }
    void print_stats() const {
        printf("======== Stats ===========\n");
        if (USE_WMCD) {
            printf("\t wmcd_success_times = %lld\n", stats.wmcd_success_times);
            printf("\t wmcd_broken_times = %lld\n", stats.wmcd_broken_times);
        }
        #if COLLECT_TIME
        printf("\t time_scan_and_destory_tree = %lf\n", stats.time_scan_and_destory_tree);
        printf("\t time_build_tree_bulk = %lf\n", stats.time_build_tree_bulk);
        #endif
    }
    
    size_t index_size(bool total=false, bool ignore_child=true) const { 
        std::stack<Node*> s;
        s.push(root);
    
        size_t size = 0;
        while (!s.empty()) {
            Node* node = s.top(); s.pop();
            bool has_child = false;
            if(ignore_child == false) {
                size += sizeof(*node);
            }
            for (int i = 0; i < node->num_items; i ++) {
                if (ignore_child == true) {
                    size += sizeof(Item);
                    has_child = true;
                } else {
                    if (total) size += sizeof(Item);
                }
                if (BITMAP_GET(node->child_bitmap, i) == 1) {
                    if (!total) size += sizeof(Item);
                    s.push(node->items[i].comp.child);
                }
            }
            if (ignore_child == true && has_child) {
                size += sizeof(*node);
            }
        }
        return size;
    }

        size_t model_size() const {
        std::stack<Node*> s;
        s.push(root);

        size_t size = 0;
        while (!s.empty()) {
            Node* node = s.top(); s.pop();
            bool has_child = false;
            for (int i = 0; i < node->num_items; i ++) {
                if (BITMAP_GET(node->child_bitmap, i) == 1) {
                    size += sizeof(Item);
                    has_child = true;
                    s.push(node->items[i].comp.child);
                }
            }
            if (has_child) size += sizeof(*node);
        }
        return size;
    }

private:
    struct Node;
    struct Item
    {
        union {
            struct {
                T key;
                P value;
                W weight;
            } data;
            Node* child;
        } comp;
    };
    struct Node
    {
        int is_two; // is special node for only two keys
        int build_size; // tree size (include sub nodes) when node created 
        int size; // current tree size (include sub nodes) 
        int fixed; // fixed node will not trigger rebuild-
        int num_inserts, num_insert_to_data;
        int num_items; // size of items
        double size_w, build_size_w;
        double num_inserts_w, num_insert_to_data_w;
        LinearModel<T> model;
        Item* items;
        bitmap_t* none_bitmap; // 1 means None, 0 means Data or Child
        bitmap_t* child_bitmap; // 1 means Child. will always be 0 when none_bitmap is 1
    };

    Node* root;
    std::stack<Node*> pending_two;   //?

    std::allocator<Node> node_allocator;
    Node* new_nodes(int n)
    {
        Node* p = node_allocator.allocate(n);
        RT_ASSERT(p != NULL && p != (Node*)(-1));
        return p;
    }
    void delete_nodes(Node* p, int n)
    {
        node_allocator.deallocate(p, n);
    }

    std::allocator<Item> item_allocator;
    Item* new_items(int n)
    {
        Item* p = item_allocator.allocate(n);
        RT_ASSERT(p != NULL && p != (Item*)(-1));
        return p;
    }
    void delete_items(Item* p, int n)
    {
        item_allocator.deallocate(p, n);
    }

    std::allocator<bitmap_t> bitmap_allocator;
    bitmap_t* new_bitmap(int n)
    {
        bitmap_t* p = bitmap_allocator.allocate(n);
        RT_ASSERT(p != NULL && p != (bitmap_t*)(-1));
        return p;
    }
    void delete_bitmap(bitmap_t* p, int n)
    {
        bitmap_allocator.deallocate(p, n);
    }

    /// build an empty tree
    Node* build_tree_none()
    {
        Node* node = new_nodes(1);
        node->is_two = 0;
        node->build_size = 0;
        node->size = 0;
        node->fixed = 0;
        node->num_inserts = node->num_insert_to_data = 0;
        node->num_items = 1;
        node->model.a = node->model.b = 0;
        node->items = new_items(1);
        node->none_bitmap = new_bitmap(1);
        node->none_bitmap[0] = 0;
        BITMAP_SET(node->none_bitmap, 0);
        node->child_bitmap = new_bitmap(1);
        node->child_bitmap[0] = 0;
        
        node->build_size_w = 0;
        node->size_w = 0;
        node->num_inserts_w = node->num_insert_to_data_w = 0;

        return node;
    }
    /// build a tree with two keys
    Node* build_tree_two(T key1, P value1,W weight1, T key2, P value2, W weight2)
    {
        if (key1 > key2) {
            std::swap(key1, key2);
            std::swap(value1, value2);
            std::swap(weight1, weight2);
        }
        RT_ASSERT(key1 < key2);
        static_assert(BITMAP_WIDTH == 8);

        Node* node = NULL;
        if (pending_two.empty()) {
            node = new_nodes(1);
            node->is_two = 1;
            node->build_size = 2;
            node->size = 2;
            node->fixed = 0;
            node->num_inserts = node->num_insert_to_data = 0;
            
            node->num_items = 8;
            node->items = new_items(node->num_items);
            node->none_bitmap = new_bitmap(1);
            node->child_bitmap = new_bitmap(1);
            node->none_bitmap[0] = 0xff;
            node->child_bitmap[0] = 0;
            
            node->num_inserts_w = node->num_insert_to_data_w = 0;
        } else {
            node = pending_two.top(); pending_two.pop();
        }

        const long double mid1_key = key1;
        const long double mid2_key = key2;

        const double mid1_target = node->num_items / 3;
        const double mid2_target = node->num_items * 2 / 3;

        node->model.a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
        node->model.b = mid1_target - node->model.a * mid1_key;
        RT_ASSERT(isfinite(node->model.a));
        RT_ASSERT(isfinite(node->model.b));

        { // insert key1&value1
            int pos = PREDICT_POS(node, key1);
            RT_ASSERT(BITMAP_GET(node->none_bitmap, pos) == 1);
            BITMAP_CLEAR(node->none_bitmap, pos);
            node->items[pos].comp.data.key = key1;
            node->items[pos].comp.data.value = value1;
            node->items[pos].comp.data.weight = weight1;
        }
        { // insert key2&value2
            int pos = PREDICT_POS(node, key2);
            RT_ASSERT(BITMAP_GET(node->none_bitmap, pos) == 1);
            BITMAP_CLEAR(node->none_bitmap, pos);
            node->items[pos].comp.data.key = key2;
            node->items[pos].comp.data.value = value2;
            node->items[pos].comp.data.weight = weight2;
        }

        node->build_size_w = weight1+weight2;
        node->size_w = weight1+weight2;

        return node;
    }
    /// bulk build, _keys must be sorted in asc order.
    Node* build_tree_bulk(T* _keys, P* _values, W* _weights, int _size)
    {
        if (USE_WMCD) {
            return build_tree_bulk_wmcd(_keys, _values, _weights, _size);
        } else {
            return build_tree_bulk_fast(_keys, _values, _weights, _size);
        }
    }
    /// bulk build, _keys must be sorted in asc order.
    /// split keys into three parts at each node.
    Node* build_tree_bulk_fast(T* _keys, P* _values, W* _weights, int _size)
    {
        RT_ASSERT(_size > 1);

        typedef struct {
            int begin;
            int end;
            int level; // top level = 1
            Node* node;
        } Segment;
        std::stack<Segment> s;

        Node* ret = new_nodes(1);
        s.push((Segment){0, _size, 1, ret});

        while (!s.empty()) {
            const int begin = s.top().begin;
            const int end = s.top().end;
            const int level = s.top().level;
            Node* node = s.top().node;
            s.pop();

            RT_ASSERT(end - begin >= 2);
            if (end - begin == 2) {
                Node* _ = build_tree_two(_keys[begin], _values[begin], _weights[begin], _keys[begin+1], _values[begin+1], _weights[begin+1]);
                memcpy(node, _, sizeof(Node));
                delete_nodes(_, 1);
            } else {
                T* keys = _keys + begin;
                P* values = _values + begin;
                W* weights = _weights + begin;
                const int size = end - begin;
                const int BUILD_GAP_CNT = compute_gap_count(size);

                node->is_two = 0;
                node->build_size = size;
                node->size = size;
                node->fixed = 0;
                node->num_inserts = node->num_insert_to_data = 0;

                int mid1_pos = (size - 1) / 3;
                int mid2_pos = (size - 1) * 2 / 3;

                RT_ASSERT(0 <= mid1_pos);
                RT_ASSERT(mid1_pos < mid2_pos);
                RT_ASSERT(mid2_pos < size - 1);

                const long double mid1_key =
                        (static_cast<long double>(keys[mid1_pos]) + static_cast<long double>(keys[mid1_pos + 1])) / 2;
                const long double mid2_key =
                        (static_cast<long double>(keys[mid2_pos]) + static_cast<long double>(keys[mid2_pos + 1])) / 2;

                node->num_items = size * static_cast<int>(BUILD_GAP_CNT + 1);
                const double mid1_target = mid1_pos * static_cast<int>(BUILD_GAP_CNT + 1) + static_cast<int>(BUILD_GAP_CNT + 1) / 2;
                const double mid2_target = mid2_pos * static_cast<int>(BUILD_GAP_CNT + 1) + static_cast<int>(BUILD_GAP_CNT + 1) / 2;

                node->model.a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
                node->model.b = mid1_target - node->model.a * mid1_key;
                RT_ASSERT(isfinite(node->model.a));
                RT_ASSERT(isfinite(node->model.b));

                const int lr_remains = static_cast<int>(size * BUILD_LR_REMAIN);
                node->model.b += lr_remains;
                node->num_items += lr_remains * 2;

                if (size > 1e6) {
                    node->fixed = 1;
                }

                node->items = new_items(node->num_items);
                const int bitmap_size = BITMAP_SIZE(node->num_items);
                node->none_bitmap = new_bitmap(bitmap_size);
                node->child_bitmap = new_bitmap(bitmap_size);
                memset(node->none_bitmap, 0xff, sizeof(bitmap_t) * bitmap_size);
                memset(node->child_bitmap, 0, sizeof(bitmap_t) * bitmap_size);

                for (int item_i = PREDICT_POS(node, keys[0]), offset = 0; offset < size; ) {
                    int next = offset + 1, next_i = -1;
                    while (next < size) {
                        next_i = PREDICT_POS(node, keys[next]);
                        if (next_i == item_i) {
                            next ++;
                        } else {
                            break;
                        }
                    }
                    if (next == offset + 1) {
                        BITMAP_CLEAR(node->none_bitmap, item_i);
                        node->items[item_i].comp.data.key = keys[offset];
                        node->items[item_i].comp.data.value = values[offset];
                        node->items[item_i].comp.data.weight = weights[offset];
                    } else {
                        // ASSERT(next - offset <= (size+2) / 3);
                        BITMAP_CLEAR(node->none_bitmap, item_i);
                        BITMAP_SET(node->child_bitmap, item_i);
                        node->items[item_i].comp.child = new_nodes(1);
                        s.push((Segment){begin + offset, begin + next, level + 1, node->items[item_i].comp.child});
                    }
                    if (next >= size) {
                        break;
                    } else {
                        item_i = next_i;
                        offset = next;
                    }
                }
            }
        }

        return ret;
    }
    /// bulk build, _keys must be sorted in asc order.
    /// WMCD method.
    Node* build_tree_bulk_wmcd(T* _keys, P* _values, W* _weights, int _size)
    {
        RT_ASSERT(_size > 1);

        typedef struct {
            int begin;
            int end;
            int level; // top level = 1
            Node* node;
        } Segment;
        std::stack<Segment> s;

        Node* ret = new_nodes(1);
        s.push((Segment){0, _size, 1, ret});

        while (!s.empty()) {
            const int begin = s.top().begin;
            const int end = s.top().end;
            const int level = s.top().level;
            Node* node = s.top().node;
            s.pop();

            RT_ASSERT(end - begin >= 2);
            if (end - begin == 2) {
                Node* _ = build_tree_two(_keys[begin], _values[begin], _weights[begin], _keys[begin + 1], _values[begin + 1], _weights[begin + 1]);
                memcpy(node, _, sizeof(Node));
                delete_nodes(_, 1);
            }
            else {
                T* keys = _keys + begin;
                P* values = _values + begin;
                W* weights = _weights + begin;
                const int size = end - begin;
                const int BUILD_GAP_CNT = compute_gap_count(size);

                node->is_two = 0;
                node->build_size = size;
                node->size = size;
                node->fixed = 0;
                node->num_inserts = node->num_insert_to_data = 0;

                node->build_size_w = 0;
                node->size_w = 0;
                node->num_inserts_w = node->num_insert_to_data_w = 0;
            {
                // WMCD method

                const int L = size * static_cast<int>(BUILD_GAP_CNT + 1);
                 int i = 0;
                 double D = 2;     
                 int x, y = 0;
                 double X=0, Y=0, Z = 0;  
                 int a = 0;
                 int l = 0;


                for (a = 0; a < size; a++) {
                    if (X <= (D - static_cast<double>(weights[a])))
                    {
                        
                        X += static_cast<double>(weights[a]);
                    }
                    else{
                        break;
                    }
                }
                x = a;
                l = x;
                Z = X; 

                for (a = size - 1; a >= 0; a--) {
                    if (Y <= (D - static_cast<double>(weights[a])))
                    {
                        Y += static_cast<double>(weights[a]);
                    }
                    else{
                        break;
                    }
                }
                y = size -1 - a;                                  


                //if (size - 1 - y < x) break;
                //RT_ASSERT(x <= size - 1 - y);
                double Ut = (static_cast<long double>(keys[size - 1 - y]) - static_cast<long double>(keys[x])) /
                    (static_cast<double>(L - 2)) + 1e-6;

                while (i < size - 1 - y) {
                    if( Ut < 0 ) break;
                    while (i + l < size && static_cast<double>(keys[i + l]) - static_cast<double>(keys[i]) >= Ut) {
                        i++;
                        Z = Z - static_cast<double>(weights[i - 1]) + static_cast<double>(weights[i + l - 1]);
                        
                        while (Z < D && i+l <size)  {
                           if(Z + static_cast<double>(weights[i + l]) <= D )
                           {
                            Z += static_cast<double>(weights[i + l]);
                            l++;
                           }    
                           else{
                            break;
                           }    
                        }
                        while (Z > D) { //&& (Z - static_cast<double>weights[i + l -1]) > D
                            Z -= static_cast<double>(weights[i + l - 1]);
                            l--;
                        }

                    }
                    if (i + y >= size) {
                        break;
                    }

                    D = D + 1; 

                    while (X <= D - static_cast<double>(weights[x]))                        
                    {
                        
                        X += static_cast<double>(weights[x]);
                        x++;
                        
                    }
                    while (Y <= D - static_cast<double>(weights[size - y - 1]))
                    {
                        Y += static_cast<double>(weights[size - y - 1]);
                        y++;
                    }

                    if (D * 3 > size) break;                                                   
                    if (size - 1 - y < x) break;

                    Ut = (static_cast<long double>(keys[size - 1 - y]) - static_cast<long double>(keys[x])) /
                        (static_cast<double>(L - 2)) + 1e-6;
                    
                 
                }

                if (size - 1 - y > x && D * 3 <= size && Ut >0) {                                                                    
                    stats.wmcd_success_times++;

                    node->model.a = 1.0 / Ut;
                    node->model.b = (L - node->model.a * (static_cast<long double>(keys[size - 1 - y]) +
                        static_cast<long double>(keys[x]))) / 2;
                    RT_ASSERT(isfinite(node->model.a));
                    RT_ASSERT(isfinite(node->model.b));
                    node->num_items = L;


                }
                else {
                    stats.wmcd_broken_times++;

                    int mid1_pos = (size - 1) / 3;
                    int mid2_pos = (size - 1) * 2 / 3;

                    RT_ASSERT(0 <= mid1_pos);
                    RT_ASSERT(mid1_pos < mid2_pos);
                    RT_ASSERT(mid2_pos < size - 1);

                    const long double mid1_key = (static_cast<long double>(keys[mid1_pos]) +
                        static_cast<long double>(keys[mid1_pos + 1])) / 2;
                    const long double mid2_key = (static_cast<long double>(keys[mid2_pos]) +
                        static_cast<long double>(keys[mid2_pos + 1])) / 2;

                    node->num_items = size * static_cast<int>(BUILD_GAP_CNT + 1);
                    const double mid1_target = mid1_pos * static_cast<int>(BUILD_GAP_CNT + 1) + static_cast<int>(BUILD_GAP_CNT + 1) / 2;
                    const double mid2_target = mid2_pos * static_cast<int>(BUILD_GAP_CNT + 1) + static_cast<int>(BUILD_GAP_CNT + 1) / 2;

                    node->model.a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
                    node->model.b = mid1_target - node->model.a * mid1_key;
                    RT_ASSERT(isfinite(node->model.a));
                    RT_ASSERT(isfinite(node->model.b));
                }
            }
                RT_ASSERT(node->model.a >= 0);
                const int lr_remains = static_cast<int>(size * BUILD_LR_REMAIN);
                node->model.b += lr_remains;
                node->num_items += lr_remains * 2;

                if (size > 1e6) {
                    node->fixed = 1;
                }

                node->items = new_items(node->num_items);
                const int bitmap_size = BITMAP_SIZE(node->num_items);
                node->none_bitmap = new_bitmap(bitmap_size);
                node->child_bitmap = new_bitmap(bitmap_size);
                memset(node->none_bitmap, 0xff, sizeof(bitmap_t) * bitmap_size);
                memset(node->child_bitmap, 0, sizeof(bitmap_t) * bitmap_size);

                for (int item_i = PREDICT_POS(node, keys[0]), offset = 0; offset < size; ) {
                    int next = offset + 1, next_i = -1;
                    while (next < size) {
                        next_i = PREDICT_POS(node, keys[next]);
                        if (next_i == item_i) {
                            next ++;
                        } else {
                            break;
                        }
                    }
                    RT_ASSERT(item_i <= node->num_items);  
                    if (next == offset + 1) {
                        BITMAP_CLEAR(node->none_bitmap, item_i);
                        
                        node->items[item_i].comp.data.key = keys[offset];
                        node->items[item_i].comp.data.value = values[offset];
                        node->items[item_i].comp.data.weight = weights[offset];

                        node->build_size_w = node->size_w += weights[offset];

                    } else {
                        // ASSERT(next - offset <= (size+2) / 3);
                        BITMAP_CLEAR(node->none_bitmap, item_i);
                        BITMAP_SET(node->child_bitmap, item_i);
                        node->items[item_i].comp.child = new_nodes(1);
                        s.push((Segment){begin + offset, begin + next, level + 1, node->items[item_i].comp.child});
                    }
                    if (next >= size) {
                        break;
                    } else {
                        item_i = next_i;
                        offset = next;
                    }
                }
            }
        }

        return ret;
    }

    void destory_pending()
    {
        while (!pending_two.empty()) {
            Node* node = pending_two.top(); pending_two.pop();

            delete_items(node->items, node->num_items);
            const int bitmap_size = BITMAP_SIZE(node->num_items);
            delete_bitmap(node->none_bitmap, bitmap_size);
            delete_bitmap(node->child_bitmap, bitmap_size);
            delete_nodes(node, 1);
        }
    }

    void destroy_tree(Node* root)
    {
        std::stack<Node*> s;
        s.push(root);
        while (!s.empty()) {
            Node* node = s.top(); s.pop();

            for (int i = 0; i < node->num_items; i ++) {
                if (BITMAP_GET(node->child_bitmap, i) == 1) {
                    s.push(node->items[i].comp.child);
                }
            }

            if (node->is_two) {
                RT_ASSERT(node->build_size == 2);
                RT_ASSERT(node->num_items == 8);
                node->size = 2;
                node->num_inserts = node->num_insert_to_data = 0;
                node->none_bitmap[0] = 0xff;
                node->child_bitmap[0] = 0;
                pending_two.push(node);
            } else {
                delete_items(node->items, node->num_items);
                const int bitmap_size = BITMAP_SIZE(node->num_items);
                delete_bitmap(node->none_bitmap, bitmap_size);
                delete_bitmap(node->child_bitmap, bitmap_size);
                delete_nodes(node, 1);
            }
        }
    }

    void scan_and_destory_tree(Node* _root, T* keys, P* values, W* weights, bool destory = true)
    {
        typedef std::pair<int, Node*> Segment; // <begin, Node*>
        std::stack<Segment> s;

        s.push(Segment(0, _root));
        while (!s.empty()) {
            int begin = s.top().first;
            Node* node = s.top().second;
            const int SHOULD_END_POS = begin + node->size;
            s.pop();

            for (int i = 0; i < node->num_items; i ++) {
                if (BITMAP_GET(node->none_bitmap, i) == 0) {
                    if (BITMAP_GET(node->child_bitmap, i) == 0) {
                        keys[begin] = node->items[i].comp.data.key;
                        values[begin] = node->items[i].comp.data.value;
                        weights[begin] = node->items[i].comp.data.weight;
                        begin ++;
                    } else {
                        s.push(Segment(begin, node->items[i].comp.child));
                        begin += node->items[i].comp.child->size;
                    }
                }
            }
            RT_ASSERT(SHOULD_END_POS == begin);

            if (destory) {
                if (node->is_two) {
                    RT_ASSERT(node->build_size == 2);
                    RT_ASSERT(node->num_items == 8);
                    node->size = 2;
                    node->num_inserts = node->num_insert_to_data = 0;
                    node->none_bitmap[0] = 0xff;
                    node->child_bitmap[0] = 0;
                    pending_two.push(node);
                } else {
                    delete_items(node->items, node->num_items);
                    const int bitmap_size = BITMAP_SIZE(node->num_items);
                    delete_bitmap(node->none_bitmap, bitmap_size);
                    delete_bitmap(node->child_bitmap, bitmap_size);
                    delete_nodes(node, 1);
                }
            }
        }
    }

    Node* insert_tree(Node* _node, const T& key, const P& value, const W& weight)
    {
        constexpr int MAX_DEPTH = 128;
        Node* path[MAX_DEPTH];
        int path_size = 0;
        int insert_to_data = 0;
        
        double insert_to_data_w = 0;

        for (Node* node = _node; ; ) {
            RT_ASSERT(path_size < MAX_DEPTH);
            path[path_size ++] = node;

            node->size ++;
            node->num_inserts ++;

            node->size_w +=weight;
            node->num_inserts_w +=weight;

            int pos = PREDICT_POS(node, key);
            if (BITMAP_GET(node->none_bitmap, pos) == 1) {
                BITMAP_CLEAR(node->none_bitmap, pos);
                node->items[pos].comp.data.key = key;
                node->items[pos].comp.data.value = value;
                node->items[pos].comp.data.weight = weight;

                // node->size_w +=weight;
                // node->num_inserts_w +=weight;

                break;
            } else if (BITMAP_GET(node->child_bitmap, pos) == 0) {
                BITMAP_SET(node->child_bitmap, pos);
                node->items[pos].comp.child = build_tree_two(key, value, weight, node->items[pos].comp.data.key, node->items[pos].comp.data.value, node->items[pos].comp.data.weight);
                insert_to_data = 1;
                
                insert_to_data_w = weight;
                
                break;
            } else {
                node = node->items[pos].comp.child;
            }
        }
        for (int i = 0; i < path_size; i ++) {
            path[i]->num_insert_to_data += insert_to_data;

            path[i]->num_insert_to_data_w += insert_to_data_w;
        }

        for (int i = 0; i < path_size; i ++) {
            Node* node = path[i];
            const int num_inserts = node->num_inserts;
            const int num_insert_to_data = node->num_insert_to_data;

            const double num_inserts_w = node->num_inserts_w;
            const double num_insert_to_data_w = node->num_insert_to_data_w;

            const bool need_rebuild = node->fixed == 0 && node->size >= node->build_size * 4 && node->size >= 64 && num_insert_to_data * 10 >= num_inserts;
            const bool need_rebuild_w = node->fixed == 0 && node->size_w >= node->build_size_w * 4 && node->size >= 64 && num_insert_to_data_w * 10 >= num_inserts_w;

            if (  need_rebuild ||  need_rebuild_w) {   
                const int ESIZE = node->size;
                T* keys = new T[ESIZE];
                P* values = new P[ESIZE];
                W* weights = new W[ESIZE];

                #if COLLECT_TIME
                auto start_time_scan = std::chrono::high_resolution_clock::now();
                #endif
                scan_and_destory_tree(node, keys, values, weights);
                #if COLLECT_TIME
                auto end_time_scan = std::chrono::high_resolution_clock::now();
                auto duration_scan = end_time_scan - start_time_scan;
                stats.time_scan_and_destory_tree += std::chrono::duration_cast<std::chrono::nanoseconds>(duration_scan).count() * 1e-9;
                #endif

                #if COLLECT_TIME
                auto start_time_build = std::chrono::high_resolution_clock::now();
                #endif
                Node* new_node = build_tree_bulk(keys, values, weights, ESIZE);
                #if COLLECT_TIME
                auto end_time_build = std::chrono::high_resolution_clock::now();
                auto duration_build = end_time_build - start_time_build;
                stats.time_build_tree_bulk += std::chrono::duration_cast<std::chrono::nanoseconds>(duration_build).count() * 1e-9;
                #endif

                delete[] keys;
                delete[] values;
                delete[] weights;

                path[i] = new_node;
                if (i > 0) {
                    int pos = PREDICT_POS(path[i-1], key);
                    path[i-1]->items[pos].comp.child = new_node;
                }

                break;
            }
        }

        return path[0];
    }

    Node* rebuild_tree(Node* node)
    {
                const int ESIZE = node->size;
                T* keys = new T[ESIZE];
                P* values = new P[ESIZE];
                W* weights = new W[ESIZE];

                #if COLLECT_TIME
                auto start_time_scan = std::chrono::high_resolution_clock::now();
                #endif
                scan_and_destory_tree(node, keys, values, weights);
                #if COLLECT_TIME
                auto end_time_scan = std::chrono::high_resolution_clock::now();
                auto duration_scan = end_time_scan - start_time_scan;
                stats.time_scan_and_destory_tree += std::chrono::duration_cast<std::chrono::nanoseconds>(duration_scan).count() * 1e-9;
                #endif

                #if COLLECT_TIME
                auto start_time_build = std::chrono::high_resolution_clock::now();
                #endif
                Node* new_node = build_tree_bulk(keys, values, weights, ESIZE);
                #if COLLECT_TIME
                auto end_time_build = std::chrono::high_resolution_clock::now();
                auto duration_build = end_time_build - start_time_build;
                stats.time_build_tree_bulk += std::chrono::duration_cast<std::chrono::nanoseconds>(duration_build).count() * 1e-9;
                #endif

                delete[] keys;
                delete[] values;
                delete[] weights;

        return new_node;
    }
};

#endif // __WALDEN_H__
