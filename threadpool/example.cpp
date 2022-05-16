#include <iostream>
#include <vector>

#include "thread_pool.h"

int main() {

    ThreadPool pool(4);
    std::vector< std::future<int> > results;

    for(int i = 0; i < 8; ++i) {
        results.emplace_back(
        pool.enqueue([i] {
            std::cout << "thread " << i << std::endl;
            return i;
        })
        );
    }

    for(auto && result: results)
        std::cout << result.get() << ' ';
    std::cout << std::endl;

    return 0;
}