#include <iostream>
#include <vector>

#include <unistd.h>

#include "thread_pool.h"

int main() {

    ThreadPool pool(2);
    std::vector< std::future<int> > results;

#if 0
    for(int i = 0; i < 8; ++i) {
        results.emplace_back(
        pool.enqueue([i] {
            std::cout << "thread " << i << std::endl;
            return i;
        })
        );
    }
#endif
    int i = 0;

    pool.enqueue([i] {
        while (1) {
            sleep(1);
            std::cout << "thread 0" << std::endl;
        }
        return 0;
    });

    i = 1;

    pool.enqueue([i] {
        while (1) {
            sleep(1);
            std::cout << "thread 1" << std::endl;
        }
        return 1;
    });

#if 0
    for(auto && result: results)
        std::cout << result.get() << ' ';
    std::cout << std::endl;
#endif
    while (1) {
        sleep(1);
        std::cout << "main process." << std::endl;
    }
    return 0;
}
