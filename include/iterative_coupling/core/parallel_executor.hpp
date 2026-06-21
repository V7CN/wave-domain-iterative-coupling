#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace iterative_coupling {

struct ExecutionOptions {
  /// Zero selects min(hardware_concurrency, subsystem_count). One is serial.
  std::size_t worker_count = 0;
};

/// Small fixed-size C++17 executor reused across all macro-steps.
class ParallelExecutor {
 public:
  ParallelExecutor(std::size_t requested_workers, std::size_t maximum_useful_workers)
      : worker_count_(resolve_worker_count(requested_workers, maximum_useful_workers)) {
    if (worker_count_ > 1) {
      workers_.reserve(worker_count_);
      for (std::size_t i = 0; i < worker_count_; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
      }
    }
  }

  ~ParallelExecutor() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_all();
    for (std::thread& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  ParallelExecutor(const ParallelExecutor&) = delete;
  ParallelExecutor& operator=(const ParallelExecutor&) = delete;

  std::size_t worker_count() const { return worker_count_; }
  bool is_parallel() const { return worker_count_ > 1; }

  /// Run index-independent work concurrently and collect results by index.
  /// All tasks finish before the lowest-index exception is rethrown.
  template <typename Function>
  auto map_indexed(std::size_t count, Function&& function)
      -> std::vector<std::invoke_result_t<Function, std::size_t>> {
    using Result = std::invoke_result_t<Function, std::size_t>;
    if (!is_parallel()) {
      std::vector<std::optional<Result>> slots(count);
      std::exception_ptr first_exception;
      for (std::size_t i = 0; i < count; ++i) {
        try {
          slots[i] = function(i);
        } catch (...) {
          if (!first_exception) {
            first_exception = std::current_exception();
          }
        }
      }
      if (first_exception) {
        std::rethrow_exception(first_exception);
      }
      std::vector<Result> results;
      results.reserve(count);
      for (std::optional<Result>& slot : slots) {
        results.push_back(std::move(slot.value()));
      }
      return results;
    }

    auto shared_function =
        std::make_shared<std::decay_t<Function>>(std::forward<Function>(function));
    std::vector<std::future<Result>> futures;
    futures.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      futures.push_back(submit([shared_function, i] { return (*shared_function)(i); }));
    }

    std::vector<std::optional<Result>> slots(count);
    std::exception_ptr first_exception;
    for (std::size_t i = 0; i < count; ++i) {
      try {
        slots[i] = futures[i].get();
      } catch (...) {
        if (!first_exception) {
          first_exception = std::current_exception();
        }
      }
    }
    if (first_exception) {
      std::rethrow_exception(first_exception);
    }

    std::vector<Result> results;
    results.reserve(count);
    for (std::optional<Result>& slot : slots) {
      results.push_back(std::move(slot.value()));
    }
    return results;
  }

 private:
  template <typename Function>
  auto submit(Function&& function) -> std::future<std::invoke_result_t<Function>> {
    using Result = std::invoke_result_t<Function>;
    auto task = std::make_shared<std::packaged_task<Result()>>(std::forward<Function>(function));
    std::future<Result> future = task->get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_) {
        throw std::runtime_error("Cannot submit work to a stopping ParallelExecutor");
      }
      tasks_.emplace([task] { (*task)(); });
    }
    ready_.notify_one();
    return future;
  }

  static std::size_t resolve_worker_count(std::size_t requested, std::size_t maximum_useful) {
    maximum_useful = std::max<std::size_t>(1, maximum_useful);
    if (requested == 0) {
      const unsigned int hardware = std::thread::hardware_concurrency();
      requested = hardware == 0 ? 1 : static_cast<std::size_t>(hardware);
    }
    return std::max<std::size_t>(1, std::min(requested, maximum_useful));
  }

  void worker_loop() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
        if (stopping_ && tasks_.empty()) {
          return;
        }
        task = std::move(tasks_.front());
        tasks_.pop();
      }
      task();
    }
  }

  std::size_t worker_count_ = 1;
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable ready_;
  bool stopping_ = false;
};

}  // namespace iterative_coupling
