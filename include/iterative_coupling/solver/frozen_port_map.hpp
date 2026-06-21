#pragma once

#include "iterative_coupling/core/parallel_executor.hpp"
#include "iterative_coupling/integrators/phi_map.hpp"

namespace iterative_coupling {

/// Frozen subsystem port map S_i^n defined in Eq. (12).
///
/// For a fixed committed state x_i^n and StepContext, evaluating Phi_i^{Delta t}
/// at an incident wave a_i produces a trial pair (x_i^{trial}, b_i). The frozen
/// map returns only b_i = S_i^n(a_i); the trial state is intentionally discarded.
/// This separation permits repeated DRS queries without advancing physical time.
class FrozenPortMap {
 public:
  FrozenPortMap(PhiMap phi, State x_frozen, StepContext ctx)
      : phi_(std::move(phi)), x_frozen_(std::move(x_frozen)), ctx_(ctx) {
    require_size(x_frozen_, phi_.state_dimension(), "FrozenPortMap frozen state");
    ctx_.validate();
  }

  /// Query S_i^n(a_i) during the inner iteration. No committed state is modified.
  Wave operator()(const Wave& a) const { return phi_.step(x_frozen_, a, ctx_).b_next; }

  /// Evaluate the same Phi_i^{Delta t} from x_i^n for the final incident wave.
  /// The caller may commit this result only after the DRS inner iteration terminates.
  StepResult advance(const Wave& a) const { return phi_.step(x_frozen_, a, ctx_); }

  double frozen_energy() const { return phi_.energy(x_frozen_); }
  double energy(const State& x) const { return phi_.energy(x); }
  bool has_energy() const { return phi_.has_energy(); }

  int state_dimension() const { return phi_.state_dimension(); }
  int port_dimension() const { return phi_.port_dimension(); }
  const State& x_frozen() const { return x_frozen_; }
  const StepContext& context() const { return ctx_; }

 private:
  PhiMap phi_;
  State x_frozen_;
  StepContext ctx_;
};

/// Block-diagonal frozen map S^n = diag(S_1^n, ..., S_N^n).
///
/// Each block depends only on its own incident-wave segment, so the subsystem
/// responses are independent and may be evaluated in parallel.
class StackedFrozenPortMap {
 public:
  explicit StackedFrozenPortMap(std::vector<FrozenPortMap> maps,
                                ParallelExecutor* executor = nullptr)
      : maps_(std::move(maps)), executor_(executor) {
    if (maps_.empty()) {
      throw std::invalid_argument("StackedFrozenPortMap requires at least one map");
    }
    total_port_dimension_ = 0;
    for (const auto& map : maps_) {
      offsets_.push_back(total_port_dimension_);
      total_port_dimension_ += map.port_dimension();
    }
  }

  Wave operator()(const Wave& a_stacked) const {
    require_size(a_stacked, total_port_dimension_, "StackedFrozenPortMap input");
    if (executor_ != nullptr && executor_->is_parallel() && maps_.size() > 1) {
      const std::vector<Wave> blocks = executor_->map_indexed(
          maps_.size(), [this, &a_stacked](std::size_t index) {
            return maps_[index](
                a_stacked.segment(offsets_[index], maps_[index].port_dimension()));
          });
      Wave b(total_port_dimension_);
      Eigen::Index offset = 0;
      for (std::size_t i = 0; i < blocks.size(); ++i) {
        b.segment(offset, maps_[i].port_dimension()) = blocks[i];
        offset += maps_[i].port_dimension();
      }
      return b;
    }
    Wave b(total_port_dimension_);
    Eigen::Index offset = 0;
    for (const auto& map : maps_) {
      const int m = map.port_dimension();
      b.segment(offset, m) = map(a_stacked.segment(offset, m));
      offset += m;
    }
    return b;
  }

  const FrozenPortMap& map(std::size_t index) const { return maps_.at(index); }
  const std::vector<FrozenPortMap>& maps() const { return maps_; }
  int total_port_dimension() const { return total_port_dimension_; }
  std::size_t size() const { return maps_.size(); }

 private:
  std::vector<FrozenPortMap> maps_;
  std::vector<Eigen::Index> offsets_;
  ParallelExecutor* executor_ = nullptr;
  int total_port_dimension_ = 0;
};

}  // namespace iterative_coupling
