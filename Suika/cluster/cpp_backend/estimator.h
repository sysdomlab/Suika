#ifndef ESTIMATOR_H
#define ESTIMATOR_H 1

#include <string>
#include <map>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <queue>

#include "resource.h"
#include "plan_config.h"

#define FEASIBILITY_MARGIN 0.1
#define EPS 1e-3
#define MAX_PREDECESSOR 8

#define SIMULATION true

#if not SIMULATION
  #define ENABLE_LIMITED_EXPAND true
  #define ENABLE_LIMITED_PREDECESOR false
  #define ENABLE_SELF_ADJUSTMENT true
  #define ENABLE_FRACTIONAL_LB false
#else
  #define ENABLE_LIMITED_EXPAND true
  #define ENABLE_LIMITED_PREDECESOR true  
  #define ENABLE_SELF_ADJUSTMENT true
  #define ENABLE_FRACTIONAL_LB false
#endif

template<class T>
using vec = std::vector<T>;
using str = std::string;

template<class T>
vec<T> flatten(const vec2D<T>& v) {
  vec<T> result;
  for (const auto& sub_v : v) {
    result.insert(result.end(), sub_v.begin(), sub_v.end());
  }
  return result;
}

template<class T>
vec<T> flatten(const vec3D<T>& v) {
  vec<T> result;
  for (const auto& sub_v : v) {
    for (const auto& sub_sub_v : sub_v) {
      result.insert(result.end(), sub_sub_v.begin(), sub_sub_v.end());
    }
  }
  return result;
}

template<class T>
vec<T> concat(const vec<T>& a, const vec<T>& b) {
  vec<T> result(a);
  result.insert(result.end(), b.begin(), b.end());
  return result;
}


#define OPTIMIZE_LOG_ONLY true

// JobOptimizer in C++ combines Estimator and JobOptimizer in Python. the only interface exposed is optimize_all.
class CppJobOptimizer {
  ResourceManager rm;
  std::map<str, float> C;
  std::ofstream log_stream;

 public:
  CppJobOptimizer(ResourceManager resource_manager, std::map<str, float> C, str log_file = "")
      : rm(resource_manager), C(C) {
    if (!log_file.empty())
      log_stream.open(log_file, std::ios_base::app);
  }

  ~CppJobOptimizer() {
    if (log_stream.is_open())
      log_stream.close();
  }

  
  /*
    Estimator
  */
  float eval_latency(const PlanConfig& pc) {

    // there are no feaibility or initialization test.
    // use eval_throughput as a proxy for feasibility
    
    float latency = _pp_latency(
      pc.acc_mesh,
      pc.lbsz_mesh,
      pc.layer_partition,
      pc.num_micro_batch
    );

    if (log_stream.is_open() && !OPTIMIZE_LOG_ONLY)
      log_stream << "estimated latency: " << latency << std::endl;
    
    return latency;
  }

  float eval_throughput(const PlanConfig& pc, bool test_feasible = true) {
    if (log_stream.is_open() && !OPTIMIZE_LOG_ONLY)
      log_stream << "estimate throughput" << std::endl;

    if (pc.is_None() || !pc.validate() || !initialized() || (test_feasible && !feasible(pc))) {
      if (log_stream.is_open() && !OPTIMIZE_LOG_ONLY)
        log_stream << "infeasible: None or invalid PlanConfig or uninitialized constants or infeasible" << " " 
        << pc.is_None() << !pc.validate() << !initialized() << '?' << std::endl;
      return 0.0f;
    }

    float latency = eval_latency(pc);
    float throughput = 1 / latency;
    if (log_stream.is_open() && !OPTIMIZE_LOG_ONLY)
      log_stream << "estimated throughput: " << throughput << std::endl;
    return throughput;
  }

  bool feasible(const PlanConfig& pc) {
    if (log_stream.is_open() && !OPTIMIZE_LOG_ONLY)
      log_stream << "feasibility test" << std::endl;
    if (pc.is_None() || !pc.validate() || !initialized()) {
      if (log_stream.is_open() && !OPTIMIZE_LOG_ONLY)
        log_stream << "infeasible: None or invalid PlanConfig or uninitialized constants" << " " 
        << pc.is_None() << !pc.validate() << !initialized() << std::endl;
      return false;
    }
    
    float peak_usage = 0;
    std::tuple<int,int,int> peak_usage_at;
    
    int pp_deg = pc.acc_mesh.size();
    int num_micro_batch = pc.num_micro_batch;
    
    for (int i = 0; i < pp_deg; i++) {
      int dp_deg = pc.acc_mesh[i].size();
      float num_layer = pc.layer_partition[i];
      
      for (int j = 0; j < dp_deg; j++) {
        int tp_deg = pc.acc_mesh[i][j].size();
        float lbsz = pc.lbsz_mesh[i][j];
        
        for (int k = 0; k < tp_deg; k++) {
          Accelerator acc = pc.acc_mesh[i][j][k];
          int available = rm.get_memory(acc);
          float activation_per_layer = lbsz * std::min(pp_deg - i, num_micro_batch) * (C["activation_mem_p"] / tp_deg + C["activation_mem_np"]),
                param_per_layer = C["param_optim_mem"] / tp_deg,
                grad_per_layer = (num_micro_batch <= pp_deg - i) ? C["param"] / tp_deg : 0,
                required = num_layer * (activation_per_layer + param_per_layer - grad_per_layer);

          required = required * (1 + FEASIBILITY_MARGIN);

          if (available < required) {
            if (log_stream.is_open() && !OPTIMIZE_LOG_ONLY) {
              log_stream << "infeasible@(" << i << "," << j << "," << k << "), "
                        << "available=" << available << ", "
                        << "required=" << required << " with " 
                        << num_layer << " layers, " << lbsz << " lbsz, " 
                        << tp_deg << " tp_deg" << std::endl;
            }
            return false;
          } else if (required / available > peak_usage) {
            peak_usage = required / available;
            peak_usage_at = {i, j, k};
          }
        }
      }
    }
    
    if (log_stream.is_open() && !OPTIMIZE_LOG_ONLY) {
      log_stream << "feasible" << std::endl;
      log_stream << "peak usage@(" << std::get<0>(peak_usage_at) << ","
                 << std::get<1>(peak_usage_at) << "," 
                 << std::get<2>(peak_usage_at) << ") "
                 << peak_usage * 100 << "%" << std::endl;
    }
    
    return true;
  }

  bool initialized() const {
    const std::vector<str> required_constants = {
        "comp",
        "optimizer",
        "param",
        "activation",
        "backward",
        "comp_commu_overlap",
        "param_optim_mem",
        "activation_mem_np",
        "activation_mem_p"
    };
    
    for (const auto& key : required_constants) {
      if (C.find(key) == C.end()) return false;
    }
    return true;
  }


  std::tuple<float, float, float> _tp_latency(const vec<Accelerator>& mesh, float lbsz, float num_layer, str log_prefix) {
    float scaling = lbsz * num_layer;
    int tp_deg = mesh.size();

    float min_TFLOPS = std::numeric_limits<float>::max();
    for (const auto& acc : mesh)
      min_TFLOPS = std::min(min_TFLOPS, rm.get_TFLOPS(acc));
    
    float TP_comp_fwd = 1 * C["comp"] / min_TFLOPS / tp_deg;
    float TP_comp_bwd = TP_comp_fwd * C["backward"];
    float TP_commu = 0;

    if (tp_deg > 1) {
      float volume = 2 * C["activation"] * 2 * (tp_deg - 1) / tp_deg;
      float bandwidth = std::numeric_limits<float>::max();
      for (int i = 0; i < tp_deg; i++)
        for (int j = i + 1; j < tp_deg; j++)
          bandwidth = std::min(bandwidth, rm.get_bandwidth(mesh[i], mesh[j], volume / 2.0));

      TP_commu = volume / bandwidth;
    }

    TP_comp_fwd *= scaling;
    TP_comp_bwd *= scaling;
    TP_commu *= scaling;

    float TP_fwd = TP_comp_fwd + TP_commu;
    float TP_bwd = TP_comp_bwd + TP_commu;
    float TP_optim = C["optimizer"] / min_TFLOPS / tp_deg * num_layer;

    if (log_stream.is_open() && !OPTIMIZE_LOG_ONLY) {
      log_stream << log_prefix << ": " << "TP_comp_fwd: " << TP_comp_fwd << ", TP_comp_bwd: " << TP_comp_bwd << ", TP_commu: " << TP_commu;
      log_stream << "\n TP_fwd: " << TP_fwd << ", TP_bwd: " << TP_bwd << ", TP_optim: " << TP_optim;
      log_stream << "\n tp_deg=" << tp_deg << ", lbsz=" << lbsz << ", num_layer=" << num_layer << std::endl;
    }

    return std::make_tuple(TP_fwd, TP_bwd, TP_optim);
  }

  std::tuple<float, float, float> _dp_latency(const vec2D<Accelerator>& mesh, 
                                      const vec<float>& lbszs, 
                                      float num_layer,
                                      str log_prefix
                                    ) {
    int dp_deg = mesh.size();

    float DP_comp_fwd = 0, DP_comp_bwd = 0, DP_optim = 0;
    for (int i = 0; i < dp_deg; i++) {
      float TP_fwd, TP_bwd, TP_optim;
      std::tie(TP_fwd, TP_bwd, TP_optim) = 
        _tp_latency(
          mesh[i], 
          lbszs[i],
          num_layer,
          log_prefix + "(" + std::to_string(i) + ")-TP");
      DP_comp_fwd = std::max(DP_comp_fwd, TP_fwd);
      DP_comp_bwd = std::max(DP_comp_bwd, TP_bwd);
      DP_optim = std::max(DP_optim, TP_optim);
    }

    float DP_commu = 0;
    if (dp_deg > 1) {
      float total_volume = num_layer * C["param"] * 2 * (dp_deg - 1) / dp_deg;
      auto get_dp_commu_latency = 
        [&](const vec<Accelerator>& mesh1, const vec<Accelerator>& mesh2) {
          int tp_deg1 = mesh1.size(), tp_deg2 = mesh2.size();
          const vec<Accelerator>  *mesh1_ptr = &mesh1, *mesh2_ptr = &mesh2;
          if (tp_deg1 > tp_deg2) {
            std::swap(mesh1_ptr, mesh2_ptr);
            std::swap(tp_deg1, tp_deg2);
          }

          int scale = tp_deg2 / tp_deg1;
          float bandwidth = std::numeric_limits<float>::max();
          for (int i = 0; i < tp_deg1; i++)
            for (int j = 0; j < scale; j++) {
              bandwidth = std::min(bandwidth, rm.get_bandwidth((*mesh1_ptr)[i], (*mesh2_ptr)[i * scale + j]));
            }
          return total_volume / tp_deg1 / bandwidth;
        };
      
      for (int i = 0; i < dp_deg; i++)
        for (int j = i + 1; j < dp_deg; j++)
          DP_commu = std::max(DP_commu, get_dp_commu_latency(mesh[i], mesh[j]));
    }

    float DP_commu_optim_no_olp = _commu_overlapped_latency(DP_comp_bwd, DP_commu) + DP_optim;

    if (log_stream.is_open() && !OPTIMIZE_LOG_ONLY) {
      log_stream << log_prefix << ": " << "DP_comp_fwd: " << DP_comp_fwd << ", DP_comp_bwd: " << DP_comp_bwd << ", DP_optim: " << DP_optim;
      log_stream << "\n DP_commu: " << DP_commu << ", DP_commu_optim_no_olp: " << DP_commu_optim_no_olp;
      log_stream << "\n dp_deg=" << dp_deg << std::endl;
    }

    return std::make_tuple(DP_comp_fwd, DP_comp_bwd, DP_commu_optim_no_olp);
  }

  float _pp_latency(
      const vec3D<Accelerator>& mesh,
      const vec2D<float>& lbsz_mesh,
      const vec<float>& layer_partition,
      int num_micro_batch,
      str log_prefix = ""
  ) {
    log_prefix += "PP";
    
    int pp_deg = mesh.size();
    vec<float> DP_comp_fwds(pp_deg), DP_comp_bwds(pp_deg), DP_commu_optim_no_olps(pp_deg);
    vec<float> DP_comps(pp_deg);

    for (int i = 0; i < pp_deg; i++) {
      float DP_comp_fwd, DP_comp_bwd, DP_commu_optim_no_olp;
      std::tie(DP_comp_fwd, DP_comp_bwd, DP_commu_optim_no_olp) = 
        _dp_latency(
            mesh[i], 
            lbsz_mesh[i], 
            layer_partition[i],
            log_prefix + "(" + std::to_string(i) + ")-DP"
        );
      DP_comp_fwds[i] = DP_comp_fwd;
      DP_comp_bwds[i] = DP_comp_bwd;
      DP_commu_optim_no_olps[i] = DP_commu_optim_no_olp;

      DP_comps[i] = DP_comp_fwd + DP_comp_bwd;
    }

    float sum_DP_comp = std::accumulate(DP_comps.begin(), DP_comps.end(), 0.0f);
    float max_DP_comp = 0, max_DP_commu_optim_no_olp = 0, DP_comp_bwd_prefix_sum = 0;
    for (int i = 0; i < pp_deg; i++) {
      max_DP_comp = std::max(max_DP_comp, DP_comps[i]);
      max_DP_commu_optim_no_olp = std::max(max_DP_commu_optim_no_olp, DP_commu_optim_no_olps[i] - DP_comp_bwd_prefix_sum);

      DP_comp_bwd_prefix_sum += DP_comp_bwds[i];
    }

    float latency = sum_DP_comp + (num_micro_batch - 1) * max_DP_comp + max_DP_commu_optim_no_olp;

    if (log_stream.is_open() && !OPTIMIZE_LOG_ONLY) {
      log_stream << log_prefix << ": " << "sum_DP_comp: " << sum_DP_comp << ", max_DP_comp=" << max_DP_comp;
      log_stream << "\n max_DP_commu_optim_no_olp: " << max_DP_commu_optim_no_olp;
      log_stream << "\n latency: " << latency;
      log_stream << "\n pp_deg=" << pp_deg << ", num_mirco_batch=" << num_micro_batch << std::endl;
    }
    return latency;
  }

  float _commu_overlapped_latency(float T_comp_bwd, float T_commu) {
    float k = C["comp_commu_overlap"];
    T_comp_bwd = std::max(T_comp_bwd, 0.0f);
    T_commu = std::max(T_commu, 0.0f);
    return pow(pow(T_comp_bwd, k) + pow(T_commu, k), 1.0 / k) - T_comp_bwd;
  }

  /*
    JobOptimizer
  */

  std::pair<
    vec<std::pair<PlanConfig, float>>,
    vec<int>
  > 
  optimize_all(const PlanConfig& cur_plan_config, const vec2D<Accelerator>& new_acc_units) {
    if (log_stream.is_open())
      log_stream << "optimize_all: " << std::endl;
    vec<Accelerator> cur_accs = flatten(cur_plan_config.acc_mesh);

    vec<int> sort_id(new_acc_units.size());
    std::iota(sort_id.begin(), sort_id.end(), 0);

    vec<std::tuple<float, int, int, int>> sort_key(new_acc_units.size());

    int acc_units_granularity = 1;
    for (int i = 0; i < (int)new_acc_units.size(); i++) {
      acc_units_granularity = std::max(acc_units_granularity, (int)new_acc_units[i].size());

      Accelerator new_acc = new_acc_units[i][0]; 
      float bdw = 0;
      int relative_rank = 0;
      for (int j = 0; j < (int)cur_accs.size(); j++) {
        Accelerator cur_acc = cur_accs[j];
        bdw = std::max(bdw, rm.get_bandwidth(cur_acc, new_acc));
        if (cur_acc.node_id == new_acc.node_id)
          relative_rank = j;
      }
      sort_key[i] = std::make_tuple(
        bdw, -relative_rank, -new_acc.node_id, -new_acc.local_id
      );
    }

    std::sort(sort_id.begin(), sort_id.end(), [&sort_key](int a, int b) {
      return sort_key[a] > sort_key[b];
    });

    if (log_stream.is_open())
      for (int i = 0; i < (int)sort_id.size(); i++) {
        log_stream << "sort_id[" << i << "] = " << sort_id[i] 
                   << ", bdw: " << std::get<0>(sort_key[sort_id[i]])
                   << ", relative_rank: " << -std::get<1>(sort_key[sort_id[i]])
                   << ", node_id: " << -std::get<2>(sort_key[sort_id[i]])
                   << ", local_id: " << -std::get<3>(sort_key[sort_id[i]])
                   << std::endl;
      }

    vec2D<Accelerator> sorted_acc_units(new_acc_units.size());
    for (int i = 0; i < (int)new_acc_units.size(); i++) {
      sorted_acc_units[i] = new_acc_units[sort_id[i]];
    }

    sorted_acc_units.insert(sorted_acc_units.begin(), vec<Accelerator>());
    sort_id.insert(sort_id.begin(), -1);

    int n = sorted_acc_units.size(), world_size = cur_plan_config.world_size();
    // int n = sorted_acc_units.size();
    vec<std::pair<PlanConfig, float>> f(n);

    f[0] = std::make_pair(
      cur_plan_config, 
      eval_throughput(cur_plan_config, true)
    );

    int i_start = (ENABLE_SELF_ADJUSTMENT ? 0 : 1);
    int i_stop = (ENABLE_LIMITED_EXPAND ? 
                  std::min(
                          n, 
                          std::max(NUM_ACC_PER_NODE, world_size) / acc_units_granularity + 1
                  ) 
                  : n);
    if (log_stream.is_open())
      log_stream << "i_start: " << i_start << ", i_stop: " << i_stop << std::endl;
    for (int i = i_start; i < i_stop; i++) {
      int j_start = (ENABLE_LIMITED_PREDECESOR ? std::max(0, i - MAX_PREDECESSOR) : 0);
      int j_stop = i + ENABLE_SELF_ADJUSTMENT;
      for (int j = j_start; j < j_stop; j++) {
        const PlanConfig& cur_pc = f[j].first;
        PlanConfig& new_pc = f[i].first;
        float& throughput = f[i].second;
        if (cur_pc.is_None())
          continue;
        vec2D<Accelerator> units;
        units.insert(units.end(), sorted_acc_units.begin() + j + 1, sorted_acc_units.begin() + i + 1);
        vec<Accelerator> unit = flatten(units);

        PlanConfig tp_plan = _get_best_tp_plan(cur_pc, unit);
        PlanConfig dp_plan = _get_best_dp_plan(cur_pc, unit);
        PlanConfig pp_plan = _get_best_pp_plan(cur_pc, units);

        _compare_max(tp_plan, new_pc, throughput);
        _compare_max(dp_plan, new_pc, throughput);
        _compare_max(pp_plan, new_pc, throughput);

        
        if (log_stream.is_open()) {
          log_stream << "(" << j + 1 << ", " << i << ") TP throughput: " << eval_throughput(tp_plan) << ", tp_plan: " << tp_plan.to_string() << std::endl;
          log_stream << "(" << j + 1 << ", " << i << ") DP throughput: " << eval_throughput(dp_plan) << ", dp_plan: " << dp_plan.to_string() << std::endl;
          log_stream << "(" << j + 1 << ", " << i << ") PP throughput: " << eval_throughput(pp_plan) << ", pp_plan: " << pp_plan.to_string() << std::endl;
        }
      }

      if (log_stream.is_open())
        log_stream << "index: " << i << ", throughput: " << f[i].second << ", plan: " << f[i].first.to_string() << std::endl;
    }

    // f.erase(f.begin());
    for (auto& [pc, throughput] : f) {
      pc = load_balance(
        pc.acc_mesh, 
        pc.global_micro_batch_size(), 
        pc.total_num_layer(), 
        pc.num_micro_batch,
        pc.max_tp_allowed,
        false
      );
      throughput = eval_throughput(pc, true);
    }

    return std::make_pair(f, sort_id);
  }

  PlanConfig load_balance(const vec3D<Accelerator>& acc_mesh, int gmbs, int n_layer, int n_mb, int max_tp_allowed, bool fractional_lb) {

    if (log_stream.is_open() && !OPTIMIZE_LOG_ONLY) {
    // if (log_stream.is_open()) {
        log_stream << "load_balance: " << std::endl;
      log_stream << "  gmbs: " << gmbs << ", n_layer: " << n_layer << ", n_mb: " << n_mb << ", fractional_lb: " << fractional_lb <<std::endl;
    }

    int pp_deg = acc_mesh.size();

    // lbsz
    vec2D<float> lbsz_mesh(pp_deg);
    for (int i = 0; i < pp_deg; i++) {
      vec2D<Accelerator> dp_mesh = acc_mesh[i];
      int dp_deg = dp_mesh.size();


      vec<float> tp_latencies(dp_deg);
      for (int j = 0; j < dp_deg; j++) {
        float TP_fwd, TP_bwd, TP_optim;
        std::tie(TP_fwd, TP_bwd, TP_optim) = _tp_latency(dp_mesh[j], 1, 1, "LB-TP");
        tp_latencies[j] = TP_fwd +TP_bwd;
      }

      vec<float> lbszs;
      if (fractional_lb) {
        lbszs = _minimize_with_proportion(tp_latencies, gmbs);
      } else {
        lbszs = _minimize_with_priority_queue(tp_latencies, gmbs);
      }

      lbsz_mesh[i] = lbszs;
    }

    // layer
    vec<float> dp_latencies(pp_deg);
    for (int i = 0; i < pp_deg; i++) {
      float DP_comp_fwd, DP_comp_bwd, DP_commu_optim_no_olp;
      std::tie(DP_comp_fwd, DP_comp_bwd, DP_commu_optim_no_olp) = 
        _dp_latency(
            acc_mesh[i], 
            lbsz_mesh[i], 
            1,
            "LB-DP"
        );
      dp_latencies[i] = DP_comp_fwd + DP_comp_bwd;
    }

    vec<float> layer_partition;
    if (fractional_lb)
      layer_partition = _minimize_with_proportion(dp_latencies, n_layer);
    else
      layer_partition = _minimize_with_priority_queue(dp_latencies, n_layer);

    PlanConfig pc = PlanConfig(
      acc_mesh,
      lbsz_mesh,
      layer_partition,
      n_mb,
      max_tp_allowed
    );

    if (log_stream.is_open() && !OPTIMIZE_LOG_ONLY) {
    // if (log_stream.is_open()) {
        log_stream << "load_balance result: " << pc.to_string() << std::endl;
    }

    return pc;
  }

  vec<float> _minimize_with_proportion(const vec<float>& weight, float N) {
    int n = std::round(N) + 0.5;
    int m = weight.size();
    vec<float> inv(m, 0);
    float sum_inv = 0;
    for (int i = 0; i < m; i++) {
      inv[i] = 1.0f / weight[i];
      sum_inv += inv[i];
    }

    vec<float> result(weight.size());
    for (int i = 0; i < m; i++) {
      result[i] = n * inv[i] / sum_inv;
    }

    return result;
  }

  vec<float> _minimize_with_priority_queue(const vec<float>& weight, float N) {
    int n = std::round(N) + 0.5;
    int m = weight.size(), remain = n;
    vec<float> result(m, 0.0f);
    using pq_key_type = std::pair<float, int>;
    std::priority_queue<pq_key_type, vec<pq_key_type>, std::greater<pq_key_type>> pq;

    for (int i = 0; i < m; i++)
      pq.push({weight[i] * (result[i] + 1), i});

    while (remain > 0) {
      auto [w, idx] = pq.top();
      pq.pop();
      result[idx]++;
      pq.push({weight[idx] * (result[idx] + 1), idx});
      remain--;
    }

    return result;
  }

  vec3D<Accelerator> _get_next_tp_plan(
    vec3D<Accelerator> acc_mesh,
    const vec<Accelerator>& new_acc_unit,
    int& pp_idx,
    int& dp_idx
  ) {
    if (pp_idx < 0 || dp_idx < 0)
      throw std::out_of_range("_get_next_tp_plan exhausted");
    auto& tp_mesh = acc_mesh[pp_idx][dp_idx];
    tp_mesh.insert(tp_mesh.end(), new_acc_unit.begin(), new_acc_unit.end());

    dp_idx++;
    if (dp_idx >= (int)acc_mesh[pp_idx].size()) {
      dp_idx = 0;
      pp_idx++;
    }
    if (pp_idx >= (int)acc_mesh.size()) {
      pp_idx = -1;
      dp_idx = -1;
    }
    return acc_mesh;
  }

  vec3D<Accelerator> _get_next_dp_plan(
    vec3D<Accelerator> acc_mesh,
    const vec<Accelerator>& new_acc_unit,
    int& pp_idx
  ) {
    if (pp_idx < 0)
      throw std::out_of_range("_get_next_dp_plan exhausted");
    
    auto& dp_mesh = acc_mesh[pp_idx];
    dp_mesh.insert(dp_mesh.end(), new_acc_unit);

    pp_idx++;
    if (pp_idx >= (int)acc_mesh.size()) {
      pp_idx = -1;
    }
    return acc_mesh;
  }

  vec3D<Accelerator> _get_next_pp_plan(
    vec3D<Accelerator> acc_mesh,
    const vec2D<Accelerator>& new_acc_units
  ) {
    acc_mesh.push_back(new_acc_units);
    return acc_mesh;
  }

  static vec2D<Accelerator> _to_target_TP_degree(
    const vec<Accelerator>& accs, 
    int target_tp_deg
  ) {
    std::map<int, vec<Accelerator>> group_by_node_id;
    for (const Accelerator& acc : accs) {
      group_by_node_id[acc.node_id].push_back(acc);
    }

    vec2D<Accelerator> new_mesh2D;
    for (const auto& [node_id, accs] : group_by_node_id) {
      if (accs.size() % target_tp_deg != 0) {
        return vec2D<Accelerator>(); // equiv to None
      }
      for (int i = 0; i < (int)accs.size(); i += target_tp_deg) {
        vec<Accelerator> new_accs(accs.begin() + i, accs.begin() + i + target_tp_deg);
        new_mesh2D.push_back(new_accs);
      }
    }

    return new_mesh2D;
  }

  PlanConfig _half_TP_degree(PlanConfig plan_config) {
    if (plan_config.is_None())
      return PlanConfig();

    PlanConfig best_plan_config = plan_config;
    float best_throughput = eval_throughput(plan_config, true);

    int pp_deg = plan_config.acc_mesh.size();
    vec<int> tp_degs(pp_deg);
    for (int i = 0; i < pp_deg; i++)
      tp_degs[i] = plan_config.acc_mesh[i][0].size();
    vec<int> tp_degs_order(pp_deg);
    std::iota(tp_degs_order.begin(), tp_degs_order.end(), 0);
    std::sort(tp_degs_order.begin(), tp_degs_order.end(), 
              [&tp_degs](int a, int b) { return tp_degs[a] > tp_degs[b]; });

    for (int i = 0; i < pp_deg; i++) {
      int pp_idx = tp_degs_order[i];
      int cur_tp_deg = tp_degs[pp_idx];
      if (cur_tp_deg <= 1) continue;

      vec<Accelerator> accs = flatten(best_plan_config.acc_mesh[pp_idx]);
      vec2D<Accelerator> new_mesh2D = _to_target_TP_degree(accs, cur_tp_deg / 2);
      if (new_mesh2D.empty()) continue;

      vec3D<Accelerator> new_acc_mesh = best_plan_config.acc_mesh;
      new_acc_mesh[pp_idx] = new_mesh2D;
      _lb_compare_max(new_acc_mesh, 
                      plan_config, 
                      best_plan_config,
                      best_throughput, 
                      false);
    }

    return best_plan_config;
  }

  PlanConfig _get_best_tp_plan(
    const PlanConfig& cur_plan_config,
    const vec<Accelerator>& new_acc_unit
  ) {
    // if (new_acc_unit.size() > NUM_ACC_PER_NODE || new_acc_unit.empty()) {
    //   return PlanConfig();
    // }

    // const vec3D<Accelerator>& acc_mesh = cur_plan_config.acc_mesh;

    // PlanConfig best_plan_config = PlanConfig();
    // float best_throughput = -1;

    // int pp_idx = 0, dp_idx = 0;
    // while (pp_idx >= 0 && dp_idx >= 0) {
    //   vec3D<Accelerator> new_acc_mesh = _get_next_tp_plan(acc_mesh, new_acc_unit, pp_idx, dp_idx);
    //   _lb_compare_max(
    //     new_acc_mesh, 
    //     cur_plan_config, 
    //     best_plan_config, 
    //     best_throughput,
    //     true
    //   );
    // }

    // return best_plan_config;

    PlanConfig best_plan_config = PlanConfig();
    float best_throughput = -1;
    
    const vec3D<Accelerator>& cur_acc_mesh = cur_plan_config.acc_mesh;
    for (int pp_idx = 0; pp_idx < (int)cur_acc_mesh.size(); pp_idx++) {
      vec<Accelerator> cur_acc_unit = flatten(cur_acc_mesh[pp_idx]);
      // if (cur_acc_unit.size() > new_acc_unit.size())
      //   continue;
      
      int cur_tp_deg = cur_acc_mesh[pp_idx][0].size();
      vec2D<Accelerator> new_mesh2D = _to_target_TP_degree(
        concat(cur_acc_unit, new_acc_unit),
        cur_tp_deg * 2
      );
      if (new_mesh2D.empty()) continue;

      vec3D<Accelerator> new_acc_mesh = cur_acc_mesh;
      new_acc_mesh[pp_idx] = new_mesh2D;

      _lb_compare_max(
        new_acc_mesh, 
        cur_plan_config, 
        best_plan_config, 
        best_throughput,
        ENABLE_FRACTIONAL_LB
      );
    }

    return _half_TP_degree(best_plan_config);
  }

  PlanConfig _get_best_dp_plan(
    const PlanConfig& cur_plan_config,
    const vec<Accelerator>& new_acc_unit
  ) {
    // const vec3D<Accelerator>& acc_mesh = cur_plan_config.acc_mesh;

    // PlanConfig best_plan_config = PlanConfig();
    // float best_throughput = -1;

    // int pp_idx = 0;
    // while (pp_idx >= 0) {
    //   vec3D<Accelerator> new_acc_mesh = _get_next_dp_plan(acc_mesh, new_acc_unit, pp_idx);

    //   _lb_compare_max(
    //     new_acc_mesh, 
    //     cur_plan_config, 
    //     best_plan_config, 
    //     best_throughput,
    //     true
    //   );
    // }

    // return best_plan_config;

    PlanConfig best_plan_config = PlanConfig();
    float best_throughput = -1;
    
    const vec3D<Accelerator>& cur_acc_mesh = cur_plan_config.acc_mesh;
    for (int pp_idx = 0; pp_idx < (int)cur_acc_mesh.size(); pp_idx++) {
      vec<Accelerator> cur_acc_unit = flatten(cur_acc_mesh[pp_idx]);
      int cur_tp_deg = cur_acc_mesh[pp_idx][0].size();
      if ((int)new_acc_unit.size() % cur_tp_deg != 0) continue;

      vec2D<Accelerator> new_mesh2D = _to_target_TP_degree(
        concat(cur_acc_unit, new_acc_unit), 
        cur_tp_deg
      );
      if (new_mesh2D.empty()) continue;

      vec3D<Accelerator> new_acc_mesh = cur_acc_mesh;
      new_acc_mesh[pp_idx] = new_mesh2D;

      _lb_compare_max(
        new_acc_mesh, 
        cur_plan_config, 
        best_plan_config, 
        best_throughput,
        ENABLE_FRACTIONAL_LB
      );
    }

    return _half_TP_degree(best_plan_config);
  }

  PlanConfig _get_best_pp_plan(
    const PlanConfig& cur_plan_config,
    const vec2D<Accelerator>& new_acc_units
  ) {

    // auto mergeable = [](const vec<Accelerator>& acc_unit1, const vec<Accelerator>& acc_unit2) -> bool {
    //   if (acc_unit1.size() != acc_unit2.size())
    //     return false;
    //   if (acc_unit1[0].acc_type != acc_unit2[0].acc_type)
    //     return false;
    //   return true;
    // };

    // auto calculate_bandwidth_sum = [this](const vec<Accelerator>& acc_unit1, const vec<Accelerator>& acc_unit2) -> float {
    //   float bw_sum = 0.0f;
    //   for (const auto& acc1 : acc_unit1) {
    //     for (const auto& acc2 : acc_unit2) {
    //       bw_sum += this->rm.get_bandwidth(acc1, acc2);
    //     }
    //   }
    //   return bw_sum;
    // };

    // auto merge = [this, mergeable, calculate_bandwidth_sum](vec2D<Accelerator>& units) -> bool {
    //   if (units.size() < 2) return false;

    //   int dp_deg = units.size();
    //   vec<float> TFLOPS_sum(dp_deg, 0.0f);
    //   for (int i = 0; i < dp_deg; i++) {
    //     for (const auto& acc : units[i]) {
    //       TFLOPS_sum[i] += this->rm.get_TFLOPS(acc);
    //     }
    //   }
    //   vec<int> TFLOPS_idx(dp_deg);
    //   std::iota(TFLOPS_idx.begin(), TFLOPS_idx.end(), 0);
    //   std::sort(TFLOPS_idx.begin(), TFLOPS_idx.end(),
    //             [&TFLOPS_sum](int a, int b) { return TFLOPS_sum[a] < TFLOPS_sum[b]; });

    //   int I = TFLOPS_idx[0];
    //   int best_J = -1;
    //   float best_bw_sum = 0.0f;
    //   for (int j = 1; j < dp_deg; j++) {
    //     int J = TFLOPS_idx[j];
    //     if (mergeable(units[I], units[J])) {
    //       float bw_sum = calculate_bandwidth_sum(units[I], units[J]);
    //       if (bw_sum > best_bw_sum) {
    //         best_bw_sum = bw_sum;
    //         best_J = J;
    //       }
    //     }
    //   }

    //   if (best_J != -1) {
    //     units[I].insert(
    //       units[I].end(), 
    //       units[best_J].begin(), 
    //       units[best_J].end()
    //     );
    //     units.erase(units.begin() + best_J);
    //     return true;
    //   } else 
    //     return false;
    // };

    // const vec3D<Accelerator>& acc_mesh = cur_plan_config.acc_mesh;

    // PlanConfig best_plan_config = PlanConfig();
    // float best_throughput = -1;

    // vec2D<Accelerator> units = new_acc_units;
    // _lb_compare_max(
    //   _get_next_pp_plan(acc_mesh, units),
    //   cur_plan_config, 
    //   best_plan_config, 
    //   best_throughput,
    //   true
    // );

    // while (merge(units)) {
    //   PlanConfig last_plan_config = best_plan_config;
    //   float last_throughput = best_throughput;
    //   _lb_compare_max(
    //     _get_next_pp_plan(acc_mesh, units),
    //     cur_plan_config, 
    //     best_plan_config, 
    //     best_throughput,
    //     true
    //   );
    //   if (best_throughput > 0 && best_throughput <= last_throughput)
    //     return last_plan_config;
    // }

    // return best_plan_config;
    PlanConfig best_plan_config = PlanConfig();
    float best_throughput = -1, last_best_throughput = -1;

    int cur_tp_deg = 1;
    vec<Accelerator> new_acc_unit = flatten(new_acc_units);
    if (new_acc_unit.empty()) return PlanConfig();

    vec2D<Accelerator> new_mesh2D = _to_target_TP_degree(
      new_acc_unit, cur_tp_deg
    );

    vec3D<Accelerator> new_acc_mesh = cur_plan_config.acc_mesh;
    new_acc_mesh.push_back(new_mesh2D);
    _lb_compare_max(
      new_acc_mesh, 
      cur_plan_config, 
      best_plan_config, 
      best_throughput,
      ENABLE_FRACTIONAL_LB
    );

    while (true) {
      cur_tp_deg *= 2;
      new_mesh2D = _to_target_TP_degree(
        new_acc_unit,
        cur_tp_deg
      );

      if (new_mesh2D.empty())
        break;

      last_best_throughput = best_throughput;
      new_acc_mesh.back() = new_mesh2D;
      _lb_compare_max(
        new_acc_mesh, 
        cur_plan_config, 
        best_plan_config, 
        best_throughput,
        ENABLE_FRACTIONAL_LB
      );
      if (best_throughput > 0 && best_throughput <= last_best_throughput)
        break;
    }

    return _half_TP_degree(best_plan_config);
  }

  void _compare_max(
    const PlanConfig& new_plan_config,
    PlanConfig& best_plan_config,
    float& best_throughput
  ) {
    float new_throughput = eval_throughput(new_plan_config, true);

    if (new_throughput > best_throughput * (1+EPS)) {
      best_plan_config = new_plan_config;
      best_throughput = new_throughput;
    }
  }

  void _lb_compare_max(
    const vec3D<Accelerator>& new_acc_mesh,
    const PlanConfig& ref_plan_config,
    PlanConfig& best_plan_config,
    float& best_throughput,
    bool fractional_lb
  ) {
    PlanConfig new_plan_config = load_balance(
      new_acc_mesh, 
      ref_plan_config.global_micro_batch_size(), 
      ref_plan_config.total_num_layer(), 
      ref_plan_config.num_micro_batch,
      ref_plan_config.max_tp_allowed,
      fractional_lb);
    
    _compare_max(new_plan_config, best_plan_config, best_throughput);

    return;
  }
};

/*
  Interface for Python
*/
float eval_throughput(
  const PlanConfig& pc,
  const ResourceManager& rm,
  const std::map<str, float>& C,
  bool test_feasible = true,
  str log_file = ""
) {
  CppJobOptimizer optimizer(rm, C, log_file);
  return optimizer.eval_throughput(pc, test_feasible);
}


std::pair<
vec<std::pair<PlanConfig, float>>,
vec<int>
> 
optimize_all(
  const PlanConfig& cur_plan_config,
  const vec2D<Accelerator>& new_acc_units,
  const ResourceManager& rm,
  const std::map<str, float>& C,
  str log_file = ""
) {
  CppJobOptimizer optimizer(rm, C, log_file);
    
  return optimizer.optimize_all(cur_plan_config, new_acc_units);
}

#endif