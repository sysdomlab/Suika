#ifndef PLAN_CONFIG_H
#define PLAN_CONFIG_H 1

#include "resource.h"
#include <vector>
#include <string>
#include <cmath>
#include <numeric>

template<class T>
using vec = std::vector<T>;
template<class T>
using vec2D = vec<vec<T>>;
template<class T>
using vec3D = vec<vec<vec<T>>>;

bool is_power_of_2(int n) {
  return n > 0 && (n & (n - 1)) == 0;
}

/*
  All PlanConfig passed to C++ is in float type
*/

template<class T>
class PlanConfigTemplate {
 public:
  vec3D<Accelerator> acc_mesh;
  vec2D<T> lbsz_mesh;
  vec<T> layer_partition;
  int num_micro_batch;
  int max_tp_allowed;

  PlanConfigTemplate() 
    : num_micro_batch(-1) {} // indicating None
  PlanConfigTemplate(vec3D<Accelerator> acc_mesh, vec2D<T> lbsz_mesh, vec<T> layer_partition, int num_micro_batch, int max_tp_allowed = NUM_ACC_PER_NODE)
    : acc_mesh(acc_mesh), lbsz_mesh(lbsz_mesh), layer_partition(layer_partition), num_micro_batch(num_micro_batch), max_tp_allowed(max_tp_allowed) {}
  PlanConfigTemplate(const PlanConfigTemplate& other) = default;

  PlanConfigTemplate& operator=(const PlanConfigTemplate& other) = default;

  bool validate() const {
    int pp_size = acc_mesh.size();

    if (pp_size != (int)lbsz_mesh.size()) {
      return false;
    }
    
    if (pp_size != (int)layer_partition.size()) {
      return false;
    }

    for (int pp_idx = 0; pp_idx < pp_size; pp_idx++) {
      int dp_size = acc_mesh[pp_idx].size();

      if (dp_size != (int)lbsz_mesh[pp_idx].size()) {
        return false;
      }
      
      for (int dp_idx = 0; dp_idx < dp_size; dp_idx++) {
        vec<Accelerator> tp_mesh = acc_mesh[pp_idx][dp_idx];
        int tp_size = tp_mesh.size();
        if (!is_power_of_2(tp_size))
          return false;
        if (tp_size > max_tp_allowed)
          return false;

        str acc_type_ref = tp_mesh[0].acc_type;
        for (const auto& acc : tp_mesh)
          if (acc.acc_type != acc_type_ref) {
            return false;
          }

        T lbsz = lbsz_mesh[pp_idx][dp_idx];

        if (lbsz < 0.5) {
        // if (lbsz < 1) {
          return false;
        }

        T num_layer = layer_partition[pp_idx];
        if (num_layer < 0.5) {
        // if (num_layer < 1) {
          return false; 
        }
      }
    }
    
    return true;
  }

  int global_batch_size() const {
    return global_micro_batch_size() * num_micro_batch;
  }

  int global_micro_batch_size() const {
    if (is_None())
      return 0;

    return std::round(
      std::accumulate(lbsz_mesh[0].begin(), lbsz_mesh[0].end(), 0.0f)
    ) + 0.5;
  }

  int total_num_layer() const {
    if (is_None())
      return 0;
    return std::round(
      std::accumulate(layer_partition.begin(), layer_partition.end(), 0.0f)
    ) + 0.5;
  }

  vec<Accelerator> acc_list() const {
    vec<Accelerator> accs;
    for (const auto& dp : acc_mesh) {
      for (const auto& tp : dp) {
        accs.insert(accs.end(), tp.begin(), tp.end());
      }
    }
    return accs;
  }

  int world_size() const {
    int size = 0;
    for (const auto& dp : acc_mesh)
      for (const auto& tp : dp)
        size += tp.size();
    return size;
  }

  bool is_None() const {
    return num_micro_batch == -1;
  }

  std::string to_string() const {
    if (is_None()) {
        return "PlanConfig(None)";
    }

  std::ostringstream oss;
    oss << "PlanConfig(\n";
    int current_indent = 2;
    
    auto indent_str = [&](int level) {
        return std::string(level, ' ');
    };

    size_t pp_deg = acc_mesh.size();
    oss << indent_str(current_indent) << "PP[" << pp_deg << "]:{\n";
    oss << indent_str(current_indent) << "  > num_micro_batch: " << num_micro_batch << "\n";

    for (size_t i = 0; i < pp_deg; ++i) {
        int pp_indent = current_indent + 2;
        size_t dp_deg = acc_mesh[i].size();
        oss << indent_str(pp_indent) << "DP(" << i << ")[" << dp_deg << "]:{\n";
        
        oss << indent_str(pp_indent) << "  > n_layer: " << layer_partition[i] << "\n";

        for (size_t j = 0; j < dp_deg; ++j) {
            int dp_indent = pp_indent + 2;
            size_t tp_deg = acc_mesh[i][j].size();
            oss << indent_str(dp_indent) << "TP(" << j << ")[" << tp_deg << "]:{\n";
            
            oss << indent_str(dp_indent) << "  > lbsz: " << lbsz_mesh[i][j] << "\n";

            for (size_t k = 0; k < tp_deg; ++k) {
                int tp_indent = dp_indent + 2;
                oss << indent_str(tp_indent) << acc_mesh[i][j][k].to_string() << "\n";
            }
            oss << indent_str(dp_indent) << "}\n"; // Close TP
        }
        oss << indent_str(pp_indent) << "}\n"; // Close DP
    }
    oss << indent_str(current_indent) << "}\n"; // Close PP
    oss << ")\n";
    return oss.str();
  }
};

using IntegerPlanConfig = PlanConfigTemplate<int>;
using PlanConfig = PlanConfigTemplate<float>;

#endif