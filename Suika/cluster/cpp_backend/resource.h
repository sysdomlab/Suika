#ifndef RESOURCE_H
#define RESOURCE_H 1

#include <string>
#include <map>
#include <sstream>


#define NUM_ACC_PER_NODE 8

using str = std::string;
template<class T1, class T2>
using pair = std::pair<T1, T2>;

class AcceleratorMeta {
 public:
  std::string type_name;
  float TFLOPS;
  int memory;

  AcceleratorMeta(std::string type_name, float TFLOPS, int memory)
      : type_name(type_name), TFLOPS(TFLOPS), memory(memory) {}
  AcceleratorMeta(const AcceleratorMeta& other) = default;
  AcceleratorMeta& operator=(const AcceleratorMeta& other) = default;
};

class Accelerator {
 public:
  std::string acc_type;
  int node_id;
  int local_id;

  Accelerator(std::string acc_type, int node_id, int local_id)
      : acc_type(acc_type), node_id(node_id), local_id(local_id) {}
  Accelerator(const Accelerator& other) = default;
  Accelerator& operator=(const Accelerator& other) = default;

  std::string to_string() const {
    std::ostringstream oss;
    oss << "Accelerator(acc_type=" << acc_type 
        << ", id=(" << node_id << ", " << local_id << "))";
    return oss.str();
  }
};

class ResourceManager {
  /*
    C++ side do not need node management info
  */
 public:
  std::map<str, AcceleratorMeta> acc_meta_table;
  std::map<str, float> intra_bw;
  std::map<str, float> intra_bw_sat;
  std::map<pair<str, str>, float> inter_bw;

  ResourceManager(
    std::map<str, AcceleratorMeta> acc_meta_table,
    std::map<str, float> intra_bw,
    std::map<str, float> intra_bw_sat,
    std::map<pair<str, str>, float> inter_bw
  ) : 
    acc_meta_table(acc_meta_table),
    intra_bw(intra_bw),
    intra_bw_sat(intra_bw_sat),
    inter_bw(inter_bw) {}

  ResourceManager(const ResourceManager& other) = default;
  ResourceManager& operator=(const ResourceManager& other) = default;

  int get_memory(const Accelerator& acc) const {
    // Get memory for an accelerator
    auto it = acc_meta_table.find(acc.acc_type);
    if (it != acc_meta_table.end()) {
      return it->second.memory;
    }
    throw std::runtime_error("Unknown accelerator type: " + acc.acc_type);
  }

  float get_TFLOPS(const Accelerator& acc) const {
    // Get TFLOPS for an accelerator
    auto it = acc_meta_table.find(acc.acc_type);
    if (it != acc_meta_table.end()) {
      return it->second.TFLOPS;
    }
    throw std::runtime_error("Unknown accelerator type: " + acc.acc_type);
  }

  float get_bandwidth(const Accelerator& acc1, const Accelerator& acc2, float volume = -1.0) const {
    // Get bandwidth between two accelerators
    if (acc1.node_id == acc2.node_id) {
      auto it = intra_bw.find(acc1.acc_type);
      if (it != intra_bw.end()) {
        float bw = it->second;
        auto sat_it = intra_bw_sat.find(acc1.acc_type);
        if (sat_it == intra_bw_sat.end() || volume < 1e-3)
          return bw;
        
        float sat = sat_it->second;
        return bw * std::min(log2(volume) / log2(sat), 1.0f);
      } else
        throw std::runtime_error("Unknown intra bandwidth for " + acc1.acc_type);
    } else {
      auto it = inter_bw.find({acc1.acc_type, acc2.acc_type});
      if (it != inter_bw.end())
        return it->second;
      else
        throw std::runtime_error("Unknown inter bandwidth between " + acc1.acc_type + " and " + acc2.acc_type);
    }
  }
};


#endif