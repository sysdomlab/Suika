#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

namespace py = pybind11;

#include "resource.h"
#include "plan_config.h"
#include "estimator.h" 

// --- Helper function to convert Python Accelerator object to C++ Accelerator struct ---
Accelerator py_to_cpp_accelerator(const py::object& py_acc_obj) {
    // Python Accelerator has attributes: acc_type (str), node_id (int), local_id (int)
    std::string acc_type_str = py_acc_obj.attr("acc_type").cast<std::string>();
    int node_id = py_acc_obj.attr("node_id").cast<int>();
    int local_id = py_acc_obj.attr("local_id").cast<int>();
    return Accelerator(acc_type_str, node_id, local_id);
}

// --- Helper function to convert Python AcceleratorMeta object to C++ AcceleratorMeta struct ---
AcceleratorMeta py_to_cpp_acc_meta(const py::object& py_acc_meta_obj) {
    // Python AcceleratorMeta has attributes: type_name (str), TFLOPS (float), memory (int)
    std::string type_name = py_acc_meta_obj.attr("type_name").cast<std::string>();
    float tflops = py_acc_meta_obj.attr("TFLOPS").cast<float>();
    int memory = py_acc_meta_obj.attr("memory").cast<int>();
    return AcceleratorMeta(type_name, tflops, memory);
}

// --- Helper function to convert Python PlanConfig object to C++ PlanConfig struct ---
PlanConfig py_to_cpp_plan_config(const py::object& py_plan_config_obj) {
    if (py_plan_config_obj.is_none()) {
        return PlanConfig(); // Uses default C++ constructor (num_micro_batch = -1)
    }

    vec3D<Accelerator> cpp_acc_mesh;
    py::list py_acc_mesh_pp = py_plan_config_obj.attr("acc_mesh").cast<py::list>();
    for (const auto& py_acc_mesh_dp_item : py_acc_mesh_pp) {
        py::list py_acc_mesh_dp = py_acc_mesh_dp_item.cast<py::list>();
        vec2D<Accelerator> cpp_acc_mesh_dp_vec;
        for (const auto& py_acc_mesh_tp_item : py_acc_mesh_dp) {
            py::list py_acc_mesh_tp = py_acc_mesh_tp_item.cast<py::list>();
            vec<Accelerator> cpp_acc_mesh_tp_vec;
            for (const auto& py_acc_obj : py_acc_mesh_tp) {
                cpp_acc_mesh_tp_vec.push_back(py_to_cpp_accelerator(py_acc_obj.cast<py::object>()));
            }
            cpp_acc_mesh_dp_vec.push_back(cpp_acc_mesh_tp_vec);
        }
        cpp_acc_mesh.push_back(cpp_acc_mesh_dp_vec);
    }

    vec2D<float> cpp_lbsz_mesh;
    py::list py_lbsz_mesh_pp = py_plan_config_obj.attr("lbsz_mesh").cast<py::list>();
    for (const auto& py_lbsz_mesh_dp_item : py_lbsz_mesh_pp) {
        py::list py_lbsz_mesh_dp = py_lbsz_mesh_dp_item.cast<py::list>();
        vec<float> cpp_lbsz_mesh_dp_vec;
        for (const auto& py_lbsz_val : py_lbsz_mesh_dp) {
            cpp_lbsz_mesh_dp_vec.push_back(py_lbsz_val.cast<float>());
        }
        cpp_lbsz_mesh.push_back(cpp_lbsz_mesh_dp_vec);
    }

    vec<float> cpp_layer_partition;
    py::list py_layer_partition = py_plan_config_obj.attr("layer_partition").cast<py::list>();
    for (const auto& py_layer_val : py_layer_partition) {
        cpp_layer_partition.push_back(py_layer_val.cast<float>());
    }

    int cpp_num_micro_batch = py_plan_config_obj.attr("num_micro_batch").cast<int>();
    int cpp_max_tp_allowed = py_plan_config_obj.attr("max_tp_allowed").cast<int>();

    return PlanConfig(cpp_acc_mesh, cpp_lbsz_mesh, cpp_layer_partition, cpp_num_micro_batch, cpp_max_tp_allowed);
}

ResourceManager py_to_cpp_resource_manager(const py::object& py_rm_obj) {
    // Extract acc_meta_table
    std::map<std::string, AcceleratorMeta> cpp_acc_meta_table;
    py::dict py_acc_meta_dict = py_rm_obj.attr("acc_meta_table").cast<py::dict>();
    for (auto item : py_acc_meta_dict) {
        std::string key = item.first.cast<std::string>();
        // Use the new helper function here
        cpp_acc_meta_table.emplace(key, py_to_cpp_acc_meta(item.second.cast<py::object>()));
    }

    // Extract bw_intra and bw_inter 
    std::map<std::string, float> cpp_intra_bw_data = 
        py_rm_obj.attr("bw_intra").cast<std::map<std::string, float>>();

    std::map<std::string, float> cpp_intra_bw_sat_data = 
        py_rm_obj.attr("bw_intra_sat").cast<std::map<std::string, float>>();
    
    std::map<std::pair<std::string, std::string>, float> cpp_inter_bw_data = 
        py_rm_obj.attr("bw_inter").cast<std::map<std::pair<std::string, std::string>, float>>();


    ResourceManager cpp_rm(cpp_acc_meta_table, cpp_intra_bw_data, cpp_intra_bw_sat_data, cpp_inter_bw_data);
    return cpp_rm;
}

// --- Helper function to convert C++ Accelerator struct to Python Accelerator object ---
py::object cpp_to_py_accelerator(const Accelerator& cpp_acc) {
    // Ensure Suika.cluster.resource is importable from where Python runs
    py::module_ resource_module = py::module_::import("Suika.cluster.resource");
    py::object PyAccelerator = resource_module.attr("Accelerator");
    return PyAccelerator(py::arg("acc_type") = cpp_acc.acc_type,
                         py::arg("node_id") = cpp_acc.node_id,
                         py::arg("local_id") = cpp_acc.local_id);
}

// --- Helper function to convert C++ PlanConfig struct to Python PlanConfig object ---
py::object cpp_to_py_plan_config(const PlanConfig& cpp_pc) {
    // Check if the C++ PlanConfig represents a "None" or invalid plan
    if (cpp_pc.is_None()) {
        return py::none();
    }

    py::list py_acc_mesh_pp;
    for (const auto& cpp_acc_mesh_dp : cpp_pc.acc_mesh) {
        py::list py_acc_mesh_dp;
        for (const auto& cpp_acc_mesh_tp : cpp_acc_mesh_dp) {
            py::list py_acc_mesh_tp;
            for (const auto& acc : cpp_acc_mesh_tp) {
                py_acc_mesh_tp.append(cpp_to_py_accelerator(acc));
            }
            py_acc_mesh_dp.append(py_acc_mesh_tp);
        }
        py_acc_mesh_pp.append(py_acc_mesh_dp);
    }

    py::list py_lbsz_mesh_pp;
    for (const auto& cpp_lbsz_mesh_dp : cpp_pc.lbsz_mesh) {
        py::list py_lbsz_mesh_dp;
        for (float lbsz : cpp_lbsz_mesh_dp) {
            py_lbsz_mesh_dp.append(static_cast<int>(std::round(lbsz) + 0.5)); // Round to int
        }
        py_lbsz_mesh_pp.append(py_lbsz_mesh_dp);
    }

    py::list py_layer_partition;
    for (float lp_val : cpp_pc.layer_partition) {
        py_layer_partition.append(static_cast<int>(std::round(lp_val) + 0.5)); // Round to int
    }

    // Ensure Suika.cluster.plan_config is importable
    py::module_ plan_config_module = py::module_::import("Suika.cluster.plan_config");
    py::object PyPlanConfig = plan_config_module.attr("PlanConfig");
    return PyPlanConfig(py_acc_mesh_pp, py_lbsz_mesh_pp, py_layer_partition, cpp_pc.num_micro_batch, cpp_pc.max_tp_allowed);
}


PYBIND11_MODULE(cpp_backend_impl, m) {
    m.doc() = "Suika C++ Estimator and Job Solver Module";

    py::add_ostream_redirect(m, "ostream_redirect"); // For C++ cout/cerr to Python sys.stdout/stderr

    // Bind the free function eval_throughput
    m.def("eval_throughput",
        [](const py::object& py_plan_config_obj, // Python PlanConfig object or None
           const py::object& py_rm_obj,          // Python ResourceManager object
           const std::map<std::string, float>& C_constants_py_data, // For CppJobOptimizer
           bool test_feasible,
           const std::string& log_file) {
            
            // 1. Convert Python PlanConfig to C++ PlanConfig
            PlanConfig cpp_pc = py_to_cpp_plan_config(py_plan_config_obj);

            // 2. Extract data from Python ResourceManager and construct C++ ResourceManager
            ResourceManager cpp_rm = py_to_cpp_resource_manager(py_rm_obj);

            // 3. Call the C++ free function `eval_throughput`
            return eval_throughput(cpp_pc, cpp_rm, C_constants_py_data, test_feasible, log_file);
        },
        py::arg("plan_config"),
        py::arg("resource_manager"), 
        py::arg("C_constants"),
        py::arg("test_feasible") = true,
        py::arg("log_file") = ""
    );

    // Bind the free function optimize_all
    m.def("optimize_all",
        [](const py::object& py_cur_plan_config_obj,    // Python PlanConfig object or None
           const py::list& py_new_acc_units,            // Python List[List[Accelerator]]
           const py::object& py_rm_obj,                 // Python ResourceManager object
           const std::map<std::string, float>& C_constants_py_data,
           const std::string& log_file) {
            // 1. Convert current Python PlanConfig to C++ PlanConfig
            PlanConfig cpp_cur_plan_config = py_to_cpp_plan_config(py_cur_plan_config_obj);

            // 2. Convert Python List[List[Accelerator]] to C++ vec2D<Accelerator>
            vec2D<Accelerator> cpp_new_acc_units;
            for (const auto& py_acc_unit_handle : py_new_acc_units) {
                py::list py_acc_unit = py_acc_unit_handle.cast<py::list>();
                vec<Accelerator> cpp_acc_unit;
                for (const auto& py_acc_obj_handle : py_acc_unit) {
                    cpp_acc_unit.push_back(py_to_cpp_accelerator(py_acc_obj_handle.cast<py::object>()));
                }
                cpp_new_acc_units.push_back(cpp_acc_unit);
            }

            // 3. Extract data from Python ResourceManager and construct C++ ResourceManager
            ResourceManager cpp_rm = py_to_cpp_resource_manager(py_rm_obj);

            // 4. Call the C++ free function `optimize_all`
            // The C_constants_py_data is already std::map<std::string, float>
            std::pair<vec<std::pair<PlanConfig, float>>, vec<int>> cpp_result;
            {
                // Release GIL for concurrency
                py::gil_scoped_release release_gil_guard;
                // Call backend
                cpp_result =
                    optimize_all(cpp_cur_plan_config, cpp_new_acc_units, cpp_rm, C_constants_py_data, log_file);
            }
            // 5. Convert the C++ result back to Python types
            py::list py_plan_throughput_list;
            for (const auto& cpp_pair : cpp_result.first) {
                py_plan_throughput_list.append(py::make_tuple(
                    cpp_to_py_plan_config(cpp_pair.first), // Convert PlanConfig
                    cpp_pair.second                        // float throughput
                ));
            }
            
            // pybind11 automatically converts vec<int> to py::list of ints
            py::list py_sort_id_list = py::cast(cpp_result.second);

            return py::make_tuple(py_plan_throughput_list, py_sort_id_list);
        },
        py::arg("cur_plan_config"),
        py::arg("new_acc_units"),
        py::arg("resource_manager"),
        py::arg("C_constants"),
        py::arg("log_file") = ""
    );
}