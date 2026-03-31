#include "graph.h"
#include "../kernel/kernel.h"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <stdexcept>

void compute_sample_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& logits_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;

    float temperature = node.params.temperature;
    float top_p = node.params.top_p;
    size_t top_k = node.params.top_k;
    size_t random_seed = node.params.random_seed;

    const float* bias_values = node.params.bias_values.empty() ? nullptr : node.params.bias_values.data();
    const uint32_t* bias_indices = node.params.bias_indices.empty() ? nullptr : node.params.bias_indices.data();
    size_t bias_count = node.params.bias_values.size();

    if (logits_buffer.shape.size() != 2) {
        throw std::runtime_error("Sample expects 2D logits tensor [seq_len, vocab_size]");
    }

    size_t seq_len = logits_buffer.shape[0];
    size_t vocab_size = logits_buffer.shape[1];
    size_t last_token_offset = (seq_len - 1) * vocab_size;

    const uint32_t* rep_tokens = node.params.rep_penalty_tokens.empty() ? nullptr : node.params.rep_penalty_tokens.data();
    size_t rep_count = node.params.rep_penalty_tokens.size();
    float rep_penalty = node.params.rep_penalty;

    if (logits_buffer.precision == Precision::FP16) {
        const __fp16* logits_fp16 = logits_buffer.data_as<__fp16>();
        // Upcast to FP32 for numerically stable sampling
        std::vector<float> logits_fp32(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            logits_fp32[i] = static_cast<float>(logits_fp16[last_token_offset + i]);
        }
        cactus_sample_f32(logits_fp32.data(), node.output_buffer.data_as<uint32_t>(),
                         vocab_size, temperature, top_p, top_k, random_seed,
                         bias_values, bias_indices, bias_count,
                         rep_tokens, rep_count, rep_penalty);
    } else {
        const float* logits_fp32 = logits_buffer.data_as<float>();
        cactus_sample_f32(logits_fp32 + last_token_offset, node.output_buffer.data_as<uint32_t>(),
                         vocab_size, temperature, top_p, top_k, random_seed,
                         bias_values, bias_indices, bias_count,
                         rep_tokens, rep_count, rep_penalty);
    }
}

void compute_topk_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    if (input_buffer.shape.size() != 2) {
        throw std::runtime_error("TopK currently only supports 2D tensors [batch, features]");
    }

    size_t k = node.params.top_k;
    size_t batch_size = input_buffer.shape[0];
    size_t feature_size = input_buffer.shape[1];
    size_t block_size = batch_size * k;

    std::vector<float> input_float(input_buffer.total_size);
    if (input_buffer.precision == Precision::INT8) {
        throw std::runtime_error("TopK currently does not support INT8 input");
    } else if (input_buffer.precision == Precision::FP16) {
        const __fp16* input_fp16 = input_buffer.data_as<__fp16>();
        for (size_t i = 0; i < input_buffer.total_size; ++i) {
            input_float[i] = static_cast<float>(input_fp16[i]);
        }
    } else {
        const float* input_fp32 = input_buffer.data_as<float>();
        std::memcpy(input_float.data(), input_fp32, input_buffer.total_size * sizeof(float));
    }

    float* output = node.output_buffer.data_as<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        const float* row = input_float.data() + b * feature_size;

        std::vector<std::pair<size_t, float>> indexed_values(feature_size);
        for (size_t i = 0; i < feature_size; ++i) {
            indexed_values[i] = {i, row[i]};
        }

        std::partial_sort(indexed_values.begin(),
                         indexed_values.begin() + k,
                         indexed_values.end(),
                         [](const auto& a, const auto& b) { return a.second > b.second; });

        float* idx_out_row = output + b * k;
        float* val_out_row = output + block_size + b * k;
        for (size_t i = 0; i < k; ++i) {
            idx_out_row[i] = static_cast<float>(indexed_values[i].first);
            val_out_row[i] = indexed_values[i].second;
        }
    }
}

void compute_scatter_topk_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& indices_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& values_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;

    if (indices_buffer.shape != values_buffer.shape) {
        throw std::runtime_error("ScatterTopK requires indices and values with identical shapes");
    }
    if (indices_buffer.shape.size() != 2) {
        throw std::runtime_error("ScatterTopK currently supports 2D tensors");
    }

    size_t batch_size = indices_buffer.shape[0];
    size_t top_k = indices_buffer.shape[1];
    size_t num_classes = node.params.num_classes;

    if (num_classes == 0) {
        throw std::runtime_error("ScatterTopK requires num_classes > 0");
    }

    float* output = node.output_buffer.data_as<float>();
    std::fill(output, output + num_classes * batch_size, 0.0f);

    if (indices_buffer.precision != Precision::FP32 || values_buffer.precision != Precision::FP32) {
        throw std::runtime_error("ScatterTopK currently expects FP32 inputs");
    }

    const float* indices_data = indices_buffer.data_as<float>();
    const float* values_data = values_buffer.data_as<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t k = 0; k < top_k; ++k) {
            float raw_index = indices_data[b * top_k + k];
            if (!std::isfinite(raw_index)) {
                throw std::runtime_error("ScatterTopK index is not finite");
            }
            size_t expert_index = static_cast<size_t>(raw_index + 0.5f);
            if (expert_index >= num_classes) {
                throw std::runtime_error("ScatterTopK index out of range");
            }
            float weight = values_data[b * top_k + k];
            output[expert_index * batch_size + b] = weight;
        }
    }
}
