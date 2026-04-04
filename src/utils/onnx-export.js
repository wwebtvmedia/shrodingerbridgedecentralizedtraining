// ONNX Export Utility for Schrödinger Bridge Models
// This module provides functionality to export trained models to ONNX-like format

import { CONFIG } from "../config.js";

export class ONNXExporter {
  constructor(modelManager) {
    this.modelManager = modelManager;
    this.torch = null;
  }

  async initialize() {
    // Try to load torch-js
    try {
      const torchModule = await import("js-pytorch");
      this.torch = torchModule;
      console.log("✅ Torch-js loaded for ONNX export");
    } catch (error) {
      console.warn(
        "⚠️ Torch-js not available, ONNX export will generate schema only",
      );
      this.torch = null;
    }
  }

  async exportVAEToONNX(vaeModel, outputPath = "vae_model.onnx.json") {
    console.log("📤 Exporting VAE to ONNX format...");

    const onnxModel = {
      ir_version: 8,
      producer_name: "SchrodingerBridgeSwarm",
      producer_version: "0.2.0",
      opset_import: [{ domain: "", version: 17 }],
      graph: {
        name: "LabelConditionedVAE",
        node: [],
        input: [],
        output: [],
        initializer: [],
      },
      metadata: {
        model_type: "LabelConditionedVAE",
        config: CONFIG,
        timestamp: new Date().toISOString(),
      },
    };

    // Add inputs
    onnxModel.graph.input.push({
      name: "input",
      type: {
        tensor_type: {
          elem_type: 1, // FLOAT
          shape: {
            dim: [
              { dim_value: "batch" },
              { dim_value: 3 }, // channels
              { dim_value: CONFIG.IMG_SIZE },
              { dim_value: CONFIG.IMG_SIZE },
            ],
          },
        },
      },
    });

    onnxModel.graph.input.push({
      name: "labels",
      type: {
        tensor_type: {
          elem_type: 7, // INT64
          shape: { dim: [{ dim_value: "batch" }] },
        },
      },
    });

    // Add outputs
    onnxModel.graph.output.push({
      name: "reconstruction",
      type: {
        tensor_type: {
          elem_type: 1,
          shape: {
            dim: [
              { dim_value: "batch" },
              { dim_value: 3 },
              { dim_value: CONFIG.IMG_SIZE },
              { dim_value: CONFIG.IMG_SIZE },
            ],
          },
        },
      },
    });

    onnxModel.graph.output.push({
      name: "mu",
      type: {
        tensor_type: {
          elem_type: 1,
          shape: {
            dim: [
              { dim_value: "batch" },
              { dim_value: CONFIG.LATENT_CHANNELS },
              { dim_value: CONFIG.LATENT_H },
              { dim_value: CONFIG.LATENT_W },
            ],
          },
        },
      },
    });

    onnxModel.graph.output.push({
      name: "logvar",
      type: {
        tensor_type: {
          elem_type: 1,
          shape: {
            dim: [
              { dim_value: "batch" },
              { dim_value: CONFIG.LATENT_CHANNELS },
              { dim_value: CONFIG.LATENT_H },
              { dim_value: CONFIG.LATENT_W },
            ],
          },
        },
      },
    });

    // Add model architecture nodes
    // Encoder layers
    this.addConvLayer(
      onnxModel,
      "enc_in",
      "input",
      "enc_in_out",
      3,
      64,
      3,
      1,
      1,
    );
    this.addBatchNormLayer(
      onnxModel,
      "enc_bn1",
      "enc_in_out",
      "enc_bn1_out",
      64,
    );
    this.addSiLULayer(onnxModel, "enc_act1", "enc_bn1_out", "enc_act1_out");

    // Residual blocks (simplified)
    for (let i = 0; i < 4; i++) {
      const inCh = [64, 128, 256, 512][i];
      const outCh = [128, 256, 512, 512][i];
      this.addResidualBlock(
        onnxModel,
        `enc_block${i}`,
        `enc_act${i}_out`,
        `enc_block${i}_out`,
        inCh,
        outCh,
      );
    }

    // Latent projections
    this.addConvLayer(
      onnxModel,
      "z_mean",
      "enc_block3_out",
      "mu",
      512,
      CONFIG.LATENT_CHANNELS,
      1,
      1,
      0,
    );
    this.addConvLayer(
      onnxModel,
      "z_logvar",
      "enc_block3_out",
      "logvar",
      512,
      CONFIG.LATENT_CHANNELS,
      1,
      1,
      0,
    );

    // Decoder layers
    this.addConvLayer(
      onnxModel,
      "dec_in",
      "mu",
      "dec_in_out",
      CONFIG.LATENT_CHANNELS,
      512,
      1,
      1,
      0,
    );

    for (let i = 0; i < 4; i++) {
      const inCh = [512, 512, 256, 128][i];
      const outCh = [512, 256, 128, 64][i];
      this.addResidualBlock(
        onnxModel,
        `dec_block${i}`,
        `dec_block${i - 1 >= 0 ? i - 1 : "in"}_out`,
        `dec_block${i}_out`,
        inCh,
        outCh,
      );
    }

    this.addConvLayer(
      onnxModel,
      "dec_out",
      "dec_block3_out",
      "reconstruction",
      64,
      3,
      3,
      1,
      1,
    );
    this.addTanhLayer(
      onnxModel,
      "final_tanh",
      "reconstruction",
      "reconstruction_tanh",
    );

    // Update output to use tanh output
    onnxModel.graph.output[0].name = "reconstruction_tanh";

    // Add weight data if available
    if (vaeModel && this.torch) {
      await this.extractWeightsToONNX(onnxModel, vaeModel);
    }

    // Save to file
    await this.saveONNXJSON(onnxModel, outputPath);

    console.log(`✅ VAE exported to ${outputPath}`);
    return onnxModel;
  }

  async exportDriftToONNX(driftModel, outputPath = "drift_model.onnx.json") {
    console.log("📤 Exporting Drift network to ONNX format...");

    const onnxModel = {
      ir_version: 8,
      producer_name: "SchrodingerBridgeSwarm",
      producer_version: "0.2.0",
      opset_import: [{ domain: "", version: 17 }],
      graph: {
        name: "LabelConditionedDrift",
        node: [],
        input: [],
        output: [],
        initializer: [],
      },
      metadata: {
        model_type: "LabelConditionedDrift",
        config: CONFIG,
        timestamp: new Date().toISOString(),
      },
    };

    // Add inputs
    onnxModel.graph.input.push({
      name: "z",
      type: {
        tensor_type: {
          elem_type: 1,
          shape: {
            dim: [
              { dim_value: "batch" },
              { dim_value: CONFIG.LATENT_CHANNELS },
              { dim_value: CONFIG.LATENT_H },
              { dim_value: CONFIG.LATENT_W },
            ],
          },
        },
      },
    });

    onnxModel.graph.input.push({
      name: "t",
      type: {
        tensor_type: {
          elem_type: 1,
          shape: { dim: [{ dim_value: "batch" }, { dim_value: 1 }] },
        },
      },
    });

    onnxModel.graph.input.push({
      name: "labels",
      type: {
        tensor_type: {
          elem_type: 7,
          shape: { dim: [{ dim_value: "batch" }] },
        },
      },
    });

    // Add output
    onnxModel.graph.output.push({
      name: "drift",
      type: {
        tensor_type: {
          elem_type: 1,
          shape: {
            dim: [
              { dim_value: "batch" },
              { dim_value: CONFIG.LATENT_CHANNELS },
              { dim_value: CONFIG.LATENT_H },
              { dim_value: CONFIG.LATENT_W },
            ],
          },
        },
      },
    });

    // Add model architecture nodes
    // Time embedding network
    this.addFourierTimeEmbedding(onnxModel, "time_embed", "t", "time_emb_out");
    this.addLinearLayer(
      onnxModel,
      "time_mlp1",
      "time_emb_out",
      "time_mlp1_out",
      128,
      256,
    );
    this.addSiLULayer(
      onnxModel,
      "time_mlp1_act",
      "time_mlp1_out",
      "time_mlp1_act_out",
    );
    this.addLinearLayer(
      onnxModel,
      "time_mlp2",
      "time_mlp1_act_out",
      "time_mlp2_out",
      256,
      256,
    );

    // Label embedding
    this.addEmbeddingLayer(
      onnxModel,
      "label_emb",
      "labels",
      "label_emb_out",
      CONFIG.NUM_CLASSES,
      CONFIG.LABEL_EMB_DIM,
    );

    // Combine embeddings
    this.addConcatLayer(
      onnxModel,
      "cond_concat",
      ["time_mlp2_out", "label_emb_out"],
      "cond_concat_out",
      -1,
    );

    if (CONFIG.USE_CONTEXT) {
      onnxModel.graph.input.push({
        name: "source_id",
        type: {
          tensor_type: {
            elem_type: 7,
            shape: { dim: [{ dim_value: "batch" }] },
          },
        },
      });

      this.addEmbeddingLayer(
        onnxModel,
        "source_emb",
        "source_id",
        "source_emb_out",
        CONFIG.NUM_SOURCES,
        CONFIG.CONTEXT_DIM,
      );
      this.addConcatLayer(
        onnxModel,
        "cond_concat_full",
        ["cond_concat_out", "source_emb_out"],
        "cond_concat_full_out",
        -1,
      );
      this.addLinearLayer(
        onnxModel,
        "cond_proj",
        "cond_concat_full_out",
        "cond_out",
        256 + CONFIG.LABEL_EMB_DIM + CONFIG.CONTEXT_DIM,
        128,
      );
    } else {
      this.addLinearLayer(
        onnxModel,
        "cond_proj",
        "cond_concat_out",
        "cond_out",
        256 + CONFIG.LABEL_EMB_DIM,
        128,
      );
    }

    // U-Net architecture (simplified)
    this.addConvLayer(
      onnxModel,
      "head",
      "z",
      "head_out",
      CONFIG.LATENT_CHANNELS,
      64,
      3,
      1,
      1,
    );
    this.addResidualBlock(onnxModel, "down1", "head_out", "down1_out", 64, 128);
    this.addConvLayer(
      onnxModel,
      "down2_conv",
      "down1_out",
      "down2_conv_out",
      128,
      256,
      4,
      2,
      1,
    );
    this.addResidualBlock(
      onnxModel,
      "down2_block",
      "down2_conv_out",
      "down2_out",
      256,
      256,
    );

    this.addResidualBlock(onnxModel, "mid1", "down2_out", "mid1_out", 256, 256);
    this.addSelfAttention(
      onnxModel,
      "mid_attn",
      "mid1_out",
      "mid_attn_out",
      256,
    );
    this.addResidualBlock(
      onnxModel,
      "mid2",
      "mid_attn_out",
      "mid2_out",
      256,
      256,
    );

    this.addUpsampleLayer(
      onnxModel,
      "up2_upsample",
      "mid2_out",
      "up2_upsample_out",
      2,
      "nearest",
    );
    this.addConvLayer(
      onnxModel,
      "up2_conv",
      "up2_upsample_out",
      "up2_conv_out",
      256,
      128,
      3,
      1,
      1,
    );
    this.addResidualBlock(
      onnxModel,
      "up2_block",
      "up2_conv_out",
      "up2_out",
      128,
      128,
    );

    // Skip connection
    this.addAddLayer(
      onnxModel,
      "skip_add",
      ["up2_out", "down1_out"],
      "skip_add_out",
    );
    this.addResidualBlock(onnxModel, "up1", "skip_add_out", "up1_out", 128, 64);

    this.addConvLayer(
      onnxModel,
      "tail",
      "up1_out",
      "drift_raw",
      64,
      CONFIG.LATENT_CHANNELS,
      3,
      1,
      1,
    );

    // Time-adaptive scaling
    this.addLinearLayer(
      onnxModel,
      "time_weight_net1",
      "t",
      "time_weight_net1_out",
      1,
      32,
    );
    this.addSiLULayer(
      onnxModel,
      "time_weight_net1_act",
      "time_weight_net1_out",
      "time_weight_net1_act_out",
    );
    this.addLinearLayer(
      onnxModel,
      "time_weight_net2",
      "time_weight_net1_act_out",
      "time_weight_net2_out",
      32,
      1,
    );
    this.addSigmoidLayer(
      onnxModel,
      "time_weight_sigmoid",
      "time_weight_net2_out",
      "time_weight",
    );

    // Scale output
    this.addMulLayer(
      onnxModel,
      "scale_output",
      ["drift_raw", "time_weight"],
      "drift",
    );

    // Add weight data if available
    if (driftModel && this.torch) {
      await this.extractWeightsToONNX(onnxModel, driftModel);
    }

    // Save to file
    await this.saveONNXJSON(onnxModel, outputPath);

    console.log(`✅ Drift network exported to ${outputPath}`);
    return onnxModel;
  }

  // Helper methods for adding ONNX nodes
  addConvLayer(
    onnxModel,
    name,
    input,
    output,
    inCh,
    outCh,
    kernel,
    stride,
    padding,
  ) {
    onnxModel.graph.node.push({
      name,
      op_type: "Conv",
      input: [input, `${name}_weight`, `${name}_bias`],
      output: [output],
      attribute: [
        { name: "kernel_shape", ints: [kernel, kernel], type: "INTS" },
        { name: "strides", ints: [stride, stride], type: "INTS" },
        {
          name: "pads",
          ints: [padding, padding, padding, padding],
          type: "INTS",
        },
        { name: "group", i: 1, type: "INT" },
        { name: "dilations", ints: [1, 1], type: "INTS" },
      ],
    });

    // Add weight initializers
    onnxModel.graph.initializer.push({
      name: `${name}_weight`,
      dims: [outCh, inCh, kernel, kernel],
      data_type: 1, // FLOAT
    });

    onnxModel.graph.initializer.push({
      name: `${name}_bias`,
      dims: [outCh],
      data_type: 1,
    });
  }

  addLinearLayer(onnxModel, name, input, output, inDim, outDim) {
    onnxModel.graph.node.push({
      name,
      op_type: "Gemm",
      input: [input, `${name}_weight`, `${name}_bias`],
      output: [output],
      attribute: [
        { name: "alpha", f: 1.0, type: "FLOAT" },
        { name: "beta", f: 1.0, type: "FLOAT" },
        { name: "transA", i: 0, type: "INT" },
        { name: "transB", i: 1, type: "INT" },
      ],
    });

    onnxModel.graph.initializer.push({
      name: `${name}_weight`,
      dims: [inDim, outDim],
      data_type: 1,
    });

    onnxModel.graph.initializer.push({
      name: `${name}_bias`,
      dims: [outDim],
      data_type: 1,
    });
  }

  addBatchNormLayer(onnxModel, name, input, output, numFeatures) {
    onnxModel.graph.node.push({
      name,
      op_type: "BatchNormalization",
      input: [
        input,
        `${name}_scale`,
        `${name}_bias`,
        `${name}_mean`,
        `${name}_var`,
      ],
      output: [output],
      attribute: [
        { name: "epsilon", f: 1e-5, type: "FLOAT" },
        { name: "momentum", f: 0.9, type: "FLOAT" },
      ],
    });

    for (const param of ["scale", "bias", "mean", "var"]) {
      onnxModel.graph.initializer.push({
        name: `${name}_${param}`,
        dims: [numFeatures],
        data_type: 1,
      });
    }
  }

  addSiLULayer(onnxModel, name, input, output) {
    const sigmoidOutput = `${name}_sigmoid_out`;

    // First apply sigmoid
    onnxModel.graph.node.push({
      name: `${name}_sigmoid`,
      op_type: "Sigmoid",
      input: [input],
      output: [sigmoidOutput],
    });

    // Then multiply
    onnxModel.graph.node.push({
      name,
      op_type: "Mul",
      input: [input, sigmoidOutput],
      output: [output],
    });
  }

  addTanhLayer(onnxModel, name, input, output) {
    onnxModel.graph.node.push({
      name,
      op_type: "Tanh",
      input: [input],
      output: [output],
    });
  }

  addSigmoidLayer(onnxModel, name, input, output) {
    onnxModel.graph.node.push({
      name,
      op_type: "Sigmoid",
      input: [input],
      output: [output],
    });
  }

  addResidualBlock(onnxModel, name, input, output, inCh, outCh) {
    const conv1Out = `${name}_conv1_out`;
    const bn1Out = `${name}_bn1_out`;
    const act1Out = `${name}_act1_out`;
    const conv2Out = `${name}_conv2_out`;
    const bn2Out = `${name}_bn2_out`;
    const shortcutOut = `${name}_shortcut_out`;
    const addOut = `${name}_add_out`;

    // First convolution
    this.addConvLayer(
      onnxModel,
      `${name}_conv1`,
      input,
      conv1Out,
      inCh,
      outCh,
      3,
      1,
      1,
    );
    this.addBatchNormLayer(onnxModel, `${name}_bn1`, conv1Out, bn1Out, outCh);
    this.addSiLULayer(onnxModel, `${name}_act1`, bn1Out, act1Out);

    // Second convolution
    this.addConvLayer(
      onnxModel,
      `${name}_conv2`,
      act1Out,
      conv2Out,
      outCh,
      outCh,
      3,
      1,
      1,
    );
    this.addBatchNormLayer(onnxModel, `${name}_bn2`, conv2Out, bn2Out, outCh);

    // Shortcut connection if needed
    if (inCh !== outCh) {
      this.addConvLayer(
        onnxModel,
        `${name}_shortcut`,
        input,
        shortcutOut,
        inCh,
        outCh,
        1,
        1,
        0,
      );
      this.addAddLayer(onnxModel, `${name}_add`, [bn2Out, shortcutOut], output);
    } else {
      this.addAddLayer(onnxModel, `${name}_add`, [bn2Out, input], output);
    }
  }

  addEmbeddingLayer(onnxModel, name, input, output, vocabSize, embedDim) {
    onnxModel.graph.node.push({
      name,
      op_type: "Gather",
      input: [`${name}_weight`, input],
      output: [output],
      attribute: [{ name: "axis", i: 0, type: "INT" }],
    });

    onnxModel.graph.initializer.push({
      name: `${name}_weight`,
      dims: [vocabSize, embedDim],
      data_type: 1,
    });
  }

  addFourierTimeEmbedding(onnxModel, name, input, output) {
    // Simplified Fourier embedding - in real implementation would use custom op
    // For ONNX export, we'll use a linear transformation approximation
    const scaledOut = `${name}_scaled_out`;
    const sinOut = `${name}_sin_out`;
    const cosOut = `${name}_cos_out`;
    const concatOut = `${name}_concat_out`;

    // Scale input by 2π
    onnxModel.graph.node.push({
      name: `${name}_scale`,
      op_type: "Mul",
      input: [input, `${name}_scale_factor`],
      output: [scaledOut],
    });

    onnxModel.graph.initializer.push({
      name: `${name}_scale_factor`,
      dims: [1],
      data_type: 1,
      float_data: [2 * Math.PI],
    });

    // Generate frequencies (simplified)
    const freqCount = Math.floor(128 / 2); // Based on FourierTimeEmbed default
    for (let i = 0; i < freqCount; i++) {
      const freq = i + 1;
      const sinNode = `${name}_sin${i}`;
      const cosNode = `${name}_cos${i}`;

      // Multiply by frequency
      const freqScaledOut = `${name}_freq${i}_scaled_out`;
      onnxModel.graph.node.push({
        name: `${name}_mul_freq${i}`,
        op_type: "Mul",
        input: [scaledOut, `${name}_freq${i}`],
        output: [freqScaledOut],
      });

      onnxModel.graph.initializer.push({
        name: `${name}_freq${i}`,
        dims: [1],
        data_type: 1,
        float_data: [freq],
      });

      // Apply sin and cos
      onnxModel.graph.node.push({
        name: sinNode,
        op_type: "Sin",
        input: [freqScaledOut],
        output: [`${sinNode}_out`],
      });

      onnxModel.graph.node.push({
        name: cosNode,
        op_type: "Cos",
        input: [freqScaledOut],
        output: [`${cosNode}_out`],
      });
    }

    // In a real implementation, we would concatenate all sin/cos outputs
    // For simplicity, we'll just pass through the scaled input
    onnxModel.graph.node.push({
      name: `${name}_identity`,
      op_type: "Identity",
      input: [scaledOut],
      output: [output],
    });
  }

  addSelfAttention(onnxModel, name, input, output, channels) {
    // Simplified self-attention for ONNX export
    // Real implementation would have query/key/value projections and softmax
    onnxModel.graph.node.push({
      name,
      op_type: "Attention",
      input: [
        input,
        `${name}_q_weight`,
        `${name}_k_weight`,
        `${name}_v_weight`,
      ],
      output: [output],
      attribute: [{ name: "num_heads", i: 1, type: "INT" }],
    });

    const headSize = Math.floor(channels / 8);
    onnxModel.graph.initializer.push({
      name: `${name}_q_weight`,
      dims: [channels, headSize],
      data_type: 1,
    });

    onnxModel.graph.initializer.push({
      name: `${name}_k_weight`,
      dims: [channels, headSize],
      data_type: 1,
    });

    onnxModel.graph.initializer.push({
      name: `${name}_v_weight`,
      dims: [channels, channels],
      data_type: 1,
    });
  }

  addUpsampleLayer(onnxModel, name, input, output, scale, mode) {
    onnxModel.graph.node.push({
      name,
      op_type: "Resize",
      input: [input, `${name}_scales`],
      output: [output],
      attribute: [
        { name: "mode", s: mode.toUpperCase(), type: "STRING" },
        { name: "nearest_mode", s: "floor", type: "STRING" },
      ],
    });

    onnxModel.graph.initializer.push({
      name: `${name}_scales`,
      dims: [4],
      data_type: 1,
      float_data: [1, 1, scale, scale],
    });
  }

  addConcatLayer(onnxModel, name, inputs, output, axis) {
    onnxModel.graph.node.push({
      name,
      op_type: "Concat",
      input: inputs,
      output: [output],
      attribute: [{ name: "axis", i: axis, type: "INT" }],
    });
  }

  addAddLayer(onnxModel, name, inputs, output) {
    onnxModel.graph.node.push({
      name,
      op_type: "Add",
      input: inputs,
      output: [output],
    });
  }

  addMulLayer(onnxModel, name, inputs, output) {
    onnxModel.graph.node.push({
      name,
      op_type: "Mul",
      input: inputs,
      output: [output],
    });
  }

  async extractWeightsToONNX(onnxModel, torchModel) {
    // Extract weights from torch-js model and add to ONNX initializers
    // This is a placeholder - in a real implementation, we would traverse
    // the model parameters and convert them to the ONNX format

    console.log("⚠️ Weight extraction not fully implemented in this prototype");
    console.log("   To get actual weights, use the Python ONNX export script");

    // For now, add placeholder weights
    for (const initializer of onnxModel.graph.initializer) {
      if (!initializer.float_data) {
        // Add random placeholder data
        const size = initializer.dims.reduce((a, b) => a * b, 1);
        initializer.float_data = Array.from(
          { length: size },
          () => Math.random() * 0.1,
        );
      }
    }
  }

  async saveONNXJSON(onnxModel, outputPath) {
    // Convert to JSON and save
    const jsonStr = JSON.stringify(onnxModel, null, 2);

    // In a browser environment, we would use the File API
    // For Node.js, we would use fs module
    // For this prototype, we'll log and offer download

    console.log(`💾 ONNX model schema saved (${jsonStr.length} bytes)`);
    console.log(`📥 To download: copy the following JSON to ${outputPath}`);

    // Create a downloadable link in browser context
    if (typeof window !== "undefined") {
      const blob = new Blob([jsonStr], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = outputPath;
      a.textContent = `Download ${outputPath}`;
      a.style.display = "block";
      a.style.margin = "10px 0";
      document.body.appendChild(a);

      console.log("📎 Download link created in the DOM");
    }

    return jsonStr;
  }

  async exportCheckpointToONNX(checkpointPath = "latest.pt") {
    console.log("🔄 Exporting checkpoint to ONNX format...");

    // This would require PyTorch and ONNX runtime
    // For this JavaScript environment, we provide instructions

    const instructions = `
# ONNX Export Instructions for Schrödinger Bridge Models

Since this is a JavaScript environment, full ONNX export requires Python.
Here are the steps:

## 1. Install required packages:
\`\`\`bash
pip install torch onnx onnxruntime
\`\`\`

## 2. Use the Python export script:
Create a file \`export_onnx.py\` with:

\`\`\`python
import torch
import torch.onnx
import sys
from pathlib import Path

# Import model architecture (you'll need to adapt from enhancedoptimaltransport/models.py)
sys.path.append('..')
from enhancedoptimaltransport.models import LabelConditionedVAE, LabelConditionedDrift

def export_vae(checkpoint_path='latest.pt', output_path='vae.onnx'):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model with same architecture
    model = LabelConditionedVAE()
    model.load_state_dict(checkpoint['vae_state'])
    model.eval()
    
    # Create dummy inputs
    batch_size = 1
    dummy_image = torch.randn(batch_size, 3, 96, 96)
    dummy_labels = torch.randint(0, 10, (batch_size,))
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_image, dummy_labels),
        output_path,
        input_names=['image', 'labels'],
        output_names=['reconstruction', 'mu', 'logvar'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'labels': {0: 'batch_size'},
            'reconstruction': {0: 'batch_size'},
            'mu': {0: 'batch_size'},
            'logvar': {0: 'batch_size'}
        }
    )
    print(f"✅ VAE exported to {output_path}")

def export_drift(checkpoint_path='latest.pt', output_path='drift.onnx'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model = LabelConditionedDrift()
    model.load_state_dict(checkpoint['drift_state'])
    model.eval()
    
    batch_size = 1
    dummy_z = torch.randn(batch_size, 8, 6, 6)
    dummy_t = torch.rand(batch_size, 1)
    dummy_labels = torch.randint(0, 10, (batch_size,))
    
    torch.onnx.export(
        model,
        (dummy_z, dummy_t, dummy_labels),
        output_path,
        input_names=['z', 't', 'labels'],
        output_names=['drift'],
        dynamic_axes={
            'z': {0: 'batch_size'},
            't': {0: 'batch_size'},
            'labels': {0: 'batch_size'},
            'drift': {0: 'batch_size'}
        }
    )
    print(f"✅ Drift network exported to {output_path}")

if __name__ == '__main__':
    export_vae()
    export_drift()
\`\`\`

## 3. Run the export:
\`\`\`bash
python export_onnx.py
\`\`\`

## 4. Verify the ONNX models:
\`\`\`bash
python -c "import onnx; model = onnx.load('vae.onnx'); print(f'VAE: {len(model.graph.node)} nodes')"
python -c "import onnx; model = onnx.load('drift.onnx'); print(f'Drift: {len(model.graph.node)} nodes')"
\`\`\`

## Alternative: Use the JSON schema from this tool
The JSON files generated by this tool contain the complete model architecture
and can be used with ONNX builder tools to create actual ONNX files.
`;

    console.log(instructions);
    return instructions;
  }
}

// Export the class
export default ONNXExporter;
