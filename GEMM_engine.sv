// ============================================================================
// gemm_engine.sv — 16-MAC Dot-Product Engine for FlashAttention
// ============================================================================
// Computes one dot product per cycle (after pipeline fill):
//   out = Σ_{k=0}^{15} row_buf_a[k] * b_row[k]
//
// Usage:
//   Phase A (QKᵀ): controller loads SR1 with Qi[r,:], streams Kj rows on b_row
//   Phase C (PV):  controller loads SR1 with P̃ij[r,:], streams Vj_T rows on b_row
//
// The engine is mode-agnostic. It sees row_buf_a (stable) and b_row (changes
// each cycle) and produces a Q16.16 dot product after pipeline latency.
//
// Pipeline: Stage 0 — 16 multiplies (registered)
//           Stage 1 — add pairs: 16 → 8
//           Stage 2 — add pairs: 8 → 4
//           Stage 3 — add pairs: 4 → 2
//           Stage 4 — add pairs: 2 → 1  → out_data
//
// Latency: 5 cycles (1 multiply + 4 adder tree stages)
// Throughput: 1 dot product / cycle
// ============================================================================

module gemm_engine (
    input  logic        clk,
    input  logic        rst_n,

    // Control
    input  logic        in_valid,       // assert each cycle a new b_row is presented

    // Operands
    input  logic signed [15:0] row_buf_a [16], // SR1: stable Q8.8 row (held by controller)
    input  logic signed [15:0] b_row     [16], // streaming Q8.8 row from Kj or Vj_T BRAM

    // Result
    output logic        out_valid,
    output logic signed [31:0] out_data         // Q16.16 dot product
);

    // ========================================================================
    // Stage 0: 16 parallel multiplies (Q8.8 × Q8.8 → Q16.16)
    // ========================================================================
    logic signed [31:0] mul_out [16];
    logic               mul_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mul_valid <= 1'b0;
            for (int i = 0; i < 16; i++)
                mul_out[i] <= 32'sd0;
        end else begin
            mul_valid <= in_valid;
            for (int i = 0; i < 16; i++)
                mul_out[i] <= row_buf_a[i] * b_row[i];   // signed 16×16 → 32
        end
    end

    // ========================================================================
    // Adder tree: 4 registered stages, 16 → 8 → 4 → 2 → 1
    // ========================================================================

    // Stage 1: 16 → 8
    logic signed [31:0] add_s1 [8];
    logic               valid_s1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s1 <= 1'b0;
            for (int i = 0; i < 8; i++)
                add_s1[i] <= 32'sd0;
        end else begin
            valid_s1 <= mul_valid;
            for (int i = 0; i < 8; i++)
                add_s1[i] <= mul_out[2*i] + mul_out[2*i + 1];
        end
    end

    // Stage 2: 8 → 4
    logic signed [31:0] add_s2 [4];
    logic               valid_s2;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s2 <= 1'b0;
            for (int i = 0; i < 4; i++)
                add_s2[i] <= 32'sd0;
        end else begin
            valid_s2 <= valid_s1;
            for (int i = 0; i < 4; i++)
                add_s2[i] <= add_s1[2*i] + add_s1[2*i + 1];
        end
    end

    // Stage 3: 4 → 2
    logic signed [31:0] add_s3 [2];
    logic               valid_s3;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s3 <= 1'b0;
            for (int i = 0; i < 2; i++)
                add_s3[i] <= 32'sd0;
        end else begin
            valid_s3 <= valid_s2;
            for (int i = 0; i < 2; i++)
                add_s3[i] <= add_s2[2*i] + add_s2[2*i + 1];
        end
    end

    // Stage 4: 2 → 1
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_data  <= 32'sd0;
        end else begin
            out_valid <= valid_s3;
            out_data  <= add_s3[0] + add_s3[1];
        end
    end

endmodule