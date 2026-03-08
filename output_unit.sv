// ============================================================================
// output_unit.sv — Output Row Accumulator for FlashAttention
// ============================================================================
// Manages a single O row register file with five operations:
//
//   1. LOAD:       o_row[k] ← oi_bram_rdata          (16 cycles)
//   2. RESCALE:    o_row[k] ← alpha × o_row[k]       (16 cycles, streaming)
//   3. ACCUMULATE: o_row[k] += gemm_out               (1/cycle from GEMM engine)
//   4. STORE:      oi_bram_wdata ← o_row[k]           (16 cycles)
//   5. NORMALIZE:  o_row[k] ← o_row[k] × recip_ell   (16 cycles, post-loop)
//
// Sequence per row (Phase B → C):
//   1. cmd_load     — o_row ← Oi BRAM               (16 cycles)
//   2. cmd_rescale  — o_row *= alpha                  (16 cycles)
//   3. cmd_acc      — o_row[idx] += gemm_out          (1/cycle, ONLY when idle)
//   4. cmd_store    — Oi BRAM ← o_row                 (16 cycles)
// Post-loop:
//   5. cmd_load + cmd_norm + cmd_store for final O /= ℓ
//
// IMPORTANT: cmd_acc must NOT overlap with load/rescale/store/norm.
//
// Q formats:
//   o_row:     Q16.16 (32-bit signed)
//   alpha:     Q8.8   (16-bit unsigned)
//   gemm_out:  Q16.16 (32-bit signed, from gemm_engine)
//   recip_ell: Q0.16  (16-bit unsigned, 1/ℓ precomputed externally)
//   BRAM data: Q16.16 (32-bit)
//
// Parameters: D = 16 (head dimension = o_row width)
// ============================================================================

module output_unit #(
    parameter D = 16
)(
    input  logic        clk,
    input  logic        rst_n,

    // Command interface
    input  logic        cmd_load,       // load o_row from BRAM (pulse to start)
    input  logic        cmd_rescale,    // rescale o_row by alpha (pulse to start)
    input  logic        cmd_acc,        // accumulate gemm result at idx
    input  logic        cmd_store,      // store o_row to BRAM (pulse to start)
    input  logic        cmd_norm,       // normalize o_row by recip_ell (pulse to start)

    // Alpha for rescale (Q8.8 unsigned, from softmax)
    input  logic [15:0] alpha,

    // GEMM accumulation port
    input  logic signed [31:0] gemm_out,       // Q16.16 dot product result
    input  logic [3:0]         gemm_idx,        // which o_row element to accumulate into

    // Reciprocal of ℓ for final normalization (Q0.16 unsigned)
    input  logic [15:0] recip_ell,

    // BRAM interface for load/store
    output logic [3:0]  bram_addr,       // 0..D-1
    output logic        bram_we,
    output logic signed [31:0] bram_wdata,  // Q16.16
    input  logic signed [31:0] bram_rdata,  // Q16.16

    // Status
    output logic        busy,
    output logic        done             // pulses when multi-cycle op finishes
);

    // ========================================================================
    // O row register file — 16 × 32-bit Q16.16
    // ========================================================================
    logic signed [31:0] o_row [D];

    // ========================================================================
    // Streaming counter for multi-cycle ops (load/rescale/store/norm)
    // ========================================================================
    logic [3:0] cnt;
    logic       cnt_running;
    logic       cnt_last;

    assign cnt_last = (cnt == D - 1);

    // Track which multi-cycle operation is active
    typedef enum logic [2:0] {
        OP_IDLE,
        OP_LOAD,
        OP_RESCALE,
        OP_STORE,
        OP_NORM
    } op_t;

    op_t op_active;

    // ========================================================================
    // Multi-cycle operation FSM
    // ========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            op_active <= OP_IDLE;
            cnt       <= 4'd0;
        end else begin
            case (op_active)
                OP_IDLE: begin
                    cnt <= 4'd0;
                    if (cmd_load)
                        op_active <= OP_LOAD;
                    else if (cmd_rescale)
                        op_active <= OP_RESCALE;
                    else if (cmd_store)
                        op_active <= OP_STORE;
                    else if (cmd_norm)
                        op_active <= OP_NORM;
                end

                OP_LOAD, OP_RESCALE, OP_STORE, OP_NORM: begin
                    if (cnt_last) begin
                        op_active <= OP_IDLE;
                        cnt       <= 4'd0;
                    end else begin
                        cnt <= cnt + 4'd1;
                    end
                end

                default: op_active <= OP_IDLE;
            endcase
        end
    end

    assign cnt_running = (op_active != OP_IDLE);
    assign busy        = cnt_running;
    assign done        = cnt_running && cnt_last;

    // ========================================================================
    // Rescale multiply: alpha × o_row[cnt]
    //   Q8.8 × Q16.16 → Q24.24, take [39:8] → Q16.16
    // ========================================================================
    logic signed [47:0] rescale_product;
    logic signed [31:0] rescale_result;

    always_comb begin
        rescale_product = $signed({1'b0, alpha}) * o_row[cnt];  // 17b × 32b → 48b
        rescale_result  = rescale_product[39:8];                 // Q24.24 → Q16.16
    end

    // ========================================================================
    // Normalize multiply: recip_ell × o_row[cnt]
    //   Q0.16 × Q16.16 → Q16.32, take [47:16] → Q16.16
    // ========================================================================
    logic signed [47:0] norm_product;
    logic signed [31:0] norm_result;

    always_comb begin
        norm_product = $signed({1'b0, recip_ell}) * o_row[cnt];  // 17b × 32b → 48b
        norm_result  = norm_product[47:16];                       // Q16.32 → Q16.16
    end

    // ========================================================================
    // O row register file update
    // ========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < D; i++)
                o_row[i] <= 32'sd0;
        end else begin
            // Multi-cycle ops: load / rescale / normalize (mutually exclusive with acc)
            case (op_active)
                OP_LOAD:    o_row[cnt] <= bram_rdata;
                OP_RESCALE: o_row[cnt] <= rescale_result;
                OP_NORM:    o_row[cnt] <= norm_result;
                default: begin
                    // Accumulate GEMM output — only when no multi-cycle op is active
                    if (cmd_acc)
                        o_row[gemm_idx] <= o_row[gemm_idx] + gemm_out;
                end
            endcase
        end
    end

    // ========================================================================
    // BRAM address and write interface
    // ========================================================================
    always_comb begin
        bram_addr  = cnt;
        bram_we    = (op_active == OP_STORE);
        bram_wdata = o_row[cnt];
    end

endmodule