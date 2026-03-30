// ============================================================================
// flash_attention_top.sv — Top-Level FlashAttention Accelerator
// ============================================================================
// Instantiates:
//   - 4 M10K BRAMs: Q(256×64×16b), K(256×64×16b), V_T(64×256×16b), O(256×64×32b)
//   - 5 tile register files: Qi, Kj, Vj, Sij(32b), [Oi via output_unit]
//   - GEMM engine (16-MAC dot product)
//   - Softmax pipeline (online softmax with exp2, row_max, row_sum)
//   - Output unit (o_row register + rescale/acc/norm)
//   - Reciprocal unit (16-cycle restoring divider)
//   - Controller FSM (master sequencer)
//
// External interface: start, done, and M10K initialization ports.
// For testing, M10K BRAMs use simple behavioral models.
// For synthesis, replace with Quartus IP.
// ============================================================================

module flash_attention_top #(
    parameter BR      = 16,
    parameter BC      = 16,
    parameter D_TILE  = 16,
    parameter D_FULL  = 64,
    parameter NUM_KK  = 4,
    parameter SEQ_LEN = 256
)(
    input  logic clk,
    input  logic rst_n,

    // Control
    input  logic start,
    output logic busy,
    output logic done,

    // --- M10K initialization ports (active before start) ---
    // Q BRAM
    input  logic [13:0]        init_q_addr,
    input  logic signed [15:0] init_q_data,
    input  logic               init_q_we,

    // K BRAM
    input  logic [13:0]        init_k_addr,
    input  logic signed [15:0] init_k_data,
    input  logic               init_k_we,

    // V_T BRAM
    input  logic [13:0]        init_vt_addr,
    input  logic signed [15:0] init_vt_data,
    input  logic               init_vt_we,

    // O BRAM read port (for reading results after done)
    input  logic [13:0]        read_o_addr,
    output logic signed [31:0] read_o_data
);

    // ========================================================================
    // M10K BRAM signals
    // ========================================================================

    // Q BRAM
    logic [13:0]        q_addr;
    logic signed [15:0] q_rdata;
    logic [13:0]        q_compute_addr;

    // K BRAM
    logic [13:0]        k_addr;
    logic signed [15:0] k_rdata;
    logic [13:0]        k_compute_addr;

    // V_T BRAM
    logic [13:0]        vt_addr;
    logic signed [15:0] vt_rdata;
    logic [13:0]        vt_compute_addr;

    // O BRAM
    logic [13:0]        o_addr;
    logic               o_we;
    logic signed [31:0] o_wdata;
    logic signed [31:0] o_rdata;
    logic [13:0]        o_compute_addr;
    logic               o_compute_we;
    logic signed [31:0] o_compute_wdata;

    // Mux: init vs compute access
    assign q_addr  = busy ? q_compute_addr  : init_q_addr;
    assign k_addr  = busy ? k_compute_addr  : init_k_addr;
    assign vt_addr = busy ? vt_compute_addr : init_vt_addr;

    assign o_addr  = busy ? o_compute_addr  : (done ? read_o_addr : 14'd0);
    assign o_we    = busy ? o_compute_we    : 1'b0;
    assign o_wdata = busy ? o_compute_wdata : 32'sd0;

    // O read-back when idle
    assign read_o_data = o_rdata;

    // ========================================================================
    // M10K BRAM instantiations (behavioral for simulation)
    // Replace with Quartus M10K IP for synthesis
    // ========================================================================

    // Q BRAM: 16384 × 16b
    logic signed [15:0] q_mem [0:16383];
    always_ff @(posedge clk) begin
        if (init_q_we && !busy)
            q_mem[init_q_addr] <= init_q_data;
    end
    assign q_rdata = q_mem[q_addr];

    // K BRAM: 16384 × 16b
    logic signed [15:0] k_mem [0:16383];
    always_ff @(posedge clk) begin
        if (init_k_we && !busy)
            k_mem[init_k_addr] <= init_k_data;
    end
    assign k_rdata = k_mem[k_addr];

    // V_T BRAM: 16384 × 16b
    logic signed [15:0] vt_mem [0:16383];
    always_ff @(posedge clk) begin
        if (init_vt_we && !busy)
            vt_mem[init_vt_addr] <= init_vt_data;
    end
    assign vt_rdata = vt_mem[vt_addr];

    // O BRAM: 16384 × 32b
    logic signed [31:0] o_mem [0:16383];
    always_ff @(posedge clk) begin
        if (o_we)
            o_mem[o_addr] <= o_wdata;
    end
    assign o_rdata = o_mem[o_addr];

    // ========================================================================
    // Tile register files
    // ========================================================================

    // --- Qi tile reg (16×16, 16b) ---
    logic               qi_serial_we;
    logic [3:0]         qi_serial_row, qi_serial_col;
    logic signed [15:0] qi_serial_wdata;
    logic [3:0]         qi_row_rd_idx;
    logic signed [15:0] qi_row_rdata [D_TILE];

    tile_reg #(.WIDTH(16)) u_qi_reg (
        .clk(clk), .rst_n(rst_n),
        .clear(1'b0),
        .serial_we(qi_serial_we), .serial_row(qi_serial_row),
        .serial_col(qi_serial_col), .serial_wdata(qi_serial_wdata),
        .serial_rd_row(4'd0), .serial_rd_col(4'd0), .serial_rdata(),
        .row_rd_idx(qi_row_rd_idx), .row_rdata(qi_row_rdata),
        .row_we(1'b0), .row_wr_idx(4'd0), .row_wdata('{default: '0}),
        .elem_we(1'b0), .elem_row(4'd0), .elem_col(4'd0), .elem_wdata('0)
    );

    // --- Kj tile reg (16×16, 16b) ---
    logic               kj_serial_we;
    logic [3:0]         kj_serial_row, kj_serial_col;
    logic signed [15:0] kj_serial_wdata;
    logic [3:0]         kj_row_rd_idx;
    logic signed [15:0] kj_row_rdata [D_TILE];

    tile_reg #(.WIDTH(16)) u_kj_reg (
        .clk(clk), .rst_n(rst_n),
        .clear(1'b0),
        .serial_we(kj_serial_we), .serial_row(kj_serial_row),
        .serial_col(kj_serial_col), .serial_wdata(kj_serial_wdata),
        .serial_rd_row(4'd0), .serial_rd_col(4'd0), .serial_rdata(),
        .row_rd_idx(kj_row_rd_idx), .row_rdata(kj_row_rdata),
        .row_we(1'b0), .row_wr_idx(4'd0), .row_wdata('{default: '0}),
        .elem_we(1'b0), .elem_row(4'd0), .elem_col(4'd0), .elem_wdata('0)
    );

    // --- Vj tile reg (16×16, 16b) ---
    logic               vj_serial_we;
    logic [3:0]         vj_serial_row, vj_serial_col;
    logic signed [15:0] vj_serial_wdata;
    logic [3:0]         vj_row_rd_idx;
    logic signed [15:0] vj_row_rdata [D_TILE];

    tile_reg #(.WIDTH(16)) u_vj_reg (
        .clk(clk), .rst_n(rst_n),
        .clear(1'b0),
        .serial_we(vj_serial_we), .serial_row(vj_serial_row),
        .serial_col(vj_serial_col), .serial_wdata(vj_serial_wdata),
        .serial_rd_row(4'd0), .serial_rd_col(4'd0), .serial_rdata(),
        .row_rd_idx(vj_row_rd_idx), .row_rdata(vj_row_rdata),
        .row_we(1'b0), .row_wr_idx(4'd0), .row_wdata('{default: '0}),
        .elem_we(1'b0), .elem_row(4'd0), .elem_col(4'd0), .elem_wdata('0)
    );

    // --- Sij tile reg (16×16, 32b, accumulator) ---
    logic               sij_clear;
    logic               sij_elem_we;
    logic [3:0]         sij_elem_row, sij_elem_col;
    logic signed [31:0] sij_elem_wdata;
    logic [3:0]         sij_row_rd_idx;
    logic signed [31:0] sij_row_rdata [D_TILE];
    logic               sij_row_we;
    logic [3:0]         sij_row_wr_idx;
    logic signed [31:0] sij_row_wdata [D_TILE];

    tile_reg #(.WIDTH(32)) u_sij_reg (
        .clk(clk), .rst_n(rst_n),
        .clear(sij_clear),
        .serial_we(1'b0), .serial_row(4'd0),
        .serial_col(4'd0), .serial_wdata('0),
        .serial_rd_row(4'd0), .serial_rd_col(4'd0), .serial_rdata(),
        .row_rd_idx(sij_row_rd_idx), .row_rdata(sij_row_rdata),
        .row_we(sij_row_we), .row_wr_idx(sij_row_wr_idx), .row_wdata(sij_row_wdata),
        .elem_we(sij_elem_we), .elem_row(sij_elem_row),
        .elem_col(sij_elem_col), .elem_wdata(sij_elem_wdata)
    );

    // ========================================================================
    // GEMM engine
    // ========================================================================
    logic signed [15:0] gemm_a [D_TILE];
    logic signed [15:0] gemm_b [D_TILE];
    logic               gemm_in_valid;
    logic               gemm_out_valid;
    logic signed [31:0] gemm_out_data;

    gemm_engine u_gemm (
        .clk(clk), .rst_n(rst_n),
        .in_valid(gemm_in_valid),
        .row_buf_a(gemm_a),
        .b_row(gemm_b),
        .out_valid(gemm_out_valid),
        .out_data(gemm_out_data)
    );

    // ========================================================================
    // Softmax pipeline
    // ========================================================================
    logic               sm_start, sm_init_tile, sm_done;
    logic [3:0]         sm_row_idx;
    logic signed [15:0] sm_score_row [BC];
    logic signed [15:0] sm_m_new_out;
    logic        [15:0] sm_alpha;
    logic        [15:0] sm_ell_tile_out;
    logic        [15:0] sm_p_tilde [BC];
    logic        [31:0] sm_ell_read;

    softmax_online #(.BR(BR), .BC(BC)) u_softmax (
        .clk(clk), .rst_n(rst_n),
        .start(sm_start),
        .init_tile(sm_init_tile),
        .row_idx(sm_row_idx),
        .score_row(sm_score_row),
        .done(sm_done),
        .m_new_out(sm_m_new_out),
        .alpha_out(sm_alpha),
        .ell_tile_out(sm_ell_tile_out),
        .p_tilde(sm_p_tilde),
        .ell_read(sm_ell_read)
    );

    // ========================================================================
    // Output unit
    // ========================================================================
    logic        ou_cmd_load, ou_cmd_rescale, ou_cmd_acc;
    logic        ou_cmd_store, ou_cmd_norm;
    logic [15:0] ou_alpha;
    logic signed [31:0] ou_gemm_out;
    logic [3:0]  ou_gemm_idx;
    logic [15:0] ou_recip_ell;
    logic        ou_busy, ou_done;
    logic [3:0]  ou_bram_addr;
    logic        ou_bram_we;
    logic signed [31:0] ou_bram_wdata;

    output_unit #(.D(D_TILE)) u_output (
        .clk(clk), .rst_n(rst_n),
        .cmd_load(ou_cmd_load),
        .cmd_rescale(ou_cmd_rescale),
        .cmd_acc(ou_cmd_acc),
        .cmd_store(ou_cmd_store),
        .cmd_norm(ou_cmd_norm),
        .alpha(ou_alpha),
        .gemm_out(ou_gemm_out),
        .gemm_idx(ou_gemm_idx),
        .recip_ell(ou_recip_ell),
        .bram_addr(ou_bram_addr),
        .bram_we(ou_bram_we),
        .bram_wdata(ou_bram_wdata),
        .bram_rdata(o_rdata),       // direct read from O M10K
        .busy(ou_busy),
        .done(ou_done)
    );

    // ========================================================================
    // Reciprocal unit
    // ========================================================================
    logic        recip_start, recip_done;
    logic [31:0] recip_ell_in;
    logic [15:0] recip_out;

    recip_unit u_recip (
        .clk(clk), .rst_n(rst_n),
        .start(recip_start),
        .ell_in(recip_ell_in),
        .done(recip_done),
        .recip_out(recip_out)
    );

    // ========================================================================
    // Controller FSM
    // ========================================================================
    controller_fsm #(
        .BR(BR), .BC(BC), .D_TILE(D_TILE), .D_FULL(D_FULL),
        .NUM_KK(NUM_KK), .SEQ_LEN(SEQ_LEN)
    ) u_ctrl (
        .clk(clk), .rst_n(rst_n),
        .start(start),
        .busy(busy),
        .all_done(done),

        // M10K
        .q_m10k_addr(q_compute_addr),
        .q_m10k_rdata(q_rdata),
        .k_m10k_addr(k_compute_addr),
        .k_m10k_rdata(k_rdata),
        .vt_m10k_addr(vt_compute_addr),
        .vt_m10k_rdata(vt_rdata),
        .o_m10k_addr(o_compute_addr),
        .o_m10k_we(o_compute_we),
        .o_m10k_wdata(o_compute_wdata),
        .o_m10k_rdata(o_rdata),

        // Qi tile reg
        .qi_serial_we(qi_serial_we),
        .qi_serial_row(qi_serial_row),
        .qi_serial_col(qi_serial_col),
        .qi_serial_wdata(qi_serial_wdata),
        .qi_row_rd_idx(qi_row_rd_idx),
        .qi_row_rdata(qi_row_rdata),

        // Kj tile reg
        .kj_serial_we(kj_serial_we),
        .kj_serial_row(kj_serial_row),
        .kj_serial_col(kj_serial_col),
        .kj_serial_wdata(kj_serial_wdata),
        .kj_row_rd_idx(kj_row_rd_idx),
        .kj_row_rdata(kj_row_rdata),

        // Vj tile reg
        .vj_serial_we(vj_serial_we),
        .vj_serial_row(vj_serial_row),
        .vj_serial_col(vj_serial_col),
        .vj_serial_wdata(vj_serial_wdata),
        .vj_row_rd_idx(vj_row_rd_idx),
        .vj_row_rdata(vj_row_rdata),

        // Sij tile reg
        .sij_clear(sij_clear),
        .sij_elem_we(sij_elem_we),
        .sij_elem_row(sij_elem_row),
        .sij_elem_col(sij_elem_col),
        .sij_elem_wdata(sij_elem_wdata),
        .sij_row_rd_idx(sij_row_rd_idx),
        .sij_row_rdata(sij_row_rdata),
        .sij_row_we(sij_row_we),
        .sij_row_wr_idx(sij_row_wr_idx),
        .sij_row_wdata(sij_row_wdata),

        // GEMM
        .gemm_a(gemm_a),
        .gemm_b(gemm_b),
        .gemm_in_valid(gemm_in_valid),
        .gemm_out_valid(gemm_out_valid),
        .gemm_out_data(gemm_out_data),

        // Softmax
        .sm_start(sm_start),
        .sm_init_tile(sm_init_tile),
        .sm_row_idx(sm_row_idx),
        .sm_score_row(sm_score_row),
        .sm_done(sm_done),
        .sm_alpha(sm_alpha),
        .sm_p_tilde(sm_p_tilde),
        .sm_ell_read(sm_ell_read),

        // Output unit
        .ou_cmd_load(ou_cmd_load),
        .ou_cmd_rescale(ou_cmd_rescale),
        .ou_cmd_acc(ou_cmd_acc),
        .ou_cmd_store(ou_cmd_store),
        .ou_cmd_norm(ou_cmd_norm),
        .ou_alpha(ou_alpha),
        .ou_gemm_out(ou_gemm_out),
        .ou_gemm_idx(ou_gemm_idx),
        .ou_recip_ell(ou_recip_ell),
        .ou_busy(ou_busy),
        .ou_done(ou_done),
        .ou_bram_addr(ou_bram_addr),
        .ou_bram_we(ou_bram_we),
        .ou_bram_wdata(ou_bram_wdata),

        // Recip
        .recip_start(recip_start),
        .recip_ell_in(recip_ell_in),
        .recip_done(recip_done),
        .recip_out(recip_out)
    );

endmodule
