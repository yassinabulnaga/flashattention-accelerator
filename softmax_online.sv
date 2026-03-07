// ============================================================================
// softmax_online.sv — Online Softmax for One Query Tile Row
// ============================================================================
// Owns m[] and ell[] register files. Processes one score row per start pulse.
//
// Inputs are latched on start so controller can freely change them after.
//
// FSM advances on exp2 out_valid signals, not hardcoded counters.
//
// Sequence per row:
//   S_ALPHA:       latch inputs, feed (m_old - m_new) into exp2 lane[0]
//   S_ALPHA_WAIT:  wait for exp_out_valid[0] → capture alpha
//   S_PTILDE:      feed (score[c] - m_new) into all 16 exp2 lanes
//   S_PTILDE_WAIT: wait for exp_out_valid[0] → capture P_tilde[0:15]
//   S_DONE:        compute ell_new, commit m and ell, assert done
//
// Controller responsibilities:
//   - Present score_row and row_idx before asserting start
//   - Read alpha_out and p_tilde[] after done
//   - Rescale SR3 by alpha, write p_tilde back to Sij BRAM
// ============================================================================

module softmax_online #(
    parameter BR = 16,
    parameter BC = 16
)(
    input  logic        clk,
    input  logic        rst_n,

    // Control
    input  logic        start,
    input  logic        init_tile,      // reset m/ell for new query tile
    input  logic [3:0]  row_idx,

    // Score row input
    input  logic signed [15:0] score_row [BC],  // Q8.8

    // Outputs — valid when done is asserted
    output logic        done,
    output logic signed [15:0] m_new_out,       // Q8.8
    output logic        [15:0] alpha_out,       // Q8.8 unsigned
    output logic        [15:0] ell_tile_out,    // Q8.8 unsigned
    output logic        [15:0] p_tilde [BC]     // Q8.8 unsigned
);

    // ========================================================================
    // Latched inputs — captured on start, stable throughout FSM
    // ========================================================================
    logic [3:0]         row_idx_lat;
    logic signed [15:0] score_lat [BC];

    // ========================================================================
    // State register files
    // ========================================================================
    logic signed [15:0] m_reg   [BR];
    logic        [31:0] ell_reg [BR];

    logic fsm_commit;
    logic signed [15:0] m_new;
    logic        [31:0] ell_new;

    integer k;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (k = 0; k < BR; k++) begin
                m_reg[k]   <= -16'sd32768; //most negative value possible
                ell_reg[k] <= 32'd0;
            end
        end else if (init_tile) begin //if new tile basically reset
            for (k = 0; k < BR; k++) begin
                m_reg[k]   <= -16'sd32768;
                ell_reg[k] <= 32'd0;
            end
        end else if (fsm_commit) begin //when fsm says row done
            m_reg[row_idx_lat]   <= m_new;
            ell_reg[row_idx_lat] <= ell_new;
        end
    end

    // Read current state (from latched row index)
    logic signed [15:0] m_old;
    logic        [31:0] ell_old;

    assign m_old   = m_reg[row_idx_lat];
    assign ell_old = ell_reg[row_idx_lat];

    // ========================================================================
    // Row max + m_new (combinational, operates on latched scores)
    // ========================================================================
    logic signed [15:0] m_tile;

    row_max_reduce u_row_max (
        .din  (score_lat),
        .dout (m_tile)
    );

    assign m_new = (m_old > m_tile) ? m_old : m_tile;

    // ========================================================================
    // FSM
    // ========================================================================
    typedef enum logic [2:0] {
        S_IDLE,
        S_ALPHA,
        S_ALPHA_WAIT,
        S_PTILDE,
        S_PTILDE_WAIT,
        S_DONE
    } state_t;

    state_t state, state_next;

    // Exp2 signals
    logic signed [15:0] exp_in       [BC];
    logic               exp_in_valid [BC];
    logic        [15:0] exp_out      [BC];
    logic               exp_out_valid[BC];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= S_IDLE;
        else
            state <= state_next;
    end

    always_comb begin
        state_next = state;
        fsm_commit = 1'b0;
        done       = 1'b0;

        case (state)
            S_IDLE: begin
                if (start)
                    state_next = S_ALPHA;
            end

            S_ALPHA: begin
                // Inputs latched in always_ff below. Feed lane[0] this cycle.
                state_next = S_ALPHA_WAIT;
            end

            S_ALPHA_WAIT: begin
                // Advance when lane[0] produces a valid output
                if (exp_out_valid[0])
                    state_next = S_PTILDE;
            end

            S_PTILDE: begin
                // Feed all 16 lanes this cycle
                state_next = S_PTILDE_WAIT;
            end

            S_PTILDE_WAIT: begin
                // Advance when lanes produce valid outputs
                if (exp_out_valid[0])
                    state_next = S_DONE;
            end

            S_DONE: begin
                done       = 1'b1;
                fsm_commit = 1'b1;
                state_next = S_IDLE;
            end

            default: state_next = S_IDLE;
        endcase
    end

    // ========================================================================
    // Latch inputs on S_IDLE → S_ALPHA transition
    // ========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            row_idx_lat <= 4'd0;
            for (int i = 0; i < BC; i++)
                score_lat[i] <= 16'sd0;
        end else if (state == S_IDLE && start) begin
            row_idx_lat <= row_idx;
            for (int i = 0; i < BC; i++)
                score_lat[i] <= score_row[i];
        end
    end

    // ========================================================================
    // Exp2 units (16 lanes)
    // ========================================================================
    genvar g;
    generate
        for (g = 0; g < BC; g++) begin : g_exp2
            exp2_unit u_exp2 (
                .clk       (clk),
                .rst_n     (rst_n),
                .in_valid  (exp_in_valid[g]),
                .x_in      (exp_in[g]),
                .out_valid (exp_out_valid[g]),
                .out_exp   (exp_out[g])
            );
        end
    endgenerate

    // ========================================================================
    // Exp2 input steering
    // ========================================================================
    always_comb begin
        for (int i = 0; i < BC; i++) begin
            exp_in[i]       = 16'sd0;
            exp_in_valid[i] = 1'b0;
        end

        case (state)
            S_ALPHA: begin
                exp_in[0]       = m_old - m_new;        // ≤ 0
                exp_in_valid[0] = 1'b1;
            end

            S_PTILDE: begin
                for (int i = 0; i < BC; i++) begin
                    exp_in[i]       = score_lat[i] - m_new;  // ≤ 0
                    exp_in_valid[i] = 1'b1;
                end
            end

            default: ;
        endcase
    end

    // ========================================================================
    // Capture exp2 outputs
    // ========================================================================
    logic [15:0] alpha_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            alpha_reg <= 16'h0100;
        else if (exp_out_valid[0] && state == S_ALPHA_WAIT)
            alpha_reg <= exp_out[0];
    end

    logic [15:0] p_tilde_reg [BC];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < BC; i++)
                p_tilde_reg[i] <= 16'd0;
        end else begin
            for (int i = 0; i < BC; i++) begin
                if (exp_out_valid[i] && state == S_PTILDE_WAIT)
                    p_tilde_reg[i] <= exp_out[i];
            end
        end
    end

    // ========================================================================
    // Row sum of P_tilde → ell_tile
    // ========================================================================
    logic [15:0] ell_tile;

    row_sum_reduce u_row_sum (
        .din  (p_tilde_reg),
        .dout (ell_tile)
    );

    // ========================================================================
    // ell update: ell_new = alpha * ell_old + ell_tile (internal)
    // ========================================================================
    logic [47:0] alpha_ell_product;
    logic [31:0] alpha_ell_scaled;
    logic [31:0] ell_tile_q16_16;

    always_comb begin
        alpha_ell_product = alpha_reg * ell_old;         // Q8.8 × Q16.16 → Q24.24
        alpha_ell_scaled  = alpha_ell_product[39:8];     // → Q16.16
        ell_tile_q16_16   = {8'd0, ell_tile, 8'd0};     // Q8.8 → Q16.16
        ell_new           = alpha_ell_scaled + ell_tile_q16_16;
    end

    // ========================================================================
    // Output assignments
    // ========================================================================
    assign m_new_out    = m_new;
    assign alpha_out    = alpha_reg;
    assign ell_tile_out = ell_tile;

    generate
        for (g = 0; g < BC; g++) begin : g_ptilde_out
            assign p_tilde[g] = p_tilde_reg[g];
        end
    endgenerate

endmodule