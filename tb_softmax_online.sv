// ============================================================================
// tb_softmax_online.sv — Testbench for softmax_online
// ============================================================================
// Test 1: Single tile, row 0. Scores = {2.0, 1.0, 0, 0, ..., 0}
//   m starts at -128 (init), so m_new = 2.0
//   alpha = exp(-128 - 2.0) → clamped to 0  (first tile, old state is garbage)
//   P_tilde[0] = exp(2-2) = 1.0
//   P_tilde[1] = exp(1-2) = exp(-1) ≈ 0.368
//   P_tilde[2:15] = exp(0-2) = exp(-2) ≈ 0.135
//   ell_tile ≈ 1.0 + 0.368 + 14×0.135 ≈ 3.258
//   ell_new = 0*ell_old + ell_tile ≈ 3.258 (alpha=0, so old ell discarded)
//
// Test 2: Second tile, same row. Scores = {3.0, 0, 0, ..., 0}
//   m_old = 2.0 (from test 1), m_new = 3.0
//   alpha = exp(2-3) = exp(-1) ≈ 0.368
//   P_tilde[0] = exp(3-3) = 1.0
//   P_tilde[1:15] = exp(0-3) = exp(-3) ≈ 0.050
//   ell_tile ≈ 1.0 + 15×0.050 ≈ 1.750
//   ell_new = 0.368 × ell_old_from_test1 + ell_tile
// ============================================================================

`timescale 1ns / 1ps

module tb_softmax_online;

    localparam BR = 16;
    localparam BC = 16;

    logic        clk, rst_n;
    logic        start, init_tile;
    logic [3:0]  row_idx;
    logic signed [15:0] score_row [BC];

    logic        done;
    logic signed [15:0] m_new_out;
    logic        [15:0] alpha_out;
    logic        [15:0] ell_tile_out;
    logic        [15:0] p_tilde [BC];

    softmax_online #(.BR(BR), .BC(BC)) dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    int cycle;
    always_ff @(posedge clk) begin
        if (!rst_n) cycle <= 0;
        else        cycle <= cycle + 1;
    end

    // Print results on done
    always_ff @(posedge clk) begin
        if (done) begin
            $display("cycle %0d: DONE", cycle);
            $display("  m_new     = 0x%04h (%.4f)", m_new_out, real'(m_new_out) / 256.0);
            $display("  alpha     = 0x%04h (%.4f)", alpha_out, real'(alpha_out) / 256.0);
            $display("  ell_tile  = 0x%04h (%.4f)", ell_tile_out, real'(ell_tile_out) / 256.0);
            $display("  P_tilde[0] = 0x%04h (%.4f)", p_tilde[0], real'(p_tilde[0]) / 256.0);
            $display("  P_tilde[1] = 0x%04h (%.4f)", p_tilde[1], real'(p_tilde[1]) / 256.0);
            $display("  P_tilde[2] = 0x%04h (%.4f)", p_tilde[2], real'(p_tilde[2]) / 256.0);
            $display("  P_tilde[3] = 0x%04h (%.4f)", p_tilde[3], real'(p_tilde[3]) / 256.0);
        end
    end

    initial begin
        rst_n     = 0;
        start     = 0;
        init_tile = 0;
        row_idx   = 0;
        for (int i = 0; i < BC; i++) score_row[i] = 0;

        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // Init state for new query tile
        init_tile = 1;
        @(posedge clk);
        init_tile = 0;
        @(posedge clk);

        // ── Test 1: first tile ──
        $display("\n=== Test 1: scores = {2.0, 1.0, 0, ..., 0} ===");
        row_idx = 4'd0;
        score_row[0] = 16'sh0200;    // 2.0
        score_row[1] = 16'sh0100;    // 1.0
        for (int i = 2; i < BC; i++) score_row[i] = 16'sh0000;

        start = 1;
        $display("cycle %0d: start", cycle);
        @(posedge clk);
        start = 0;

        wait (done);
        @(posedge clk);

        repeat (3) @(posedge clk);

        // ── Test 2: second tile, online update ──
        $display("\n=== Test 2: scores = {3.0, 0, ..., 0} ===");
        score_row[0] = 16'sh0300;    // 3.0
        score_row[1] = 16'sh0000;

        start = 1;
        $display("cycle %0d: start", cycle);
        @(posedge clk);
        start = 0;

        wait (done);
        @(posedge clk);

        repeat (3) @(posedge clk);

        // ── Test 3: different row to verify register file independence ──
        $display("\n=== Test 3: row 5, scores = {1.0, 1.0, 1.0, ..., 1.0} ===");
        row_idx = 4'd5;
        for (int i = 0; i < BC; i++) score_row[i] = 16'sh0100;  // all 1.0

        start = 1;
        $display("cycle %0d: start", cycle);
        @(posedge clk);
        start = 0;

        wait (done);
        @(posedge clk);

        repeat (3) @(posedge clk);
        $display("\n=== Done ===");
        $finish;
    end

endmodule