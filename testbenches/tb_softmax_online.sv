// ============================================================================
// tb_softmax_online.sv — Testbench for softmax_online
// ============================================================================
// Test 1: First tile, row 0. Scores = {2.0, 1.0, 0, ..., 0}
//   m_old = -128 (init), m_new = 2.0
//   alpha = exp(-128 - 2) → clamped to 0 (first tile, old state is junk)
//   P[0] = exp(0) = 1.0,  P[1] = exp(-1) ≈ 0.368,  P[2:15] = exp(-2) ≈ 0.135
//   ell_tile ≈ 1.0 + 0.368 + 14×0.135 ≈ 3.26
//   ell_new = 0 × 0 + ell_tile ≈ 3.26
//   After: m[0] = 2.0, ell[0] ≈ 3.26
//
// Test 2: Second tile, same row. Scores = {3.0, 1.0, 0, ..., 0}
//   m_old = 2.0, m_new = 3.0  →  **m actually changes**
//   alpha = exp(2 - 3) = exp(-1) ≈ 0.368
//   P[0] = exp(3-3) = 1.0,  P[1] = exp(1-3) = exp(-2) ≈ 0.135
//   P[2:15] = exp(0-3) = exp(-3) ≈ 0.050
//   ell_tile ≈ 1.0 + 0.135 + 14×0.050 ≈ 1.835
//   ell_new = 0.368 × 3.26 + 1.835 ≈ 3.03
//   After: m[0] = 3.0, ell[0] ≈ 3.03
//
// Test 3: Third tile, same row. Scores = {3.0, 3.0, 3.0, ..., 3.0}
//   m_old = 3.0, m_new = 3.0  →  **m does NOT change**
//   alpha = exp(3-3) = exp(0) = 1.0  (no rescaling needed)
//   P[all] = exp(3-3) = 1.0
//   ell_tile = 16 × 1.0 = 16.0
//   ell_new = 1.0 × 3.03 + 16.0 ≈ 19.03
//   After: m[0] = 3.0, ell[0] ≈ 19.03
//
// Test 4: Different row (row 7) to verify register file independence.
//   Scores = {1.0, 1.0, ..., 1.0}
//   m_old = -128 (untouched row), m_new = 1.0
//   alpha = 0 (clamped)
//   P[all] = exp(0) = 1.0
//   ell_tile = 16.0
//   Row 0 state should be unaffected.
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

    // Print on done
    always_ff @(posedge clk) begin
        if (done) begin
            $display("cycle %0d: DONE", cycle);
            $display("  m_new      = 0x%04h (%.4f)", m_new_out, real'(m_new_out) / 256.0);
            $display("  alpha      = 0x%04h (%.4f)", alpha_out, real'(alpha_out) / 256.0);
            $display("  ell_tile   = 0x%04h (%.4f)", ell_tile_out, real'(ell_tile_out) / 256.0);
            $display("  P_tilde[0] = 0x%04h (%.4f)", p_tilde[0], real'(p_tilde[0]) / 256.0);
            $display("  P_tilde[1] = 0x%04h (%.4f)", p_tilde[1], real'(p_tilde[1]) / 256.0);
            $display("  P_tilde[2] = 0x%04h (%.4f)", p_tilde[2], real'(p_tilde[2]) / 256.0);
            $display("  P_tilde[15]= 0x%04h (%.4f)", p_tilde[15], real'(p_tilde[15]) / 256.0);
        end
    end

    // Run one softmax pass and wait for done
    task run_softmax();
        start = 1;
        $display("cycle %0d: start (row %0d)", cycle, row_idx);
        @(posedge clk);
        start = 0;
        wait (done);
        @(posedge clk);
        repeat (2) @(posedge clk);
    endtask

    initial begin
        rst_n     = 0;
        start     = 0;
        init_tile = 0;
        row_idx   = 0;
        for (int i = 0; i < BC; i++) score_row[i] = 0;

        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // Init
        init_tile = 1;
        @(posedge clk);
        init_tile = 0;
        @(posedge clk);

        // ── Test 1: first tile, alpha should be 0 ──
        $display("\n=== Test 1: first tile, scores={2.0, 1.0, 0,...} ===");
        $display("  Expect: m_new=2.0, alpha≈0, P[0]=1.0, P[1]≈0.37, P[2:]≈0.13");
        row_idx = 4'd0;
        score_row[0] = 16'sh0200;    // 2.0
        score_row[1] = 16'sh0100;    // 1.0
        for (int i = 2; i < BC; i++) score_row[i] = 16'sh0000;
        run_softmax();

        // ── Test 2: second tile, m shifts from 2.0 → 3.0, alpha = exp(-1) ──
        $display("\n=== Test 2: second tile, scores={3.0, 1.0, 0,...} ===");
        $display("  Expect: m_new=3.0, alpha≈0.37, P[0]=1.0, P[1]≈0.13, P[2:]≈0.05");
        score_row[0] = 16'sh0300;    // 3.0
        score_row[1] = 16'sh0100;    // 1.0
        for (int i = 2; i < BC; i++) score_row[i] = 16'sh0000;
        run_softmax();

        // ── Test 3: third tile, m stays at 3.0, alpha = exp(0) = 1.0 ──
        $display("\n=== Test 3: third tile, scores={3.0, 3.0, ..., 3.0} ===");
        $display("  Expect: m_new=3.0, alpha=1.0, P[all]=1.0, ell_tile=16.0");
        for (int i = 0; i < BC; i++) score_row[i] = 16'sh0300;  // all 3.0
        run_softmax();

        // ── Test 4: different row, verify independence ──
        $display("\n=== Test 4: row 7, scores={1.0, ..., 1.0} ===");
        $display("  Expect: m_new=1.0, alpha≈0 (m_old=-128), P[all]=1.0, ell_tile=16.0");
        row_idx = 4'd7;
        for (int i = 0; i < BC; i++) score_row[i] = 16'sh0100;  // all 1.0
        run_softmax();

        repeat (3) @(posedge clk);
        $display("\n=== All tests complete ===");
        $finish;
    end

endmodule