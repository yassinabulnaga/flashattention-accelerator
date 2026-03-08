// ============================================================================
// tb_output_unit.sv — Testbench for output_unit
// ============================================================================
// Tests:
//   1. LOAD:      Load o_row from BRAM with known values
//   2. RESCALE:   Multiply o_row by alpha=0.5, check values halved
//   3. ACCUMULATE: Add GEMM results to specific o_row slots
//   4. STORE:     Write o_row back to BRAM, verify contents
//   5. NORMALIZE: Multiply o_row by recip_ell, check final output
// ============================================================================

`timescale 1ns / 1ps

module tb_output_unit;

    localparam D = 16;
    localparam CLK_PERIOD = 10;

    // DUT signals
    logic        clk, rst_n;
    logic        cmd_load, cmd_rescale, cmd_acc, cmd_store, cmd_norm;
    logic [15:0] alpha;
    logic signed [31:0] gemm_out;
    logic [3:0]  gemm_idx;
    logic [15:0] recip_ell;
    logic [3:0]  bram_addr;
    logic        bram_we;
    logic signed [31:0] bram_wdata;
    logic signed [31:0] bram_rdata;
    logic        busy, done;

    // Simulated BRAM (Oi row)
    logic signed [31:0] oi_bram [D];

    // BRAM read model: combinational read at bram_addr
    assign bram_rdata = oi_bram[bram_addr];

    // BRAM write model: capture on clock when bram_we
    always_ff @(posedge clk) begin
        if (bram_we)
            oi_bram[bram_addr] <= bram_wdata;
    end

    // DUT
    output_unit #(.D(D)) u_dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .cmd_load  (cmd_load),
        .cmd_rescale(cmd_rescale),
        .cmd_acc   (cmd_acc),
        .cmd_store (cmd_store),
        .cmd_norm  (cmd_norm),
        .alpha     (alpha),
        .gemm_out  (gemm_out),
        .gemm_idx  (gemm_idx),
        .recip_ell (recip_ell),
        .bram_addr (bram_addr),
        .bram_we   (bram_we),
        .bram_wdata(bram_wdata),
        .bram_rdata(bram_rdata),
        .busy      (busy),
        .done      (done)
    );

    // Clock
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Helper: wait for done
    task automatic wait_done();
        while (!done) @(posedge clk);
        @(posedge clk);  // one more cycle for op to return to IDLE
    endtask

    // Helper: deassert all commands
    task automatic clear_cmds();
        cmd_load    = 0;
        cmd_rescale = 0;
        cmd_acc     = 0;
        cmd_store   = 0;
        cmd_norm    = 0;
    endtask

    // Helper: print Q16.16 as float
    function real q16_to_real(input logic signed [31:0] val);
        q16_to_real = real'(val) / 65536.0;
    endfunction

    // Helper: print Q8.8 as float
    function real q8_to_real(input logic [15:0] val);
        q8_to_real = real'(val) / 256.0;
    endfunction

    initial begin
        // ============================================================
        // Reset
        // ============================================================
        rst_n = 0;
        clear_cmds();
        alpha     = 16'h0100;  // 1.0
        gemm_out  = 32'sd0;
        gemm_idx  = 4'd0;
        recip_ell = 16'd0;

        // Initialize BRAM with known pattern: oi_bram[k] = (k+1) << 16 = (k+1).0 in Q16.16
        for (int i = 0; i < D; i++)
            oi_bram[i] = (i + 1) <<< 16;  // 1.0, 2.0, ..., 16.0

        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // ============================================================
        // Test 1: LOAD — o_row ← BRAM
        // ============================================================
        $display("\n=== Test 1: LOAD o_row from BRAM ===");
        cmd_load = 1;
        @(posedge clk);
        cmd_load = 0;  // pulse for 1 cycle, FSM latches it
        wait_done();
        $display("  LOAD complete");

        // ============================================================
        // Test 2: RESCALE — o_row[k] *= alpha (alpha = 0.5 = 0x0080)
        // ============================================================
        $display("\n=== Test 2: RESCALE by alpha=0.5 ===");
        alpha = 16'h0080;  // 0.5 in Q8.8
        cmd_rescale = 1;
        @(posedge clk);
        cmd_rescale = 0;
        wait_done();

        // Store to BRAM to check values
        cmd_store = 1;
        @(posedge clk);
        cmd_store = 0;
        wait_done();

        $display("  After rescale by 0.5:");
        for (int i = 0; i < D; i++) begin
            $display("    o_row[%0d] = 0x%08x (%.4f), expect %.4f",
                     i, oi_bram[i], q16_to_real(oi_bram[i]),
                     (i + 1) * 0.5);
        end

        // ============================================================
        // Test 3: ACCUMULATE — Add GEMM values to specific slots
        // ============================================================
        $display("\n=== Test 3: ACCUMULATE gemm results ===");

        // Add 10.0 (Q16.16 = 0x000A_0000) to o_row[0]
        gemm_out = 32'sh000A_0000;
        gemm_idx = 4'd0;
        cmd_acc  = 1;
        @(posedge clk);

        // Add 5.0 to o_row[7]
        gemm_out = 32'sh0005_0000;
        gemm_idx = 4'd7;
        @(posedge clk);

        cmd_acc = 0;
        gemm_out = 32'sd0;
        @(posedge clk);

        // Store to check
        cmd_store = 1;
        @(posedge clk);
        cmd_store = 0;
        wait_done();

        $display("  After accumulate:");
        $display("    o_row[0]  = 0x%08x (%.4f), expect %.4f (0.5 + 10.0)", 
                 oi_bram[0], q16_to_real(oi_bram[0]), 0.5 + 10.0);
        $display("    o_row[7]  = 0x%08x (%.4f), expect %.4f (4.0 + 5.0)", 
                 oi_bram[7], q16_to_real(oi_bram[7]), 4.0 + 5.0);
        $display("    o_row[1]  = 0x%08x (%.4f), expect %.4f (unchanged)", 
                 oi_bram[1], q16_to_real(oi_bram[1]), 1.0);

        // ============================================================
        // Test 4: NORMALIZE — o_row[k] *= recip_ell
        //   Load fresh values first: all o_row = 8.0
        //   recip_ell = 1/16 = 0x1000 in Q0.16 (4096/65536 = 0.0625)
        //   Expect: 8.0 × 0.0625 = 0.5
        // ============================================================
        $display("\n=== Test 4: NORMALIZE ===");

        // Reload BRAM with uniform 8.0
        for (int i = 0; i < D; i++)
            oi_bram[i] = 32'sh0008_0000;  // 8.0 in Q16.16

        // Load into o_row
        cmd_load = 1;
        @(posedge clk);
        cmd_load = 0;
        wait_done();

        // Normalize by 1/16
        recip_ell = 16'h1000;  // Q0.16: 4096 = 1/16 × 65536
        cmd_norm = 1;
        @(posedge clk);
        cmd_norm = 0;
        wait_done();

        // Store result
        cmd_store = 1;
        @(posedge clk);
        cmd_store = 0;
        wait_done();

        $display("  After normalize by 1/16:");
        for (int i = 0; i < 4; i++) begin
            $display("    o_row[%0d] = 0x%08x (%.4f), expect 0.5000",
                     i, oi_bram[i], q16_to_real(oi_bram[i]));
        end

        // ============================================================
        // Test 5: Full sequence — load, rescale, accumulate, normalize
        //   Simulates one row of Phase B→C:
        //   O_old = {4.0, 4.0, ...}, alpha = 0.25
        //   After rescale: {1.0, 1.0, ...}
        //   GEMM adds 3.0 to each slot → {4.0, 4.0, ...}
        //   Normalize by 1/4 → {1.0, 1.0, ...}
        // ============================================================
        $display("\n=== Test 5: Full sequence (load→rescale→acc→norm→store) ===");

        // Setup BRAM = 4.0 everywhere
        for (int i = 0; i < D; i++)
            oi_bram[i] = 32'sh0004_0000;

        // Load
        cmd_load = 1;
        @(posedge clk);
        cmd_load = 0;
        wait_done();

        // Rescale by 0.25 (Q8.8 = 0x0040)
        alpha = 16'h0040;
        cmd_rescale = 1;
        @(posedge clk);
        cmd_rescale = 0;
        wait_done();

        // Accumulate 3.0 into each slot
        gemm_out = 32'sh0003_0000;  // 3.0 in Q16.16
        for (int i = 0; i < D; i++) begin
            gemm_idx = i[3:0];
            cmd_acc  = 1;
            @(posedge clk);
        end
        cmd_acc  = 0;
        gemm_out = 32'sd0;
        @(posedge clk);

        // Normalize by 1/4 (Q0.16 = 16384)
        recip_ell = 16'h4000;  // 16384/65536 = 0.25
        cmd_norm = 1;
        @(posedge clk);
        cmd_norm = 0;
        wait_done();

        // Store
        cmd_store = 1;
        @(posedge clk);
        cmd_store = 0;
        wait_done();

        $display("  After full sequence:");
        for (int i = 0; i < 4; i++) begin
            $display("    o_row[%0d] = 0x%08x (%.4f), expect 1.0000",
                     i, oi_bram[i], q16_to_real(oi_bram[i]));
        end
        $display("    o_row[15]= 0x%08x (%.4f), expect 1.0000",
                 oi_bram[15], q16_to_real(oi_bram[15]));

        // ============================================================
        $display("\n=== All tests complete ===");
        $finish;
    end

endmodule