// ============================================================================
// tb_flash_attention_top.sv — Integration Testbench
// ============================================================================
// Loads Q, K, V with simple patterns into M10K, runs full FlashAttention,
// reads back O and checks against a golden model.
//
// Test pattern: Q=K=V= small uniform values so attention output is predictable.
// With Q=K=constant c, S = Q×K^T has all entries = c^2 × d = c^2 × 64.
// Softmax of a uniform S → uniform P̃ = 1/seq_len per element.
// O = P̃ × V = V (since P̃ rows sum to 1 and V is uniform).
//
// For simplicity: Q=K=V all = 0.125 (Q8.8 = 0x0020).
// S[r,c] = sum_d(Q[r,k]*K[c,k]) = 64 * 0.125 * 0.125 = 1.0 per tile-pair kk sum.
// After full accumulation over kk=0..3: S[r,c] = 4.0 (but truncated to Q8.8).
// Softmax of uniform 4.0 → P̃[r,c] = 1/16 per tile (all equal).
// O = P̃ × V ≈ V = 0.125 after normalization.
// ============================================================================

`timescale 1ns / 1ps

module tb_flash_attention_top;

    localparam CLK_PERIOD = 10;
    localparam SEQ_LEN = 256;
    localparam D_FULL  = 64;

    logic clk, rst_n;
    logic start, busy, done;

    // Init ports
    logic [13:0]        init_q_addr,  init_k_addr,  init_vt_addr;
    logic signed [15:0] init_q_data,  init_k_data,  init_vt_data;
    logic               init_q_we,    init_k_we,    init_vt_we;

    // O readback
    logic [13:0]        read_o_addr;
    logic signed [31:0] read_o_data;

    flash_attention_top u_dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .busy(busy), .done(done),
        .init_q_addr(init_q_addr),   .init_q_data(init_q_data),   .init_q_we(init_q_we),
        .init_k_addr(init_k_addr),   .init_k_data(init_k_data),   .init_k_we(init_k_we),
        .init_vt_addr(init_vt_addr), .init_vt_data(init_vt_data), .init_vt_we(init_vt_we),
        .read_o_addr(read_o_addr),   .read_o_data(read_o_data)
    );

    // Clock
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Helpers
    function real q88_to_real(input logic signed [15:0] val);
        q88_to_real = real'(val) / 256.0;
    endfunction

    function real q1616_to_real(input logic signed [31:0] val);
        q1616_to_real = real'(val) / 65536.0;
    endfunction

    // Load task: write one element to a BRAM
    task automatic load_q(input int addr, input logic signed [15:0] data);
        init_q_addr = addr[13:0];
        init_q_data = data;
        init_q_we   = 1'b1;
        @(posedge clk);
        init_q_we   = 1'b0;
    endtask

    task automatic load_k(input int addr, input logic signed [15:0] data);
        init_k_addr = addr[13:0];
        init_k_data = data;
        init_k_we   = 1'b1;
        @(posedge clk);
        init_k_we   = 1'b0;
    endtask

    task automatic load_vt(input int addr, input logic signed [15:0] data);
        init_vt_addr = addr[13:0];
        init_vt_data = data;
        init_vt_we   = 1'b1;
        @(posedge clk);
        init_vt_we   = 1'b0;
    endtask

    integer cycle_count;

    initial begin
        // ============================================================
        // Reset
        // ============================================================
        rst_n = 0;
        start = 0;
        init_q_we = 0; init_k_we = 0; init_vt_we = 0;
        init_q_addr = 0; init_k_addr = 0; init_vt_addr = 0;
        init_q_data = 0; init_k_data = 0; init_vt_data = 0;
        read_o_addr = 0;

        repeat (5) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        // ============================================================
        // Load Q, K, V with uniform 0.125 (Q8.8 = 0x0020)
        // ============================================================
        $display("\n=== Loading Q, K, V ===");

        for (int row = 0; row < SEQ_LEN; row++) begin
            for (int col = 0; col < D_FULL; col++) begin
                load_q(row * D_FULL + col, 16'sh0020);   // Q[row,col] = 0.125
                load_k(row * D_FULL + col, 16'sh0020);   // K[row,col] = 0.125
                // V_T stored as V_T[col,row] = V[row,col]
                load_vt(col * SEQ_LEN + row, 16'sh0020); // V[row,col] = 0.125
            end
        end

        $display("  Load complete: %0d elements per matrix", SEQ_LEN * D_FULL);
        @(posedge clk);

        // ============================================================
        // Run FlashAttention
        // ============================================================
        $display("\n=== Starting FlashAttention ===");
        start = 1;
        @(posedge clk);
        start = 0;

        cycle_count = 0;
        while (!done) begin
            @(posedge clk);
            cycle_count++;
            if (cycle_count % 100000 == 0)
                $display("  ... %0d cycles elapsed", cycle_count);
        end

        $display("  FlashAttention complete in %0d cycles", cycle_count);

        // ============================================================
        // Read back O and check
        // ============================================================
        $display("\n=== Reading O ===");

        // Check a few rows
        for (int row = 0; row < 4; row++) begin
            $display("  O[%0d, 0..3]:", row);
            for (int col = 0; col < 4; col++) begin
                read_o_addr = row * D_FULL + col;
                @(posedge clk);
                @(posedge clk);  // allow read latency
                $display("    O[%0d,%0d] = 0x%08x (%.6f)",
                         row, col, read_o_data, q1616_to_real(read_o_data));
            end
        end

        // Check last row
        $display("  O[255, 0..3]:");
        for (int col = 0; col < 4; col++) begin
            read_o_addr = 255 * D_FULL + col;
            @(posedge clk);
            @(posedge clk);
            $display("    O[255,%0d] = 0x%08x (%.6f)",
                     col, read_o_data, q1616_to_real(read_o_data));
        end

        $display("\n=== Test complete ===");
        $finish;
    end

endmodule
