// ============================================================================
// tb_recip_unit.sv — Testbench for recip_unit
// ============================================================================
// Tests 1/ℓ for known values. ℓ is Q16.16, result is Q0.16.
//
// Power-of-two cases should be exact.
// Non-power-of-two cases will have ≤1 LSB error (1/65536 ≈ 0.0000153).
// ============================================================================

`timescale 1ns / 1ps

module tb_recip_unit;

    localparam CLK_PERIOD = 10;

    logic        clk, rst_n;
    logic        start;
    logic [31:0] ell_in;
    logic        done;
    logic [15:0] recip_out;

    recip_unit u_dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (start),
        .ell_in    (ell_in),
        .done      (done),
        .recip_out (recip_out)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    function real q016_to_real(input logic [15:0] val);
        q016_to_real = real'(val) / 65536.0;
    endfunction

    task automatic run_test(input real ell_real);
        automatic real expected = 1.0 / ell_real;
        automatic real got;
        automatic real err;

        // Convert ell_real to Q16.16
        ell_in = $rtoi(ell_real * 65536.0);
        start  = 1;
        @(posedge clk);
        start  = 0;

        while (!done) @(posedge clk);
        got = q016_to_real(recip_out);
        err = got - expected;

        $display("  ell=%6.3f → recip=0x%04x (%.6f), expect=%.6f, err=%+.6f",
                 ell_real, recip_out, got, expected, err);
        @(posedge clk);
    endtask

    initial begin
        rst_n = 0;
        start = 0;
        ell_in = 32'd0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        $display("\n=== Reciprocal Unit Tests ===");

        $display("\n--- Power-of-two (should be exact) ---");
        run_test(1.0);      // → 1.0    = 0xFFFF (capped) or 0x10000 overflow → check
        run_test(2.0);      // → 0.5    = 0x8000
        run_test(4.0);      // → 0.25   = 0x4000
        run_test(8.0);      // → 0.125  = 0x0800
        run_test(16.0);     // → 0.0625 = 0x1000

        $display("\n--- Non-power-of-two ---");
        run_test(3.0);      // → 0.33333
        run_test(5.0);      // → 0.2
        run_test(7.0);      // → 0.14286
        run_test(10.0);     // → 0.1
        run_test(12.0);     // → 0.08333
        run_test(1.5);      // → 0.66667
        run_test(6.25);     // → 0.16

        $display("\n--- Typical FlashAttention ℓ values ---");
        run_test(3.222);    // from softmax test 1
        run_test(1.789);    // from softmax test 2

        $display("\n=== All tests complete ===");
        $finish;
    end

endmodule