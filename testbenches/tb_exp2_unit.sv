// ============================================================================
// tb_exp2_unit.sv — Testbench for exp2_unit
// ============================================================================
// Tests exp(x) at several known points in Q8.8:
//   exp(0)    = 1.0        x = 0x0000
//   exp(-1)   ≈ 0.3679     x = 0xFF00  (-256 in Q8.8)
//   exp(-2)   ≈ 0.1353     x = 0xFE00  (-512)
//   exp(-0.5) ≈ 0.6065     x = 0xFF80  (-128)
//   exp(-5)   ≈ 0.0067     x = 0xFB00  (-1280)
//   exp(-10)  ≈ 0.0000454  x = 0xF600  (-2560) → expect ~0 (clamp)
//   exp(1)    = 2.718...   x = 0x0100  (+256) → expect clamp to 1.0
//   exp(-0.25)≈ 0.7788     x = 0xFFC0  (-64)
// ============================================================================

`timescale 1ns / 1ps

module tb_exp2_unit;

    logic        clk, rst_n;
    logic        in_valid;
    logic signed [15:0] x_in;
    logic        out_valid;
    logic        [15:0] out_exp;

    exp2_unit dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    int cycle;
    always_ff @(posedge clk) begin
        if (!rst_n) cycle <= 0;
        else        cycle <= cycle + 1;
    end

    // Monitor
    always_ff @(posedge clk) begin
        if (out_valid)
            $display("cycle %0d: out_exp=0x%04h (Q8.8 = %.4f)",
                     cycle, out_exp, real'(out_exp) / 256.0);
    end

    // Drive one test vector
    task test_exp(input logic signed [15:0] x, input string label, input real expected);
        @(posedge clk);
        x_in     = x;
        in_valid = 1;
        $display("cycle %0d: input x=0x%04h (%0d = Q8.8 %.4f)  [%s, expect %.4f]",
                 cycle, x, x, real'(x) / 256.0, label, expected);
        @(posedge clk);
        in_valid = 0;
        x_in     = 0;
    endtask

    initial begin
        rst_n    = 0;
        in_valid = 0;
        x_in     = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        $display("\n=== exp2_unit tests ===\n");

        test_exp(16'sh0000, "exp(0)",     1.0);
        repeat (6) @(posedge clk);

        test_exp(-16'sd128, "exp(-0.5)",  0.6065);
        repeat (6) @(posedge clk);

        test_exp(-16'sd64,  "exp(-0.25)", 0.7788);
        repeat (6) @(posedge clk);

        test_exp(-16'sd256, "exp(-1)",    0.3679);
        repeat (6) @(posedge clk);

        test_exp(-16'sd512, "exp(-2)",    0.1353);
        repeat (6) @(posedge clk);

        test_exp(-16'sd1280,"exp(-5)",    0.0067);
        repeat (6) @(posedge clk);

        test_exp(-16'sd2560,"exp(-10)",   0.0000);
        repeat (6) @(posedge clk);

        test_exp(16'sh0100, "exp(+1) CLAMP", 1.0);
        repeat (6) @(posedge clk);

        // Back-to-back: verify pipeline throughput
        $display("\n=== Back-to-back: exp(-1) then exp(-2) ===");
        @(posedge clk);
        x_in = -16'sd256; in_valid = 1;
        $display("cycle %0d: beat 1 exp(-1)", cycle);
        @(posedge clk);
        x_in = -16'sd512;
        $display("cycle %0d: beat 2 exp(-2)", cycle);
        @(posedge clk);
        in_valid = 0; x_in = 0;
        repeat (8) @(posedge clk);

        $display("\n=== Done ===");
        $finish;
    end

endmodule