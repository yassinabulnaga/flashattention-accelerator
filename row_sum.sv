// ============================================================================
// row_sum_reduce.sv — 16-input sum-reduce tree
// ============================================================================
// Sums 16 unsigned Q8.8 values (exp outputs, each ∈ [0, 1.0]).
// Combinational (single cycle).
// Output is Q4.8 (12-bit) since max sum = 16 × 1.0 = 16.0.
// Stored in 16 bits for convenience.
// ============================================================================

// Way better timing to split into four levels.
// log2(N) critical path
module row_sum_reduce (
    input  logic [15:0] din [16],   // Q8.8 unsigned exp values
    output logic [15:0] dout        // Q8.8 (sum, max = 16.0 = 0x1000)
);

    // Level 1: 16 → 8 (each sum max = 2.0, needs 10 bits)
    logic [16:0] l1 [8];
    genvar i;
    generate
        for (i = 0; i < 8; i++) begin : g_l1
            assign l1[i] = din[2*i] + din[2*i+1];
        end
    endgenerate

    // Level 2: 8 → 4 (max = 4.0)
    logic [16:0] l2 [4];
    generate
        for (i = 0; i < 4; i++) begin : g_l2
            assign l2[i] = l1[2*i] + l1[2*i+1];
        end
    endgenerate

    // Level 3: 4 → 2 (max = 8.0)
    logic [16:0] l3 [2];
    assign l3[0] = l2[0] + l2[1];
    assign l3[1] = l2[2] + l2[3];

    // Level 4: 2 → 1 (max = 16.0)
    assign dout = l3[0] + l3[1];

endmodule