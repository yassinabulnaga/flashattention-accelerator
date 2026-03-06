// ============================================================================
// exp2_unit.sv — Compute exp(x) via base-2 trick for Q8.8 fixed-point
// ============================================================================
// exp(x) = 2^(x * log2(e))
//
// Input:  x  — Q8.8 signed, expected ≤ 0
// Output: y  — Q8.8 unsigned (exp is always ≥ 0)
//
// Saturation:
//   x ≥ 0    → clamp to 1.0 (0x0100)
//   x < -10  → clamp to 0x0000
//
// Q format chain:
//   Stage 1: y = x × LOG2E          Q8.8 × Q2.14 → Q10.22, >>> 14 → Q8.8
//   Stage 2: split y → y_int, y_frac (Q0.8)
//            tmp = C2 × y_frac      Q0.8 × Q0.8 → Q0.16
//   Stage 3: inner = C1 + tmp>>8    Q0.8
//            horner = inner × y_frac Q0.8 × Q0.8 → Q0.16
//   Stage 4: raw = {1, horner>>8}   Q1.8 (9-bit), zero-extend to Q8.8
//            out = raw >> |y_int|    Q8.8
//
// Polynomial: 2^f ≈ 1.0 + f×(C1 + f×C2)  for f ∈ [0,1)
//   C1 = 177/256 ≈ 0.6914     C2 = 61/256 ≈ 0.2383
//   The "1.0 +" is added as a hardwired bit at stage 4.
//
// 3 DSP multiplies.  4 registered stages.  1 result/cycle throughput.
// ============================================================================

module exp2_unit (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        in_valid,
    input  logic signed [15:0] x_in,    // Q8.8

    output logic        out_valid,
    output logic        [15:0] out_exp  // Q8.8 unsigned
);

    // Constants
    localparam signed [15:0] LOG2E = 16'sd23638;   // Q2.14: round(1.4427 × 16384)
    localparam signed [15:0] X_MIN = -16'sd2560;   // -10.0 in Q8.8
    localparam        [7:0]  C2    = 8'd61;        // Q0.8: round(0.2402 × 256)
    localparam        [7:0]  C1    = 8'd177;       // Q0.8: round(0.6931 × 256)

    // ========================================================================
    // Stage 1: y = x × LOG2E                                        [1 DSP]
    // ========================================================================
    // Combinational: compute product and truncated y
    logic signed [31:0] s1_product_comb;   // Q10.22
    logic signed [15:0] s1_y_comb;         // Q8.8

    always_comb begin
        s1_product_comb = x_in * LOG2E;              // Q8.8 × Q2.14 → Q10.22
        s1_y_comb       = s1_product_comb >>> 14;     // Q10.22 → Q8.8
    end

    // Registered outputs of stage 1
    logic signed [15:0] s1_y;
    logic               s1_clamp_one;
    logic               s1_clamp_zero;
    logic               s1_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_y          <= 0;
            s1_clamp_one  <= 0;
            s1_clamp_zero <= 0;
            s1_valid      <= 0;
        end else begin
            s1_valid      <= in_valid;
            s1_clamp_one  <= (x_in >= 16'sd0);
            s1_clamp_zero <= (x_in < X_MIN);
            s1_y          <= s1_y_comb;
        end
    end

    // ========================================================================
    // Stage 2: floor split + C2 × y_frac                            [1 DSP]
    // ========================================================================
    // For negative Q8.8 values, arithmetic right shift by 8 (>>>) gives
    // floor, not truncation toward zero. Two's complement arithmetic shift
    // fills with sign bits, rounding toward −∞.
    //
    //   Example: y = -371 (Q8.8 = -1.4492)
    //     Binary: 1111_1110_1001_1101
    //     >>> 8:  1111_1111_1111_1110 = -2    floor(-1.4492) = -2  ✓
    //
    //   Example: y = -256 (Q8.8 = -1.0, exact)
    //     Binary: 1111_1111_0000_0000
    //     >>> 8:  1111_1111_1111_1111 = -1    floor(-1.0) = -1  ✓
    //
    // y_frac = y[7:0], unsigned ∈ [0, 255], represents the fractional part.
    // When y_int = y >>> 8, then y = y_int × 256 + y_frac holds exactly
    // because >>> preserves this identity in two's complement.

    logic signed [15:0] s2_y_int;
    logic        [7:0]  s2_y_frac;
    logic        [15:0] s2_c2_frac;    // Q0.16
    logic               s2_clamp_one;
    logic               s2_clamp_zero;
    logic               s2_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_y_int      <= 0;
            s2_y_frac     <= 0;
            s2_c2_frac    <= 0;
            s2_clamp_one  <= 0;
            s2_clamp_zero <= 0;
            s2_valid      <= 0;
        end else begin
            s2_valid      <= s1_valid;
            s2_clamp_one  <= s1_clamp_one;
            s2_clamp_zero <= s1_clamp_zero;

            if (s1_clamp_one || s1_clamp_zero) begin
                s2_y_int   <= 0;
                s2_y_frac  <= 0;
                s2_c2_frac <= 0;
            end else begin
                s2_y_int   <= s1_y >>> 8;             // floor (arithmetic shift). Ex -1.4 -> -2 
                s2_y_frac  <= s1_y[7:0];              // unsigned fractional part
                s2_c2_frac <= C2 * s1_y[7:0];         // Q0.8 × Q0.8 → Q0.16 [DSP]
            end
        end
    end

    // ========================================================================
    // Stage 3: horner = (C1 + C2×frac>>8) × frac                   [1 DSP]
    // ========================================================================
    logic        [15:0] s3_horner;     // Q0.16
    logic signed [15:0] s3_y_int;
    logic               s3_clamp_one;
    logic               s3_clamp_zero;
    logic               s3_valid;

    logic        [7:0]  s3_inner;      // Q0.8: C1 + (C2×frac)>>8

    always_comb begin
        s3_inner = C1 + s2_c2_frac[15:8];             // max ≈ 0.69 + 0.24 = 0.93, fits Q0.8
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s3_horner     <= 0;
            s3_y_int      <= 0;
            s3_clamp_one  <= 0;
            s3_clamp_zero <= 0;
            s3_valid      <= 0;
        end else begin
            s3_valid      <= s2_valid;
            s3_clamp_one  <= s2_clamp_one;
            s3_clamp_zero <= s2_clamp_zero;
            s3_y_int      <= s2_y_int;
            s3_horner     <= s3_inner * s2_y_frac;     // Q0.8 × Q0.8 → Q0.16 [DSP] : (C1+C2*f)*f
        end
    end

    // ========================================================================
    // Stage 4: raw = 1.0 + horner>>8; barrel shift → Q8.8 output
    // ========================================================================
    // raw = {1'b1, horner[15:8]} is 9-bit Q1.8 representing 2^y_frac ∈ [1.0, ~2.0)
    // Zero-extended to 16 bits it's already in Q8.8 bit positions.
    // Right-shift by |y_int| gives 2^y = 2^y_int × 2^y_frac.

    logic [15:0] s4_raw;          // 2^y_frac in Q8.8
    logic [4:0]  s4_shift;        // |y_int|

    always_comb begin
        s4_raw   = {7'b0, 1'b1, s3_horner[15:8]}; // 2^f =  1 + (C1+C2*f)*f
        s4_shift = (-s3_y_int > 16) ? 5'd16 : 5'(-s3_y_int); //find out how much we need to shift. 
    end                                                      // since for y<0, 2^y == LSL by y

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 0;
            out_exp   <= 0;
        end else begin
            out_valid <= s3_valid;

            if (s3_clamp_one)
                out_exp <= 16'h0100;                   // 1.0 in Q8.8
            else if (s3_clamp_zero)
                out_exp <= 16'h0000;
            else if (s4_shift >= 5'd16)
                out_exp <= 16'h0000;                   // underflow
            else
                out_exp <= s4_raw >> s4_shift;
        end
    end

endmodule