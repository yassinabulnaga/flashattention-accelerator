// ============================================================================
// recip_unit.sv — Fixed-Point Reciprocal via Restoring Division
// ============================================================================
// Computes recip = 1/ℓ where:
//   ℓ input:    Q16.16 unsigned (32-bit), typical range [1.0, 16.0]
//   recip out:  Q0.16  unsigned (16-bit), range (0, 1.0]
//
// Math:
//   ℓ_real = ell_in / 2^16
//   1/ℓ_real = 2^16 / ell_in
//   In Q0.16: recip = (2^16 / ell_in) × 2^16 = 2^32 / ell_in
//
// We compute quotient = 2^32 / ell_in using a 16-cycle restoring divider
// that produces 16 quotient bits (one per cycle).
//
// Latency: 16 cycles after start. Asserts done for 1 cycle with result.
// Resources: one 33-bit subtractor, no DSPs.
// ============================================================================

module recip_unit (
    input  logic        clk,
    input  logic        rst_n,

    // Control
    input  logic        start,
    input  logic [31:0] ell_in,         // Q16.16 unsigned

    // Output
    output logic        done,
    output logic [15:0] recip_out       // Q0.16 unsigned (1/ℓ)
);

    // ========================================================================
    // Restoring divider: quotient = dividend / divisor
    //   dividend = 2^32, divisor = ell_in
    //
    // Standard algorithm (MSB-first, 16 quotient bits):
    //   R = dividend
    //   for i = 15 downto 0:
    //     R_trial = R - (divisor << i)
    //     if R_trial >= 0: Q[i] = 1, R = R_trial
    //     else:            Q[i] = 0
    //
    // Equivalently (shift-subtract form):
    //   R starts holding the dividend in upper bits
    //   Each cycle: shift R left by 1, compare upper 33 bits with divisor
    //   If >=: subtract and set quotient bit
    //
    // We use a 48-bit remainder register:
    //   Bits [47:16] hold the "working" portion compared against divisor
    //   Bits [15:0]  shift in zeros (or could hold quotient)
    // ========================================================================

    logic [3:0]  bit_cnt;
    logic        running;
    logic [32:0] rem;           // 33-bit remainder (1 extra for sign detection)
    logic [31:0] divisor;
    logic [15:0] quotient;

    // Trial subtraction (combinational)
    logic [32:0] rem_shifted;
    logic [32:0] rem_sub;
    logic        sub_positive;

    always_comb begin
        rem_shifted  = {rem[31:0], 1'b0};          // shift left by 1
        rem_sub      = rem_shifted - {1'b0, divisor};
        sub_positive = ~rem_sub[32];                // no borrow = remainder >= divisor
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running   <= 1'b0;
            done      <= 1'b0;
            bit_cnt   <= 4'd0;
            rem       <= 33'd0;
            divisor   <= 32'd0;
            quotient  <= 16'd0;
            recip_out <= 16'd0;
        end else if (start && !running) begin
            // dividend = 2^32.  In shift-subtract form, we start with
            // remainder = dividend >> 16 = 2^16 = 65536, since we'll
            // shift left 16 times to extract 16 quotient bits.
            // (Total effective dividend = rem_init << 16 = 2^32)
            running   <= 1'b1;
            done      <= 1'b0;
            bit_cnt   <= 4'd0;
            rem       <= {1'b0, 32'h0001_0000};     // 2^16
            divisor   <= (ell_in == 32'd0) ? 32'd1 : ell_in;
            quotient  <= 16'd0;
        end else if (running) begin
            // Shift-subtract step
            if (sub_positive) begin
                rem      <= rem_sub;
                quotient <= {quotient[14:0], 1'b1};
            end else begin// shift remainder try again 
                rem      <= rem_shifted;
                quotient <= {quotient[14:0], 1'b0};
            end

            if (bit_cnt == 4'd15) begin
                running   <= 1'b0;
                done      <= 1'b1;
                // Capture final quotient with this cycle's bit
                if (sub_positive)
                    recip_out <= {quotient[14:0], 1'b1};
                else
                    recip_out <= {quotient[14:0], 1'b0};
            end else begin
                bit_cnt <= bit_cnt + 4'd1;
                done    <= 1'b0;
            end
        end else begin
            done <= 1'b0;
        end
    end

endmodule