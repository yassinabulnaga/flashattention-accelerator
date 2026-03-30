// ============================================================================
// controller_fsm.sv — Master Sequencer for FlashAttention (256×64)
// ============================================================================
// Full loop: i(0..15) × j(0..15) × kk(0..3) with 16×16 register tiles.
// M10K BRAMs hold Q(256×64), K(256×64), V_T(64×256), O(256×64).
// Controller loads tiles serially from M10K into register files, runs
// compute at full speed on registers, stores results back.
//
// Parameters: BR=BC=16, D_FULL=64, D_TILE=16, NUM_KK=4
// ============================================================================

module controller_fsm #(
    parameter BR      = 16,
    parameter BC      = 16,
    parameter D_TILE  = 16,
    parameter D_FULL  = 64,
    parameter NUM_KK  = 4,     // D_FULL / D_TILE
    parameter SEQ_LEN = 256,
    parameter NUM_I   = 16,    // SEQ_LEN / BR
    parameter NUM_J   = 16,    // SEQ_LEN / BC
    parameter GEMM_LAT = 5
)(
    input  logic clk,
    input  logic rst_n,

    // Top-level control
    input  logic start,
    output logic busy,
    output logic all_done,

    // --- M10K Q BRAM (256×64, 16b, read-only) ---
    output logic [13:0]        q_m10k_addr,
    input  logic signed [15:0] q_m10k_rdata,

    // --- M10K K BRAM (256×64, 16b, read-only) ---
    output logic [13:0]        k_m10k_addr,
    input  logic signed [15:0] k_m10k_rdata,

    // --- M10K V_T BRAM (64×256, 16b, read-only) ---
    output logic [13:0]        vt_m10k_addr,
    input  logic signed [15:0] vt_m10k_rdata,

    // --- M10K O BRAM (256×64, 32b, read/write) ---
    output logic [13:0]        o_m10k_addr,
    output logic               o_m10k_we,
    output logic signed [31:0] o_m10k_wdata,
    input  logic signed [31:0] o_m10k_rdata,

    // --- Qi tile register file ---
    output logic               qi_serial_we,
    output logic [3:0]         qi_serial_row,
    output logic [3:0]         qi_serial_col,
    output logic signed [15:0] qi_serial_wdata,
    output logic [3:0]         qi_row_rd_idx,
    input  logic signed [15:0] qi_row_rdata [D_TILE],

    // --- Kj tile register file ---
    output logic               kj_serial_we,
    output logic [3:0]         kj_serial_row,
    output logic [3:0]         kj_serial_col,
    output logic signed [15:0] kj_serial_wdata,
    output logic [3:0]         kj_row_rd_idx,
    input  logic signed [15:0] kj_row_rdata [D_TILE],

    // --- Vj tile register file (V stored transposed) ---
    output logic               vj_serial_we,
    output logic [3:0]         vj_serial_row,
    output logic [3:0]         vj_serial_col,
    output logic signed [15:0] vj_serial_wdata,
    output logic [3:0]         vj_row_rd_idx,
    input  logic signed [15:0] vj_row_rdata [D_TILE],

    // --- Sij tile register file (32b, accumulates across kk) ---
    output logic               sij_clear,
    output logic               sij_elem_we,
    output logic [3:0]         sij_elem_row,
    output logic [3:0]         sij_elem_col,
    output logic signed [31:0] sij_elem_wdata,
    output logic [3:0]         sij_row_rd_idx,
    input  logic signed [31:0] sij_row_rdata [D_TILE],
    // Parallel row write (for P̃ writeback)
    output logic               sij_row_we,
    output logic [3:0]         sij_row_wr_idx,
    output logic signed [31:0] sij_row_wdata [D_TILE],

    // --- GEMM engine ---
    output logic signed [15:0] gemm_a [D_TILE],
    output logic signed [15:0] gemm_b [D_TILE],
    output logic               gemm_in_valid,
    input  logic               gemm_out_valid,
    input  logic signed [31:0] gemm_out_data,

    // --- Softmax ---
    output logic               sm_start,
    output logic               sm_init_tile,
    output logic [3:0]         sm_row_idx,
    output logic signed [15:0] sm_score_row [BC],
    input  logic               sm_done,
    input  logic        [15:0] sm_alpha,
    input  logic        [15:0] sm_p_tilde [BC],
    input  logic        [31:0] sm_ell_read,

    // --- Output unit ---
    output logic        ou_cmd_load,
    output logic        ou_cmd_rescale,
    output logic        ou_cmd_acc,
    output logic        ou_cmd_store,
    output logic        ou_cmd_norm,
    output logic [15:0] ou_alpha,
    output logic signed [31:0] ou_gemm_out,
    output logic [3:0]  ou_gemm_idx,
    output logic [15:0] ou_recip_ell,
    input  logic        ou_busy,
    input  logic        ou_done,

    // --- Output unit BRAM interface (directly to O M10K) ---
    // output_unit drives bram_addr[3:0], bram_we, bram_wdata, reads bram_rdata.
    // Controller provides the row offset. Top wires:
    //   o_m10k_addr = (i_idx*16 + row_r)*64 + kk_idx*16 + ou_bram_addr
    input  logic [3:0]         ou_bram_addr,
    input  logic               ou_bram_we,
    input  logic signed [31:0] ou_bram_wdata,

    // --- Recip unit ---
    output logic        recip_start,
    output logic [31:0] recip_ell_in,
    input  logic        recip_done,
    input  logic [15:0] recip_out
);

    // ========================================================================
    // FSM states
    // ========================================================================
    typedef enum logic [4:0] {
        S_IDLE,

        // Tile loading from M10K
        S_LOAD_QI,            // serial load Qi tile from M10K
        S_LOAD_KJ,            // serial load Kj tile from M10K
        S_CLEAR_SIJ,          // clear Sij accumulator (before kk loop)

        // Phase A: S accumulation
        S_A_GEMM,             // compute one row of S: dot(Qi[r,:], Kj[c,:])
        S_A_GEMM_WAIT,        // wait for GEMM pipeline result
        S_A_WRITE_SIJ,        // accumulate dot product into Sij_reg

        // Phase B: online softmax + rescale
        S_B_SOFTMAX,          // run softmax on Sij[r,:]
        S_B_RESCALE_LOAD_O,   // load Oi chunk for rescaling
        S_B_RESCALE,          // rescale o_row by alpha
        S_B_RESCALE_STORE_O,  // store rescaled Oi chunk
        S_B_WRITE_PTILDE,     // write P̃ row to Sij_reg

        // Phase C: O += P̃ × V
        S_LOAD_VJ,            // serial load Vj_T tile from M10K
        S_LOAD_OI,            // serial load Oi tile from M10K (via output_unit)
        S_C_GEMM,             // dot(P̃[r,:], Vj[c,:])
        S_C_GEMM_WAIT,        // wait for result
        S_C_ACC,              // accumulate into o_row
        S_C_STORE_OI,         // store Oi tile back to M10K

        // Post-loop
        S_P_LOAD_O,           // load Oi chunk
        S_P_RECIP,            // compute 1/ℓ[r]
        S_P_NORM,             // normalize o_row
        S_P_STORE_O,          // store normalized Oi chunk

        S_DONE
    } state_t;

    state_t state;

    // ========================================================================
    // Loop counters
    // ========================================================================
    logic [3:0] i_idx;        // outer Q row block (0..15)
    logic [3:0] j_idx;        // inner K/V row block (0..15)
    logic [1:0] kk_idx;       // feature chunk (0..3)
    logic [3:0] row_r;        // row within tile (0..15)
    logic [3:0] col_c;        // column within tile (0..15)
    logic [7:0] serial_cnt;   // 0..255 for serial M10K transfers
    logic [2:0] wait_cnt;     // GEMM pipeline wait

    // Rescale sub-counter for kk chunks of O
    logic [1:0] rescale_kk;

    // ========================================================================
    // First-cycle detection
    // ========================================================================
    state_t state_prev;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state_prev <= S_IDLE;
        else        state_prev <= state;
    end
    wire first_cycle = (state != state_prev);

    // ========================================================================
    // Serial counter row/col decode
    // ========================================================================
    wire [3:0] ser_row = serial_cnt[7:4];
    wire [3:0] ser_col = serial_cnt[3:0];

    // ========================================================================
    // M10K address generation
    // ========================================================================

    // Q: addr = (i_idx*16 + ser_row) * 64 + kk_idx*16 + ser_col
    wire [13:0] q_tile_addr = ({i_idx, ser_row} * 64) + ({kk_idx, ser_col});

    // K: addr = (j_idx*16 + ser_row) * 64 + kk_idx*16 + ser_col
    wire [13:0] k_tile_addr = ({j_idx, ser_row} * 64) + ({kk_idx, ser_col});

    // V_T: addr = (kk_idx*16 + ser_col) * 256 + j_idx*16 + ser_row
    // We load V_T such that reg[r][c] = V_T[kk*16+c, j*16+r] = V[j*16+r, kk*16+c]
    wire [13:0] vt_tile_addr = ({kk_idx, ser_col} * 256) + ({j_idx, ser_row});

    // O: addr = (i_idx*16 + row) * 64 + kk*16 + col
    // Used for serial load/store AND output_unit access
    // During output_unit operation, row comes from row_r, col from ou_bram_addr
    wire [13:0] o_serial_addr = ({i_idx, ser_row} * 64) + ({kk_idx, ser_col});

    // O addr when output_unit is driving (load/store/norm via output_unit)
    wire [13:0] o_ou_addr = ({i_idx, row_r} * 64) + ({kk_idx, ou_bram_addr});

    // O addr for rescale phases (different kk)
    wire [13:0] o_rescale_addr = ({i_idx, row_r} * 64) + ({rescale_kk, ou_bram_addr});

    // ========================================================================
    // Main FSM
    // ========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            i_idx      <= 0;
            j_idx      <= 0;
            kk_idx     <= 0;
            row_r      <= 0;
            col_c      <= 0;
            serial_cnt <= 0;
            wait_cnt   <= 0;
            rescale_kk <= 0;
        end else begin
            case (state)

                S_IDLE: begin
                    i_idx      <= 0;
                    j_idx      <= 0;
                    kk_idx     <= 0;
                    row_r      <= 0;
                    col_c      <= 0;
                    serial_cnt <= 0;
                    if (start)
                        state <= S_CLEAR_SIJ;
                end

                // ============================================================
                // TILE LOADING
                // ============================================================

                S_CLEAR_SIJ: begin
                    // 1 cycle, then start kk loop
                    kk_idx     <= 0;
                    serial_cnt <= 0;
                    state      <= S_LOAD_QI;
                end

                S_LOAD_QI: begin
                    // 256 serial reads from Q M10K
                    if (serial_cnt == 8'd255) begin
                        serial_cnt <= 0;
                        state      <= S_LOAD_KJ;
                    end else begin
                        serial_cnt <= serial_cnt + 8'd1;
                    end
                end

                S_LOAD_KJ: begin
                    // 256 serial reads from K M10K
                    if (serial_cnt == 8'd255) begin
                        serial_cnt <= 0;
                        row_r      <= 0;
                        col_c      <= 0;
                        state      <= S_A_GEMM;
                    end else begin
                        serial_cnt <= serial_cnt + 8'd1;
                    end
                end

                // ============================================================
                // PHASE A: S[i,j] += Qi[kk] × Kj[kk]^T
                // ============================================================

                // Compute dot(Qi[row_r,:], Kj[col_c,:]) — GEMM fires, wait for result
                S_A_GEMM: begin
                    // gemm_in_valid pulsed combinationally on first_cycle
                    wait_cnt <= 0;
                    state    <= S_A_GEMM_WAIT;
                end

                S_A_GEMM_WAIT: begin
                    if (gemm_out_valid) begin
                        state <= S_A_WRITE_SIJ;
                    end else begin
                        wait_cnt <= wait_cnt + 3'd1;
                    end
                end

                S_A_WRITE_SIJ: begin
                    // Accumulate: Sij[row_r, col_c] += gemm_out
                    // elem_we driven combinationally
                    if (col_c == 4'd15) begin
                        col_c <= 0;
                        if (row_r == 4'd15) begin
                            row_r <= 0;
                            // Done with this kk chunk
                            if (kk_idx == 2'd3) begin
                                // All kk done → Phase B
                                row_r <= 0;
                                state <= S_B_SOFTMAX;
                            end else begin
                                // Next kk chunk
                                kk_idx     <= kk_idx + 2'd1;
                                serial_cnt <= 0;
                                state      <= S_LOAD_QI;
                            end
                        end else begin
                            row_r <= row_r + 4'd1;
                            state <= S_A_GEMM;
                        end
                    end else begin
                        col_c <= col_c + 4'd1;
                        state <= S_A_GEMM;
                    end
                end

                // ============================================================
                // PHASE B: Online softmax + rescale O
                // ============================================================

                S_B_SOFTMAX: begin
                    // sm_start pulsed on first_cycle
                    // Softmax reads Sij[row_r,:] (truncated to Q8.8)
                    if (sm_done) begin
                        // Need to rescale O across ALL kk chunks
                        rescale_kk <= 0;
                        state      <= S_B_RESCALE_LOAD_O;
                    end
                end

                S_B_RESCALE_LOAD_O: begin
                    // Load O[i, row_r, rescale_kk] chunk via output_unit
                    // ou_cmd_load pulsed on first_cycle
                    if (ou_done)
                        state <= S_B_RESCALE;
                end

                S_B_RESCALE: begin
                    // ou_cmd_rescale pulsed on first_cycle
                    if (ou_done)
                        state <= S_B_RESCALE_STORE_O;
                end

                S_B_RESCALE_STORE_O: begin
                    // ou_cmd_store pulsed on first_cycle
                    if (ou_done) begin
                        if (rescale_kk == 2'd3) begin
                            // All kk chunks rescaled → write P̃
                            state <= S_B_WRITE_PTILDE;
                        end else begin
                            rescale_kk <= rescale_kk + 2'd1;
                            state      <= S_B_RESCALE_LOAD_O;
                        end
                    end
                end

                S_B_WRITE_PTILDE: begin
                    // Write P̃[row_r,:] to Sij_reg (1 cycle, row_we)
                    // Then advance to next row or Phase C
                    if (row_r == 4'd15) begin
                        // All rows softmaxed → Phase C
                        row_r  <= 0;
                        kk_idx <= 0;
                        serial_cnt <= 0;
                        state  <= S_LOAD_VJ;
                    end else begin
                        row_r <= row_r + 4'd1;
                        state <= S_B_SOFTMAX;
                    end
                end

                // ============================================================
                // PHASE C: O += P̃ × V
                // ============================================================

                S_LOAD_VJ: begin
                    // 256 serial reads from V_T M10K
                    if (serial_cnt == 8'd255) begin
                        serial_cnt <= 0;
                        state      <= S_LOAD_OI;
                    end else begin
                        serial_cnt <= serial_cnt + 8'd1;
                    end
                end

                S_LOAD_OI: begin
                    // Load Oi tile via output_unit (ou_cmd_load)
                    // Actually: output_unit loads one row at a time (16 elements)
                    // We need to load full 16×16 tile... output_unit only holds 1 row.
                    // So Phase C must be row-by-row:
                    //   for each row r: load o_row, GEMM 16 dots, acc, store o_row
                    // ou_cmd_load on first_cycle
                    if (ou_done) begin
                        col_c <= 0;
                        state <= S_C_GEMM;
                    end
                end

                S_C_GEMM: begin
                    // dot(P̃[row_r,:], Vj[col_c,:])
                    wait_cnt <= 0;
                    state    <= S_C_GEMM_WAIT;
                end

                S_C_GEMM_WAIT: begin
                    if (gemm_out_valid)
                        state <= S_C_ACC;
                    else
                        wait_cnt <= wait_cnt + 3'd1;
                end

                S_C_ACC: begin
                    // ou_cmd_acc pulsed combinationally
                    if (col_c == 4'd15) begin
                        // All columns done for this row → store
                        col_c <= 0;
                        state <= S_C_STORE_OI;
                    end else begin
                        col_c <= col_c + 4'd1;
                        state <= S_C_GEMM;
                    end
                end

                S_C_STORE_OI: begin
                    // ou_cmd_store on first_cycle
                    if (ou_done) begin
                        if (row_r == 4'd15) begin
                            // All rows done for this kk chunk
                            row_r <= 0;
                            if (kk_idx == 2'd3) begin
                                // All kk chunks done → next j or post-loop
                                kk_idx <= 0;
                                if (j_idx == 4'd15) begin
                                    // All j tiles done → post-loop
                                    j_idx  <= 0;
                                    row_r  <= 0;
                                    kk_idx <= 0;
                                    state  <= S_P_LOAD_O;
                                end else begin
                                    // Next j tile
                                    j_idx      <= j_idx + 4'd1;
                                    serial_cnt <= 0;
                                    state      <= S_CLEAR_SIJ;
                                end
                            end else begin
                                // Next kk chunk of V/O
                                kk_idx     <= kk_idx + 2'd1;
                                serial_cnt <= 0;
                                state      <= S_LOAD_VJ;
                            end
                        end else begin
                            // Next row
                            row_r <= row_r + 4'd1;
                            state <= S_LOAD_OI;
                        end
                    end
                end

                // ============================================================
                // POST-LOOP: O /= ℓ
                // ============================================================

                S_P_LOAD_O: begin
                    // Load o_row via output_unit
                    if (ou_done)
                        state <= S_P_RECIP;
                end

                S_P_RECIP: begin
                    // recip_start on first_cycle
                    if (recip_done)
                        state <= S_P_NORM;
                end

                S_P_NORM: begin
                    // ou_cmd_norm on first_cycle
                    if (ou_done)
                        state <= S_P_STORE_O;
                end

                S_P_STORE_O: begin
                    // ou_cmd_store on first_cycle
                    if (ou_done) begin
                        if (row_r == 4'd15) begin
                            row_r <= 0;
                            if (kk_idx == 2'd3) begin
                                // All kk chunks normalized
                                kk_idx <= 0;
                                if (i_idx == 4'd15) begin
                                    // ALL DONE
                                    state <= S_DONE;
                                end else begin
                                    // Next i block
                                    i_idx  <= i_idx + 4'd1;
                                    j_idx  <= 0;
                                    kk_idx <= 0;
                                    state  <= S_CLEAR_SIJ;
                                end
                            end else begin
                                kk_idx <= kk_idx + 2'd1;
                                row_r  <= 0;
                                state  <= S_P_LOAD_O;
                            end
                        end else begin
                            row_r <= row_r + 4'd1;
                            state <= S_P_LOAD_O;
                        end
                    end
                end

                S_DONE: begin
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;

            endcase
        end
    end

    // ========================================================================
    // Combinational outputs
    // ========================================================================

    // --- M10K addresses ---
    always_comb begin
        q_m10k_addr  = q_tile_addr;
        k_m10k_addr  = k_tile_addr;
        vt_m10k_addr = vt_tile_addr;
        o_m10k_addr  = 14'd0;
        o_m10k_we    = 1'b0;
        o_m10k_wdata = 32'sd0;

        // O M10K: mux between serial load/store and output_unit access
        case (state)
            S_B_RESCALE_LOAD_O, S_B_RESCALE, S_B_RESCALE_STORE_O: begin
                o_m10k_addr  = o_rescale_addr;
                o_m10k_we    = ou_bram_we;
                o_m10k_wdata = ou_bram_wdata;
            end
            S_LOAD_OI, S_C_GEMM, S_C_GEMM_WAIT, S_C_ACC, S_C_STORE_OI: begin
                o_m10k_addr  = o_ou_addr;
                o_m10k_we    = ou_bram_we;
                o_m10k_wdata = ou_bram_wdata;
            end
            S_P_LOAD_O, S_P_RECIP, S_P_NORM, S_P_STORE_O: begin
                o_m10k_addr  = o_ou_addr;
                o_m10k_we    = ou_bram_we;
                o_m10k_wdata = ou_bram_wdata;
            end
            default: ;
        endcase
    end

    // --- Qi tile reg ---
    always_comb begin
        qi_serial_we    = (state == S_LOAD_QI);
        qi_serial_row   = ser_row;
        qi_serial_col   = ser_col;
        qi_serial_wdata = q_m10k_rdata;
        qi_row_rd_idx   = row_r;
    end

    // --- Kj tile reg ---
    always_comb begin
        kj_serial_we    = (state == S_LOAD_KJ);
        kj_serial_row   = ser_row;
        kj_serial_col   = ser_col;
        kj_serial_wdata = k_m10k_rdata;
        kj_row_rd_idx   = col_c;      // reading row col_c of K for dot product
    end

    // --- Sij tile reg ---
    always_comb begin
        sij_clear       = (state == S_CLEAR_SIJ);
        sij_elem_we     = (state == S_A_WRITE_SIJ);
        sij_elem_row    = row_r;
        sij_elem_col    = col_c;
        // Accumulate: old value + new GEMM result
        sij_elem_wdata  = sij_row_rdata[col_c] + gemm_out_data;
        sij_row_rd_idx  = row_r;

        // P̃ writeback: convert P̃ (Q8.8 unsigned, 16b) to 32b for Sij reg
        sij_row_we      = (state == S_B_WRITE_PTILDE);
        sij_row_wr_idx  = row_r;
        for (int c = 0; c < BC; c++)
            sij_row_wdata[c] = {16'd0, sm_p_tilde[c]};  // zero-extend to 32b
    end

    // --- Vj tile reg ---
    always_comb begin
        vj_serial_we    = (state == S_LOAD_VJ);
        vj_serial_row   = ser_row;
        vj_serial_col   = ser_col;
        vj_serial_wdata = vt_m10k_rdata;
        vj_row_rd_idx   = col_c;      // reading row col_c of Vj_T for dot product
    end

    // --- GEMM engine ---
    always_comb begin
        for (int k = 0; k < D_TILE; k++) begin
            gemm_a[k] = 16'sd0;
            gemm_b[k] = 16'sd0;
        end
        gemm_in_valid = 1'b0;

        case (state)
            S_A_GEMM: begin
                for (int k = 0; k < D_TILE; k++) begin
                    gemm_a[k] = qi_row_rdata[k];   // Qi[row_r, :]
                    gemm_b[k] = kj_row_rdata[k];   // Kj[col_c, :]
                end
                gemm_in_valid = first_cycle;
            end
            S_C_GEMM: begin
                for (int k = 0; k < D_TILE; k++) begin
                    gemm_a[k] = sij_row_rdata[k][15:0];  // P̃[row_r, :] (lower 16b)
                    gemm_b[k] = vj_row_rdata[k];          // Vj_T[col_c, :]
                end
                gemm_in_valid = first_cycle;
            end
            default: ;
        endcase
    end

    // --- Softmax ---
    always_comb begin
        sm_start     = first_cycle && (state == S_B_SOFTMAX);
        sm_init_tile = first_cycle && (state == S_CLEAR_SIJ) && (j_idx == 4'd0);
        sm_row_idx   = row_r;
        // Truncate Sij (32b Q16.16) to Q8.8 for softmax input
        for (int c = 0; c < BC; c++)
            sm_score_row[c] = sij_row_rdata[c][23:8];  // Q16.16 → Q8.8
    end

    // --- Output unit ---
    always_comb begin
        ou_cmd_load    = first_cycle && (state == S_B_RESCALE_LOAD_O ||
                                         state == S_LOAD_OI ||
                                         state == S_P_LOAD_O);
        ou_cmd_rescale = first_cycle && (state == S_B_RESCALE);
        ou_cmd_acc     = (state == S_C_ACC);
        ou_cmd_store   = first_cycle && (state == S_B_RESCALE_STORE_O ||
                                         state == S_C_STORE_OI ||
                                         state == S_P_STORE_O);
        ou_cmd_norm    = first_cycle && (state == S_P_NORM);
        ou_alpha       = sm_alpha;
        ou_gemm_out    = gemm_out_data;
        ou_gemm_idx    = col_c;
        ou_recip_ell   = recip_out;
    end

    // --- Recip ---
    always_comb begin
        recip_start  = first_cycle && (state == S_P_RECIP);
        recip_ell_in = sm_ell_read;
    end

    // --- Status ---
    assign busy     = (state != S_IDLE);
    assign all_done = (state == S_DONE);

endmodule