// ============================================================================
// tile_reg.sv — 16×16 Tile Register File
// ============================================================================
// Generic register file for one 16×16 working tile.
// Supports:
//   - Serial load from M10K: 256 cycles, one element per cycle
//   - Serial store to M10K: 256 cycles, one element per cycle
//   - Parallel row read:  given row index, output 16 elements combinationally
//   - Parallel row write: given row index, write 16 elements in 1 cycle
//   - Single-element write: given (row, col), write 1 element
//   - Clear: zero all entries in 1 cycle
//
// Parameterized width (16b for Q/K/V/S, 32b for O/S_accum).
// ============================================================================

module tile_reg #(
    parameter WIDTH = 16,       // element width in bits
    parameter ROWS  = 16,
    parameter COLS  = 16
)(
    input  logic clk,
    input  logic rst_n,

    // --- Clear ---
    input  logic clear,                              // zero all entries, 1 cycle

    // --- Serial load from M10K (256 cycles) ---
    input  logic                   serial_we,        // write one element per cycle
    input  logic [3:0]             serial_row,
    input  logic [3:0]             serial_col,
    input  logic signed [WIDTH-1:0] serial_wdata,

    // --- Serial read to M10K (256 cycles) ---
    input  logic [3:0]             serial_rd_row,
    input  logic [3:0]             serial_rd_col,
    output logic signed [WIDTH-1:0] serial_rdata,

    // --- Parallel row read (combinational) ---
    input  logic [3:0]             row_rd_idx,
    output logic signed [WIDTH-1:0] row_rdata [COLS],

    // --- Parallel row write (1 cycle) ---
    input  logic                   row_we,
    input  logic [3:0]             row_wr_idx,
    input  logic signed [WIDTH-1:0] row_wdata [COLS],

    // --- Single-element write ---
    input  logic                   elem_we,
    input  logic [3:0]             elem_row,
    input  logic [3:0]             elem_col,
    input  logic signed [WIDTH-1:0] elem_wdata
);

    // ========================================================================
    // Storage
    // ========================================================================
    logic signed [WIDTH-1:0] mem [ROWS][COLS];

    // ========================================================================
    // Write logic (priority: clear > row_we > serial_we > elem_we)
    // ========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int r = 0; r < ROWS; r++)
                for (int c = 0; c < COLS; c++)
                    mem[r][c] <= '0;
        end else if (clear) begin
            for (int r = 0; r < ROWS; r++)
                for (int c = 0; c < COLS; c++)
                    mem[r][c] <= '0;
        end else if (row_we) begin
            for (int c = 0; c < COLS; c++)
                mem[row_wr_idx][c] <= row_wdata[c];
        end else if (serial_we) begin
            mem[serial_row][serial_col] <= serial_wdata;
        end else if (elem_we) begin
            mem[elem_row][elem_col] <= elem_wdata;
        end
    end

    // ========================================================================
    // Read logic (combinational)
    // ========================================================================

    // Parallel row read
    always_comb begin
        for (int c = 0; c < COLS; c++)
            row_rdata[c] = mem[row_rd_idx][c];
    end

    // Serial element read
    assign serial_rdata = mem[serial_rd_row][serial_rd_col];

endmodule
