# FlashAttention FPGA Accelerator — Comprehensive Architecture README

## 1. What This Is

A fully on-chip hardware implementation of the FlashAttention forward pass targeting the DE1-SoC FPGA (Intel Cyclone V 5CSEMA5F31C6). It computes exact scaled dot-product attention for a single attention head with sequence length 256 and head dimension 64:

```
O = softmax(Q × K^T) × V
```

where Q, K, V ∈ ℝ^(256×64) and O ∈ ℝ^(256×64).

No external memory access during compute. No DMA. All data lives in on-chip M10K BRAMs. Compute operates on 16×16 tiles loaded into register files.

---

## 2. Key Parameters

| Parameter  | Value | Meaning                                      |
|------------|-------|----------------------------------------------|
| SEQ_LEN    | 256   | Sequence length (number of tokens)            |
| D_FULL     | 64    | Full head dimension                           |
| BR         | 16    | Tile row size (query block rows)              |
| BC         | 16    | Tile column size (key block rows)             |
| D_TILE     | 16    | Tile feature chunk size                       |
| NUM_I      | 16    | Number of Q row blocks (SEQ_LEN / BR)         |
| NUM_J      | 16    | Number of K/V row blocks (SEQ_LEN / BC)       |
| NUM_KK     | 4     | Feature chunks per tile (D_FULL / D_TILE)     |

---

## 3. Fixed-Point Number Formats

All data uses fixed-point arithmetic. No floating point.

| Format  | Bits | Signed | Range (approx)     | Used for                           |
|---------|------|--------|--------------------|------------------------------------|
| Q7.8    | 16   | Yes    | ±127.996           | Q, K, V elements, scores, m[]      |
| Q8.8    | 16   | No     | 0 to 255.996       | alpha, P̃, ℓ_tile, exp2 outputs    |
| Q15.16  | 32   | Yes    | ±32767.99998       | GEMM outputs, Sij accumulator, O   |
| Q16.16  | 32   | No     | 0 to 65535.99998   | ℓ[] registers                      |
| Q0.16   | 16   | No     | 0 to 0.99998       | 1/ℓ (reciprocal output)            |

Note: "Q8.8 signed" is technically Q7.8 since the MSB is the sign bit. Comments in code sometimes say "Q8.8" loosely but the MSB is always the sign bit for signed values.

---

## 4. Memory Architecture

### 4.1 M10K BRAMs (Bulk Storage)

Full matrices live in M10K SRAMs. Single-port, 16b or 32b wide, serial access (one element per clock cycle).

| BRAM  | Stored As   | Word Width | Depth  | Addr Bits | M10K Blocks | Notes                    |
|-------|-------------|------------|--------|-----------|-------------|--------------------------|
| Q     | 256×64      | 16b        | 16,384 | 14        | ~26         | Row-major                |
| K     | 256×64      | 16b        | 16,384 | 14        | ~26         | Row-major                |
| V_T   | 64×256      | 16b        | 16,384 | 14        | ~26         | V stored **transposed**  |
| O     | 256×64      | 32b        | 16,384 | 14        | ~52         | Row-major, Q15.16        |
| **Total** |         |            |        |           | **~130**    | of 397 available (33%)   |

#### M10K Address Formulas

**Q and K** (row-major): `addr = row × 64 + col`

For tile block (i, kk) element (r, c):
```
addr = (i×16 + r) × 64 + kk×16 + c
```

**V_T** (transposed): V[row, col] stored at V_T[col, row]. `addr = col × 256 + row`

For tile block (kk, j) element (r, c) where we want V[j×16+r, kk×16+c]:
```
addr = (kk×16 + c) × 256 + j×16 + r
```

**O** (row-major, 32b): `addr = row × 64 + col`

Same tile indexing as Q/K.

### 4.2 Register Files (Active Working Tiles)

16×16 register arrays for parallel access during compute. Loaded serially from M10K (256 cycles per tile).

| Register File | Size   | Width | Total Bits | Purpose                                  |
|---------------|--------|-------|------------|------------------------------------------|
| Qi_reg        | 16×16  | 16b   | 4,096      | Current Q tile chunk                     |
| Kj_reg        | 16×16  | 16b   | 4,096      | Current K tile chunk                     |
| Vj_reg        | 16×16  | 16b   | 4,096      | Current V_T tile chunk                   |
| Sij_reg       | 16×16  | 32b   | 8,192      | Score accumulator (accumulates over kk)  |
| o_row         | 1×16   | 32b   | 512        | One O row (inside output_unit)           |
| m[16]         | 16     | 16b   | 256        | Running row-max (inside softmax_online)  |
| ℓ[16]         | 16     | 32b   | 512        | Running row-sum (inside softmax_online)  |
| **Total**     |        |       | **21,768** | ~22K FFs of ~64K available (34%)         |

Note: Oi is NOT a full 16×16 register file. The output_unit holds only one row (16 elements × 32b) at a time, loading/storing from O M10K per row. This is a deliberate choice to save registers.

---

## 5. Algorithm — FlashAttention Tiled Forward Pass

### 5.1 The Math

Standard attention: `O = softmax(S) × V` where `S = Q × K^T / √d`.

FlashAttention avoids materializing the full 256×256 S matrix by using tiled computation with online softmax. The key insight: softmax can be computed incrementally by tracking running statistics m (row max) and ℓ (row sum of exponentials).

For each new tile of scores, we:
1. Compute partial scores S_tile = Q_tile × K_tile^T
2. Update the running max: m_new = max(m_old, rowmax(S_tile))
3. Compute rescale factor: alpha = exp(m_old - m_new)
4. Compute P̃ = exp(S_tile - m_new)  (softmax numerators)
5. Update running sum: ℓ_new = alpha × ℓ_old + rowsum(P̃)
6. Rescale old output: O = alpha × O_old + P̃ × V_tile
7. After all tiles: O_final = O / ℓ  (final normalization)

### 5.2 Loop Structure

```
for i = 0 to 15:                         // Q row block (16 blocks of 16 rows)
  reset m[], ℓ[] to (-∞, 0)              // new query block starts fresh

  for j = 0 to 15:                       // K/V row block (16 blocks of 16 rows)

    // ─── Phase A: Compute Score Tile ───
    clear Sij_reg                         // 32b accumulator, zeroed before kk loop
    for kk = 0 to 3:                     // feature chunks (4 × 16 = 64)
      load Qi_reg ← M10K Q[i, kk]       // 256 serial reads
      load Kj_reg ← M10K K[j, kk]       // 256 serial reads
      for r = 0 to 15:
        for c = 0 to 15:
          Sij_reg[r,c] += dot(Qi_reg[r,:], Kj_reg[c,:])   // 16-MAC GEMM

    // Sij_reg now holds full Q[i]×K[j]^T in Q15.16
    // Truncate to Q7.8 for softmax input

    // ─── Phase B: Online Softmax ───
    for r = 0 to 15:
      scores[0:15] = Sij_reg[r, :] truncated to Q7.8
      softmax_online(scores, row_r) → alpha, P̃[0:15]

      // Rescale ALL O columns by alpha (4 kk passes through O M10K)
      for kk = 0 to 3:
        load o_row ← O M10K [i, r, kk]   // 16 elements, 32b each
        o_row *= alpha
        store o_row → O M10K [i, r, kk]

      write P̃ row back to Sij_reg        // overwrite scores with P̃

    // ─── Phase C: Output Accumulation ───
    for kk = 0 to 3:
      load Vj_reg ← M10K V_T[kk, j]     // 256 serial reads
      for r = 0 to 15:
        load o_row ← O M10K [i, r, kk]
        for c = 0 to 15:
          o_row[c] += dot(P̃_reg[r,:], Vj_reg[c,:])   // GEMM
        store o_row → O M10K [i, r, kk]

  // ─── Post-Loop: Final Normalization ───
  for kk = 0 to 3:
    for r = 0 to 15:
      load o_row ← O M10K [i, r, kk]
      recip = 1 / ℓ[r]                   // 16-cycle restoring divider
      o_row *= recip
      store o_row → O M10K [i, r, kk]
```

### 5.3 Why V Must Be Stored Transposed

The Phase C matmul computes O_col_k = dot(P̃[r,:], V[:,k]). This needs column k of V as a 16-element vector. In row-major storage, extracting a column requires strided access. By storing V transposed (V_T), column k of V becomes row k of V_T — a contiguous 16-element row that maps directly to the GEMM B-side input.

### 5.4 Why K Does NOT Need Transposing

S[r,c] = dot(Q[r,:], K[c,:]). We need row c of K. K stored row-major gives K[c,:] at contiguous addresses. No transposition needed.

---

## 6. Datapath Modules

### 6.1 `gemm_engine.sv` — 16-MAC Dot Product Unit

Computes `out = Σ(row_buf_a[k] × b_row[k])` for k=0..15.

- 16 parallel multipliers (Q7.8 × Q7.8 → Q15.16)
- 4-stage pipelined adder tree (16→8→4→2→1)
- **Latency**: 5 cycles
- **Throughput**: 1 dot product / cycle (after fill)
- **Mode-agnostic**: controller muxes which tile reg feeds A-side vs B-side

Interface:
```
in_valid, row_buf_a[16], b_row[16] → out_valid, out_data[31:0]
```

### 6.2 `exp2_unit.sv` — Exponential Approximation

Computes `exp(x) = 2^(x × log2(e))` for Q7.8 input, Q8.8 unsigned output.

- Piecewise polynomial: `2^f ≈ 1.0 + f×(C1 + f×C2)` for f ∈ [0,1)
- 4-stage pipeline, 3 DSP multiplies per unit
- Saturation: x ≥ 0 → 1.0, x < -10 → 0.0
- 16 instances in the softmax pipeline (one per tile column)

### 6.3 `softmax_online.sv` — Online Softmax Pipeline

Processes one score row (16 elements) per invocation. Owns the persistent m[] and ℓ[] state registers across tiles.

Sequence per row:
1. Latch inputs (score_row, row_idx)
2. Compute m_tile = rowmax(scores), m_new = max(m_old, m_tile)
3. alpha = exp2(m_old - m_new) via lane[0]
4. P̃[0:15] = exp2(score[c] - m_new) via all 16 lanes
5. ℓ_tile = rowsum(P̃)
6. ℓ_new = alpha × ℓ_old + ℓ_tile (internal update)
7. Commit m_new, ℓ_new; assert done

Submodules: `row_max_reduce`, `row_sum_reduce`, `exp2_unit[0:15]`

Key outputs consumed externally:
- `alpha_out` → output_unit for O rescaling
- `p_tilde[16]` → written to Sij_reg, later used as GEMM A-side in Phase C
- `ell_read` → read port for post-loop reciprocal computation

### 6.4 `output_unit.sv` — O Row Accumulator

Holds one row of O (16 × 32b) in the `o_row` register file. Five command-driven operations:

| Command      | Operation                    | Cycles | When                      |
|-------------|------------------------------|--------|---------------------------|
| cmd_load    | o_row ← O M10K              | 16     | Phase B (before rescale)  |
| cmd_rescale | o_row *= alpha               | 16     | Phase B (after softmax)   |
| cmd_acc     | o_row[idx] += gemm_out       | 1      | Phase C (per dot product) |
| cmd_store   | O M10K ← o_row              | 16     | Phase C end, post-loop    |
| cmd_norm    | o_row *= recip_ell           | 16     | Post-loop normalization   |

**Critical**: `cmd_acc` must NOT overlap with other commands (enforced by case statement priority in RTL).

The output_unit's BRAM interface (addr, we, wdata, rdata) connects to O M10K through the controller's address mux.

### 6.5 `recip_unit.sv` — Fixed-Point Reciprocal

Computes `1/ℓ` where ℓ is Q16.16 unsigned, output is Q0.16 unsigned.

- 16-cycle restoring divider: shift-subtract, one quotient bit per cycle
- Zero DSPs — one 33-bit subtractor
- Computes `2^32 / ell_in`, producing 16 quotient bits
- Only used in post-loop (16 times per i block), not on critical path

### 6.6 `tile_reg.sv` — Generic Tile Register File

16×16 parameterized-width register array. Supports:

| Operation       | Description                              | Cycles |
|-----------------|------------------------------------------|--------|
| clear           | Zero all entries                         | 1      |
| serial_we       | Write one element (from M10K load)       | 1/elem |
| row_rdata       | Read entire row (combinational)          | 0      |
| row_we          | Write entire row (P̃ writeback)          | 1      |
| elem_we         | Write single element (Sij accumulation)  | 1      |

Write priority: clear > row_we > serial_we > elem_we.

Instances:
- Qi_reg (16b): current Q tile chunk
- Kj_reg (16b): current K tile chunk
- Vj_reg (16b): current V_T tile chunk
- Sij_reg (32b): score accumulator

### 6.7 `row_max_reduce.sv` / `row_sum_reduce.sv`

Combinational trees (no latency):
- `row_max_reduce`: 16-input signed comparator tree → single max value
- `row_sum_reduce`: 16-input unsigned adder tree → single sum (max 16.0)

---

## 7. Controller FSM (`controller_fsm.sv`)

### 7.1 States

```
S_IDLE
├── S_CLEAR_SIJ          clear Sij accumulator
├── S_LOAD_QI             256 serial reads from Q M10K → Qi_reg
├── S_LOAD_KJ             256 serial reads from K M10K → Kj_reg
│
├── S_A_GEMM              fire GEMM (Qi row × Kj row)
├── S_A_GEMM_WAIT         wait 5 cycles for pipeline
├── S_A_WRITE_SIJ         accumulate result into Sij_reg
│
├── S_B_SOFTMAX           run softmax_online, wait for done
├── S_B_RESCALE_LOAD_O    load o_row chunk from O M10K
├── S_B_RESCALE           rescale o_row by alpha
├── S_B_RESCALE_STORE_O   store rescaled o_row
├── S_B_WRITE_PTILDE      write P̃ row to Sij_reg
│
├── S_LOAD_VJ             256 serial reads from V_T M10K → Vj_reg
├── S_LOAD_OI             load o_row from O M10K (via output_unit)
├── S_C_GEMM              fire GEMM (P̃ row × Vj row)
├── S_C_GEMM_WAIT         wait for pipeline
├── S_C_ACC               accumulate into o_row
├── S_C_STORE_OI          store o_row to O M10K
│
├── S_P_LOAD_O            post-loop: load o_row
├── S_P_RECIP             compute 1/ℓ[r]
├── S_P_NORM              normalize o_row
├── S_P_STORE_O           store normalized o_row
│
└── S_DONE                pulse all_done, return to IDLE
```

### 7.2 Loop Counter Variables

| Counter     | Range  | Purpose                                  |
|-------------|--------|------------------------------------------|
| i_idx       | 0..15  | Q row block                              |
| j_idx       | 0..15  | K/V row block                            |
| kk_idx      | 0..3   | Feature chunk                            |
| row_r       | 0..15  | Row within current tile                  |
| col_c       | 0..15  | Column within current tile               |
| serial_cnt  | 0..255 | M10K serial transfer counter             |
| rescale_kk  | 0..3   | kk sub-loop for Phase B O rescaling      |

### 7.3 Critical Timing Details

**`first_cycle` detection**: `state != state_prev`. Used to pulse single-cycle commands (sm_start, ou_cmd_load, etc.) on the first clock of each state.

**`sm_init_tile`**: Fires when entering S_CLEAR_SIJ with j_idx==0. This resets m[] and ℓ[] in the softmax module at the start of each new i block.

**Phase B rescaling**: After softmax produces alpha for row r, ALL 4 kk chunks of O must be rescaled. This is the rescale_kk sub-loop: load → rescale → store × 4. This is the most expensive part of Phase B per row.

**Sij accumulation**: elem_we writes `sij_row_rdata[col_c] + gemm_out_data` — a read-modify-write on the register file. The read (sij_row_rdata) is combinational, so the addition happens in the same cycle as the write. This works because tile_reg's elem_we path uses the combinational read output.

---

## 8. Top Level (`flash_attention_top.sv`)

### 8.1 Module Hierarchy

```
flash_attention_top
├── Q M10K BRAM (behavioral, 16384×16b)
├── K M10K BRAM (behavioral, 16384×16b)
├── V_T M10K BRAM (behavioral, 16384×16b)
├── O M10K BRAM (behavioral, 16384×32b)
│
├── tile_reg (Qi, 16×16×16b)
├── tile_reg (Kj, 16×16×16b)
├── tile_reg (Vj, 16×16×16b)
├── tile_reg (Sij, 16×16×32b)
│
├── gemm_engine (16 MACs, 5-stage pipeline)
│
├── softmax_online
│   ├── row_max_reduce
│   ├── row_sum_reduce
│   └── exp2_unit × 16
│
├── output_unit (o_row register + rescale/acc/norm)
├── recip_unit (16-cycle restoring divider)
│
└── controller_fsm (master sequencer)
```

### 8.2 External Interface

```systemverilog
// Control
input  start          // pulse to begin
output busy           // high during computation
output done           // pulses when complete

// M10K initialization (active only when !busy)
input  init_q_addr[14], init_q_data[16], init_q_we
input  init_k_addr[14], init_k_data[16], init_k_we
input  init_vt_addr[14], init_vt_data[16], init_vt_we

// O readback (active when done)
input  read_o_addr[14]
output read_o_data[32]
```

### 8.3 M10K Address Muxing

Q, K, V_T BRAMs: init ports when !busy, compute ports when busy.
O BRAM: controller muxes between multiple compute phases (rescale, Phase C, post-loop) based on current FSM state. The output_unit drives bram_addr/we/wdata, and the controller adds the row/kk offset to form the full M10K address.

---

## 9. File Inventory

### RTL Source Files

| File                    | Lines | Description                                |
|-------------------------|-------|--------------------------------------------|
| `flash_attention_top.sv`| ~310  | Top-level wiring and behavioral BRAMs      |
| `controller_fsm.sv`     | ~680  | Master sequencer FSM                       |
| `tile_reg.sv`           | ~90   | Generic 16×16 register file                |
| `gemm_engine.sv`        | ~100  | 16-MAC dot product pipeline                |
| `softmax_online.sv`     | ~300  | Online softmax with m/ℓ state              |
| `exp2_unit.sv`          | ~180  | Piecewise polynomial exp2 approximation    |
| `row_max_reduce.sv`     | ~30   | 16-input comparator tree                   |
| `row_sum_reduce.sv`     | ~40   | 16-input adder tree                        |
| `output_unit.sv`        | ~190  | O row accumulator with rescale/norm        |
| `recip_unit.sv`         | ~105  | 16-cycle restoring divider                 |

### Testbench Files

| File                       | Tests                                      |
|----------------------------|--------------------------------------------|
| `tb_exp2_unit.sv`          | exp2 accuracy across input range           |
| `tb_gemm_engine.sv`        | Dot product correctness and pipelining     |
| `tb_softmax_online.sv`     | Multi-tile online softmax (4 test cases)   |
| `tb_output_unit.sv`        | Load, rescale, acc, norm, full sequence    |
| `tb_recip_unit.sv`         | Reciprocal for power-of-2 and general ℓ   |
| `tb_flash_attention_top.sv`| Full integration with uniform Q/K/V       |

### Verified Modules (testbench passed)

- exp2_unit ✅
- gemm_engine ✅
- softmax_online ✅
- output_unit ✅
- recip_unit (pending your test run)

### Not Yet Verified

- tile_reg (new, untested)
- controller_fsm (new, untested)
- flash_attention_top (integration, untested)

---

## 10. FPGA Resource Estimates

**Cyclone V 5CSEMA5F31C6 (DE1-SoC)**:
- 32,070 ALMs (~64K FFs, ~64K LUTs)
- 397 M10K blocks (4,065,280 bits total)
- 87 DSP blocks (18×18 multipliers)

| Resource     | Used (est.)  | Available | Utilization |
|-------------|-------------|-----------|-------------|
| M10K blocks | ~130        | 397       | 33%         |
| FFs         | ~22K        | ~64K      | 34%         |
| DSPs        | ~19*        | 87        | 22%         |
| ALMs/LUTs   | ~8K (est.)  | 32,070    | 25%         |

*DSP estimate: 16 for GEMM multipliers, 3 for exp2 (shared across pipeline stages). The exp2 instances use DSPs for the polynomial stages but many can time-share since softmax processes one row at a time.

---

## 11. Estimated Cycle Counts

For one full attention computation (256×64):

| Phase              | Per Occurrence    | Occurrences           | Total Cycles (est.)  |
|--------------------|--------------------|----------------------|----------------------|
| Tile load (Q+K)    | 512 cy            | 16×16×4 = 1,024      | 524,288              |
| Phase A GEMM       | 16×16×6 = 1,536   | 16×16×4 = 1,024      | 1,572,864            |
| Phase B softmax    | ~12 cy/row         | 16×16×16 = 4,096     | 49,152               |
| Phase B rescale    | 4×48 = 192 cy/row | 16×16×16 = 4,096     | 786,432              |
| Phase B P̃ write   | 1 cy/row           | 4,096                | 4,096                |
| Tile load (V)      | 256 cy             | 16×16×4 = 1,024      | 262,144              |
| Phase C GEMM+acc   | 16×16×6 = 1,536   | 16×16×4 = 1,024      | 1,572,864            |
| Phase C O load/store| 32 cy/row         | 16×16×4×16 = 16,384  | 524,288              |
| Post-loop          | ~64 cy/row         | 16×4×16 = 1,024      | 65,536               |
| **Rough total**    |                    |                      | **~5.4M cycles**     |

At 50 MHz: ~108 ms. This is a non-optimized first implementation. Major optimization opportunities exist (pipelining tile loads with compute, wider M10K access, double-buffering).

---

## 12. Known Limitations and Future Work

### Current Limitations
1. **Serial M10K access**: Loading a 16×16 tile takes 256 cycles. This dominates total runtime.
2. **Single GEMM fire per dot product**: No pipelining of consecutive dot products — each waits for the full 5-cycle latency.
3. **Phase B rescale is expensive**: 4 O M10K load-rescale-store passes per row per j tile.
4. **Behavioral M10K**: Top-level BRAMs are behavioral arrays for simulation. Must be replaced with Quartus M10K IP for synthesis.
5. **No scaling by 1/√d**: The score computation omits the 1/√d scaling factor. Can be absorbed into Q or K during initialization.

### Optimization Opportunities
1. **Wider M10K**: Configure M10K as 128b or 256b wide to load multiple elements per cycle.
2. **Double-buffering**: Load next tile while computing current tile.
3. **GEMM pipelining**: Stream consecutive dot products without waiting for drain.
4. **Phase B/C overlap**: Start Phase C while Phase B is still processing later rows.
5. **Register-based Oi**: If FFs allow, keep full 16×16 O tile in registers to avoid per-row M10K access in Phase C.

### Path to Synthesis
1. Run `tb_flash_attention_top` in ModelSim — fix compilation errors and functional bugs
2. Replace behavioral BRAMs with Quartus M10K IP instances
3. Add pin assignments for DE1-SoC (at minimum: clk, rst, start, done, debug LEDs)
4. Add a simple loader (JTAG or UART) to write Q/K/V into M10K from a host
5. Synthesize, check timing at target frequency
6. Verify on hardware with known test vectors
